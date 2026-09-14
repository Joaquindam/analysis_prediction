"""
Temporal modeling foundation for article-period demand forecasting.

Pipeline position:

    Raw -> Bronze -> Silver -> Gold -> EDA -> Features -> Modeling

This modeling layer combines a strict evaluation protocol with a small,
general candidate set. It provides:

1. automatic leakage-aware predictor selection from feature_catalogue.csv;
2. a final temporal holdout isolated before cross-validation;
3. expanding-window cross-validation performed only on development periods;
4. causal baseline forecasting and common demand metrics;
5. HistGradientBoosting Poisson and Random Forest candidates;
6. a sequential out-of-fold adaptive selector by historical demand regime;
7. diagnostics by fold, period and historical demand regime.

The final holdout is NOT evaluated by default. Baselines and machine-learning
candidates are first compared on identical development cross-validation folds.
Only the selected competitor may later be evaluated once on the isolated test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math
import re
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ModelingPaths:
    """Output locations for modeling diagnostics and predictions."""

    tables_dir: str | Path
    reports_dir: str | Path
    predictions_dir: str | Path

    def __post_init__(self) -> None:
        self.tables_dir = Path(self.tables_dir)
        self.reports_dir = Path(self.reports_dir)
        self.predictions_dir = Path(self.predictions_dir)


@dataclass
class ModelingRules:
    """General temporal-modeling policy for article-period demand."""

    dataset_name: str = 'features_article_period'
    target_candidates: tuple[str, ...] = ('units', 'amount')
    training_eligibility_column: str = 'training_eligible'

    entity_column_candidates: tuple[str, ...] = ('article_code',)
    period_index_candidates: tuple[str, ...] = (
        'period_index',
        'period_id',
        'report_start',
    )
    period_start_candidates: tuple[str, ...] = ('report_start',)
    period_end_candidates: tuple[str, ...] = ('report_end',)
    regime_column_candidates: tuple[str, ...] = (
        'article_demand_regime_prior',
    )

    use_safe_default_features_only: bool = True
    drop_constant_features: bool = True
    drop_all_missing_features: bool = True
    max_development_missing_fraction: float = 0.98

    final_test_fraction: float = 0.15
    min_final_test_periods: int = 2

    cv_splits: int = 4
    cv_validation_fraction: float = 0.10
    min_cv_validation_periods: int = 1
    min_train_periods: int = 8

    top_k: int = 20
    nonnegative_predictions: bool = True

    # Baseline policy. The module discovers only baselines supported by the
    # feature table. Missing baseline history receives a train-only median
    # fallback so that cold-start rows remain evaluable without future leakage.
    baseline_use_lag_1: bool = True
    baseline_use_shortest_rolling_mean: bool = True
    baseline_use_expanding_mean: bool = True
    baseline_train_median_fallback: bool = True

    # Machine-learning candidates. Hyperparameters are deliberately modest:
    # the objective is robust temporal comparison, not exhaustive tuning.
    run_ml_candidates: bool = True
    use_hist_gradient_boosting_poisson: bool = True
    use_random_forest: bool = True
    random_state: int = 42

    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 250
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 1.0

    rf_n_estimators: int = 300
    rf_min_samples_leaf: int = 2
    rf_max_features: str | float | int | None = 'sqrt'
    rf_max_depth: int | None = None
    rf_n_jobs: int = -1

    # Sequential adaptive selector. For each validation fold, the model used
    # for a demand regime is chosen only from OOF evidence produced by earlier
    # validation folds. Fold 1 therefore uses a causal baseline fallback.
    run_adaptive_regime_selector: bool = True
    adaptive_min_prior_regime_rows: int = 20
    adaptive_min_prior_global_rows: int = 50
    adaptive_initial_baseline_kind: str = 'expanding'

    # Protect the final holdout during model development.
    evaluate_final_test: bool = False

    save_tables: bool = True
    save_predictions: bool = True
    save_report: bool = True

    def validate(self) -> None:
        if not 0 < self.final_test_fraction < 1:
            raise ValueError('final_test_fraction must be between 0 and 1.')

        if not 0 < self.cv_validation_fraction < 1:
            raise ValueError('cv_validation_fraction must be between 0 and 1.')

        if self.min_final_test_periods < 1:
            raise ValueError('min_final_test_periods must be >= 1.')

        if self.cv_splits < 1:
            raise ValueError('cv_splits must be >= 1.')

        if self.min_cv_validation_periods < 1:
            raise ValueError('min_cv_validation_periods must be >= 1.')

        if self.min_train_periods < 2:
            raise ValueError('min_train_periods must be >= 2.')

        if not 0 <= self.max_development_missing_fraction <= 1:
            raise ValueError(
                'max_development_missing_fraction must be between 0 and 1.'
            )

        if self.top_k < 1:
            raise ValueError('top_k must be >= 1.')

        if self.hgb_learning_rate <= 0:
            raise ValueError('hgb_learning_rate must be > 0.')

        if self.hgb_max_iter < 1:
            raise ValueError('hgb_max_iter must be >= 1.')

        if self.hgb_max_leaf_nodes < 2:
            raise ValueError('hgb_max_leaf_nodes must be >= 2.')

        if self.hgb_min_samples_leaf < 1:
            raise ValueError('hgb_min_samples_leaf must be >= 1.')

        if self.hgb_l2_regularization < 0:
            raise ValueError('hgb_l2_regularization must be >= 0.')

        if self.rf_n_estimators < 1:
            raise ValueError('rf_n_estimators must be >= 1.')

        if self.rf_min_samples_leaf < 1:
            raise ValueError('rf_min_samples_leaf must be >= 1.')

        if self.adaptive_min_prior_regime_rows < 1:
            raise ValueError(
                'adaptive_min_prior_regime_rows must be >= 1.'
            )

        if self.adaptive_min_prior_global_rows < 1:
            raise ValueError(
                'adaptive_min_prior_global_rows must be >= 1.'
            )


# =============================================================================
# CONSOLE / GENERIC HELPERS
# =============================================================================


def _print_header(title: str, verbose: bool = True) -> None:
    if not verbose:
        return

    line = '=' * 96
    print(f'\n{line}')
    print(title)
    print(line)


def _print_subheader(title: str, verbose: bool = True) -> None:
    if verbose:
        print(f'\n--- {title} ---')


def _print_message(
    message: str,
    level: str = 'INFO',
    verbose: bool = True,
) -> None:
    if verbose:
        print(f'[{level}] {message}')


def _ensure_directory(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_existing(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    return next(
        (column for column in candidates if column in dataframe.columns),
        None,
    )


def _safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype('boolean')

    lowered = series.astype('string').str.strip().str.lower()

    mapping = {
        'true': True,
        '1': True,
        'yes': True,
        'si': True,
        'sí': True,
        'false': False,
        '0': False,
        'no': False,
    }

    return lowered.map(mapping).astype('boolean')


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce')


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


# =============================================================================
# FEATURE CATALOGUE
# =============================================================================


def load_feature_catalogue(path: str | Path) -> pd.DataFrame:
    """Load and validate the feature catalogue produced by ap_features.py."""
    catalogue = pd.read_csv(path)

    required = {
        'dataset',
        'column',
        'feature_dtype',
        'role',
        'availability',
        'safe_default',
    }

    missing = required.difference(catalogue.columns)

    if missing:
        raise ValueError(
            'Feature catalogue is missing required columns: '
            f'{sorted(missing)}'
        )

    catalogue = catalogue.copy()
    catalogue['safe_default'] = _safe_bool(
        catalogue['safe_default']
    ).fillna(False)

    return catalogue


def _select_catalogue_rows(
    feature_table: pd.DataFrame,
    catalogue: pd.DataFrame,
    rules: ModelingRules,
) -> pd.DataFrame:
    subset = catalogue.loc[
        catalogue['dataset'].eq(rules.dataset_name)
    ].copy()

    if subset.empty:
        raise ValueError(
            f'No catalogue rows found for dataset {rules.dataset_name!r}.'
        )

    subset = subset.loc[
        subset['column'].isin(feature_table.columns)
    ].copy()

    if rules.use_safe_default_features_only:
        subset = subset.loc[
            subset['safe_default'].eq(True)
        ].copy()

    return subset.reset_index(drop=True)


def _infer_categorical_mask(catalogue_rows: pd.DataFrame) -> pd.Series:
    categorical_roles = {
        'categorical_identifier',
        'adaptive_demand_regime',
        'static_or_scheduled_metadata',
        'known_context',
    }

    return (
        catalogue_rows['feature_dtype'].eq('categorical_or_text')
        | catalogue_rows['feature_dtype'].eq('boolean')
        | catalogue_rows['role'].isin(categorical_roles)
    )


def select_model_features(
    feature_table: pd.DataFrame,
    catalogue: pd.DataFrame,
    development_frame: pd.DataFrame,
    rules: ModelingRules,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Select modeling predictors from catalogue metadata and development data.

    The final holdout is not used for missingness or constant-feature pruning.
    """
    selected = _select_catalogue_rows(
        feature_table,
        catalogue,
        rules,
    )

    diagnostics: list[dict[str, Any]] = []
    kept_rows: list[pd.Series] = []

    for _, row in selected.iterrows():
        column = str(row['column'])
        values = development_frame[column]

        missing_fraction = float(values.isna().mean())
        non_null = values.dropna()
        unique_non_null = int(non_null.nunique(dropna=True))

        reason = 'selected'
        keep = True

        if rules.drop_all_missing_features and non_null.empty:
            keep = False
            reason = 'all_missing_in_development'

        elif missing_fraction > rules.max_development_missing_fraction:
            keep = False
            reason = 'too_missing_in_development'

        elif rules.drop_constant_features and unique_non_null <= 1:
            keep = False
            reason = 'constant_in_development'

        diagnostics.append(
            {
                'column': column,
                'feature_dtype': row['feature_dtype'],
                'role': row['role'],
                'availability': row['availability'],
                'missing_fraction_development': missing_fraction,
                'unique_non_null_development': unique_non_null,
                'selected_for_modeling': keep,
                'selection_reason': reason,
            }
        )

        if keep:
            kept_rows.append(row)

    diagnostics_frame = pd.DataFrame(diagnostics)

    if not kept_rows:
        raise ValueError(
            'No model features remain after leakage and development-data '
            'selection rules.'
        )

    kept = pd.DataFrame(kept_rows)
    categorical_mask = _infer_categorical_mask(kept)

    categorical = kept.loc[
        categorical_mask,
        'column',
    ].astype(str).tolist()

    numeric = kept.loc[
        ~categorical_mask,
        'column',
    ].astype(str).tolist()

    return categorical, numeric, diagnostics_frame


# =============================================================================
# TEMPORAL SPLITS
# =============================================================================


def _period_table(
    frame: pd.DataFrame,
    period_column: str,
    start_column: str | None,
    end_column: str | None,
) -> pd.DataFrame:
    columns = [period_column]

    if start_column is not None and start_column not in columns:
        columns.append(start_column)

    if end_column is not None and end_column not in columns:
        columns.append(end_column)

    periods = frame[columns].drop_duplicates().copy()

    if start_column is not None:
        periods[start_column] = _to_datetime(periods[start_column])
        periods = periods.sort_values(
            [start_column, period_column]
        )
    else:
        periods = periods.sort_values(period_column)

    return periods.reset_index(drop=True)


def build_final_holdout_split(
    frame: pd.DataFrame,
    period_column: str,
    start_column: str | None,
    end_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the latest eligible periods as an untouched final holdout."""
    periods = _period_table(
        frame,
        period_column=period_column,
        start_column=start_column,
        end_column=end_column,
    )

    n_periods = len(periods)

    requested_test = max(
        rules.min_final_test_periods,
        int(math.ceil(n_periods * rules.final_test_fraction)),
    )

    max_test = n_periods - (
        rules.min_train_periods
        + rules.min_cv_validation_periods
    )

    if max_test < 1:
        raise ValueError(
            'Too few eligible periods to create a development set and an '
            'isolated final holdout.'
        )

    n_test = min(requested_test, max_test)

    test_periods = periods.iloc[-n_test:][period_column].tolist()
    development_periods = periods.iloc[:-n_test][period_column].tolist()

    development = frame.loc[
        frame[period_column].isin(development_periods)
    ].copy()

    final_test = frame.loc[
        frame[period_column].isin(test_periods)
    ].copy()

    manifest = periods.copy()
    manifest['split'] = np.where(
        manifest[period_column].isin(test_periods),
        'final_test',
        'development',
    )

    if set(development_periods).intersection(test_periods):
        raise RuntimeError('Temporal split overlap detected.')

    return development, final_test, manifest


def build_expanding_cv_splits(
    development_frame: pd.DataFrame,
    period_column: str,
    start_column: str | None,
    end_column: str | None,
    rules: ModelingRules,
) -> list[dict[str, Any]]:
    """Create expanding-window folds entirely inside development periods."""
    periods = _period_table(
        development_frame,
        period_column=period_column,
        start_column=start_column,
        end_column=end_column,
    )

    n_periods = len(periods)
    validation_size = max(
        rules.min_cv_validation_periods,
        int(math.ceil(n_periods * rules.cv_validation_fraction)),
    )

    max_splits = (
        n_periods - rules.min_train_periods
    ) // validation_size

    actual_splits = min(
        rules.cv_splits,
        max_splits,
    )

    if actual_splits < 1:
        raise ValueError(
            'Too few development periods for the requested expanding-window '
            'cross-validation policy.'
        )

    first_validation_position = (
        n_periods
        - actual_splits * validation_size
    )

    folds: list[dict[str, Any]] = []

    for fold_index in range(actual_splits):
        validation_start = (
            first_validation_position
            + fold_index * validation_size
        )
        validation_end = validation_start + validation_size

        train_ids = periods.iloc[
            :validation_start
        ][period_column].tolist()

        validation_ids = periods.iloc[
            validation_start:validation_end
        ][period_column].tolist()

        if len(train_ids) < rules.min_train_periods:
            raise RuntimeError(
                'Internal CV construction error: a fold has too little '
                'training history.'
            )

        if set(train_ids).intersection(validation_ids):
            raise RuntimeError('Cross-validation period overlap detected.')

        folds.append(
            {
                'fold': fold_index + 1,
                'train_period_ids': train_ids,
                'validation_period_ids': validation_ids,
            }
        )

    return folds


def _cv_manifest(
    folds: list[dict[str, Any]],
    period_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for fold in folds:
        for period_id in fold['train_period_ids']:
            rows.append(
                {
                    'fold': fold['fold'],
                    period_column: period_id,
                    'role': 'train',
                }
            )

        for period_id in fold['validation_period_ids']:
            rows.append(
                {
                    'fold': fold['fold'],
                    period_column: period_id,
                    'role': 'validation',
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# METRICS
# =============================================================================


def wape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(true).sum()

    if denominator == 0:
        return np.nan

    return float(
        np.abs(true - pred).sum()
        / denominator
    )


def relative_bias(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(true).sum()

    if denominator == 0:
        return np.nan

    return float(
        (pred - true).sum()
        / denominator
    )


def top_k_overlap(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    period_column: str,
    entity_column: str,
    k: int,
) -> float:
    scores: list[float] = []

    for _, group in frame.groupby(period_column, dropna=False):
        valid = group.dropna(
            subset=[target_column, prediction_column, entity_column]
        )

        if valid.empty:
            continue

        local_k = min(k, len(valid))

        actual = set(
            valid.nlargest(local_k, target_column)[entity_column]
        )
        predicted = set(
            valid.nlargest(local_k, prediction_column)[entity_column]
        )

        if not actual:
            continue

        scores.append(
            len(actual.intersection(predicted))
            / local_k
        )

    if not scores:
        return np.nan

    return float(np.mean(scores))


def evaluate_predictions(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    period_column: str,
    entity_column: str | None,
    top_k: int,
) -> dict[str, Any]:
    valid = frame.dropna(
        subset=[target_column, prediction_column]
    ).copy()

    n_total = len(frame)
    n_evaluated = len(valid)

    if n_evaluated == 0:
        return {
            'n_rows': n_total,
            'n_evaluated': 0,
            'prediction_coverage': 0.0,
            'MAE': np.nan,
            'RMSE': np.nan,
            'WAPE': np.nan,
            'Bias': np.nan,
            'actual_total': np.nan,
            'predicted_total': np.nan,
            f'Top_{top_k}_overlap_observed_panel': np.nan,
        }

    true = _to_numeric(valid[target_column]).to_numpy(dtype=float)
    pred = _to_numeric(valid[prediction_column]).to_numpy(dtype=float)

    error = pred - true

    metrics: dict[str, Any] = {
        'n_rows': n_total,
        'n_evaluated': n_evaluated,
        'prediction_coverage': n_evaluated / n_total if n_total else np.nan,
        'MAE': float(np.mean(np.abs(error))),
        'RMSE': float(np.sqrt(np.mean(np.square(error)))),
        'WAPE': wape(true, pred),
        'Bias': relative_bias(true, pred),
        'actual_total': float(np.sum(true)),
        'predicted_total': float(np.sum(pred)),
    }

    if entity_column is not None and entity_column in valid.columns:
        metrics[f'Top_{top_k}_overlap_observed_panel'] = top_k_overlap(
            valid,
            target_column=target_column,
            prediction_column=prediction_column,
            period_column=period_column,
            entity_column=entity_column,
            k=top_k,
        )
    else:
        metrics[f'Top_{top_k}_overlap_observed_panel'] = np.nan

    return metrics


# =============================================================================
# CAUSAL BASELINES
# =============================================================================


def discover_baseline_specs(
    frame: pd.DataFrame,
    target_column: str,
    rules: ModelingRules,
) -> list[dict[str, str]]:
    """Discover baseline features available for the selected target."""
    specs: list[dict[str, str]] = []

    lag_1 = f'{target_column}__lag_1'

    if rules.baseline_use_lag_1 and lag_1 in frame.columns:
        specs.append(
            {
                'model': 'Naive lag 1',
                'source_column': lag_1,
                'kind': 'lag',
            }
        )

    if rules.baseline_use_shortest_rolling_mean:
        rolling_candidates: list[tuple[int, str]] = []
        pattern = re.compile(
            rf'^{re.escape(target_column)}__rolling_mean_(\d+)$'
        )

        for column in frame.columns:
            match = pattern.match(str(column))

            if match is not None:
                rolling_candidates.append(
                    (int(match.group(1)), str(column))
                )

        if rolling_candidates:
            window, column = min(rolling_candidates)
            specs.append(
                {
                    'model': f'Rolling mean {window}',
                    'source_column': column,
                    'kind': 'rolling',
                }
            )

    expanding = f'{target_column}__expanding_mean_prior'

    if (
        rules.baseline_use_expanding_mean
        and expanding in frame.columns
    ):
        specs.append(
            {
                'model': 'Historical expanding mean',
                'source_column': expanding,
                'kind': 'expanding',
            }
        )

    if not specs:
        raise ValueError(
            'No supported causal baseline features were found in the feature '
            'table.'
        )

    return specs


def _baseline_predictions(
    train_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    target_column: str,
    source_column: str,
    rules: ModelingRules,
) -> tuple[pd.Series, pd.Series, float | None]:
    prediction = _to_numeric(
        prediction_frame[source_column]
    ).copy()
    used_fallback = prediction.isna()

    fallback_value: float | None = None

    if rules.baseline_train_median_fallback and used_fallback.any():
        train_target = _to_numeric(
            train_frame[target_column]
        ).dropna()

        if train_target.empty:
            raise ValueError(
                'Cannot compute baseline fallback: training target is empty.'
            )

        fallback_value = float(train_target.median())
        prediction = prediction.fillna(fallback_value)

    if rules.nonnegative_predictions:
        prediction = prediction.clip(lower=0)

    return prediction, used_fallback.astype('boolean'), fallback_value


def cross_validate_baselines(
    development_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    baseline_specs: list[dict[str, str]],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    for fold in folds:
        train = development_frame.loc[
            development_frame[period_column].isin(
                fold['train_period_ids']
            )
        ].copy()

        validation = development_frame.loc[
            development_frame[period_column].isin(
                fold['validation_period_ids']
            )
        ].copy()

        for spec in baseline_specs:
            pred, fallback_mask, fallback_value = _baseline_predictions(
                train,
                validation,
                target_column=target_column,
                source_column=spec['source_column'],
                rules=rules,
            )

            scored = validation.copy()
            scored['prediction'] = pred
            scored['used_fallback'] = fallback_mask

            metrics = evaluate_predictions(
                scored,
                target_column=target_column,
                prediction_column='prediction',
                period_column=period_column,
                entity_column=entity_column,
                top_k=rules.top_k,
            )

            metrics_rows.append(
                {
                    'fold': fold['fold'],
                    'model': spec['model'],
                    'source_column': spec['source_column'],
                    'train_periods': len(fold['train_period_ids']),
                    'validation_periods': len(
                        fold['validation_period_ids']
                    ),
                    'fallback_value': fallback_value,
                    'fallback_fraction': float(fallback_mask.mean()),
                    **metrics,
                }
            )

            prediction_columns = [
                period_column,
                target_column,
                'prediction',
                'used_fallback',
            ]

            for optional in [
                entity_column,
                regime_column,
                'report_start',
                'report_end',
            ]:
                if (
                    optional is not None
                    and optional in scored.columns
                    and optional not in prediction_columns
                ):
                    prediction_columns.append(optional)

            predictions = scored[prediction_columns].copy()
            predictions.insert(0, 'model', spec['model'])
            predictions.insert(0, 'fold', fold['fold'])
            prediction_frames.append(predictions)

            if regime_column is not None and regime_column in scored.columns:
                for regime, group in scored.groupby(
                    regime_column,
                    dropna=False,
                ):
                    regime_metrics = evaluate_predictions(
                        group,
                        target_column=target_column,
                        prediction_column='prediction',
                        period_column=period_column,
                        entity_column=entity_column,
                        top_k=rules.top_k,
                    )

                    regime_rows.append(
                        {
                            'fold': fold['fold'],
                            'model': spec['model'],
                            'regime': regime,
                            **regime_metrics,
                        }
                    )

            for period_id, group in scored.groupby(
                period_column,
                dropna=False,
            ):
                period_metrics = evaluate_predictions(
                    group,
                    target_column=target_column,
                    prediction_column='prediction',
                    period_column=period_column,
                    entity_column=entity_column,
                    top_k=rules.top_k,
                )

                period_rows.append(
                    {
                        'fold': fold['fold'],
                        'model': spec['model'],
                        period_column: period_id,
                        **period_metrics,
                    }
                )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    regime_frame = pd.DataFrame(regime_rows)
    period_frame = pd.DataFrame(period_rows)

    return (
        metrics_frame,
        predictions_frame,
        regime_frame,
        period_frame,
    )


def summarize_cv_metrics(
    metrics: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    top_column = f'Top_{top_k}_overlap_observed_panel'

    summary = (
        metrics
        .groupby('model', as_index=False)
        .agg(
            folds=('fold', 'nunique'),
            MAE_mean=('MAE', 'mean'),
            MAE_std=('MAE', 'std'),
            RMSE_mean=('RMSE', 'mean'),
            RMSE_std=('RMSE', 'std'),
            WAPE_mean=('WAPE', 'mean'),
            WAPE_std=('WAPE', 'std'),
            Bias_mean=('Bias', 'mean'),
            Bias_std=('Bias', 'std'),
            prediction_coverage_mean=('prediction_coverage', 'mean'),
            fallback_fraction_mean=('fallback_fraction', 'mean'),
            TopK_mean=(top_column, 'mean'),
            TopK_std=(top_column, 'std'),
        )
        .sort_values(
            ['WAPE_mean', 'MAE_mean'],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    summary.insert(
        0,
        'cv_rank',
        np.arange(1, len(summary) + 1),
    )

    return summary



# =============================================================================
# MACHINE-LEARNING CANDIDATES
# =============================================================================


def discover_ml_candidate_specs(
    development_frame: pd.DataFrame,
    target_column: str,
    rules: ModelingRules,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Discover model candidates supported by the current target.

    Poisson boosting is enabled only for non-negative targets. Unsupported
    candidates are reported rather than failing silently.
    """
    specs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    target = _to_numeric(
        development_frame[target_column]
    ).dropna()

    if rules.use_hist_gradient_boosting_poisson:
        if target.empty:
            skipped.append(
                {
                    'model': 'HistGradientBoosting Poisson',
                    'reason': 'development_target_empty',
                }
            )
        elif bool((target < 0).any()):
            skipped.append(
                {
                    'model': 'HistGradientBoosting Poisson',
                    'reason': 'negative_target_values_present',
                }
            )
        elif float(target.sum()) <= 0:
            skipped.append(
                {
                    'model': 'HistGradientBoosting Poisson',
                    'reason': 'non_positive_target_sum',
                }
            )
        else:
            specs.append(
                {
                    'model': 'HistGradientBoosting Poisson',
                    'family': 'hist_gradient_boosting',
                    'loss': 'poisson',
                }
            )

    if rules.use_random_forest:
        specs.append(
            {
                'model': 'Random Forest',
                'family': 'random_forest',
                'loss': 'squared_error',
            }
        )

    return specs, skipped


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OHE with compatibility across sklearn versions."""
    try:
        return OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown='ignore',
            sparse=False,
            dtype=np.float32,
        )


def _make_numeric_imputer() -> SimpleImputer:
    """Median imputation learned only from the training fold."""
    try:
        return SimpleImputer(
            strategy='median',
            keep_empty_features=True,
        )
    except TypeError:
        return SimpleImputer(
            strategy='median',
        )


def _normalise_model_matrix(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    """
    Prepare dtypes without learning statistics from validation/test rows.

    Categorical missingness is represented explicitly. Numeric coercion leaves
    NaN values for the train-fitted median imputer.
    """
    columns = categorical_features + numeric_features
    matrix = frame[columns].copy()

    for column in categorical_features:
        matrix[column] = (
            matrix[column]
            .astype('string')
            .fillna('__MISSING__')
            .astype(str)
        )

    for column in numeric_features:
        matrix[column] = _to_numeric(
            matrix[column]
        ).astype(float)

    return matrix


def _build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        transformers.append(
            (
                'numeric',
                _make_numeric_imputer(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                'categorical',
                _make_one_hot_encoder(),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError('No predictors are available for ML preprocessing.')

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False,
    )


def _build_ml_pipeline(
    spec: dict[str, Any],
    categorical_features: list[str],
    numeric_features: list[str],
    rules: ModelingRules,
) -> Pipeline:
    preprocessor = _build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    family = spec['family']

    if family == 'hist_gradient_boosting':
        estimator = HistGradientBoostingRegressor(
            loss='poisson',
            learning_rate=rules.hgb_learning_rate,
            max_iter=rules.hgb_max_iter,
            max_leaf_nodes=rules.hgb_max_leaf_nodes,
            min_samples_leaf=rules.hgb_min_samples_leaf,
            l2_regularization=rules.hgb_l2_regularization,
            early_stopping=False,
            random_state=rules.random_state,
        )

    elif family == 'random_forest':
        estimator = RandomForestRegressor(
            n_estimators=rules.rf_n_estimators,
            min_samples_leaf=rules.rf_min_samples_leaf,
            max_features=rules.rf_max_features,
            max_depth=rules.rf_max_depth,
            random_state=rules.random_state,
            n_jobs=rules.rf_n_jobs,
            criterion='squared_error',
        )

    else:
        raise ValueError(
            f'Unsupported ML family: {family!r}.'
        )

    return Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('model', estimator),
        ]
    )


def _unseen_category_fraction(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    categorical_features: list[str],
) -> float:
    """
    Fraction of validation rows containing at least one unseen category.

    This is diagnostic only and never changes validation data.
    """
    if not categorical_features or validation_frame.empty:
        return 0.0

    train_matrix = _normalise_model_matrix(
        train_frame,
        categorical_features=categorical_features,
        numeric_features=[],
    )
    validation_matrix = _normalise_model_matrix(
        validation_frame,
        categorical_features=categorical_features,
        numeric_features=[],
    )

    unseen_any = pd.Series(
        False,
        index=validation_frame.index,
        dtype=bool,
    )

    for column in categorical_features:
        known = set(
            train_matrix[column].dropna().unique().tolist()
        )
        unseen_any = (
            unseen_any
            | ~validation_matrix[column].isin(known)
        )

    return float(
        unseen_any.mean()
    )


def _encoded_feature_count(
    pipeline: Pipeline,
) -> int | None:
    try:
        preprocessor = pipeline.named_steps['preprocess']
        names = preprocessor.get_feature_names_out()
        return int(len(names))
    except Exception:
        return None


def cross_validate_ml_candidates(
    development_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    candidate_specs: list[dict[str, Any]],
    categorical_features: list[str],
    numeric_features: list[str],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/evaluate all ML candidates on exactly the same temporal CV folds.

    Every preprocessing statistic and model parameter fit occurs inside the
    training part of each fold. The final holdout is never accessed.
    """
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    for fold in folds:
        train = development_frame.loc[
            development_frame[period_column].isin(
                fold['train_period_ids']
            )
        ].copy()

        validation = development_frame.loc[
            development_frame[period_column].isin(
                fold['validation_period_ids']
            )
        ].copy()

        x_train = _normalise_model_matrix(
            train,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
        x_validation = _normalise_model_matrix(
            validation,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )

        y_train = _to_numeric(
            train[target_column]
        ).astype(float)

        unseen_fraction = _unseen_category_fraction(
            train,
            validation,
            categorical_features=categorical_features,
        )

        for spec in candidate_specs:
            pipeline = _build_ml_pipeline(
                spec,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                rules=rules,
            )

            start = time.perf_counter()
            pipeline.fit(
                x_train,
                y_train,
            )
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            prediction = pd.Series(
                pipeline.predict(
                    x_validation
                ),
                index=validation.index,
                dtype=float,
            )
            predict_seconds = time.perf_counter() - start

            if rules.nonnegative_predictions:
                prediction = prediction.clip(
                    lower=0
                )

            scored = validation.copy()
            scored['prediction'] = prediction
            scored['used_fallback'] = False

            metrics = evaluate_predictions(
                scored,
                target_column=target_column,
                prediction_column='prediction',
                period_column=period_column,
                entity_column=entity_column,
                top_k=rules.top_k,
            )

            metrics_rows.append(
                {
                    'fold': fold['fold'],
                    'model': spec['model'],
                    'model_family': spec['family'],
                    'train_periods': len(
                        fold['train_period_ids']
                    ),
                    'validation_periods': len(
                        fold['validation_period_ids']
                    ),
                    'fallback_value': np.nan,
                    'fallback_fraction': 0.0,
                    'unseen_category_fraction': unseen_fraction,
                    'encoded_feature_count': _encoded_feature_count(
                        pipeline
                    ),
                    'fit_seconds': fit_seconds,
                    'predict_seconds': predict_seconds,
                    **metrics,
                }
            )

            prediction_columns = [
                period_column,
                target_column,
                'prediction',
                'used_fallback',
            ]

            for optional in [
                entity_column,
                regime_column,
                'report_start',
                'report_end',
            ]:
                if (
                    optional is not None
                    and optional in scored.columns
                    and optional not in prediction_columns
                ):
                    prediction_columns.append(
                        optional
                    )

            predictions = scored[
                prediction_columns
            ].copy()
            predictions.insert(
                0,
                'model_family',
                spec['family'],
            )
            predictions.insert(
                0,
                'model',
                spec['model'],
            )
            predictions.insert(
                0,
                'fold',
                fold['fold'],
            )
            prediction_frames.append(
                predictions
            )

            if (
                regime_column is not None
                and regime_column in scored.columns
            ):
                for regime, group in scored.groupby(
                    regime_column,
                    dropna=False,
                ):
                    regime_metrics = evaluate_predictions(
                        group,
                        target_column=target_column,
                        prediction_column='prediction',
                        period_column=period_column,
                        entity_column=entity_column,
                        top_k=rules.top_k,
                    )

                    regime_rows.append(
                        {
                            'fold': fold['fold'],
                            'model': spec['model'],
                            'model_family': spec['family'],
                            'regime': regime,
                            **regime_metrics,
                        }
                    )

            for period_id, group in scored.groupby(
                period_column,
                dropna=False,
            ):
                period_metrics = evaluate_predictions(
                    group,
                    target_column=target_column,
                    prediction_column='prediction',
                    period_column=period_column,
                    entity_column=entity_column,
                    top_k=rules.top_k,
                )

                period_rows.append(
                    {
                        'fold': fold['fold'],
                        'model': spec['model'],
                        'model_family': spec['family'],
                        period_column: period_id,
                        **period_metrics,
                    }
                )

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    predictions_frame = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        if prediction_frames
        else pd.DataFrame()
    )

    regime_frame = pd.DataFrame(
        regime_rows
    )
    period_frame = pd.DataFrame(
        period_rows
    )

    return (
        metrics_frame,
        predictions_frame,
        regime_frame,
        period_frame,
    )


def combine_model_summaries(
    baseline_metrics: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one development-CV leaderboard across all competitors."""
    baseline = baseline_metrics.copy()

    if not baseline.empty:
        baseline['model_family'] = 'baseline'

        if 'unseen_category_fraction' not in baseline.columns:
            baseline['unseen_category_fraction'] = np.nan

        if 'encoded_feature_count' not in baseline.columns:
            baseline['encoded_feature_count'] = np.nan

        if 'fit_seconds' not in baseline.columns:
            baseline['fit_seconds'] = 0.0

        if 'predict_seconds' not in baseline.columns:
            baseline['predict_seconds'] = 0.0

    all_metrics = pd.concat(
        [
            frame
            for frame in [
                baseline,
                ml_metrics,
            ]
            if not frame.empty
        ],
        ignore_index=True,
        sort=False,
    )

    summary = summarize_cv_metrics(
        all_metrics,
        top_k=top_k,
    )

    if not summary.empty:
        family_map = (
            all_metrics[
                [
                    'model',
                    'model_family',
                ]
            ]
            .drop_duplicates()
            .set_index(
                'model'
            )[
                'model_family'
            ]
            .to_dict()
        )

        summary.insert(
            2,
            'model_family',
            summary['model'].map(
                family_map
            ),
        )

    return all_metrics, summary


def _fit_predict_ml(
    train_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    spec: dict[str, Any],
    categorical_features: list[str],
    numeric_features: list[str],
    target_column: str,
    rules: ModelingRules,
) -> tuple[pd.Series, Pipeline]:
    pipeline = _build_ml_pipeline(
        spec,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        rules=rules,
    )

    x_train = _normalise_model_matrix(
        train_frame,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )
    x_prediction = _normalise_model_matrix(
        prediction_frame,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )
    y_train = _to_numeric(
        train_frame[target_column]
    ).astype(float)

    pipeline.fit(
        x_train,
        y_train,
    )

    prediction = pd.Series(
        pipeline.predict(
            x_prediction
        ),
        index=prediction_frame.index,
        dtype=float,
    )

    if rules.nonnegative_predictions:
        prediction = prediction.clip(
            lower=0
        )

    return prediction, pipeline



# =============================================================================
# SEQUENTIAL ADAPTIVE REGIME SELECTOR
# =============================================================================


ADAPTIVE_MODEL_NAME = 'Adaptive regime selector'


def _adaptive_default_model(
    baseline_specs: list[dict[str, str]],
    rules: ModelingRules,
) -> str:
    """
    Pick a deterministic causal baseline for the first adaptive fold.

    No validation performance is available before fold 1, so the selector must
    not learn a winner from that same fold. The preferred baseline kind is
    configurable; otherwise the first supported baseline is used.
    """
    for spec in baseline_specs:
        if spec.get('kind') == rules.adaptive_initial_baseline_kind:
            return str(spec['model'])

    if not baseline_specs:
        raise ValueError(
            'Adaptive selector requires at least one causal baseline.'
        )

    return str(baseline_specs[0]['model'])


def _normalise_regime_key(value: Any) -> str:
    if pd.isna(value):
        return '__MISSING_REGIME__'

    return str(value)


def _candidate_score_table(
    predictions: pd.DataFrame,
    target_column: str,
    minimum_rows: int,
) -> pd.DataFrame:
    """
    Rank candidate models using pooled OOF WAPE, then MAE.

    The table is built exclusively from already-observed validation predictions
    supplied by the caller. It therefore supports causal sequential selection.
    """
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for model_name, group in predictions.groupby(
        'model',
        dropna=False,
    ):
        valid = group.dropna(
            subset=[
                target_column,
                'prediction',
            ]
        ).copy()

        n_rows = len(
            valid
        )

        if n_rows < minimum_rows:
            continue

        true = _to_numeric(
            valid[target_column]
        ).to_numpy(
            dtype=float
        )
        pred = _to_numeric(
            valid['prediction']
        ).to_numpy(
            dtype=float
        )

        actual_total = float(
            np.sum(
                true
            )
        )

        if actual_total <= 0:
            continue

        error = pred - true

        rows.append(
            {
                'model': str(
                    model_name
                ),
                'n_rows': n_rows,
                'actual_total': actual_total,
                'WAPE': wape(
                    true,
                    pred,
                ),
                'MAE': float(
                    np.mean(
                        np.abs(
                            error
                        )
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                'WAPE',
                'MAE',
                'model',
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def _combine_candidate_predictions(
    baseline_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not baseline_predictions.empty:
        baseline = baseline_predictions.copy()
        baseline['model_family'] = 'baseline'
        frames.append(
            baseline
        )

    if not ml_predictions.empty:
        frames.append(
            ml_predictions.copy()
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def build_sequential_adaptive_regime_selector(
    baseline_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    folds: list[dict[str, Any]],
    baseline_specs: list[dict[str, str]],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build an honest prequential selector by demand regime.

    For validation fold f:
      * model choices may use OOF predictions only from folds < f;
      * no performance from fold f is used to choose its model;
      * if a regime lacks enough historical OOF evidence, the selector falls
        back to the best global model from earlier folds;
      * if even global evidence is unavailable, fold 1 uses a predefined
        causal baseline.

    This is intentionally stricter than selecting a regime winner from all CV
    folds and evaluating on those same folds.
    """
    if regime_column is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    candidates = _combine_candidate_predictions(
        baseline_predictions=baseline_predictions,
        ml_predictions=ml_predictions,
    )

    if candidates.empty or regime_column not in candidates.columns:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    default_model = _adaptive_default_model(
        baseline_specs=baseline_specs,
        rules=rules,
    )

    candidates = candidates.copy()
    candidates[
        '__adaptive_regime_key'
    ] = candidates[
        regime_column
    ].map(
        _normalise_regime_key
    )

    candidate_models = set(
        candidates[
            'model'
        ].astype(
            str
        ).unique()
    )

    if default_model not in candidate_models:
        raise ValueError(
            f'Adaptive default model {default_model!r} has no OOF '
            'predictions.'
        )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    ordered_folds = sorted(
        folds,
        key=lambda item: int(
            item['fold']
        ),
    )

    for fold in ordered_folds:
        fold_id = int(
            fold['fold']
        )

        history = candidates.loc[
            candidates[
                'fold'
            ].astype(
                int
            )
            < fold_id
        ].copy()

        current = candidates.loc[
            candidates[
                'fold'
            ].astype(
                int
            )
            == fold_id
        ].copy()

        if current.empty:
            continue

        current_regimes = (
            current[
                '__adaptive_regime_key'
            ]
            .drop_duplicates()
            .tolist()
        )

        global_scores = _candidate_score_table(
            history,
            target_column=target_column,
            minimum_rows=rules.adaptive_min_prior_global_rows,
        )

        if not global_scores.empty:
            global_best_model = str(
                global_scores.iloc[
                    0
                ][
                    'model'
                ]
            )
        else:
            global_best_model = default_model

        selected_chunks: list[pd.DataFrame] = []

        for regime_key in current_regimes:
            regime_history = history.loc[
                history[
                    '__adaptive_regime_key'
                ].eq(
                    regime_key
                )
            ].copy()

            regime_scores = _candidate_score_table(
                regime_history,
                target_column=target_column,
                minimum_rows=rules.adaptive_min_prior_regime_rows,
            )

            if not regime_scores.empty:
                selected_model = str(
                    regime_scores.iloc[
                        0
                    ][
                        'model'
                    ]
                )
                selection_source = 'prior_regime_oof'
                prior_rows = int(
                    regime_scores.iloc[
                        0
                    ][
                        'n_rows'
                    ]
                )
                prior_wape = float(
                    regime_scores.iloc[
                        0
                    ][
                        'WAPE'
                    ]
                )
            elif not global_scores.empty:
                selected_model = global_best_model
                selection_source = 'prior_global_oof'

                selected_global = global_scores.loc[
                    global_scores[
                        'model'
                    ].eq(
                        selected_model
                    )
                ].iloc[
                    0
                ]

                prior_rows = int(
                    selected_global[
                        'n_rows'
                    ]
                )
                prior_wape = float(
                    selected_global[
                        'WAPE'
                    ]
                )
            else:
                selected_model = default_model
                selection_source = 'initial_causal_baseline'
                prior_rows = 0
                prior_wape = np.nan

            regime_current = current.loc[
                current[
                    '__adaptive_regime_key'
                ].eq(
                    regime_key
                )
                & current[
                    'model'
                ].astype(
                    str
                ).eq(
                    selected_model
                )
            ].copy()

            if regime_current.empty:
                regime_current = current.loc[
                    current[
                        '__adaptive_regime_key'
                    ].eq(
                        regime_key
                    )
                    & current[
                        'model'
                    ].astype(
                        str
                    ).eq(
                        default_model
                    )
                ].copy()

                selection_source = (
                    f'{selection_source}__prediction_unavailable_'
                    'fallback_default'
                )
                selected_model = default_model

            if regime_current.empty:
                raise RuntimeError(
                    'Adaptive selector could not obtain predictions for '
                    f'fold={fold_id}, regime={regime_key!r}.'
                )

            regime_current[
                'prediction_source_model'
            ] = selected_model
            regime_current[
                'selection_source'
            ] = selection_source

            selected_chunks.append(
                regime_current
            )

            selection_rows.append(
                {
                    'fold': fold_id,
                    'regime': regime_key,
                    'selected_model': selected_model,
                    'selection_source': selection_source,
                    'prior_validation_folds': max(
                        0,
                        fold_id - 1,
                    ),
                    'prior_selected_model_rows': prior_rows,
                    'prior_selected_model_wape': prior_wape,
                    'current_rows': int(
                        len(
                            regime_current
                        )
                    ),
                }
            )

        scored = pd.concat(
            selected_chunks,
            ignore_index=True,
            sort=False,
        )

        duplicate_subset = [
            'fold',
            period_column,
        ]

        if (
            entity_column is not None
            and entity_column in scored.columns
        ):
            duplicate_subset.append(
                entity_column
            )

        if scored.duplicated(
            subset=duplicate_subset,
            keep=False,
        ).any():
            raise RuntimeError(
                'Adaptive selector produced duplicate predictions for the '
                'same validation observation.'
            )

        scored[
            'model'
        ] = ADAPTIVE_MODEL_NAME
        scored[
            'model_family'
        ] = 'adaptive_selector'

        metrics = evaluate_predictions(
            scored,
            target_column=target_column,
            prediction_column='prediction',
            period_column=period_column,
            entity_column=entity_column,
            top_k=rules.top_k,
        )

        metrics_rows.append(
            {
                'fold': fold_id,
                'model': ADAPTIVE_MODEL_NAME,
                'model_family': 'adaptive_selector',
                'train_periods': len(
                    fold[
                        'train_period_ids'
                    ]
                ),
                'validation_periods': len(
                    fold[
                        'validation_period_ids'
                    ]
                ),
                'fallback_value': np.nan,
                'fallback_fraction': float(
                    _safe_bool(
                        scored[
                            'used_fallback'
                        ]
                    ).fillna(
                        False
                    ).mean()
                ),
                'unseen_category_fraction': np.nan,
                'encoded_feature_count': np.nan,
                'fit_seconds': 0.0,
                'predict_seconds': 0.0,
                **metrics,
            }
        )

        keep_columns = [
            'fold',
            'model',
            'model_family',
            period_column,
            target_column,
            'prediction',
            'used_fallback',
            'prediction_source_model',
            'selection_source',
        ]

        for optional in [
            entity_column,
            regime_column,
            'report_start',
            'report_end',
        ]:
            if (
                optional is not None
                and optional in scored.columns
                and optional not in keep_columns
            ):
                keep_columns.append(
                    optional
                )

        prediction_frames.append(
            scored[
                keep_columns
            ].copy()
        )

        for regime, group in scored.groupby(
            regime_column,
            dropna=False,
        ):
            regime_metrics = evaluate_predictions(
                group,
                target_column=target_column,
                prediction_column='prediction',
                period_column=period_column,
                entity_column=entity_column,
                top_k=rules.top_k,
            )

            regime_rows.append(
                {
                    'fold': fold_id,
                    'model': ADAPTIVE_MODEL_NAME,
                    'model_family': 'adaptive_selector',
                    'regime': regime,
                    **regime_metrics,
                }
            )

        for period_id, group in scored.groupby(
            period_column,
            dropna=False,
        ):
            period_metrics = evaluate_predictions(
                group,
                target_column=target_column,
                prediction_column='prediction',
                period_column=period_column,
                entity_column=entity_column,
                top_k=rules.top_k,
            )

            period_rows.append(
                {
                    'fold': fold_id,
                    'model': ADAPTIVE_MODEL_NAME,
                    'model_family': 'adaptive_selector',
                    period_column: period_id,
                    **period_metrics,
                }
            )

    metrics_frame = pd.DataFrame(
        metrics_rows
    )
    predictions_frame = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
            sort=False,
        )
        if prediction_frames
        else pd.DataFrame()
    )
    regime_frame = pd.DataFrame(
        regime_rows
    )
    period_frame = pd.DataFrame(
        period_rows
    )
    selection_frame = pd.DataFrame(
        selection_rows
    )

    return (
        metrics_frame,
        predictions_frame,
        regime_frame,
        period_frame,
        selection_frame,
    )


# =============================================================================
# OPTIONAL FINAL TEST EVALUATION
# =============================================================================


def evaluate_selected_baseline_on_final_test(
    development_frame: pd.DataFrame,
    final_test_frame: pd.DataFrame,
    selected_baseline: dict[str, str],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred, fallback_mask, fallback_value = _baseline_predictions(
        development_frame,
        final_test_frame,
        target_column=target_column,
        source_column=selected_baseline['source_column'],
        rules=rules,
    )

    scored = final_test_frame.copy()
    scored['prediction'] = pred
    scored['used_fallback'] = fallback_mask

    metrics = evaluate_predictions(
        scored,
        target_column=target_column,
        prediction_column='prediction',
        period_column=period_column,
        entity_column=entity_column,
        top_k=rules.top_k,
    )

    metrics_frame = pd.DataFrame(
        [
            {
                'model': selected_baseline['model'],
                'source_column': selected_baseline['source_column'],
                'fallback_value': fallback_value,
                'fallback_fraction': float(fallback_mask.mean()),
                **metrics,
            }
        ]
    )

    keep_columns = [
        period_column,
        target_column,
        'prediction',
        'used_fallback',
    ]

    for optional in [
        entity_column,
        regime_column,
        'report_start',
        'report_end',
    ]:
        if (
            optional is not None
            and optional in scored.columns
            and optional not in keep_columns
        ):
            keep_columns.append(optional)

    predictions = scored[keep_columns].copy()

    return metrics_frame, predictions



def evaluate_selected_competitor_on_final_test(
    development_frame: pd.DataFrame,
    final_test_frame: pd.DataFrame,
    selected_model_name: str,
    baseline_specs: list[dict[str, str]],
    ml_specs: list[dict[str, Any]],
    categorical_features: list[str],
    numeric_features: list[str],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the single CV-selected competitor on the isolated final holdout.

    This function is called only when evaluate_final_test=True.
    """
    baseline_lookup = {
        spec['model']: spec
        for spec in baseline_specs
    }
    ml_lookup = {
        spec['model']: spec
        for spec in ml_specs
    }

    scored = final_test_frame.copy()
    model_family: str
    fallback_value: float | None = None
    fallback_fraction = 0.0

    if selected_model_name in baseline_lookup:
        spec = baseline_lookup[
            selected_model_name
        ]
        prediction, fallback_mask, fallback_value = _baseline_predictions(
            development_frame,
            final_test_frame,
            target_column=target_column,
            source_column=spec['source_column'],
            rules=rules,
        )
        scored['prediction'] = prediction
        scored['used_fallback'] = fallback_mask
        fallback_fraction = float(
            fallback_mask.mean()
        )
        model_family = 'baseline'

    elif selected_model_name in ml_lookup:
        spec = ml_lookup[
            selected_model_name
        ]
        prediction, _ = _fit_predict_ml(
            development_frame,
            final_test_frame,
            spec=spec,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            target_column=target_column,
            rules=rules,
        )
        scored['prediction'] = prediction
        scored['used_fallback'] = False
        model_family = spec['family']

    else:
        raise KeyError(
            f'Selected model {selected_model_name!r} has no candidate spec.'
        )

    metrics = evaluate_predictions(
        scored,
        target_column=target_column,
        prediction_column='prediction',
        period_column=period_column,
        entity_column=entity_column,
        top_k=rules.top_k,
    )

    metrics_frame = pd.DataFrame(
        [
            {
                'model': selected_model_name,
                'model_family': model_family,
                'fallback_value': fallback_value,
                'fallback_fraction': fallback_fraction,
                **metrics,
            }
        ]
    )

    keep_columns = [
        period_column,
        target_column,
        'prediction',
        'used_fallback',
    ]

    for optional in [
        entity_column,
        regime_column,
        'report_start',
        'report_end',
    ]:
        if (
            optional is not None
            and optional in scored.columns
            and optional not in keep_columns
        ):
            keep_columns.append(
                optional
            )

    predictions = scored[
        keep_columns
    ].copy()
    predictions.insert(
        0,
        'model_family',
        model_family,
    )
    predictions.insert(
        0,
        'model',
        selected_model_name,
    )

    return metrics_frame, predictions



def freeze_adaptive_regime_mapping(
    baseline_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    baseline_specs: list[dict[str, str]],
    target_column: str,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[pd.DataFrame, str]:
    """
    Freeze the adaptive regime mapping using Development OOF evidence only.
    """
    if regime_column is None:
        return pd.DataFrame(), _adaptive_default_model(
            baseline_specs,
            rules,
        )

    candidates = _combine_candidate_predictions(
        baseline_predictions=baseline_predictions,
        ml_predictions=ml_predictions,
    )

    if candidates.empty or regime_column not in candidates.columns:
        return pd.DataFrame(), _adaptive_default_model(
            baseline_specs,
            rules,
        )

    default_model = _adaptive_default_model(
        baseline_specs,
        rules,
    )

    global_scores = _candidate_score_table(
        candidates,
        target_column=target_column,
        minimum_rows=rules.adaptive_min_prior_global_rows,
    )

    frozen_global_model = (
        str(global_scores.iloc[0]['model'])
        if not global_scores.empty
        else default_model
    )

    candidates = candidates.copy()
    candidates[
        '__frozen_regime_key'
    ] = candidates[
        regime_column
    ].map(
        _normalise_regime_key
    )

    rows: list[dict[str, Any]] = []

    for regime_key in (
        candidates[
            '__frozen_regime_key'
        ]
        .drop_duplicates()
        .tolist()
    ):
        regime_history = candidates.loc[
            candidates[
                '__frozen_regime_key'
            ].eq(
                regime_key
            )
        ].copy()

        regime_scores = _candidate_score_table(
            regime_history,
            target_column=target_column,
            minimum_rows=rules.adaptive_min_prior_regime_rows,
        )

        if not regime_scores.empty:
            selected = regime_scores.iloc[0]
            selected_model = str(
                selected['model']
            )
            source = 'development_oof_regime'
            evidence_rows = int(
                selected['n_rows']
            )
            evidence_wape = float(
                selected['WAPE']
            )

        elif not global_scores.empty:
            selected = global_scores.loc[
                global_scores[
                    'model'
                ].eq(
                    frozen_global_model
                )
            ].iloc[0]

            selected_model = frozen_global_model
            source = 'development_oof_global_fallback'
            evidence_rows = int(
                selected['n_rows']
            )
            evidence_wape = float(
                selected['WAPE']
            )

        else:
            selected_model = default_model
            source = 'initial_causal_baseline_fallback'
            evidence_rows = 0
            evidence_wape = np.nan

        rows.append(
            {
                'regime': regime_key,
                'selected_model': selected_model,
                'freeze_source': source,
                'development_oof_rows': evidence_rows,
                'development_oof_wape': evidence_wape,
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            'regime'
        )
        .reset_index(
            drop=True
        ),
        frozen_global_model,
    )


def _score_final_prediction_frame(
    scored: pd.DataFrame,
    model_name: str,
    model_family: str,
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics = evaluate_predictions(
        scored,
        target_column=target_column,
        prediction_column='prediction',
        period_column=period_column,
        entity_column=entity_column,
        top_k=rules.top_k,
    )

    overall = {
        'model': model_name,
        'model_family': model_family,
        **metrics,
    }

    regime_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    if (
        regime_column is not None
        and regime_column in scored.columns
    ):
        for regime, group in scored.groupby(
            regime_column,
            dropna=False,
        ):
            regime_rows.append(
                {
                    'model': model_name,
                    'model_family': model_family,
                    'regime': regime,
                    **evaluate_predictions(
                        group,
                        target_column=target_column,
                        prediction_column='prediction',
                        period_column=period_column,
                        entity_column=entity_column,
                        top_k=rules.top_k,
                    ),
                }
            )

    for period_id, group in scored.groupby(
        period_column,
        dropna=False,
    ):
        period_rows.append(
            {
                'model': model_name,
                'model_family': model_family,
                period_column: period_id,
                **evaluate_predictions(
                    group,
                    target_column=target_column,
                    prediction_column='prediction',
                    period_column=period_column,
                    entity_column=entity_column,
                    top_k=rules.top_k,
                ),
            }
        )

    return overall, regime_rows, period_rows


def evaluate_frozen_models_on_final_test(
    development_frame: pd.DataFrame,
    final_test_frame: pd.DataFrame,
    baseline_specs: list[dict[str, str]],
    ml_specs: list[dict[str, Any]],
    frozen_adaptive_mapping: pd.DataFrame,
    frozen_global_model: str,
    primary_model_name: str,
    categorical_features: list[str],
    numeric_features: list[str],
    target_column: str,
    period_column: str,
    entity_column: str | None,
    regime_column: str | None,
    rules: ModelingRules,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate the frozen Development decision once on the final temporal test.

    The preselected Development winner remains the primary model. Other test
    results are diagnostic comparisons only and cannot be used to re-select it.
    """
    candidate_frames: list[pd.DataFrame] = []

    for spec in baseline_specs:
        prediction, fallback_mask, _ = _baseline_predictions(
            development_frame,
            final_test_frame,
            target_column=target_column,
            source_column=spec['source_column'],
            rules=rules,
        )

        scored = final_test_frame.copy()
        scored['prediction'] = prediction
        scored['used_fallback'] = fallback_mask
        scored['model'] = spec['model']
        scored['model_family'] = 'baseline'
        scored['prediction_source_model'] = spec['model']
        candidate_frames.append(
            scored
        )

    for spec in ml_specs:
        prediction, _ = _fit_predict_ml(
            development_frame,
            final_test_frame,
            spec=spec,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            target_column=target_column,
            rules=rules,
        )

        scored = final_test_frame.copy()
        scored['prediction'] = prediction
        scored['used_fallback'] = False
        scored['model'] = spec['model']
        scored['model_family'] = spec['family']
        scored['prediction_source_model'] = spec['model']
        candidate_frames.append(
            scored
        )

    if not candidate_frames:
        raise RuntimeError(
            'No frozen competitors are available for final-test evaluation.'
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
        sort=False,
    )

    if (
        not frozen_adaptive_mapping.empty
        and regime_column is not None
        and regime_column in final_test_frame.columns
    ):
        mapping_lookup = (
            frozen_adaptive_mapping.set_index(
                'regime'
            )[
                'selected_model'
            ].to_dict()
        )

        adaptive_chunks: list[pd.DataFrame] = []

        test_keyed = final_test_frame.copy()
        test_keyed[
            '__final_regime_key'
        ] = test_keyed[
            regime_column
        ].map(
            _normalise_regime_key
        )

        for regime_key, group in test_keyed.groupby(
            '__final_regime_key',
            dropna=False,
        ):
            selected_model = mapping_lookup.get(
                regime_key,
                frozen_global_model,
            )

            key_columns = [
                period_column,
            ]

            if (
                entity_column is not None
                and entity_column in group.columns
            ):
                key_columns.append(
                    entity_column
                )

            chosen = candidates.loc[
                candidates[
                    'model'
                ].astype(
                    str
                ).eq(
                    str(
                        selected_model
                    )
                )
            ].copy()

            chosen = chosen.merge(
                group[
                    key_columns
                ].drop_duplicates(),
                on=key_columns,
                how='inner',
            )

            if chosen.empty:
                raise RuntimeError(
                    'Frozen adaptive mapping has no final-test predictions '
                    f'for regime={regime_key!r}, model={selected_model!r}.'
                )

            chosen[
                'selection_source'
            ] = (
                'frozen_development_regime_mapping'
                if regime_key in mapping_lookup
                else 'frozen_global_model_for_unseen_regime'
            )

            adaptive_chunks.append(
                chosen
            )

        adaptive = pd.concat(
            adaptive_chunks,
            ignore_index=True,
            sort=False,
        )
        adaptive['model'] = ADAPTIVE_MODEL_NAME
        adaptive['model_family'] = 'adaptive_selector'

        duplicate_keys = [
            period_column,
        ]

        if (
            entity_column is not None
            and entity_column in adaptive.columns
        ):
            duplicate_keys.append(
                entity_column
            )

        if adaptive.duplicated(
            duplicate_keys,
            keep=False,
        ).any():
            raise RuntimeError(
                'Frozen adaptive final-test predictions contain duplicates.'
            )

        candidates = pd.concat(
            [
                candidates,
                adaptive,
            ],
            ignore_index=True,
            sort=False,
        )

    metrics_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    for model_name, scored in candidates.groupby(
        'model',
        sort=False,
    ):
        model_family = str(
            scored[
                'model_family'
            ].iloc[0]
        )

        overall, model_regime_rows, model_period_rows = (
            _score_final_prediction_frame(
                scored=scored,
                model_name=str(
                    model_name
                ),
                model_family=model_family,
                target_column=target_column,
                period_column=period_column,
                entity_column=entity_column,
                regime_column=regime_column,
                rules=rules,
            )
        )

        overall[
            'evaluation_role'
        ] = (
            'primary_frozen_from_development'
            if str(model_name) == primary_model_name
            else 'comparison_only'
        )

        metrics_rows.append(
            overall
        )
        regime_rows.extend(
            model_regime_rows
        )
        period_rows.extend(
            model_period_rows
        )

    metrics_frame = pd.DataFrame(
        metrics_rows
    )

    if not metrics_frame.empty:
        metrics_frame[
            'test_rank_by_wape'
        ] = metrics_frame[
            'WAPE'
        ].rank(
            method='min'
        ).astype(
            'Int64'
        )

        metrics_frame = metrics_frame.sort_values(
            [
                'WAPE',
                'MAE',
            ],
            ascending=[
                True,
                True,
            ],
        ).reset_index(
            drop=True
        )

    keep_columns = [
        'model',
        'model_family',
        period_column,
        target_column,
        'prediction',
        'used_fallback',
        'prediction_source_model',
    ]

    if 'selection_source' in candidates.columns:
        keep_columns.append(
            'selection_source'
        )

    for optional in [
        entity_column,
        regime_column,
        'report_start',
        'report_end',
    ]:
        if (
            optional is not None
            and optional in candidates.columns
            and optional not in keep_columns
        ):
            keep_columns.append(
                optional
            )

    return (
        metrics_frame,
        candidates[
            keep_columns
        ].copy(),
        pd.DataFrame(
            regime_rows
        ),
        pd.DataFrame(
            period_rows
        ),
    )


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================


def run_modeling_foundation(
    feature_datasets: dict[str, pd.DataFrame],
    feature_catalogue: pd.DataFrame | str | Path,
    paths: ModelingPaths,
    rules: ModelingRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run temporal split construction, feature selection, baselines and ML.

    The final holdout is isolated before any cross-validation. All candidate
    comparisons are performed on the development folds only unless final-test
    evaluation is explicitly enabled after model selection is frozen.
    """
    rules = rules if rules is not None else ModelingRules()
    rules.validate()

    if rules.dataset_name not in feature_datasets:
        raise KeyError(
            f'Feature dataset {rules.dataset_name!r} is not available.'
        )

    feature_table = feature_datasets[rules.dataset_name].copy()

    if isinstance(feature_catalogue, (str, Path)):
        catalogue = load_feature_catalogue(feature_catalogue)
    else:
        catalogue = feature_catalogue.copy()
        required_catalogue_columns = {
            'dataset',
            'column',
            'feature_dtype',
            'role',
            'availability',
            'safe_default',
        }
        missing_catalogue = required_catalogue_columns.difference(
            catalogue.columns
        )
        if missing_catalogue:
            raise ValueError(
                'Feature catalogue is missing required columns: '
                f'{sorted(missing_catalogue)}'
            )
        catalogue['safe_default'] = _safe_bool(
            catalogue['safe_default']
        ).fillna(False)

    target_column = _first_existing(
        feature_table,
        rules.target_candidates,
    )

    if target_column is None:
        raise KeyError(
            'No modeling target candidate was found in the feature table.'
        )

    entity_column = _first_existing(
        feature_table,
        rules.entity_column_candidates,
    )
    period_column = _first_existing(
        feature_table,
        rules.period_index_candidates,
    )
    start_column = _first_existing(
        feature_table,
        rules.period_start_candidates,
    )
    end_column = _first_existing(
        feature_table,
        rules.period_end_candidates,
    )
    regime_column = _first_existing(
        feature_table,
        rules.regime_column_candidates,
    )

    if period_column is None:
        raise KeyError(
            'No period index/time column was found for temporal splitting.'
        )

    if rules.training_eligibility_column in feature_table.columns:
        eligible = _safe_bool(
            feature_table[rules.training_eligibility_column]
        ).fillna(False)
        modeling_frame = feature_table.loc[eligible].copy()
    else:
        modeling_frame = feature_table.loc[
            feature_table[target_column].notna()
        ].copy()

    if modeling_frame.empty:
        raise ValueError('No training-eligible modeling rows are available.')

    if start_column is not None:
        modeling_frame[start_column] = _to_datetime(
            modeling_frame[start_column]
        )

    if end_column is not None:
        modeling_frame[end_column] = _to_datetime(
            modeling_frame[end_column]
        )

    _print_header(
        'FEATURES -> TEMPORAL MODELING FOUNDATION',
        verbose=verbose,
    )

    _print_subheader('Temporal holdout', verbose=verbose)

    development, final_test, split_manifest = build_final_holdout_split(
        modeling_frame,
        period_column=period_column,
        start_column=start_column,
        end_column=end_column,
        rules=rules,
    )

    development_periods = development[period_column].nunique()
    final_test_periods = final_test[period_column].nunique()

    _print_message(
        f'Development: {len(development):,} rows | '
        f'{development_periods} periods.',
        verbose=verbose,
    )
    _print_message(
        f'Final test RESERVED: {len(final_test):,} rows | '
        f'{final_test_periods} periods | evaluated={rules.evaluate_final_test}.',
        verbose=verbose,
    )

    _print_subheader('Leakage-aware feature selection', verbose=verbose)

    categorical_features, numeric_features, feature_selection = (
        select_model_features(
            feature_table=modeling_frame,
            catalogue=catalogue,
            development_frame=development,
            rules=rules,
        )
    )

    _print_message(
        f'Selected predictors: '
        f'{len(categorical_features)} categorical + '
        f'{len(numeric_features)} numeric = '
        f'{len(categorical_features) + len(numeric_features)} total.',
        verbose=verbose,
    )

    _print_subheader('Expanding-window cross-validation', verbose=verbose)

    folds = build_expanding_cv_splits(
        development,
        period_column=period_column,
        start_column=start_column,
        end_column=end_column,
        rules=rules,
    )
    cv_manifest = _cv_manifest(
        folds,
        period_column=period_column,
    )

    for fold in folds:
        _print_message(
            f'Fold {fold["fold"]}: '
            f'{len(fold["train_period_ids"])} train periods -> '
            f'{len(fold["validation_period_ids"])} validation periods.',
            verbose=verbose,
        )

    _print_subheader('Causal baselines', verbose=verbose)

    baseline_specs = discover_baseline_specs(
        modeling_frame,
        target_column=target_column,
        rules=rules,
    )

    for spec in baseline_specs:
        _print_message(
            f'Baseline: {spec["model"]} <- {spec["source_column"]}',
            verbose=verbose,
        )

    (
        cv_metrics,
        cv_predictions,
        regime_metrics,
        period_metrics,
    ) = cross_validate_baselines(
        development_frame=development,
        folds=folds,
        baseline_specs=baseline_specs,
        target_column=target_column,
        period_column=period_column,
        entity_column=entity_column,
        regime_column=regime_column,
        rules=rules,
    )

    cv_summary = summarize_cv_metrics(
        cv_metrics,
        top_k=rules.top_k,
    )

    if cv_summary.empty:
        raise RuntimeError('Baseline cross-validation produced no metrics.')

    selected_baseline_name = str(
        cv_summary.iloc[0]['model']
    )

    _print_message(
        f'Best baseline by development CV WAPE: {selected_baseline_name} '
        f'(WAPE={cv_summary.iloc[0]["WAPE_mean"]:.3f}).',
        verbose=verbose,
    )

    ml_specs: list[dict[str, Any]] = []
    skipped_ml_specs: list[dict[str, str]] = []
    ml_metrics = pd.DataFrame()
    ml_predictions = pd.DataFrame()
    ml_regime_metrics = pd.DataFrame()
    ml_period_metrics = pd.DataFrame()
    ml_summary = pd.DataFrame()

    if rules.run_ml_candidates:
        _print_subheader(
            'Machine-learning candidates',
            verbose=verbose,
        )

        ml_specs, skipped_ml_specs = discover_ml_candidate_specs(
            development,
            target_column=target_column,
            rules=rules,
        )

        for skipped in skipped_ml_specs:
            _print_message(
                f'Skipped {skipped["model"]}: {skipped["reason"]}.',
                level='WARNING',
                verbose=verbose,
            )

        for spec in ml_specs:
            _print_message(
                f'Candidate: {spec["model"]}.',
                verbose=verbose,
            )

        if ml_specs:
            (
                ml_metrics,
                ml_predictions,
                ml_regime_metrics,
                ml_period_metrics,
            ) = cross_validate_ml_candidates(
                development_frame=development,
                folds=folds,
                candidate_specs=ml_specs,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target_column=target_column,
                period_column=period_column,
                entity_column=entity_column,
                regime_column=regime_column,
                rules=rules,
            )

            ml_summary = summarize_cv_metrics(
                ml_metrics,
                top_k=rules.top_k,
            )

            for _, row in ml_summary.iterrows():
                _print_message(
                    f'{row["model"]}: '
                    f'WAPE={row["WAPE_mean"]:.3f} | '
                    f'MAE={row["MAE_mean"]:.3f} | '
                    f'RMSE={row["RMSE_mean"]:.3f} | '
                    f'Bias={row["Bias_mean"]:.3f}.',
                    verbose=verbose,
                )

    adaptive_metrics = pd.DataFrame()
    adaptive_predictions = pd.DataFrame()
    adaptive_regime_metrics = pd.DataFrame()
    adaptive_period_metrics = pd.DataFrame()
    adaptive_selection = pd.DataFrame()
    adaptive_summary = pd.DataFrame()

    if (
        rules.run_adaptive_regime_selector
        and regime_column is not None
    ):
        _print_subheader(
            'Sequential adaptive regime selector',
            verbose=verbose,
        )

        (
            adaptive_metrics,
            adaptive_predictions,
            adaptive_regime_metrics,
            adaptive_period_metrics,
            adaptive_selection,
        ) = build_sequential_adaptive_regime_selector(
            baseline_predictions=cv_predictions,
            ml_predictions=ml_predictions,
            folds=folds,
            baseline_specs=baseline_specs,
            target_column=target_column,
            period_column=period_column,
            entity_column=entity_column,
            regime_column=regime_column,
            rules=rules,
        )

        if not adaptive_metrics.empty:
            adaptive_summary = summarize_cv_metrics(
                adaptive_metrics,
                top_k=rules.top_k,
            )

            adaptive_row = adaptive_summary.iloc[
                0
            ]

            _print_message(
                f'{ADAPTIVE_MODEL_NAME}: '
                f'WAPE={adaptive_row["WAPE_mean"]:.3f} | '
                f'MAE={adaptive_row["MAE_mean"]:.3f} | '
                f'RMSE={adaptive_row["RMSE_mean"]:.3f} | '
                f'Bias={adaptive_row["Bias_mean"]:.3f}.',
                verbose=verbose,
            )

            initial_folds = adaptive_selection.loc[
                adaptive_selection[
                    'selection_source'
                ].eq(
                    'initial_causal_baseline'
                ),
                'fold',
            ].nunique()

            _print_message(
                'Regime choices use only OOF evidence from earlier folds; '
                f'{initial_folds} fold(s) required the initial causal '
                'baseline because no earlier OOF evidence existed.',
                verbose=verbose,
            )

    cv_model_metrics, cv_model_summary = combine_model_summaries(
        baseline_metrics=cv_metrics,
        ml_metrics=ml_metrics,
        top_k=rules.top_k,
    )

    if not adaptive_metrics.empty:
        cv_model_metrics = pd.concat(
            [
                cv_model_metrics,
                adaptive_metrics,
            ],
            ignore_index=True,
            sort=False,
        )

        cv_model_summary = summarize_cv_metrics(
            cv_model_metrics,
            top_k=rules.top_k,
        )

        family_map = (
            cv_model_metrics[
                [
                    'model',
                    'model_family',
                ]
            ]
            .drop_duplicates()
            .set_index(
                'model'
            )[
                'model_family'
            ]
            .to_dict()
        )

        cv_model_summary.insert(
            2,
            'model_family',
            cv_model_summary[
                'model'
            ].map(
                family_map
            ),
        )

    if cv_model_summary.empty:
        raise RuntimeError(
            'No development-CV competitors produced valid metrics.'
        )

    selected_name = str(
        cv_model_summary.iloc[0]['model']
    )
    selected_family = str(
        cv_model_summary.iloc[0]['model_family']
    )

    _print_subheader(
        'Development CV leaderboard',
        verbose=verbose,
    )

    for _, row in cv_model_summary.iterrows():
        _print_message(
            f'#{int(row["cv_rank"])} {row["model"]} '
            f'[{row["model_family"]}] -> '
            f'WAPE={row["WAPE_mean"]:.3f} | '
            f'MAE={row["MAE_mean"]:.3f}.',
            verbose=verbose,
        )

    _print_message(
        f'Current development winner: {selected_name} '
        f'[{selected_family}] with WAPE='
        f'{cv_model_summary.iloc[0]["WAPE_mean"]:.3f}.',
        verbose=verbose,
    )

    frozen_adaptive_mapping = pd.DataFrame()
    frozen_global_model = _adaptive_default_model(
        baseline_specs,
        rules,
    )

    if (
        rules.run_adaptive_regime_selector
        and regime_column is not None
    ):
        frozen_adaptive_mapping, frozen_global_model = (
            freeze_adaptive_regime_mapping(
                baseline_predictions=cv_predictions,
                ml_predictions=ml_predictions,
                baseline_specs=baseline_specs,
                target_column=target_column,
                regime_column=regime_column,
                rules=rules,
            )
        )

    final_metrics = pd.DataFrame()
    final_predictions = pd.DataFrame()
    final_regime_metrics = pd.DataFrame()
    final_period_metrics = pd.DataFrame()

    if rules.evaluate_final_test:
        _print_subheader(
            'FINAL TEST | frozen development decision',
            verbose=verbose,
        )

        if not frozen_adaptive_mapping.empty:
            _print_message(
                'Adaptive regime mapping frozen using Development OOF only:',
                verbose=verbose,
            )

            for _, row in frozen_adaptive_mapping.iterrows():
                _print_message(
                    f'{row["regime"]} -> {row["selected_model"]} '
                    f'(source={row["freeze_source"]}).',
                    verbose=verbose,
                )

        (
            final_metrics,
            final_predictions,
            final_regime_metrics,
            final_period_metrics,
        ) = evaluate_frozen_models_on_final_test(
            development_frame=development,
            final_test_frame=final_test,
            baseline_specs=baseline_specs,
            ml_specs=ml_specs,
            frozen_adaptive_mapping=frozen_adaptive_mapping,
            frozen_global_model=frozen_global_model,
            primary_model_name=selected_name,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            target_column=target_column,
            period_column=period_column,
            entity_column=entity_column,
            regime_column=regime_column,
            rules=rules,
        )

        primary_row = final_metrics.loc[
            final_metrics[
                'evaluation_role'
            ].eq(
                'primary_frozen_from_development'
            )
        ]

        if not primary_row.empty:
            row = primary_row.iloc[0]

            _print_message(
                f'PRIMARY FINAL TEST | {row["model"]}: '
                f'WAPE={row["WAPE"]:.3f} | '
                f'MAE={row["MAE"]:.3f} | '
                f'RMSE={row["RMSE"]:.3f} | '
                f'Bias={row["Bias"]:.3f}.',
                level='WARNING',
                verbose=verbose,
            )

        _print_message(
            'Final holdout has now been opened. Comparator test metrics are '
            'diagnostic only and MUST NOT be used to re-select the primary '
            'model.',
            level='WARNING',
            verbose=verbose,
        )
    else:
        _print_message(
            'Final holdout remains untouched. The development winner and '
            'adaptive mapping may now be frozen before enabling final-test '
            'evaluation.',
            verbose=verbose,
        )

    tables_dir = _ensure_directory(paths.tables_dir)
    reports_dir = _ensure_directory(paths.reports_dir)
    predictions_dir = _ensure_directory(paths.predictions_dir)

    outputs: dict[str, pd.DataFrame] = {
        'split_manifest': split_manifest,
        'cv_manifest': cv_manifest,
        'feature_selection': feature_selection,
        'cv_baseline_metrics': cv_metrics,
        'cv_baseline_summary': cv_summary,
        'cv_regime_metrics': regime_metrics,
        'cv_period_metrics': period_metrics,
        'cv_baseline_predictions': cv_predictions,
        'cv_ml_metrics': ml_metrics,
        'cv_ml_summary': ml_summary,
        'cv_ml_regime_metrics': ml_regime_metrics,
        'cv_ml_period_metrics': ml_period_metrics,
        'cv_ml_predictions': ml_predictions,
        'cv_adaptive_metrics': adaptive_metrics,
        'cv_adaptive_summary': adaptive_summary,
        'cv_adaptive_regime_metrics': adaptive_regime_metrics,
        'cv_adaptive_period_metrics': adaptive_period_metrics,
        'cv_adaptive_selection': adaptive_selection,
        'cv_adaptive_predictions': adaptive_predictions,
        'cv_model_metrics': cv_model_metrics,
        'cv_model_summary': cv_model_summary,
        'frozen_adaptive_mapping': frozen_adaptive_mapping,
    }

    if not final_metrics.empty:
        outputs['final_test_metrics'] = final_metrics
        outputs['final_test_predictions'] = final_predictions
        outputs['final_test_regime_metrics'] = final_regime_metrics
        outputs['final_test_period_metrics'] = final_period_metrics

    saved_tables: dict[str, str] = {}
    saved_predictions: dict[str, str] = {}

    if rules.save_tables:
        table_names = [
            'split_manifest',
            'cv_manifest',
            'feature_selection',
            'cv_baseline_metrics',
            'cv_baseline_summary',
            'cv_regime_metrics',
            'cv_period_metrics',
            'cv_ml_metrics',
            'cv_ml_summary',
            'cv_ml_regime_metrics',
            'cv_ml_period_metrics',
            'cv_adaptive_metrics',
            'cv_adaptive_summary',
            'cv_adaptive_regime_metrics',
            'cv_adaptive_period_metrics',
            'cv_adaptive_selection',
            'cv_model_metrics',
            'cv_model_summary',
            'frozen_adaptive_mapping',
            'final_test_metrics',
            'final_test_regime_metrics',
            'final_test_period_metrics',
        ]

        for name in table_names:
            if name not in outputs or outputs[name].empty:
                continue

            path = tables_dir / f'{name}.csv'
            outputs[name].to_csv(path, index=False)
            saved_tables[name] = str(path)

    if rules.save_predictions:
        prediction_names = [
            'cv_baseline_predictions',
            'cv_ml_predictions',
            'cv_adaptive_predictions',
            'final_test_predictions',
        ]

        for name in prediction_names:
            if name not in outputs or outputs[name].empty:
                continue

            path = predictions_dir / f'{name}.parquet'
            outputs[name].to_parquet(path, index=False)
            saved_predictions[name] = str(path)

    report = {
        'rules': asdict(rules),
        'dataset_name': rules.dataset_name,
        'target_column': target_column,
        'entity_column': entity_column,
        'period_column': period_column,
        'period_start_column': start_column,
        'period_end_column': end_column,
        'regime_column': regime_column,
        'eligible_rows': len(modeling_frame),
        'eligible_periods': int(modeling_frame[period_column].nunique()),
        'development_rows': len(development),
        'development_periods': int(development_periods),
        'final_test_rows': len(final_test),
        'final_test_periods': int(final_test_periods),
        'final_test_evaluated': bool(rules.evaluate_final_test),
        'final_test_primary_model': (
            selected_name
            if rules.evaluate_final_test
            else None
        ),
        'actual_cv_splits': len(folds),
        'selected_predictors': {
            'categorical': categorical_features,
            'numeric': numeric_features,
            'total': len(categorical_features) + len(numeric_features),
        },
        'baselines': baseline_specs,
        'best_baseline_by_cv_wape': {
            'model': selected_baseline_name,
        },
        'best_baseline_cv_metrics': _json_safe(
            cv_summary.iloc[0].to_dict()
        ),
        'ml_candidates': ml_specs,
        'ml_candidates_skipped': skipped_ml_specs,
        'adaptive_selector': {
            'enabled': bool(
                rules.run_adaptive_regime_selector
            ),
            'model_name': ADAPTIVE_MODEL_NAME,
            'initial_default_model': (
                _adaptive_default_model(
                    baseline_specs,
                    rules,
                )
                if baseline_specs
                else None
            ),
            'selection_rows': int(
                len(
                    adaptive_selection
                )
            ),
            'cv_metrics': (
                _json_safe(
                    adaptive_summary.iloc[
                        0
                    ].to_dict()
                )
                if not adaptive_summary.empty
                else None
            ),
        },
        'best_model_by_development_cv_wape': {
            'model': selected_name,
            'model_family': selected_family,
        },
        'best_model_cv_metrics': _json_safe(
            cv_model_summary.iloc[0].to_dict()
        ),
        'frozen_adaptive_mapping': _json_safe(
            frozen_adaptive_mapping.to_dict(
                orient='records'
            )
        ),
        'frozen_adaptive_global_fallback_model': frozen_global_model,
        'final_test_primary_metrics': (
            _json_safe(
                final_metrics.loc[
                    final_metrics[
                        'evaluation_role'
                    ].eq(
                        'primary_frozen_from_development'
                    )
                ].iloc[
                    0
                ].to_dict()
            )
            if (
                rules.evaluate_final_test
                and not final_metrics.empty
                and bool(
                    final_metrics[
                        'evaluation_role'
                    ].eq(
                        'primary_frozen_from_development'
                    ).any()
                )
            )
            else None
        ),
        'development_leaderboard': _json_safe(
            cv_model_summary.to_dict(
                orient='records'
            )
        ),
        'methodology': {
            'final_test_isolation': (
                'The final temporal holdout is removed before cross-validation '
                'and is not evaluated unless explicitly enabled.'
            ),
            'feature_selection': (
                'Predictors originate from safe_default catalogue entries and '
                'development-only missingness/constant checks.'
            ),
            'cv': (
                'Expanding-window validation uses only development periods; '
                'train periods always precede validation periods.'
            ),
            'baseline_fallback': (
                'When a causal baseline history feature is missing, the '
                'training-fold target median is used if enabled. The fallback '
                'fraction is reported explicitly.'
            ),
            'ml_preprocessing': (
                'Numeric imputation and categorical one-hot encoding are fit '
                'inside each training fold only. Unknown validation categories '
                'are ignored by the encoder and their incidence is reported.'
            ),
            'hgb_internal_validation': (
                'HistGradientBoosting uses early_stopping=False so it does not '
                'create an internal random validation split that would violate '
                'the temporal evaluation design.'
            ),
            'candidate_comparison': (
                'Baselines and machine-learning candidates are evaluated on '
                'the same development folds and ranked by mean WAPE.'
            ),
            'adaptive_regime_selector': (
                'For each validation fold, the regime-specific model is '
                'selected only from out-of-fold evidence produced by earlier '
                'validation folds. If regime evidence is insufficient, the '
                'selector falls back to earlier global OOF evidence; before '
                'any OOF evidence exists it uses a predefined causal baseline.'
            ),
            'adaptive_selector_no_same_fold_selection': (
                'The validation performance of a fold is never used to choose '
                'the model applied to that same fold.'
            ),
            'final_adaptive_freeze': (
                'Before opening the final test, each demand regime is assigned '
                'a model using pooled Development OOF evidence only. This '
                'mapping is then frozen.'
            ),
            'final_test_interpretation': (
                'The Development-CV winner is the primary final-test model. '
                'Other competitors may be reported on test for diagnostic '
                'comparison only and cannot replace the preselected primary '
                'model after test is opened.'
            ),
            'top_k_scope': (
                'Top-K overlap is computed on the observed article-period '
                'panel only; absent article-period rows are not assumed zero.'
            ),
        },
        'saved_tables': saved_tables,
        'saved_predictions': saved_predictions,
    }

    if rules.save_report:
        report_path = reports_dir / 'modeling_development_report.json'

        with report_path.open('w', encoding='utf-8') as handle:
            json.dump(
                _json_safe(report),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        _print_message(
            f'Modeling foundation report: {report_path}',
            verbose=verbose,
        )

    _print_message(
        (
            'Modeling completed. Development selection was frozen before the '
            'final holdout was evaluated.'
            if rules.evaluate_final_test
            else
            'Model development completed. Baselines, ML candidates and the '
            'sequential adaptive selector were compared without using '
            'final-test information.'
        ),
        verbose=verbose,
    )

    return outputs
