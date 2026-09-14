"""
Operational next-period forecasting for article demand.

This module converts the frozen modeling decision into an actionable forecast
table for the next article-reporting period.

Important methodological separation
-------------------------------------
* Model selection and the adaptive regime mapping are frozen before final-test
  evaluation.
* After evaluation is finished, deployment models may be refit on all eligible
  historical observations, including the former final-test period, because
  those observations are now part of the known past.
* The frozen model-selection rule is NOT changed using final-test performance.

The operational target is article demand (`units`) per future report period.
Ingredient quantities cannot be inferred unless a recipe/BOM mapping from
articles to ingredients is supplied separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .ap_features import (
        FeatureRules,
        build_article_period_feature_table,
    )
    from .ap_modeling import (
        ADAPTIVE_MODEL_NAME,
        ModelingRules,
        _fit_predict_ml,
        discover_baseline_specs,
        discover_ml_candidate_specs,
    )
except ImportError:
    from ap_features import (
        FeatureRules,
        build_article_period_feature_table,
    )
    from ap_modeling import (
        ADAPTIVE_MODEL_NAME,
        ModelingRules,
        _fit_predict_ml,
        discover_baseline_specs,
        discover_ml_candidate_specs,
    )


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class OperationalPaths:
    forecasts_dir: Path
    reports_dir: Path


@dataclass
class OperationalRules:
    """
    Rules for the deployment-style forecast.

    `expected_period_days=None` means infer the modal positive duration from
    historical report periods.
    """

    expected_period_days: int | None = None
    article_scope: str = 'all_observed'

    round_prediction_to_units: bool = True
    abc_threshold_a: float = 0.80
    abc_threshold_b: float = 0.95

    default_fallback_model: str = 'Historical expanding mean'

    save_forecast: bool = True
    save_summary: bool = True
    save_report: bool = True

    def validate(self) -> None:
        if (
            self.expected_period_days is not None
            and self.expected_period_days < 1
        ):
            raise ValueError(
                'expected_period_days must be >= 1 or None.'
            )

        if self.article_scope not in {
            'all_observed',
            'latest_period_observed',
        }:
            raise ValueError(
                "article_scope must be 'all_observed' or "
                "'latest_period_observed'."
            )

        if not 0 < self.abc_threshold_a < self.abc_threshold_b < 1:
            raise ValueError(
                'ABC thresholds must satisfy 0 < A < B < 1.'
            )


# =============================================================================
# SMALL HELPERS
# =============================================================================


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        errors='coerce',
    )


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.astype(
            'boolean'
        )

    normalised = (
        series
        .astype('string')
        .str.strip()
        .str.lower()
    )

    mapping = {
        'true': True,
        '1': True,
        'yes': True,
        'y': True,
        'si': True,
        'sí': True,
        'false': False,
        '0': False,
        'no': False,
        'n': False,
    }

    return normalised.map(
        mapping
    ).astype(
        'boolean'
    )


def _json_safe(value: Any) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        if pd.isna(
            value
        ):
            return None

        return value.item()

    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        if pd.isna(
            value
        ):
            return None

        return pd.Timestamp(
            value
        ).isoformat()

    if pd.isna(
        value
    ) if not isinstance(
        value,
        (str, bytes, Path)
    ) else False:
        return None

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    return value


def _ensure_dirs(
    paths: OperationalPaths,
) -> tuple[Path, Path]:
    forecasts_dir = Path(
        paths.forecasts_dir
    )
    reports_dir = Path(
        paths.reports_dir
    )

    forecasts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return forecasts_dir, reports_dir


def _infer_expected_period_days(
    article_gold: pd.DataFrame,
    rules: OperationalRules,
) -> int:
    if rules.expected_period_days is not None:
        return int(
            rules.expected_period_days
        )

    if 'period_days' in article_gold.columns:
        duration = _to_numeric(
            article_gold[
                'period_days'
            ]
        )
    else:
        duration = (
            _to_datetime(
                article_gold[
                    'report_end'
                ]
            )
            - _to_datetime(
                article_gold[
                    'report_start'
                ]
            )
        ).dt.days + 1

    duration = duration.loc[
        duration.gt(
            0
        )
        & duration.notna()
    ].round().astype(
        int
    )

    if duration.empty:
        raise ValueError(
            'Could not infer a positive historical report-period duration.'
        )

    modes = duration.mode()

    if modes.empty:
        return int(
            round(
                float(
                    duration.median()
                )
            )
        )

    return int(
        modes.iloc[
            0
        ]
    )



def _period_table(
    article_gold: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per historical report period."""
    columns = [
        'report_start',
        'report_end',
    ]

    optional = [
        column
        for column in [
            'period_days',
            'period_is_complete',
        ]
        if column in article_gold.columns
    ]

    periods = (
        article_gold[
            columns + optional
        ]
        .copy()
    )

    periods[
        'report_start'
    ] = _to_datetime(
        periods[
            'report_start'
        ]
    )
    periods[
        'report_end'
    ] = _to_datetime(
        periods[
            'report_end'
        ]
    )

    aggregations: dict[str, Any] = {}

    if 'period_days' in periods.columns:
        aggregations[
            'period_days'
        ] = 'first'

    if 'period_is_complete' in periods.columns:
        periods[
            'period_is_complete'
        ] = _safe_bool(
            periods[
                'period_is_complete'
            ]
        )
        aggregations[
            'period_is_complete'
        ] = 'all'

    if aggregations:
        periods = (
            periods
            .groupby(
                [
                    'report_start',
                    'report_end',
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                aggregations
            )
        )
    else:
        periods = periods.drop_duplicates(
            subset=[
                'report_start',
                'report_end',
            ]
        )

    return periods.sort_values(
        [
            'report_start',
            'report_end',
        ]
    ).reset_index(
        drop=True
    )


def _complete_period_table(
    article_gold: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer explicitly complete periods for deployment-history features."""
    periods = _period_table(
        article_gold
    )

    if 'period_is_complete' not in periods.columns:
        return periods

    complete = periods.loc[
        _safe_bool(
            periods[
                'period_is_complete'
            ]
        ).fillna(
            False
        )
    ].copy()

    return (
        complete
        if not complete.empty
        else periods
    )


def _infer_period_cadence_days(
    article_gold: pd.DataFrame,
    expected_period_days: int,
) -> int:
    """
    Infer start-to-start cadence from complete historical periods.

    This preserves the restaurant's reporting alignment. For example, a
    Monday-Sunday history remains Monday-Sunday even when the latest raw
    period is only Monday-Thursday.
    """
    periods = _complete_period_table(
        article_gold
    )

    starts = (
        periods[
            'report_start'
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    deltas = starts.diff().dt.days

    deltas = deltas.loc[
        deltas.gt(
            0
        )
    ]

    if deltas.empty:
        return int(
            expected_period_days
        )

    mode = deltas.mode()

    if not mode.empty:
        cadence = int(
            mode.iloc[
                0
            ]
        )
    else:
        cadence = int(
            round(
                float(
                    deltas.median()
                )
            )
        )

    return max(
        1,
        cadence,
    )


def _next_aligned_period(
    article_gold: pd.DataFrame,
    expected_period_days: int,
) -> dict[str, Any]:
    """
    Find the next standard report period after all currently observed data.

    The anchor is the latest complete historical period, not an incomplete
    trailing period. Start-to-start cadence is inferred from complete history.
    """
    all_periods = _period_table(
        article_gold
    )
    complete_periods = _complete_period_table(
        article_gold
    )

    if all_periods.empty or complete_periods.empty:
        raise ValueError(
            'Cannot infer the next aligned report period.'
        )

    latest_observed_end = pd.Timestamp(
        all_periods[
            'report_end'
        ].max()
    )

    latest_complete = complete_periods.sort_values(
        [
            'report_start',
            'report_end',
        ]
    ).iloc[
        -1
    ]

    cadence_days = _infer_period_cadence_days(
        article_gold,
        expected_period_days=expected_period_days,
    )

    future_start = pd.Timestamp(
        latest_complete[
            'report_start'
        ]
    )

    while future_start <= latest_observed_end:
        future_start = (
            future_start
            + pd.Timedelta(
                days=cadence_days
            )
        )

    future_end = (
        future_start
        + pd.Timedelta(
            days=expected_period_days - 1
        )
    )

    return {
        'forecast_period_start': future_start,
        'forecast_period_end': future_end,
        'cadence_days': cadence_days,
        'latest_observed_end': latest_observed_end,
        'latest_complete_start': pd.Timestamp(
            latest_complete[
                'report_start'
            ]
        ),
        'latest_complete_end': pd.Timestamp(
            latest_complete[
                'report_end'
            ]
        ),
    }


def _complete_history_rows(
    article_gold: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Exclude incomplete historical periods from deployment lag/rolling history.

    They remain in Gold for auditability, but are not treated as full-period
    demand when building the next complete-period forecast.
    """
    if 'period_is_complete' not in article_gold.columns:
        return (
            article_gold.copy(),
            {
                'excluded_incomplete_periods': 0,
                'excluded_incomplete_rows': 0,
            },
        )

    complete_mask = _safe_bool(
        article_gold[
            'period_is_complete'
        ]
    ).fillna(
        False
    )

    excluded = article_gold.loc[
        ~complete_mask
    ].copy()

    return (
        article_gold.loc[
            complete_mask
        ].copy(),
        {
            'excluded_incomplete_periods': int(
                excluded[
                    [
                        'report_start',
                        'report_end',
                    ]
                ]
                .drop_duplicates()
                .shape[
                    0
                ]
            ),
            'excluded_incomplete_rows': int(
                len(
                    excluded
                )
            ),
        },
    )


def _latest_historical_period(
    article_gold: pd.DataFrame,
) -> dict[str, Any]:
    periods = (
        article_gold[
            [
                'report_start',
                'report_end',
            ]
        ]
        .copy()
        .drop_duplicates()
    )

    periods[
        'report_start'
    ] = _to_datetime(
        periods[
            'report_start'
        ]
    )
    periods[
        'report_end'
    ] = _to_datetime(
        periods[
            'report_end'
        ]
    )

    periods = periods.sort_values(
        [
            'report_end',
            'report_start',
        ]
    ).reset_index(
        drop=True
    )

    if periods.empty:
        raise ValueError(
            'Article Gold contains no report periods.'
        )

    latest = periods.iloc[
        -1
    ]

    mask = (
        _to_datetime(
            article_gold[
                'report_start'
            ]
        ).eq(
            latest[
                'report_start'
            ]
        )
        & _to_datetime(
            article_gold[
                'report_end'
            ]
        ).eq(
            latest[
                'report_end'
            ]
        )
    )

    period_rows = article_gold.loc[
        mask
    ]

    complete = None

    if 'period_is_complete' in period_rows.columns:
        complete_values = _safe_bool(
            period_rows[
                'period_is_complete'
            ]
        ).dropna()

        if not complete_values.empty:
            complete = bool(
                complete_values.all()
            )

    return {
        'report_start': latest[
            'report_start'
        ],
        'report_end': latest[
            'report_end'
        ],
        'period_is_complete': complete,
        'rows': int(
            len(
                period_rows
            )
        ),
    }


def _latest_rows_per_article(
    article_gold: pd.DataFrame,
) -> pd.DataFrame:
    data = article_gold.copy()

    data[
        'report_start'
    ] = _to_datetime(
        data[
            'report_start'
        ]
    )
    data[
        'report_end'
    ] = _to_datetime(
        data[
            'report_end'
        ]
    )

    return (
        data
        .sort_values(
            [
                'article_code',
                'report_end',
                'report_start',
            ]
        )
        .groupby(
            'article_code',
            as_index=False,
            sort=False,
        )
        .tail(
            1
        )
        .reset_index(
            drop=True
        )
    )


def _forecast_article_universe(
    article_gold: pd.DataFrame,
    rules: OperationalRules,
) -> pd.DataFrame:
    if rules.article_scope == 'all_observed':
        return _latest_rows_per_article(
            article_gold
        )

    latest_period = _latest_historical_period(
        article_gold
    )

    mask = (
        _to_datetime(
            article_gold[
                'report_start'
            ]
        ).eq(
            latest_period[
                'report_start'
            ]
        )
        & _to_datetime(
            article_gold[
                'report_end'
            ]
        ).eq(
            latest_period[
                'report_end'
            ]
        )
    )

    return article_gold.loc[
        mask
    ].copy().reset_index(
        drop=True
    )


def _build_future_gold_scaffold(
    article_gold: pd.DataFrame,
    feature_rules: FeatureRules,
    operational_rules: OperationalRules,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build one future Gold-like row per article without inventing demand.

    Static/current metadata are carried from the latest observed article row.
    Current-period outcomes are cleared before feature engineering.
    """
    required = {
        'report_start',
        'report_end',
        'article_code',
    }

    missing = required.difference(
        article_gold.columns
    )

    if missing:
        raise KeyError(
            'Article Gold is missing required columns: '
            f'{sorted(missing)}'
        )

    expected_days = _infer_expected_period_days(
        article_gold,
        operational_rules,
    )
    latest = _latest_historical_period(
        article_gold
    )
    aligned = _next_aligned_period(
        article_gold,
        expected_period_days=expected_days,
    )

    future_start = aligned[
        'forecast_period_start'
    ]
    future_end = aligned[
        'forecast_period_end'
    ]

    future = _forecast_article_universe(
        article_gold,
        operational_rules,
    )

    future[
        'report_start'
    ] = future_start
    future[
        'report_end'
    ] = future_end

    if 'period_days' in future.columns:
        future[
            'period_days'
        ] = expected_days

    if 'period_is_complete' in future.columns:
        future[
            'period_is_complete'
        ] = True

    if 'period_is_complete_week' in future.columns:
        future[
            'period_is_complete_week'
        ] = (
            expected_days
            == 7
        )

    # Clear current-period outcomes. These values are unknown for the future.
    outcome_columns = set(
        feature_rules.article_target_candidates
    )
    outcome_columns.update(
        {
            'invitation',
            'invitations',
            'invited_units',
        }
    )

    for column in outcome_columns:
        if column in future.columns:
            future[
                column
            ] = np.nan

    # Quality flags attached to unknown outcomes must not be copied forward.
    for column in future.columns:
        name = str(
            column
        ).lower()

        if (
            (
                'units' in name
                or 'amount' in name
            )
            and (
                '__negative' in name
                or '__outlier' in name
                or '__invalid' in name
            )
        ):
            future[
                column
            ] = False

    # Avoid pretending that provenance of the last historical row belongs to
    # the future row. Provenance is not used for prediction.
    for column in future.columns:
        if (
            'source_file' in str(
                column
            ).lower()
            or 'terminal_start' in str(
                column
            ).lower()
            or 'terminal_end' in str(
                column
            ).lower()
        ):
            future[
                column
            ] = pd.NA

    report = {
        'expected_period_days': expected_days,
        'latest_historical_period': latest,
        'forecast_period_start': future_start,
        'forecast_period_end': future_end,
        'forecast_articles': int(
            future[
                'article_code'
            ].nunique(
                dropna=True
            )
        ),
        'article_scope': operational_rules.article_scope,
        'trailing_period_complete': latest[
            'period_is_complete'
        ],
        'inferred_cadence_days': aligned[
            'cadence_days'
        ],
        'latest_observed_end': aligned[
            'latest_observed_end'
        ],
        'latest_complete_period_start': aligned[
            'latest_complete_start'
        ],
        'latest_complete_period_end': aligned[
            'latest_complete_end'
        ],
        'forecast_anchor_policy': (
            'next cadence-aligned complete period after latest observed date'
        ),
    }

    return future, report


def _merge_daily_context(
    daily_gold: pd.DataFrame | None,
    future_daily_context: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if daily_gold is None and future_daily_context is None:
        return None

    frames: list[pd.DataFrame] = []

    if daily_gold is not None:
        frames.append(
            daily_gold.copy()
        )

    if future_daily_context is not None:
        frames.append(
            future_daily_context.copy()
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    if 'date' in combined.columns:
        combined[
            'date'
        ] = _to_datetime(
            combined[
                'date'
            ]
        )

        # Explicit future context is appended last and therefore wins.
        combined = (
            combined
            .sort_index()
            .drop_duplicates(
                subset=[
                    'date',
                ],
                keep='last',
            )
            .sort_values(
                'date'
            )
            .reset_index(
                drop=True
            )
        )

    return combined


def build_next_period_features(
    article_gold: pd.DataFrame,
    daily_gold: pd.DataFrame | None,
    feature_rules: FeatureRules,
    operational_rules: OperationalRules,
    future_daily_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build the future article-period feature rows with the same feature engine
    used for historical modeling.
    """
    future_gold, forecast_report = _build_future_gold_scaffold(
        article_gold=article_gold,
        feature_rules=feature_rules,
        operational_rules=operational_rules,
    )

    complete_history, exclusion_report = _complete_history_rows(
        article_gold
    )

    forecast_report.update(
        exclusion_report
    )

    combined_gold = pd.concat(
        [
            complete_history,
            future_gold,
        ],
        ignore_index=True,
        sort=False,
    )

    context = _merge_daily_context(
        daily_gold=daily_gold,
        future_daily_context=future_daily_context,
    )

    feature_table, catalogue, feature_report = (
        build_article_period_feature_table(
            combined_gold,
            daily_gold=context,
            rules=feature_rules,
        )
    )

    future_mask = (
        _to_datetime(
            feature_table[
                'report_start'
            ]
        ).eq(
            pd.Timestamp(
                forecast_report[
                    'forecast_period_start'
                ]
            )
        )
        & _to_datetime(
            feature_table[
                'report_end'
            ]
        ).eq(
            pd.Timestamp(
                forecast_report[
                    'forecast_period_end'
                ]
            )
        )
    )

    future_features = feature_table.loc[
        future_mask
    ].copy().reset_index(
        drop=True
    )

    if future_features.empty:
        raise RuntimeError(
            'Future feature engineering produced no rows.'
        )

    forecast_report[
        'feature_rows'
    ] = int(
        len(
            future_features
        )
    )
    forecast_report[
        'feature_columns'
    ] = int(
        len(
            future_features.columns
        )
    )
    forecast_report[
        'feature_target_column'
    ] = feature_report.get(
        'target_column'
    )

    if 'context_coverage_fraction' in future_features.columns:
        context_coverage = _to_numeric(
            future_features[
                'context_coverage_fraction'
            ]
        )

        forecast_report[
            'future_context_coverage_mean'
        ] = (
            float(
                context_coverage.mean()
            )
            if context_coverage.notna().any()
            else None
        )
    else:
        forecast_report[
            'future_context_coverage_mean'
        ] = None

    return future_features, catalogue, forecast_report


# =============================================================================
# FROZEN MODEL RULES
# =============================================================================


def _load_frame(
    value: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    if isinstance(
        value,
        pd.DataFrame,
    ):
        return value.copy()

    return pd.read_csv(
        value
    )


def _load_json(
    value: dict[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )

    with Path(
        value
    ).open(
        'r',
        encoding='utf-8',
    ) as handle:
        return json.load(
            handle
        )


def _selected_model_features(
    feature_selection: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    required = {
        'column',
        'selected_for_modeling',
    }

    missing = required.difference(
        feature_selection.columns
    )

    if missing:
        raise KeyError(
            'feature_selection is missing required columns: '
            f'{sorted(missing)}'
        )

    selected_mask = _safe_bool(
        feature_selection[
            'selected_for_modeling'
        ]
    ).fillna(
        False
    )

    selected = feature_selection.loc[
        selected_mask
    ].copy()

    if selected.empty:
        raise ValueError(
            'No frozen modeling features are selected.'
        )

    if 'feature_dtype' not in selected.columns:
        selected[
            'feature_dtype'
        ] = 'numeric'

    if 'role' not in selected.columns:
        selected[
            'role'
        ] = 'feature'

    categorical_roles = {
        'categorical_identifier',
        'adaptive_demand_regime',
        'static_or_scheduled_metadata',
        'known_context',
    }

    categorical_mask = (
        selected[
            'feature_dtype'
        ].eq(
            'categorical_or_text'
        )
        | selected[
            'feature_dtype'
        ].eq(
            'boolean'
        )
        | selected[
            'role'
        ].isin(
            categorical_roles
        )
    )

    categorical = selected.loc[
        categorical_mask,
        'column',
    ].astype(
        str
    ).tolist()

    numeric = selected.loc[
        ~categorical_mask,
        'column',
    ].astype(
        str
    ).tolist()

    all_features = categorical + numeric

    return categorical, numeric, all_features


def _frozen_mapping_lookup(
    frozen_mapping: pd.DataFrame,
) -> dict[str, str]:
    required = {
        'regime',
        'selected_model',
    }

    missing = required.difference(
        frozen_mapping.columns
    )

    if missing:
        raise KeyError(
            'Frozen adaptive mapping is missing required columns: '
            f'{sorted(missing)}'
        )

    return {
        str(
            regime
        ): str(
            model
        )
        for regime, model in zip(
            frozen_mapping[
                'regime'
            ],
            frozen_mapping[
                'selected_model'
            ],
        )
    }



def _load_optional_frame(
    value: pd.DataFrame | str | Path | None,
) -> pd.DataFrame | None:
    if value is None:
        return None

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return value.copy()

    path = Path(
        value
    )

    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(
            path
        )

    return pd.read_csv(
        path
    )


def _apply_article_scope(
    future_features: pd.DataFrame,
    article_scope_audit: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply a previously audited forecast universe.

    The audit is authoritative for scope only when supplied explicitly.
    This function does not infer inactivity from recency on its own.
    """
    if article_scope_audit is None:
        return (
            future_features.copy(),
            {
                'scope_applied': False,
                'articles_before_scope': int(
                    len(
                        future_features
                    )
                ),
                'articles_after_scope': int(
                    len(
                        future_features
                    )
                ),
                'manual_review_articles': None,
            },
        )

    required = {
        'article_code',
        'forecast_eligible_recommended',
    }

    missing = required.difference(
        article_scope_audit.columns
    )

    if missing:
        raise KeyError(
            'Article-scope audit is missing required columns: '
            f'{sorted(missing)}'
        )

    audit = article_scope_audit.copy()
    audit[
        'forecast_eligible_recommended'
    ] = _safe_bool(
        audit[
            'forecast_eligible_recommended'
        ]
    ).fillna(
        False
    )

    eligible = audit.loc[
        audit[
            'forecast_eligible_recommended'
        ]
    ].copy()

    metadata_columns = [
        column
        for column in [
            'article_code',
            'activity_status',
            'manual_review_required',
            'forecast_scope_reason',
            'periods_since_last_observed',
            'periods_observed',
            'last_observed_period_end',
        ]
        if column in eligible.columns
    ]

    eligible = eligible[
        metadata_columns
    ].drop_duplicates(
        subset=[
            'article_code',
        ],
        keep='last',
    )

    before = int(
        len(
            future_features
        )
    )

    scoped = future_features.merge(
        eligible,
        on='article_code',
        how='inner',
        validate='many_to_one',
    )

    after = int(
        len(
            scoped
        )
    )

    manual_review_articles = None

    if 'manual_review_required' in scoped.columns:
        manual_review_articles = int(
            _safe_bool(
                scoped[
                    'manual_review_required'
                ]
            ).fillna(
                False
            ).sum()
        )

    return (
        scoped.reset_index(
            drop=True
        ),
        {
            'scope_applied': True,
            'articles_before_scope': before,
            'articles_after_scope': after,
            'articles_excluded_by_scope': before - after,
            'manual_review_articles': manual_review_articles,
        },
    )


# =============================================================================
# OPERATIONAL PREDICTION
# =============================================================================


def _baseline_prediction_for_future(
    training_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    modeling_rules: ModelingRules,
) -> tuple[pd.Series, pd.Series]:
    prediction = _to_numeric(
        future_frame[
            source_column
        ]
    ).astype(
        float
    )

    fallback_mask = prediction.isna()

    if (
        modeling_rules.baseline_train_median_fallback
        and fallback_mask.any()
    ):
        fallback_value = float(
            _to_numeric(
                training_frame[
                    target_column
                ]
            )
            .dropna()
            .median()
        )

        prediction = prediction.fillna(
            fallback_value
        )

    if modeling_rules.nonnegative_predictions:
        prediction = prediction.clip(
            lower=0
        )

    return prediction, fallback_mask


def _candidate_future_predictions(
    training_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    target_column: str,
    categorical_features: list[str],
    numeric_features: list[str],
    modeling_rules: ModelingRules,
) -> dict[str, pd.DataFrame]:
    predictions: dict[str, pd.DataFrame] = {}

    baseline_specs = discover_baseline_specs(
        future_frame,
        target_column=target_column,
        rules=modeling_rules,
    )

    for spec in baseline_specs:
        prediction, fallback_mask = _baseline_prediction_for_future(
            training_frame=training_frame,
            future_frame=future_frame,
            source_column=spec[
                'source_column'
            ],
            target_column=target_column,
            modeling_rules=modeling_rules,
        )

        frame = future_frame.copy()
        frame[
            'prediction'
        ] = prediction
        frame[
            'used_fallback'
        ] = fallback_mask
        predictions[
            spec[
                'model'
            ]
        ] = frame

    ml_specs, _ = discover_ml_candidate_specs(
        training_frame,
        target_column=target_column,
        rules=modeling_rules,
    )

    for spec in ml_specs:
        prediction, _ = _fit_predict_ml(
            train_frame=train_frame_with_columns(
                training_frame,
                categorical_features + numeric_features,
            ),
            prediction_frame=train_frame_with_columns(
                future_frame,
                categorical_features + numeric_features,
            ),
            spec=spec,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            target_column=target_column,
            rules=modeling_rules,
        )

        frame = future_frame.copy()
        frame[
            'prediction'
        ] = prediction
        frame[
            'used_fallback'
        ] = False
        predictions[
            spec[
                'model'
            ]
        ] = frame

    return predictions


def train_frame_with_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Ensure a frame contains the frozen predictor set.

    Missing operational context features are left as NaN so the train-fitted
    preprocessing pipeline can impute them without fabricating future values.
    """
    result = frame.copy()

    for column in feature_columns:
        if column not in result.columns:
            result[
                column
            ] = np.nan

    return result


def _assign_operational_prediction(
    future_frame: pd.DataFrame,
    candidate_predictions: dict[str, pd.DataFrame],
    frozen_mapping: pd.DataFrame,
    global_fallback_model: str,
    regime_column: str,
) -> pd.DataFrame:
    mapping = _frozen_mapping_lookup(
        frozen_mapping
    )

    result = future_frame.copy()

    result[
        'selected_model'
    ] = result[
        regime_column
    ].astype(
        'string'
    ).map(
        mapping
    ).fillna(
        global_fallback_model
    )

    prediction = pd.Series(
        np.nan,
        index=result.index,
        dtype=float,
    )
    used_fallback = pd.Series(
        False,
        index=result.index,
        dtype=bool,
    )

    for model_name in result[
        'selected_model'
    ].dropna().unique():
        model_name = str(
            model_name
        )

        if model_name not in candidate_predictions:
            raise KeyError(
                f'Frozen model {model_name!r} has no operational prediction.'
            )

        mask = result[
            'selected_model'
        ].astype(
            str
        ).eq(
            model_name
        )

        source = candidate_predictions[
            model_name
        ]

        prediction.loc[
            mask
        ] = _to_numeric(
            source.loc[
                mask,
                'prediction',
            ]
        )

        used_fallback.loc[
            mask
        ] = _safe_bool(
            source.loc[
                mask,
                'used_fallback',
            ]
        ).fillna(
            False
        ).astype(
            bool
        )

    result[
        'predicted_units'
    ] = prediction.clip(
        lower=0
    )
    result[
        'used_prediction_fallback'
    ] = used_fallback

    return result


def _add_operational_ranking(
    forecast: pd.DataFrame,
    rules: OperationalRules,
) -> pd.DataFrame:
    result = forecast.copy()

    result = result.sort_values(
        [
            'predicted_units',
            'article_code',
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    result[
        'forecast_rank'
    ] = np.arange(
        1,
        len(
            result
        ) + 1,
        dtype='int64',
    )

    total = float(
        _to_numeric(
            result[
                'predicted_units'
            ]
        ).fillna(
            0
        ).sum()
    )

    if total > 0:
        result[
            'forecast_share'
        ] = (
            _to_numeric(
                result[
                    'predicted_units'
                ]
            ).fillna(
                0
            )
            / total
        )

        result[
            'forecast_cumulative_share'
        ] = result[
            'forecast_share'
        ].cumsum()
    else:
        result[
            'forecast_share'
        ] = 0.0
        result[
            'forecast_cumulative_share'
        ] = 0.0

    cumulative_before = (
        result[
            'forecast_cumulative_share'
        ]
        - result[
            'forecast_share'
        ]
    )

    result[
        'planning_class'
    ] = np.select(
        [
            cumulative_before.lt(
                rules.abc_threshold_a
            ),
            cumulative_before.lt(
                rules.abc_threshold_b
            ),
        ],
        [
            'A',
            'B',
        ],
        default='C',
    )

    if rules.round_prediction_to_units:
        result[
            'predicted_units_rounded'
        ] = (
            result[
                'predicted_units'
            ]
            .round()
            .clip(
                lower=0
            )
            .astype(
                'Int64'
            )
        )

    return result


def _compact_operational_columns(
    forecast: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    preferred = [
        'report_start',
        'report_end',
        'article_code',
        'article_name',
        'department_code',
        'department_name',
        'article_demand_regime_prior',
        'activity_status',
        'manual_review_required',
        'forecast_scope_reason',
        'periods_since_last_observed',
        'selected_model',
        'predicted_units',
        'predicted_units_rounded',
        'forecast_rank',
        'forecast_share',
        'forecast_cumulative_share',
        'planning_class',
        'used_prediction_fallback',
        f'{target_column}__lag_1',
        f'{target_column}__rolling_mean_4',
        f'{target_column}__expanding_mean_prior',
        'history_observation_rate',
        'periods_since_last_observation',
        'context_coverage_fraction',
    ]

    existing = [
        column
        for column in preferred
        if column in forecast.columns
    ]

    extras = [
        column
        for column in forecast.columns
        if column not in existing
        and column.startswith(
            'context_'
        )
    ]

    return forecast[
        existing + extras
    ].copy()


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def run_operational_forecast(
    article_gold: pd.DataFrame,
    daily_gold: pd.DataFrame | None,
    historical_feature_table: pd.DataFrame,
    feature_selection: pd.DataFrame | str | Path,
    frozen_adaptive_mapping: pd.DataFrame | str | Path,
    paths: OperationalPaths,
    feature_rules: FeatureRules | None = None,
    modeling_rules: ModelingRules | None = None,
    operational_rules: OperationalRules | None = None,
    modeling_report: dict[str, Any] | str | Path | None = None,
    future_daily_context: pd.DataFrame | None = None,
    article_scope_audit: pd.DataFrame | str | Path | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Refit the frozen architecture on all eligible history and forecast the next
    report period.

    This function is intended to run only after the final evaluation protocol
    has been completed and the model-selection rule is frozen.
    """
    feature_rules = (
        FeatureRules()
        if feature_rules is None
        else feature_rules
    )
    modeling_rules = (
        ModelingRules()
        if modeling_rules is None
        else modeling_rules
    )
    operational_rules = (
        OperationalRules()
        if operational_rules is None
        else operational_rules
    )

    operational_rules.validate()

    feature_selection_frame = _load_frame(
        feature_selection
    )
    frozen_mapping_frame = _load_frame(
        frozen_adaptive_mapping
    )
    modeling_report_data = _load_json(
        modeling_report
    )
    article_scope_frame = _load_optional_frame(
        article_scope_audit
    )

    (
        categorical_features,
        numeric_features,
        selected_features,
    ) = _selected_model_features(
        feature_selection_frame
    )

    target_column = None

    for candidate in feature_rules.article_target_candidates:
        if candidate in historical_feature_table.columns:
            target_column = candidate
            break

    if target_column is None:
        raise KeyError(
            'No operational target candidate exists in historical features.'
        )

    training_frame = historical_feature_table.copy()

    if 'training_eligible' in training_frame.columns:
        eligible = _safe_bool(
            training_frame[
                'training_eligible'
            ]
        ).fillna(
            False
        )

        training_frame = training_frame.loc[
            eligible
        ].copy()

    training_frame = training_frame.loc[
        _to_numeric(
            training_frame[
                target_column
            ]
        ).notna()
    ].copy()

    if training_frame.empty:
        raise ValueError(
            'No eligible historical rows are available for operational refit.'
        )

    future_features, _, forecast_report = build_next_period_features(
        article_gold=article_gold,
        daily_gold=daily_gold,
        feature_rules=feature_rules,
        operational_rules=operational_rules,
        future_daily_context=future_daily_context,
    )

    future_features, scope_report = _apply_article_scope(
        future_features=future_features,
        article_scope_audit=article_scope_frame,
    )

    if future_features.empty:
        raise ValueError(
            'Article-scope filtering removed every future article.'
        )

    forecast_report[
        'article_scope'
    ] = scope_report

    # Keep the exact frozen feature set. Missing future context remains NaN and
    # is handled by train-fitted preprocessing.
    training_frame = train_frame_with_columns(
        training_frame,
        selected_features,
    )
    future_features = train_frame_with_columns(
        future_features,
        selected_features,
    )

    candidate_predictions = _candidate_future_predictions(
        training_frame=training_frame,
        future_frame=future_features,
        target_column=target_column,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        modeling_rules=modeling_rules,
    )

    global_fallback_model = (
        modeling_report_data.get(
            'frozen_adaptive_global_fallback_model'
        )
        or operational_rules.default_fallback_model
    )

    if global_fallback_model not in candidate_predictions:
        if operational_rules.default_fallback_model in candidate_predictions:
            global_fallback_model = (
                operational_rules.default_fallback_model
            )
        else:
            global_fallback_model = next(
                iter(
                    candidate_predictions
                )
            )

    regime_column = 'article_demand_regime_prior'

    if regime_column not in future_features.columns:
        raise KeyError(
            f'Future feature table does not contain {regime_column!r}.'
        )

    forecast = _assign_operational_prediction(
        future_frame=future_features,
        candidate_predictions=candidate_predictions,
        frozen_mapping=frozen_mapping_frame,
        global_fallback_model=str(
            global_fallback_model
        ),
        regime_column=regime_column,
    )

    forecast = _add_operational_ranking(
        forecast,
        rules=operational_rules,
    )

    compact = _compact_operational_columns(
        forecast,
        target_column=target_column,
    )

    model_usage = (
        compact[
            'selected_model'
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            'selected_model'
        )
        .reset_index(
            name='article_count',
        )
    )

    regime_usage = (
        compact[
            regime_column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            'demand_regime'
        )
        .reset_index(
            name='article_count',
        )
    )

    summary = pd.DataFrame(
        [
            {
                'forecast_period_start': forecast_report[
                    'forecast_period_start'
                ],
                'forecast_period_end': forecast_report[
                    'forecast_period_end'
                ],
                'forecast_articles': len(
                    compact
                ),
                'scope_applied': bool(
                    scope_report[
                        'scope_applied'
                    ]
                ),
                'articles_excluded_by_scope': int(
                    scope_report.get(
                        'articles_excluded_by_scope',
                        0,
                    )
                ),
                'manual_review_articles': (
                    scope_report.get(
                        'manual_review_articles'
                    )
                ),
                'predicted_total_units': float(
                    _to_numeric(
                        compact[
                            'predicted_units'
                        ]
                    ).sum()
                ),
                'top_20_predicted_units': float(
                    _to_numeric(
                        compact.head(
                            20
                        )[
                            'predicted_units'
                        ]
                    ).sum()
                ),
                'top_20_predicted_share': float(
                    _to_numeric(
                        compact.head(
                            20
                        )[
                            'forecast_share'
                        ]
                    ).sum()
                ),
                'fallback_articles': int(
                    _safe_bool(
                        compact[
                            'used_prediction_fallback'
                        ]
                    ).fillna(
                        False
                    ).sum()
                ),
                'future_context_coverage_mean': forecast_report.get(
                    'future_context_coverage_mean'
                ),
            }
        ]
    )

    forecasts_dir, reports_dir = _ensure_dirs(
        paths
    )

    saved: dict[str, str] = {}

    if operational_rules.save_forecast:
        forecast_path = (
            forecasts_dir
            / 'next_period_article_forecast.csv'
        )
        compact.to_csv(
            forecast_path,
            index=False,
        )
        saved[
            'forecast'
        ] = str(
            forecast_path
        )

    if operational_rules.save_summary:
        summary_path = (
            forecasts_dir
            / 'next_period_forecast_summary.csv'
        )
        summary.to_csv(
            summary_path,
            index=False,
        )
        saved[
            'summary'
        ] = str(
            summary_path
        )

        model_usage_path = (
            forecasts_dir
            / 'next_period_model_usage.csv'
        )
        model_usage.to_csv(
            model_usage_path,
            index=False,
        )
        saved[
            'model_usage'
        ] = str(
            model_usage_path
        )

        regime_usage_path = (
            forecasts_dir
            / 'next_period_regime_usage.csv'
        )
        regime_usage.to_csv(
            regime_usage_path,
            index=False,
        )
        saved[
            'regime_usage'
        ] = str(
            regime_usage_path
        )

    operational_report = {
        'rules': asdict(
            operational_rules
        ),
        'target_column': target_column,
        'deployment_refit_rows': int(
            len(
                training_frame
            )
        ),
        'deployment_refit_periods': int(
            training_frame[
                'period_index'
            ].nunique()
            if 'period_index' in training_frame.columns
            else 0
        ),
        'frozen_selected_feature_count': int(
            len(
                selected_features
            )
        ),
        'categorical_feature_count': int(
            len(
                categorical_features
            )
        ),
        'numeric_feature_count': int(
            len(
                numeric_features
            )
        ),
        'frozen_adaptive_mapping': _json_safe(
            frozen_mapping_frame.to_dict(
                orient='records'
            )
        ),
        'article_scope': _json_safe(
            scope_report
        ),
        'global_fallback_model': str(
            global_fallback_model
        ),
        'forecast_period': _json_safe(
            forecast_report
        ),
        'forecast_summary': _json_safe(
            summary.iloc[
                0
            ].to_dict()
        ),
        'model_usage': _json_safe(
            model_usage.to_dict(
                orient='records'
            )
        ),
        'regime_usage': _json_safe(
            regime_usage.to_dict(
                orient='records'
            )
        ),
        'methodology': {
            'selection_rule_frozen': (
                'Model selection and the adaptive regime mapping are not '
                'changed using final-test performance.'
            ),
            'deployment_refit': (
                'After final evaluation, the frozen architecture is refit on '
                'all eligible historical rows because they are now known past.'
            ),
            'future_context': (
                'Unknown future context is left missing and handled by '
                'train-fitted preprocessing. Known future events/holidays or '
                'weather forecasts can be supplied through future_daily_context.'
            ),
            'article_scope': (
                'If an article-scope audit is supplied, only rows explicitly '
                'recommended as forecast-eligible are predicted. Stale '
                'historical articles can remain forecastable while being '
                'flagged for manual review.'
            ),
            'ingredient_scope': (
                'Predictions are article units. Ingredient purchasing requires '
                'an explicit article-to-recipe/BOM mapping and is not inferred.'
            ),
        },
        'warnings': [],
        'saved_outputs': saved,
    }

    if forecast_report.get(
        'trailing_period_complete'
    ) is False:
        operational_report[
            'warnings'
        ].append(
            'The latest raw report period is incomplete. It has been excluded '
            'from full-period lag/rolling demand history, and the forecast was '
            'aligned to the next standard reporting period.'
        )

    context_coverage = forecast_report.get(
        'future_context_coverage_mean'
    )

    if (
        context_coverage is None
        or float(
            context_coverage
        ) < 1.0
    ):
        operational_report[
            'warnings'
        ].append(
            'Future daily context coverage is incomplete. Supply known future '
            'events/holidays and, when available, weather forecasts through '
            'future_daily_context for richer operational predictions.'
        )

    if operational_rules.save_report:
        report_path = (
            reports_dir
            / 'operational_forecast_report.json'
        )

        with report_path.open(
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(
                _json_safe(
                    operational_report
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        saved[
            'report'
        ] = str(
            report_path
        )

    if verbose:
        print()
        print(
            '=' * 96
        )
        print(
            'FROZEN MODEL -> NEXT-PERIOD OPERATIONAL FORECAST'
        )
        print(
            '=' * 96
        )
        print(
            f'[INFO] Forecast period: '
            f'{pd.Timestamp(forecast_report["forecast_period_start"]).date()} '
            f'-> '
            f'{pd.Timestamp(forecast_report["forecast_period_end"]).date()}'
        )
        print(
            f'[INFO] Reporting cadence: '
            f'{forecast_report["inferred_cadence_days"]} days | '
            f'last complete period: '
            f'{pd.Timestamp(forecast_report["latest_complete_period_start"]).date()} '
            f'-> '
            f'{pd.Timestamp(forecast_report["latest_complete_period_end"]).date()}.'
        )
        print(
            f'[INFO] Deployment refit: '
            f'{len(training_frame):,} eligible historical rows.'
        )
        print(
            f'[INFO] Forecasted articles: {len(compact):,}.'
        )

        if scope_report[
            'scope_applied'
        ]:
            print(
                f'[INFO] Article scope applied: '
                f'{scope_report["articles_before_scope"]:,} -> '
                f'{scope_report["articles_after_scope"]:,} articles | '
                f'excluded={scope_report["articles_excluded_by_scope"]:,} | '
                f'manual review='
                f'{scope_report["manual_review_articles"]:,}.'
            )
        print(
            f'[INFO] Predicted total units: '
            f'{summary.iloc[0]["predicted_total_units"]:,.1f}.'
        )
        print(
            f'[INFO] Frozen predictors: '
            f'{len(categorical_features)} categorical + '
            f'{len(numeric_features)} numeric = '
            f'{len(selected_features)} total.'
        )

        if operational_report[
            'warnings'
        ]:
            for warning in operational_report[
                'warnings'
            ]:
                print(
                    f'[WARNING] {warning}'
                )

        print()
        print(
            'MODEL USAGE'
        )
        print(
            model_usage.to_string(
                index=False
            )
        )

        print()
        print(
            'TOP 20 FORECAST'
        )

        preview_columns = [
            column
            for column in [
                'forecast_rank',
                'article_code',
                'article_name',
                'article_demand_regime_prior',
                'activity_status',
                'manual_review_required',
                'selected_model',
                'predicted_units',
                'predicted_units_rounded',
                'planning_class',
            ]
            if column in compact.columns
        ]

        print(
            compact[
                preview_columns
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    return {
        'next_period_article_forecast': compact,
        'next_period_forecast_summary': summary,
        'next_period_model_usage': model_usage,
        'next_period_regime_usage': regime_usage,
    }
