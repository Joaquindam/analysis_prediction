
"""
Leakage-aware feature engineering for the Analysis Prediction framework.

Pipeline position:

    Raw -> Bronze -> Silver -> Gold -> EDA -> Features

This module converts integrated Gold datasets into modelling-ready feature
tables while preserving temporal causality. The main design rule is simple:
features used by default must be either known in advance or computed strictly
from past observations.

The module does not train models. It produces candidate feature tables plus a
feature catalogue that records temporal availability and leakage risk.

Public entry point
------------------
run_feature_engineering()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FeaturePaths:
    """Output locations used by feature engineering."""

    feature_dir: str | Path
    reports_dir: str | Path

    def __post_init__(self) -> None:
        self.feature_dir = Path(self.feature_dir)
        self.reports_dir = Path(self.reports_dir)


@dataclass
class FeatureRules:
    """
    User-facing feature-engineering policy.

    Defaults are conservative and leakage-aware. Same-period business outcomes
    are preserved in the output for auditability but are not marked as safe
    model inputs.
    """

    build_daily_features: bool = True
    build_article_period_features: bool = True

    # Targets are selected from the first available canonical candidate.
    daily_target_candidates: tuple[str, ...] = (
        'facturacion',
        'facturacion_tickets',
    )
    daily_activity_candidates: tuple[str, ...] = (
        'num_tickets',
        'num_tickets_unicos',
    )
    article_target_candidates: tuple[str, ...] = (
        'units',
        'amount',
    )

    # Past-only history.
    daily_lags: tuple[int, ...] = (
        1,
        7,
        14,
        28,
    )
    daily_rolling_windows: tuple[int, ...] = (
        7,
        14,
        28,
    )
    article_lags: tuple[int, ...] = (
        1,
        2,
        4,
        8,
    )
    article_rolling_windows: tuple[int, ...] = (
        4,
        8,
    )

    # Additional daily columns for which past-only history is useful if they
    # exist. Current-period values remain leakage-risk / snapshot-dependent.
    daily_history_candidates: tuple[str, ...] = (
        'num_tickets',
        'num_tickets_unicos',
        'reservas_total',
        'comensales_reservados_total',
    )

    # Calendar and context.
    include_calendar_features: bool = True
    include_cyclical_calendar_features: bool = True
    include_current_known_context: bool = True
    aggregate_daily_context_to_article_period: bool = True

    # Actual weather is useful historically but future use requires a forecast.
    # It is therefore generated/preserved but not safe-by-default.
    include_weather_context: bool = True

    # Training eligibility. Rows are flagged, never silently deleted here.
    require_complete_article_period_for_training: bool = True
    exclude_negative_targets_from_training: bool = True

    # Past-only demand regime. This is computed separately at every time step.
    article_min_history_for_regime: int = 4
    article_frequent_threshold: float = 0.75
    article_regular_threshold: float = 0.50
    article_intermittent_threshold: float = 0.25

    # Persistence.
    save_feature_tables: bool = True
    save_feature_catalogue: bool = True
    save_feature_report: bool = True

    def validate(self) -> None:
        if not self.build_daily_features and not self.build_article_period_features:
            raise ValueError(
                'At least one feature table must be enabled.'
            )

        for field_name in [
            'include_calendar_features',
            'include_cyclical_calendar_features',
            'include_current_known_context',
            'aggregate_daily_context_to_article_period',
            'include_weather_context',
            'require_complete_article_period_for_training',
            'exclude_negative_targets_from_training',
            'save_feature_tables',
            'save_feature_catalogue',
            'save_feature_report',
        ]:
            if not isinstance(
                getattr(self, field_name),
                bool,
            ):
                raise TypeError(
                    f'{field_name} must be True or False.'
                )

        for field_name in [
            'daily_lags',
            'daily_rolling_windows',
            'article_lags',
            'article_rolling_windows',
        ]:
            values = getattr(
                self,
                field_name,
            )

            if any(
                int(value) < 1
                for value in values
            ):
                raise ValueError(
                    f'All values in {field_name} must be >= 1.'
                )

        if self.article_min_history_for_regime < 1:
            raise ValueError(
                'article_min_history_for_regime must be >= 1.'
            )

        thresholds = [
            self.article_frequent_threshold,
            self.article_regular_threshold,
            self.article_intermittent_threshold,
        ]

        if any(
            not 0 <= value <= 1
            for value in thresholds
        ):
            raise ValueError(
                'Article regime thresholds must be between 0 and 1.'
            )

        if not (
            self.article_frequent_threshold
            >= self.article_regular_threshold
            >= self.article_intermittent_threshold
        ):
            raise ValueError(
                'Article regime thresholds must be descending: '
                'frequent >= regular >= intermittent.'
            )


# =============================================================================
# CONSTANTS
# =============================================================================


FEATURE_FILENAMES = {
    'features_daily': 'features_daily.parquet',
    'features_article_period': 'features_article_period.parquet',
}

CALENDAR_FEATURE_NAMES = {
    'cal_year',
    'cal_month',
    'cal_quarter',
    'cal_day_of_month',
    'cal_day_of_week',
    'cal_week_of_year',
    'cal_is_weekend',
    'cal_is_month_start',
    'cal_is_month_end',
    'cal_day_of_year',
    'cal_dow_sin',
    'cal_dow_cos',
    'cal_month_sin',
    'cal_month_cos',
    'cal_doy_sin',
    'cal_doy_cos',
}


# =============================================================================
# CONSOLE HELPERS
# =============================================================================


def _print_header(
    title: str,
    verbose: bool = True,
) -> None:
    if not verbose:
        return

    line = '=' * 96
    print(f'\n{line}')
    print(title)
    print(line)


def _print_subheader(
    title: str,
    verbose: bool = True,
) -> None:
    if verbose:
        print(f'\n--- {title} ---')


def _print_message(
    message: str,
    level: str = 'INFO',
    verbose: bool = True,
) -> None:
    if verbose:
        print(
            f'[{level}] {message}'
        )


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _ensure_directory(
    directory: str | Path,
) -> Path:
    path = Path(
        directory
    )
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def _first_existing(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    return next(
        (
            column
            for column in candidates
            if column in dataframe.columns
        ),
        None,
    )


def _to_datetime(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors='coerce',
        dayfirst=True,
    )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        if np.isnan(
            value
        ):
            return None

        return float(
            value
        )

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

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item in value.items()
        }

    try:
        if pd.isna(
            value
        ):
            return None
    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


def _safe_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(
            'boolean'
        )

    lowered = (
        series
        .astype(
            'string'
        )
        .str.strip()
        .str.lower()
    )

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

    return lowered.map(
        mapping
    ).astype(
        'boolean'
    )


def _feature_dtype(
    series: pd.Series,
) -> str:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return 'boolean'

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return 'numeric'

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return 'datetime'

    return 'categorical_or_text'


# =============================================================================
# TEMPORAL AVAILABILITY / LEAKAGE CLASSIFICATION
# =============================================================================


def _is_weather_column(
    column: str,
) -> bool:
    name = column.lower()

    patterns = (
        'temperature',
        'temperatura',
        'precipitation',
        'precipitacion',
        'rain',
        'lluvia',
        'snow',
        'nieve',
        'wind',
        'viento',
        'sunshine',
        'solar',
        'humidity',
        'humedad',
        'weather',
        'meteo',
    )

    return any(
        token in name
        for token in patterns
    )


def _is_event_or_holiday_column(
    column: str,
) -> bool:
    name = column.lower()

    patterns = (
        'event',
        'evento',
        'festiv',
        'holiday',
    )

    return any(
        token in name
        for token in patterns
    )


def _is_reservation_column(
    column: str,
) -> bool:
    name = column.lower()

    patterns = (
        'reserva',
        'reservation',
        'comensal',
        'guest',
        'lead_time',
        'tamano_grupo',
        'group_size',
        'no_show',
        'cancel_rate',
        'walk_in',
    )

    return any(
        token in name
        for token in patterns
    )


def _is_transaction_outcome_column(
    column: str,
) -> bool:
    name = column.lower()

    patterns = (
        'facturacion',
        'revenue',
        'ticket_',
        'num_ticket',
        'factura',
        'invoice',
        'comprobante',
        'receipt',
        'efectivo',
        'tarjeta',
        'cash_',
        'card_',
    )

    return any(
        token in name
        for token in patterns
    )


def _is_quality_column(
    column: str,
) -> bool:
    name = column.lower()

    return (
        name.startswith(
            'quality_'
        )
        or '__outlier' in name
        or '__negative' in name
        or '__invalid' in name
        or 'conflict' in name
        or '__imputed' in name
    )


def _classify_original_daily_column(
    column: str,
    target_column: str | None,
) -> dict[str, Any]:
    """
    Classify an original Daily Gold column by temporal availability.

    The classifier is deliberately conservative. When availability depends on
    a future forecast or a prediction-time snapshot, safe_default is False.
    """
    name = column.lower()

    if target_column is not None and column == target_column:
        return {
            'role': 'target',
            'availability': 'target',
            'safe_default': False,
            'reason': 'Prediction target.',
        }

    if column == 'date':
        return {
            'role': 'time_index',
            'availability': 'known_in_advance',
            'safe_default': False,
            'reason': 'Temporal index; derived calendar features are preferred.',
        }

    if column in CALENDAR_FEATURE_NAMES or name.startswith(
        'cal_'
    ):
        return {
            'role': 'calendar_feature',
            'availability': 'known_in_advance',
            'safe_default': True,
            'reason': 'Deterministic calendar information.',
        }

    if _is_quality_column(
        column
    ):
        return {
            'role': 'quality_metadata',
            'availability': 'historical_metadata',
            'safe_default': False,
            'reason': 'Quality diagnostic, not a default predictor.',
        }

    if _is_weather_column(
        column
    ):
        return {
            'role': 'weather_context',
            'availability': 'requires_forecast',
            'safe_default': False,
            'reason': (
                'Observed future weather is unavailable at prediction time; '
                'use only when an equivalent forecast is supplied.'
            ),
        }

    if _is_event_or_holiday_column(
        column
    ):
        dtype_role = (
            'known_context'
            if not (
                'nombre' in name
                or 'name' in name
                or 'categoria' in name
                or 'categor' in name
            )
            else 'context_metadata'
        )

        return {
            'role': dtype_role,
            'availability': 'known_in_advance',
            'safe_default': dtype_role == 'known_context',
            'reason': (
                'Calendar/event context can be known before the business day.'
            ),
        }

    if _is_reservation_column(
        column
    ):
        return {
            'role': 'reservation_context',
            'availability': 'requires_prediction_time_snapshot',
            'safe_default': False,
            'reason': (
                'Daily reservation aggregates may contain information created '
                'or resolved after the intended prediction cutoff.'
            ),
        }

    if _is_transaction_outcome_column(
        column
    ):
        return {
            'role': 'same_period_business_outcome',
            'availability': 'same_period_outcome',
            'safe_default': False,
            'reason': (
                'Same-day transaction/outcome information is not available '
                'before the target period is completed.'
            ),
        }

    if name in {
        'year',
        'month',
        'quarter',
        'day_of_month',
        'day_of_week_num',
        'day_of_week',
        'week_of_year',
        'is_weekend',
        'is_month_start',
        'is_month_end',
        'week_sin',
        'week_cos',
    }:
        return {
            'role': 'calendar_feature',
            'availability': 'known_in_advance',
            'safe_default': True,
            'reason': 'Deterministic calendar information.',
        }

    return {
        'role': 'unclassified_original',
        'availability': 'unknown',
        'safe_default': False,
        'reason': (
            'Availability cannot be guaranteed automatically; review before '
            'using as a predictor.'
        ),
    }


def _classify_original_article_column(
    column: str,
    target_column: str | None,
) -> dict[str, Any]:
    name = column.lower()

    if target_column is not None and column == target_column:
        return {
            'role': 'target',
            'availability': 'target',
            'safe_default': False,
            'reason': 'Prediction target.',
        }

    if column == 'article_code':
        return {
            'role': 'categorical_identifier',
            'availability': 'known_in_advance',
            'safe_default': True,
            'reason': (
                'Stable article identity; model layer must encode it '
                'categorically rather than as a continuous quantity.'
            ),
        }

    if column in {
        'report_start',
        'report_end',
        'period_id',
    }:
        return {
            'role': 'time_index',
            'availability': 'known_in_advance',
            'safe_default': False,
            'reason': 'Temporal identifier; derived period features are preferred.',
        }

    if column in {
        'period_days',
        'period_is_complete',
        'period_is_complete_week',
    }:
        return {
            'role': 'period_structure',
            'availability': 'known_in_advance',
            'safe_default': True,
            'reason': 'Period definition is known before prediction.',
        }

    if _is_quality_column(
        column
    ):
        return {
            'role': 'quality_metadata',
            'availability': 'historical_metadata',
            'safe_default': False,
            'reason': 'Quality diagnostic, not a default predictor.',
        }

    if column == 'amount' and target_column == 'units':
        return {
            'role': 'same_period_business_outcome',
            'availability': 'same_period_outcome',
            'safe_default': False,
            'reason': (
                'Same-period monetary sales are not known before unit demand.'
            ),
        }

    if column == 'units' and target_column == 'amount':
        return {
            'role': 'same_period_business_outcome',
            'availability': 'same_period_outcome',
            'safe_default': False,
            'reason': (
                'Same-period units are not known before monetary demand.'
            ),
        }

    if name in {
        'department_code',
        'department_name',
    } or 'department_' in name:
        is_text = (
            'name' in name
        )

        return {
            'role': (
                'static_or_scheduled_metadata'
                if not is_text
                else 'metadata_text'
            ),
            'availability': 'known_if_catalog_available',
            'safe_default': not is_text,
            'reason': (
                'Article catalogue metadata is usable when the future catalogue '
                'assignment is known.'
            ),
        }

    if 'article_name' in name:
        return {
            'role': 'metadata_text',
            'availability': 'known_if_catalog_available',
            'safe_default': False,
            'reason': (
                'Article text is metadata; explicit encoding is required.'
            ),
        }

    if column == 'es_invitacion':
        return {
            'role': 'same_period_business_outcome',
            'availability': 'same_period_outcome',
            'safe_default': False,
            'reason': (
                'Invitation status is a realized transaction attribute.'
            ),
        }

    return {
        'role': 'unclassified_original',
        'availability': 'unknown',
        'safe_default': False,
        'reason': (
            'Availability cannot be guaranteed automatically; review before '
            'using as a predictor.'
        ),
    }


# =============================================================================
# FEATURE REGISTRY
# =============================================================================


def _register_feature(
    registry: dict[str, dict[str, Any]],
    column: str,
    role: str,
    availability: str,
    safe_default: bool,
    reason: str,
    origin: str = 'engineered',
) -> None:
    registry[
        column
    ] = {
        'origin': origin,
        'role': role,
        'availability': availability,
        'safe_default': bool(
            safe_default
        ),
        'reason': reason,
    }


def _build_catalogue(
    dataframe: pd.DataFrame,
    dataset_name: str,
    registry: dict[str, dict[str, Any]],
    target_column: str | None,
    original_classifier,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        metadata = registry.get(
            column
        )

        if metadata is None:
            metadata = original_classifier(
                column,
                target_column,
            )
            metadata = {
                'origin': 'gold_original',
                **metadata,
            }

        rows.append(
            {
                'dataset': dataset_name,
                'column': column,
                'dtype': str(
                    dataframe[
                        column
                    ].dtype
                ),
                'feature_dtype': _feature_dtype(
                    dataframe[
                        column
                    ]
                ),
                **metadata,
            }
        )

    return pd.DataFrame(
        rows
    )



def _mark_catalogue_rows_not_default(
    catalogue: pd.DataFrame,
    mask: pd.Series,
    role: str,
    reason: str,
) -> pd.DataFrame:
    """Update catalogue rows that should remain available but not default."""
    result = catalogue.copy()

    if not mask.any():
        return result

    result.loc[
        mask,
        'safe_default',
    ] = False
    result.loc[
        mask,
        'role',
    ] = role
    result.loc[
        mask,
        'reason',
    ] = reason

    return result


def _apply_daily_catalogue_redundancy_policy(
    catalogue: pd.DataFrame,
    include_calendar_features: bool,
) -> pd.DataFrame:
    """
    Avoid selecting duplicate calendar encodings by default.

    Gold may already contain year/month/day-of-week variables. When this module
    creates canonical `cal_*` features, the source calendar columns remain in
    the output for traceability but are not selected twice.
    """
    if not include_calendar_features:
        return catalogue

    source_calendar_columns = {
        'year',
        'month',
        'quarter',
        'day_of_month',
        'day_of_week_num',
        'day_of_week',
        'week_of_year',
        'is_weekend',
        'is_month_start',
        'is_month_end',
        'week_sin',
        'week_cos',
    }

    mask = (
        catalogue[
            'dataset'
        ].eq(
            'features_daily'
        )
        & catalogue[
            'origin'
        ].eq(
            'gold_original'
        )
        & catalogue[
            'column'
        ].isin(
            source_calendar_columns
        )
    )

    return _mark_catalogue_rows_not_default(
        catalogue,
        mask=mask,
        role='calendar_source_redundant',
        reason=(
            'Canonical cal_* calendar features are generated by ap_features; '
            'the original Gold calendar column is retained only for traceability.'
        ),
    )


def _apply_article_catalogue_redundancy_policy(
    catalogue: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep compatibility/master-enrichment columns without selecting duplicates.

    This is schema-driven rather than restaurant-specific.
    """
    result = catalogue.copy()

    # Legacy weekly compatibility field: the general field is
    # `period_is_complete`.
    if (
        'period_is_complete' in dataframe.columns
        and 'period_is_complete_week' in dataframe.columns
    ):
        mask = (
            result[
                'dataset'
            ].eq(
                'features_article_period'
            )
            & result[
                'column'
            ].eq(
                'period_is_complete_week'
            )
        )

        result = _mark_catalogue_rows_not_default(
            result,
            mask=mask,
            role='compatibility_metadata',
            reason=(
                'Legacy weekly compatibility field; the general '
                'period_is_complete feature is preferred.'
            ),
        )

    # Enriched master columns can duplicate an already-present canonical
    # column. Preserve them for auditability but do not select both by default.
    for column in dataframe.columns:
        match = re.fullmatch(
            r'(.+)__(articles|departments)_master',
            str(
                column
            ),
        )

        if match is None:
            continue

        base_column = match.group(
            1
        )

        if base_column not in dataframe.columns:
            continue

        mask = (
            result[
                'dataset'
            ].eq(
                'features_article_period'
            )
            & result[
                'column'
            ].eq(
                column
            )
        )

        result = _mark_catalogue_rows_not_default(
            result,
            mask=mask,
            role='redundant_master_metadata',
            reason=(
                f'Redundant with canonical column {base_column!r}; retained '
                'for auditability but excluded from default predictors.'
            ),
        )

    return result


# =============================================================================
# CALENDAR FEATURES
# =============================================================================


def _add_calendar_features(
    dataframe: pd.DataFrame,
    date_column: str,
    registry: dict[str, dict[str, Any]],
    cyclical: bool,
) -> pd.DataFrame:
    data = dataframe.copy()

    dates = _to_datetime(
        data[
            date_column
        ]
    )

    values = {
        'cal_year': dates.dt.year.astype(
            'Int64'
        ),
        'cal_month': dates.dt.month.astype(
            'Int64'
        ),
        'cal_quarter': dates.dt.quarter.astype(
            'Int64'
        ),
        'cal_day_of_month': dates.dt.day.astype(
            'Int64'
        ),
        'cal_day_of_week': dates.dt.dayofweek.astype(
            'Int64'
        ),
        'cal_week_of_year': dates.dt.isocalendar().week.astype(
            'Int64'
        ),
        'cal_is_weekend': dates.dt.dayofweek.isin(
            [
                5,
                6,
            ]
        ).astype(
            'boolean'
        ),
        'cal_is_month_start': dates.dt.is_month_start.astype(
            'boolean'
        ),
        'cal_is_month_end': dates.dt.is_month_end.astype(
            'boolean'
        ),
        'cal_day_of_year': dates.dt.dayofyear.astype(
            'Int64'
        ),
    }

    for column, series in values.items():
        data[
            column
        ] = series

        _register_feature(
            registry,
            column=column,
            role='calendar_feature',
            availability='known_in_advance',
            safe_default=True,
            reason='Deterministic calendar feature.',
        )

    if cyclical:
        dow = _to_numeric(
            data[
                'cal_day_of_week'
            ]
        )
        month = _to_numeric(
            data[
                'cal_month'
            ]
        )
        doy = _to_numeric(
            data[
                'cal_day_of_year'
            ]
        )

        cyclical_values = {
            'cal_dow_sin': np.sin(
                2 * np.pi * dow / 7
            ),
            'cal_dow_cos': np.cos(
                2 * np.pi * dow / 7
            ),
            'cal_month_sin': np.sin(
                2 * np.pi * (
                    month - 1
                ) / 12
            ),
            'cal_month_cos': np.cos(
                2 * np.pi * (
                    month - 1
                ) / 12
            ),
            'cal_doy_sin': np.sin(
                2 * np.pi * (
                    doy - 1
                ) / 365.25
            ),
            'cal_doy_cos': np.cos(
                2 * np.pi * (
                    doy - 1
                ) / 365.25
            ),
        }

        for column, series in cyclical_values.items():
            data[
                column
            ] = series.astype(
                'Float64'
            )

            _register_feature(
                registry,
                column=column,
                role='calendar_feature',
                availability='known_in_advance',
                safe_default=True,
                reason='Cyclical encoding of deterministic calendar position.',
            )

    return data


# =============================================================================
# PAST-ONLY HISTORY FEATURES
# =============================================================================


def _add_lag_features(
    dataframe: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...],
    registry: dict[str, dict[str, Any]],
    role_prefix: str,
) -> pd.DataFrame:
    data = dataframe.copy()

    for source_column in columns:
        if source_column not in data.columns:
            continue

        numeric = _to_numeric(
            data[
                source_column
            ]
        )

        for lag in sorted(
            set(
                int(value)
                for value in lags
            )
        ):
            feature_name = (
                f'{source_column}__lag_{lag}'
            )

            data[
                feature_name
            ] = numeric.shift(
                lag
            )

            _register_feature(
                registry,
                column=feature_name,
                role=f'{role_prefix}_lag',
                availability='past_only',
                safe_default=True,
                reason=(
                    f'{lag}-step lag computed strictly from prior observations.'
                ),
            )

    return data


def _add_rolling_features(
    dataframe: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...],
    registry: dict[str, dict[str, Any]],
    role_prefix: str,
) -> pd.DataFrame:
    data = dataframe.copy()

    for source_column in columns:
        if source_column not in data.columns:
            continue

        shifted = _to_numeric(
            data[
                source_column
            ]
        ).shift(
            1
        )

        for window in sorted(
            set(
                int(value)
                for value in windows
            )
        ):
            rolling = shifted.rolling(
                window=window,
                min_periods=1,
            )

            definitions = {
                f'{source_column}__rolling_mean_{window}': rolling.mean(),
                f'{source_column}__rolling_std_{window}': rolling.std(),
                f'{source_column}__rolling_count_{window}': rolling.count(),
            }

            for feature_name, series in definitions.items():
                data[
                    feature_name
                ] = series

                _register_feature(
                    registry,
                    column=feature_name,
                    role=f'{role_prefix}_rolling',
                    availability='past_only',
                    safe_default=True,
                    reason=(
                        'Rolling statistic computed after shifting one step, '
                        'so the current target period is excluded.'
                    ),
                )

    return data


def _add_daily_same_weekday_history(
    dataframe: pd.DataFrame,
    target_column: str | None,
    activity_column: str | None,
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    data = dataframe.copy()

    if 'cal_day_of_week' not in data.columns:
        return data

    weekday = data[
        'cal_day_of_week'
    ]

    if target_column is not None:
        target = _to_numeric(
            data[
                target_column
            ]
        )

        temp = pd.DataFrame(
            {
                'weekday': weekday,
                'value': target,
            },
            index=data.index,
        )

        feature = (
            temp
            .groupby(
                'weekday',
                dropna=False,
            )[
                'value'
            ]
            .transform(
                lambda series: (
                    series
                    .shift(
                        1
                    )
                    .expanding(
                        min_periods=1
                    )
                    .mean()
                )
            )
        )

        count = (
            temp
            .groupby(
                'weekday',
                dropna=False,
            )[
                'value'
            ]
            .transform(
                lambda series: (
                    series
                    .shift(
                        1
                    )
                    .expanding(
                        min_periods=1
                    )
                    .count()
                )
            )
        )

        data[
            f'{target_column}__same_weekday_mean_prior'
        ] = feature
        data[
            f'{target_column}__same_weekday_count_prior'
        ] = count

        for column in [
            f'{target_column}__same_weekday_mean_prior',
            f'{target_column}__same_weekday_count_prior',
        ]:
            _register_feature(
                registry,
                column=column,
                role='target_history_by_weekday',
                availability='past_only',
                safe_default=True,
                reason=(
                    'Uses only earlier observations from the same weekday.'
                ),
            )

    if activity_column is not None:
        activity = (
            _to_numeric(
                data[
                    activity_column
                ]
            )
            .gt(
                0
            )
            .astype(
                'float64'
            )
        )

        temp = pd.DataFrame(
            {
                'weekday': weekday,
                'value': activity,
            },
            index=data.index,
        )

        feature = (
            temp
            .groupby(
                'weekday',
                dropna=False,
            )[
                'value'
            ]
            .transform(
                lambda series: (
                    series
                    .shift(
                        1
                    )
                    .expanding(
                        min_periods=1
                    )
                    .mean()
                )
            )
        )

        name = (
            f'{activity_column}__same_weekday_positive_rate_prior'
        )

        data[
            name
        ] = feature

        _register_feature(
            registry,
            column=name,
            role='activity_schedule_history',
            availability='past_only',
            safe_default=True,
            reason=(
                'Historical positive-activity rate for the weekday, computed '
                'without using future observations.'
            ),
        )

    return data


# =============================================================================
# DAILY FEATURES
# =============================================================================


def _structural_fill_known_context(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill only context values whose zero meaning is structurally unambiguous.

    Event intensity is set to zero only on rows explicitly marked as having no
    event. Unknown intensity for an existing event remains missing.
    """
    data = dataframe.copy()

    if 'tiene_evento' in data.columns:
        has_event = _safe_bool(
            data[
                'tiene_evento'
            ]
        )

        for column in [
            'event_intensity_mean',
            'event_intensity_max',
        ]:
            if column not in data.columns:
                continue

            numeric = _to_numeric(
                data[
                    column
                ]
            )

            data[
                column
            ] = numeric.mask(
                has_event.eq(
                    False
                )
                & numeric.isna(),
                0.0,
            )

    return data


def build_daily_feature_table(
    daily_gold: pd.DataFrame,
    rules: FeatureRules,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build leakage-aware daily features.

    Current-period operational outcomes are retained for traceability but are
    not safe-by-default in the generated feature catalogue.
    """
    if 'date' not in daily_gold.columns:
        raise KeyError(
            "Daily Gold requires a 'date' column."
        )

    data = daily_gold.copy()
    data[
        'date'
    ] = _to_datetime(
        data[
            'date'
        ]
    )
    data = data.sort_values(
        'date'
    ).reset_index(
        drop=True
    )

    target_column = _first_existing(
        data,
        rules.daily_target_candidates,
    )
    activity_column = _first_existing(
        data,
        rules.daily_activity_candidates,
    )

    registry: dict[str, dict[str, Any]] = {}

    if rules.include_current_known_context:
        data = _structural_fill_known_context(
            data
        )

    if rules.include_calendar_features:
        data = _add_calendar_features(
            data,
            date_column='date',
            registry=registry,
            cyclical=rules.include_cyclical_calendar_features,
        )

    history_columns: list[str] = []

    if target_column is not None:
        history_columns.append(
            target_column
        )

    for column in rules.daily_history_candidates:
        if (
            column in data.columns
            and column not in history_columns
        ):
            history_columns.append(
                column
            )

    data = _add_lag_features(
        data,
        columns=history_columns,
        lags=rules.daily_lags,
        registry=registry,
        role_prefix='daily_history',
    )

    data = _add_rolling_features(
        data,
        columns=history_columns,
        windows=rules.daily_rolling_windows,
        registry=registry,
        role_prefix='daily_history',
    )

    data = _add_daily_same_weekday_history(
        data,
        target_column=target_column,
        activity_column=activity_column,
        registry=registry,
    )

    target_observed_name = 'target_observed'
    training_eligible_name = 'training_eligible'

    if target_column is not None:
        target = _to_numeric(
            data[
                target_column
            ]
        )

        data[
            target_observed_name
        ] = target.notna().astype(
            'boolean'
        )

        training_eligible = target.notna()

        if rules.exclude_negative_targets_from_training:
            negative_flag = (
                f'{target_column}__negative'
            )

            if negative_flag in data.columns:
                negative = _safe_bool(
                    data[
                        negative_flag
                    ]
                ).fillna(
                    False
                )
                training_eligible = (
                    training_eligible
                    & ~negative
                )
            else:
                training_eligible = (
                    training_eligible
                    & target.ge(
                        0
                    )
                )

        data[
            training_eligible_name
        ] = training_eligible.astype(
            'boolean'
        )
    else:
        data[
            target_observed_name
        ] = pd.Series(
            False,
            index=data.index,
            dtype='boolean',
        )
        data[
            training_eligible_name
        ] = pd.Series(
            False,
            index=data.index,
            dtype='boolean',
        )

    _register_feature(
        registry,
        column=target_observed_name,
        role='training_metadata',
        availability='target_metadata',
        safe_default=False,
        reason='Indicates whether a supervised target is present.',
    )
    _register_feature(
        registry,
        column=training_eligible_name,
        role='training_metadata',
        availability='target_metadata',
        safe_default=False,
        reason='Row-level supervised-training eligibility flag.',
    )

    catalogue = _build_catalogue(
        data,
        dataset_name='features_daily',
        registry=registry,
        target_column=target_column,
        original_classifier=_classify_original_daily_column,
    )

    catalogue = _apply_daily_catalogue_redundancy_policy(
        catalogue,
        include_calendar_features=rules.include_calendar_features,
    )

    safe_columns = (
        catalogue.loc[
            catalogue[
                'safe_default'
            ].eq(
                True
            ),
            'column',
        ]
        .tolist()
    )

    report = {
        'available': True,
        'rows': len(
            data
        ),
        'columns': len(
            data.columns
        ),
        'target_column': target_column,
        'activity_column': activity_column,
        'target_observed_rows': (
            int(
                data[
                    target_observed_name
                ].sum()
            )
            if target_column is not None
            else 0
        ),
        'training_eligible_rows': int(
            data[
                training_eligible_name
            ].sum()
        ),
        'safe_default_feature_count': len(
            safe_columns
        ),
        'safe_default_features': safe_columns,
        'history_columns': history_columns,
        'daily_lags': list(
            rules.daily_lags
        ),
        'daily_rolling_windows': list(
            rules.daily_rolling_windows
        ),
        'leakage_policy': (
            'Current-period transactions and reservation aggregates are not '
            'safe-by-default. Only calendar/known context and past-only '
            'history are selected automatically.'
        ),
    }

    return data, catalogue, report


# =============================================================================
# ARTICLE-PERIOD FEATURES
# =============================================================================


def _build_period_index(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = (
        dataframe[
            [
                'report_start',
                'report_end',
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                'report_start',
                'report_end',
            ]
        )
        .reset_index(
            drop=True
        )
    )

    periods[
        'period_index'
    ] = np.arange(
        len(
            periods
        ),
        dtype='int64',
    )

    data = dataframe.merge(
        periods,
        on=[
            'report_start',
            'report_end',
        ],
        how='left',
        validate='many_to_one',
    )

    return data, periods


def _add_article_exact_lags(
    dataframe: pd.DataFrame,
    target_column: str,
    lags: tuple[int, ...],
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Add exact-period lags.

    A lag is missing when the article was not observed in that exact previous
    period. The code never treats the previous observed row as equivalent to
    the previous calendar/report period.
    """
    data = dataframe.copy()

    base = data[
        [
            'article_code',
            'period_index',
            target_column,
        ]
    ].copy()

    base[
        target_column
    ] = _to_numeric(
        base[
            target_column
        ]
    )

    for lag in sorted(
        set(
            int(value)
            for value in lags
        )
    ):
        lagged = base.copy()
        lagged[
            'period_index'
        ] = (
            lagged[
                'period_index'
            ]
            + lag
        )

        feature_name = (
            f'{target_column}__lag_{lag}'
        )
        observed_name = (
            f'{target_column}__lag_{lag}_observed'
        )

        lagged = lagged.rename(
            columns={
                target_column: feature_name,
            }
        )

        lagged[
            observed_name
        ] = lagged[
            feature_name
        ].notna().astype(
            'boolean'
        )

        data = data.merge(
            lagged,
            on=[
                'article_code',
                'period_index',
            ],
            how='left',
            validate='one_to_one',
        )

        data[
            observed_name
        ] = (
            data[
                observed_name
            ]
            .fillna(
                False
            )
            .astype(
                'boolean'
            )
        )

        _register_feature(
            registry,
            column=feature_name,
            role='article_target_lag',
            availability='past_only',
            safe_default=True,
            reason=(
                f'Exact {lag}-period lag. Missing means the article was not '
                'observed in that exact lagged period or the lag is unavailable.'
            ),
        )
        _register_feature(
            registry,
            column=observed_name,
            role='article_history_availability',
            availability='past_only',
            safe_default=True,
            reason='Indicates whether the exact lag value exists.',
        )

    return data


def _build_dense_article_history(
    dataframe: pd.DataFrame,
    periods: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    article_codes = pd.Index(
        dataframe[
            'article_code'
        ].dropna().unique(),
        name='article_code',
    )

    period_indexes = pd.Index(
        periods[
            'period_index'
        ].tolist(),
        name='period_index',
    )

    full_index = pd.MultiIndex.from_product(
        [
            article_codes,
            period_indexes,
        ],
        names=[
            'article_code',
            'period_index',
        ],
    )

    observed = (
        dataframe[
            [
                'article_code',
                'period_index',
                target_column,
            ]
        ]
        .copy()
        .set_index(
            [
                'article_code',
                'period_index',
            ]
        )
        .reindex(
            full_index
        )
        .reset_index()
    )

    observed[
        target_column
    ] = _to_numeric(
        observed[
            target_column
        ]
    )
    observed[
        '__observed'
    ] = observed[
        target_column
    ].notna().astype(
        'int64'
    )

    return observed


def _add_article_history_statistics(
    dataframe: pd.DataFrame,
    periods: pd.DataFrame,
    target_column: str,
    windows: tuple[int, ...],
    rules: FeatureRules,
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    data = dataframe.copy()

    dense = _build_dense_article_history(
        data,
        periods=periods,
        target_column=target_column,
    )

    dense = dense.sort_values(
        [
            'article_code',
            'period_index',
        ]
    ).reset_index(
        drop=True
    )

    grouped = dense.groupby(
        'article_code',
        sort=False,
        group_keys=False,
    )

    # Number of report periods available before the current one.
    dense[
        'history_periods_elapsed'
    ] = dense[
        'period_index'
    ]

    dense[
        'history_observed_periods'
    ] = grouped[
        '__observed'
    ].cumsum() - dense[
        '__observed'
    ]

    denominator = dense[
        'history_periods_elapsed'
    ].replace(
        0,
        np.nan,
    )

    dense[
        'history_observation_rate'
    ] = (
        dense[
            'history_observed_periods'
        ]
        / denominator
    )

    shifted_target = grouped[
        target_column
    ].shift(
        1
    )

    dense[
        f'{target_column}__expanding_mean_prior'
    ] = (
        shifted_target
        .groupby(
            dense[
                'article_code'
            ],
            sort=False,
        )
        .expanding(
            min_periods=1
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .sort_index()
    )

    dense[
        f'{target_column}__expanding_std_prior'
    ] = (
        shifted_target
        .groupby(
            dense[
                'article_code'
            ],
            sort=False,
        )
        .expanding(
            min_periods=2
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
        .sort_index()
    )

    last_observed = (
        dense[
            'period_index'
        ]
        .where(
            dense[
                '__observed'
            ].eq(
                1
            )
        )
    )

    prior_last_observed = (
        last_observed
        .groupby(
            dense[
                'article_code'
            ],
            sort=False,
        )
        .ffill()
        .groupby(
            dense[
                'article_code'
            ],
            sort=False,
        )
        .shift(
            1
        )
    )

    dense[
        'periods_since_last_observation'
    ] = (
        dense[
            'period_index'
        ]
        - prior_last_observed
    )

    for window in sorted(
        set(
            int(value)
            for value in windows
        )
    ):
        rolling = (
            shifted_target
            .groupby(
                dense[
                    'article_code'
                ],
                sort=False,
            )
            .rolling(
                window=window,
                min_periods=1,
            )
        )

        dense[
            f'{target_column}__rolling_mean_{window}'
        ] = (
            rolling
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        dense[
            f'{target_column}__rolling_std_{window}'
        ] = (
            rolling
            .std()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

        dense[
            f'{target_column}__rolling_count_{window}'
        ] = (
            rolling
            .count()
            .reset_index(
                level=0,
                drop=True,
            )
            .sort_index()
        )

    def classify_regime(
        row: pd.Series,
    ) -> str:
        history_count = row[
            'history_observed_periods'
        ]

        rate = row[
            'history_observation_rate'
        ]

        if (
            pd.isna(
                history_count
            )
            or history_count
            < rules.article_min_history_for_regime
        ):
            return 'cold_start'

        if pd.isna(
            rate
        ):
            return 'cold_start'

        if rate >= rules.article_frequent_threshold:
            return 'frequent'

        if rate >= rules.article_regular_threshold:
            return 'regular'

        if rate >= rules.article_intermittent_threshold:
            return 'intermittent'

        return 'sparse'

    dense[
        'article_demand_regime_prior'
    ] = dense.apply(
        classify_regime,
        axis=1,
    ).astype(
        'string'
    )

    join_columns = [
        'article_code',
        'period_index',
        'history_periods_elapsed',
        'history_observed_periods',
        'history_observation_rate',
        f'{target_column}__expanding_mean_prior',
        f'{target_column}__expanding_std_prior',
        'periods_since_last_observation',
        'article_demand_regime_prior',
    ]

    for window in sorted(
        set(
            int(value)
            for value in windows
        )
    ):
        join_columns.extend(
            [
                f'{target_column}__rolling_mean_{window}',
                f'{target_column}__rolling_std_{window}',
                f'{target_column}__rolling_count_{window}',
            ]
        )

    features = dense[
        join_columns
    ]

    data = data.merge(
        features,
        on=[
            'article_code',
            'period_index',
        ],
        how='left',
        validate='one_to_one',
    )

    for column in [
        'history_periods_elapsed',
        'history_observed_periods',
        'history_observation_rate',
        'periods_since_last_observation',
    ]:
        _register_feature(
            registry,
            column=column,
            role='article_history_structure',
            availability='past_only',
            safe_default=True,
            reason='Derived exclusively from report periods before the current one.',
        )

    for column in [
        f'{target_column}__expanding_mean_prior',
        f'{target_column}__expanding_std_prior',
    ]:
        _register_feature(
            registry,
            column=column,
            role='article_target_history',
            availability='past_only',
            safe_default=True,
            reason='Expanding statistic computed only from previous periods.',
        )

    _register_feature(
        registry,
        column='article_demand_regime_prior',
        role='adaptive_demand_regime',
        availability='past_only',
        safe_default=True,
        reason=(
            'Demand regime is recomputed at each period using only prior '
            'coverage, avoiding future-information segmentation.'
        ),
    )

    for window in sorted(
        set(
            int(value)
            for value in windows
        )
    ):
        for suffix in [
            'mean',
            'std',
            'count',
        ]:
            column = (
                f'{target_column}__rolling_{suffix}_{window}'
            )

            _register_feature(
                registry,
                column=column,
                role='article_target_history',
                availability='past_only',
                safe_default=True,
                reason=(
                    'Rolling statistic over exact prior report periods; '
                    'current-period target is excluded.'
                ),
            )

    return data


def _add_period_calendar_features(
    dataframe: pd.DataFrame,
    registry: dict[str, dict[str, Any]],
    cyclical: bool,
) -> pd.DataFrame:
    data = dataframe.copy()

    start = _to_datetime(
        data[
            'report_start'
        ]
    )
    end = _to_datetime(
        data[
            'report_end'
        ]
    )
    midpoint = start + (
        end - start
    ) / 2

    definitions = {
        'period_start_year': start.dt.year.astype(
            'Int64'
        ),
        'period_start_month': start.dt.month.astype(
            'Int64'
        ),
        'period_start_week_of_year': start.dt.isocalendar().week.astype(
            'Int64'
        ),
        'period_start_day_of_week': start.dt.dayofweek.astype(
            'Int64'
        ),
        'period_mid_month': midpoint.dt.month.astype(
            'Int64'
        ),
    }

    for column, series in definitions.items():
        data[
            column
        ] = series

        _register_feature(
            registry,
            column=column,
            role='period_calendar_feature',
            availability='known_in_advance',
            safe_default=True,
            reason='Deterministic feature of the report period.',
        )

    if cyclical:
        month = _to_numeric(
            data[
                'period_start_month'
            ]
        )
        week = _to_numeric(
            data[
                'period_start_week_of_year'
            ]
        )

        cyclical_values = {
            'period_month_sin': np.sin(
                2 * np.pi * (
                    month - 1
                ) / 12
            ),
            'period_month_cos': np.cos(
                2 * np.pi * (
                    month - 1
                ) / 12
            ),
            'period_week_sin': np.sin(
                2 * np.pi * (
                    week - 1
                ) / 52.1775
            ),
            'period_week_cos': np.cos(
                2 * np.pi * (
                    week - 1
                ) / 52.1775
            ),
        }

        for column, series in cyclical_values.items():
            data[
                column
            ] = series.astype(
                'Float64'
            )

            _register_feature(
                registry,
                column=column,
                role='period_calendar_feature',
                availability='known_in_advance',
                safe_default=True,
                reason='Cyclical encoding of report-period position.',
            )

    return data


def _period_global_history(
    dataframe: pd.DataFrame,
    target_column: str,
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    data = dataframe.copy()

    totals = (
        data
        .groupby(
            'period_index',
            as_index=False,
        )[
            target_column
        ]
        .sum(
            min_count=1
        )
        .rename(
            columns={
                target_column: '__global_target_total',
            }
        )
        .sort_values(
            'period_index'
        )
    )

    totals[
        'global_target_total__lag_1'
    ] = totals[
        '__global_target_total'
    ].shift(
        1
    )

    totals[
        'global_target_total__rolling_mean_4'
    ] = (
        totals[
            '__global_target_total'
        ]
        .shift(
            1
        )
        .rolling(
            window=4,
            min_periods=1,
        )
        .mean()
    )

    data = data.merge(
        totals[
            [
                'period_index',
                'global_target_total__lag_1',
                'global_target_total__rolling_mean_4',
            ]
        ],
        on='period_index',
        how='left',
        validate='many_to_one',
    )

    for column in [
        'global_target_total__lag_1',
        'global_target_total__rolling_mean_4',
    ]:
        _register_feature(
            registry,
            column=column,
            role='global_demand_history',
            availability='past_only',
            safe_default=True,
            reason=(
                'Aggregate demand history is shifted so the current report '
                'period is never used.'
            ),
        )

    return data


# =============================================================================
# DAILY CONTEXT -> ARTICLE PERIOD
# =============================================================================


def _daily_context_column_policy(
    dataframe: pd.DataFrame,
    target_column: str | None,
    include_weather: bool,
) -> dict[str, str]:
    """
    Return daily context columns eligible for period aggregation.

    Values are 'known' or 'forecast'. Same-period business outcomes and
    reservations are intentionally excluded.
    """
    policy: dict[str, str] = {}

    for column in dataframe.columns:
        if column == 'date':
            continue

        classification = _classify_original_daily_column(
            column,
            target_column,
        )

        availability = classification[
            'availability'
        ]

        if availability == 'known_in_advance':
            if classification[
                'role'
            ] == 'known_context':
                policy[
                    column
                ] = 'known'

        elif (
            include_weather
            and availability == 'requires_forecast'
        ):
            policy[
                column
            ] = 'forecast'

    return policy


def _aggregate_daily_context_for_periods(
    article_data: pd.DataFrame,
    daily_gold: pd.DataFrame,
    daily_target_column: str | None,
    include_weather: bool,
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Aggregate safe contextual Daily Gold information into article periods.

    Known calendar/event/holiday context is safe-by-default. Weather aggregates
    are generated but remain forecast-dependent.
    """
    if 'date' not in daily_gold.columns:
        return article_data

    daily = daily_gold.copy()
    daily[
        'date'
    ] = _to_datetime(
        daily[
            'date'
        ]
    )

    policy = _daily_context_column_policy(
        daily,
        target_column=daily_target_column,
        include_weather=include_weather,
    )

    if not policy:
        return article_data

    periods = (
        article_data[
            [
                'report_start',
                'report_end',
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                'report_start',
                'report_end',
            ]
        )
    )

    rows: list[dict[str, Any]] = []

    for period in periods.itertuples(
        index=False
    ):
        start = pd.Timestamp(
            period.report_start
        )
        end = pd.Timestamp(
            period.report_end
        )

        subset = daily[
            daily[
                'date'
            ].between(
                start,
                end,
                inclusive='both',
            )
        ]

        row: dict[str, Any] = {
            'report_start': start,
            'report_end': end,
            'context_days_available': int(
                len(
                    subset
                )
            ),
        }

        for column, category in policy.items():
            if column not in subset.columns:
                continue

            series = subset[
                column
            ]

            if pd.api.types.is_bool_dtype(
                series
            ):
                numeric = series.astype(
                    'Int64'
                )

                feature_name = (
                    f'context_{column}__days_true'
                )
                row[
                    feature_name
                ] = numeric.sum(
                    min_count=1
                )

                _register_feature(
                    registry,
                    column=feature_name,
                    role='period_context',
                    availability=(
                        'known_in_advance'
                        if category == 'known'
                        else 'requires_forecast'
                    ),
                    safe_default=category == 'known',
                    reason=(
                        'Aggregated from daily context over the report period.'
                    ),
                )

                continue

            if pd.api.types.is_numeric_dtype(
                series
            ):
                numeric = _to_numeric(
                    series
                )

                mean_name = (
                    f'context_{column}__mean'
                )
                row[
                    mean_name
                ] = numeric.mean()

                _register_feature(
                    registry,
                    column=mean_name,
                    role='period_context',
                    availability=(
                        'known_in_advance'
                        if category == 'known'
                        else 'requires_forecast'
                    ),
                    safe_default=category == 'known',
                    reason=(
                        'Mean daily context over the report period.'
                    ),
                )

                if any(
                    token in column.lower()
                    for token in (
                        'precipitation',
                        'precipitacion',
                        'rain',
                        'lluvia',
                        'snow',
                        'nieve',
                        'num_event',
                        'num_fest',
                    )
                ):
                    sum_name = (
                        f'context_{column}__sum'
                    )
                    row[
                        sum_name
                    ] = numeric.sum(
                        min_count=1
                    )

                    _register_feature(
                        registry,
                        column=sum_name,
                        role='period_context',
                        availability=(
                            'known_in_advance'
                            if category == 'known'
                            else 'requires_forecast'
                        ),
                        safe_default=category == 'known',
                        reason=(
                            'Additive daily context summed over the report period.'
                        ),
                    )

        rows.append(
            row
        )

    context = pd.DataFrame(
        rows
    )

    _register_feature(
        registry,
        column='context_days_available',
        role='context_coverage',
        availability='known_in_advance',
        safe_default=True,
        reason='Number of Daily Gold rows available inside the report period.',
    )

    merged = article_data.merge(
        context,
        on=[
            'report_start',
            'report_end',
        ],
        how='left',
        validate='many_to_one',
    )

    if 'period_days' in merged.columns:
        period_days = _to_numeric(
            merged[
                'period_days'
            ]
        ).replace(
            0,
            np.nan,
        )

        merged[
            'context_coverage_fraction'
        ] = (
            _to_numeric(
                merged[
                    'context_days_available'
                ]
            )
            / period_days
        ).clip(
            lower=0,
            upper=1,
        )

        _register_feature(
            registry,
            column='context_coverage_fraction',
            role='context_coverage',
            availability='known_in_advance',
            safe_default=True,
            reason=(
                'Fraction of the report period covered by available Daily Gold '
                'context rows.'
            ),
        )

    return merged


def build_article_period_feature_table(
    article_gold: pd.DataFrame,
    daily_gold: pd.DataFrame | None,
    rules: FeatureRules,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Build leakage-aware article-period features.

    Missing article-period rows are never densified to zero because the
    framework cannot infer whether an absent row means zero demand, unavailable
    product, menu change or missing source data.
    """
    required = {
        'report_start',
        'report_end',
        'article_code',
    }

    missing_required = required.difference(
        article_gold.columns
    )

    if missing_required:
        raise KeyError(
            'Article-period Gold is missing required columns: '
            f'{sorted(missing_required)}'
        )

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

    data = data.sort_values(
        [
            'report_start',
            'report_end',
            'article_code',
        ]
    ).reset_index(
        drop=True
    )

    target_column = _first_existing(
        data,
        rules.article_target_candidates,
    )

    if target_column is None:
        raise KeyError(
            'No article-period target candidate was found.'
        )

    registry: dict[str, dict[str, Any]] = {}

    data, periods = _build_period_index(
        data
    )

    _register_feature(
        registry,
        column='period_index',
        role='time_index',
        availability='known_in_advance',
        safe_default=False,
        reason='Ordinal period index; derived temporal features are preferred.',
    )

    if rules.include_calendar_features:
        data = _add_period_calendar_features(
            data,
            registry=registry,
            cyclical=rules.include_cyclical_calendar_features,
        )

    data = _add_article_exact_lags(
        data,
        target_column=target_column,
        lags=rules.article_lags,
        registry=registry,
    )

    data = _add_article_history_statistics(
        data,
        periods=periods,
        target_column=target_column,
        windows=rules.article_rolling_windows,
        rules=rules,
        registry=registry,
    )

    data = _period_global_history(
        data,
        target_column=target_column,
        registry=registry,
    )

    daily_target_column = None

    if daily_gold is not None:
        daily_target_column = _first_existing(
            daily_gold,
            rules.daily_target_candidates,
        )

    if (
        daily_gold is not None
        and rules.aggregate_daily_context_to_article_period
    ):
        data = _aggregate_daily_context_for_periods(
            data,
            daily_gold=daily_gold,
            daily_target_column=daily_target_column,
            include_weather=rules.include_weather_context,
            registry=registry,
        )

    target = _to_numeric(
        data[
            target_column
        ]
    )

    target_observed_mask = target.notna()

    data[
        'target_observed'
    ] = target_observed_mask.astype(
        'boolean'
    )

    if (
        rules.require_complete_article_period_for_training
        and 'period_is_complete' in data.columns
    ):
        complete_period_mask = _safe_bool(
            data[
                'period_is_complete'
            ]
        ).fillna(
            False
        )
    else:
        complete_period_mask = pd.Series(
            True,
            index=data.index,
            dtype='boolean',
        )

    if rules.exclude_negative_targets_from_training:
        negative_flag = (
            f'{target_column}__negative'
        )

        if negative_flag in data.columns:
            negative_target_mask = _safe_bool(
                data[
                    negative_flag
                ]
            ).fillna(
                False
            )
        else:
            negative_target_mask = target.lt(
                0
            ).fillna(
                False
            )
    else:
        negative_target_mask = pd.Series(
            False,
            index=data.index,
            dtype='boolean',
        )

    training_eligible = (
        target_observed_mask
        & complete_period_mask
        & ~negative_target_mask
    )

    data[
        'training_eligible'
    ] = training_eligible.astype(
        'boolean'
    )

    _register_feature(
        registry,
        column='target_observed',
        role='training_metadata',
        availability='target_metadata',
        safe_default=False,
        reason='Indicates whether a supervised target is present.',
    )
    _register_feature(
        registry,
        column='training_eligible',
        role='training_metadata',
        availability='target_metadata',
        safe_default=False,
        reason=(
            'Eligibility flag; incomplete periods/invalid targets can be '
            'excluded later without deleting them here.'
        ),
    )

    catalogue = _build_catalogue(
        data,
        dataset_name='features_article_period',
        registry=registry,
        target_column=target_column,
        original_classifier=_classify_original_article_column,
    )

    catalogue = _apply_article_catalogue_redundancy_policy(
        catalogue,
        dataframe=data,
    )

    safe_columns = (
        catalogue.loc[
            catalogue[
                'safe_default'
            ].eq(
                True
            ),
            'column',
        ]
        .tolist()
    )

    report = {
        'available': True,
        'rows': len(
            data
        ),
        'columns': len(
            data.columns
        ),
        'articles': int(
            data[
                'article_code'
            ].nunique(
                dropna=True
            )
        ),
        'periods': int(
            len(
                periods
            )
        ),
        'target_column': target_column,
        'target_observed_rows': int(
            data[
                'target_observed'
            ].sum()
        ),
        'training_eligible_rows': int(
            data[
                'training_eligible'
            ].sum()
        ),
        'training_exclusion_breakdown': {
            'target_missing': int(
                (
                    ~target_observed_mask
                ).sum()
            ),
            'incomplete_or_nonstandard_period': int(
                (
                    target_observed_mask
                    & ~complete_period_mask
                ).sum()
            ),
            'negative_target': int(
                (
                    target_observed_mask
                    & complete_period_mask
                    & negative_target_mask
                ).sum()
            ),
        },
        'safe_default_feature_count': len(
            safe_columns
        ),
        'safe_default_features': safe_columns,
        'article_lags': list(
            rules.article_lags
        ),
        'article_rolling_windows': list(
            rules.article_rolling_windows
        ),
        'sparse_panel_policy': (
            'Absent article-period rows are NOT converted to zero. Their '
            'business meaning is unknown without explicit availability/menu '
            'semantics.'
        ),
        'adaptive_regime_policy': (
            'Demand regime is computed at each row using only historical '
            'coverage prior to the current period.'
        ),
    }

    return data, catalogue, report


# =============================================================================
# PERSISTENCE
# =============================================================================


def _save_feature_table(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_parquet(
        path,
        index=False,
    )


def _save_catalogue(
    catalogue: pd.DataFrame,
    path: Path,
) -> None:
    catalogue.to_csv(
        path,
        index=False,
    )


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================


def run_feature_engineering(
    gold_datasets: dict[str, pd.DataFrame],
    paths: FeaturePaths,
    rules: FeatureRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build leakage-aware modelling feature tables from Gold datasets.

    Parameters
    ----------
    gold_datasets
        Gold dataset mapping created by integration.
    paths
        Feature output and report locations.
    rules
        Feature-engineering policy.
    verbose
        Print compact execution diagnostics.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Generated feature tables.
    """
    rules = (
        rules
        if rules is not None
        else FeatureRules()
    )
    rules.validate()

    feature_dir = _ensure_directory(
        paths.feature_dir
    )
    reports_dir = _ensure_directory(
        paths.reports_dir
    )

    _print_header(
        'GOLD -> LEAKAGE-AWARE FEATURE ENGINEERING',
        verbose=verbose,
    )

    outputs: dict[str, pd.DataFrame] = {}
    catalogues: list[pd.DataFrame] = []

    report: dict[str, Any] = {
        'rules': asdict(
            rules
        ),
        'feature_tables': {},
        'principles': {
            'temporal_causality': (
                'Default model features use only known-in-advance information '
                'or strictly past observations.'
            ),
            'no_implicit_zero_for_sparse_articles': (
                'Missing article-period rows are never assumed to be zero '
                'demand without explicit availability semantics.'
            ),
            'forecast_dependent_weather': (
                'Observed weather is not safe-by-default for future prediction; '
                'an equivalent forecast is required.'
            ),
            'reservation_snapshot_requirement': (
                'Same-day reservation aggregates are not safe-by-default '
                'without a prediction-time snapshot.'
            ),
        },
    }

    daily_gold = gold_datasets.get(
        'tabla_maestra_diaria'
    )
    article_gold = gold_datasets.get(
        'tabla_maestra_semanal_articulos'
    )

    if rules.build_daily_features:
        _print_subheader(
            'Daily features',
            verbose=verbose,
        )

        if daily_gold is None:
            _print_message(
                'Daily Gold dataset not available -> skipped.',
                level='WARNING',
                verbose=verbose,
            )

            report[
                'feature_tables'
            ][
                'features_daily'
            ] = {
                'available': False,
                'reason': 'Daily Gold dataset not available.',
            }

        else:
            daily_features, daily_catalogue, daily_report = (
                build_daily_feature_table(
                    daily_gold,
                    rules=rules,
                )
            )

            outputs[
                'features_daily'
            ] = daily_features
            catalogues.append(
                daily_catalogue
            )
            report[
                'feature_tables'
            ][
                'features_daily'
            ] = daily_report

            _print_message(
                'Daily feature table created: '
                f'{len(daily_features):,} rows x '
                f'{len(daily_features.columns):,} columns | '
                f'{daily_report["safe_default_feature_count"]} '
                'safe-by-default predictors.',
                verbose=verbose,
            )

    if rules.build_article_period_features:
        _print_subheader(
            'Article-period features',
            verbose=verbose,
        )

        if article_gold is None:
            _print_message(
                'Article-period Gold dataset not available -> skipped.',
                level='WARNING',
                verbose=verbose,
            )

            report[
                'feature_tables'
            ][
                'features_article_period'
            ] = {
                'available': False,
                'reason': 'Article-period Gold dataset not available.',
            }

        else:
            (
                article_features,
                article_catalogue,
                article_report,
            ) = build_article_period_feature_table(
                article_gold,
                daily_gold=daily_gold,
                rules=rules,
            )

            outputs[
                'features_article_period'
            ] = article_features
            catalogues.append(
                article_catalogue
            )
            report[
                'feature_tables'
            ][
                'features_article_period'
            ] = article_report

            _print_message(
                'Article-period feature table created: '
                f'{len(article_features):,} rows x '
                f'{len(article_features.columns):,} columns | '
                f'{article_report["safe_default_feature_count"]} '
                'safe-by-default predictors | '
                f'{article_report["training_eligible_rows"]:,} '
                'training-eligible rows.',
                verbose=verbose,
            )

    if rules.save_feature_tables:
        for name, dataframe in outputs.items():
            path = (
                feature_dir
                / FEATURE_FILENAMES[
                    name
                ]
            )

            _save_feature_table(
                dataframe,
                path,
            )

            report[
                'feature_tables'
            ][
                name
            ][
                'saved_path'
            ] = str(
                path
            )

            _print_message(
                f'FEATURES | {name}: {path}',
                verbose=verbose,
            )

    if catalogues:
        catalogue = pd.concat(
            catalogues,
            ignore_index=True,
        )
    else:
        catalogue = pd.DataFrame(
            columns=[
                'dataset',
                'column',
                'dtype',
                'feature_dtype',
                'origin',
                'role',
                'availability',
                'safe_default',
                'reason',
            ]
        )

    if rules.save_feature_catalogue:
        catalogue_path = (
            reports_dir
            / 'feature_catalogue.csv'
        )

        _save_catalogue(
            catalogue,
            catalogue_path,
        )

        report[
            'feature_catalogue'
        ] = str(
            catalogue_path
        )

    if rules.save_feature_report:
        report_path = (
            reports_dir
            / 'feature_report.json'
        )

        with report_path.open(
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(
                _json_safe(
                    report
                ),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        report[
            'feature_report'
        ] = str(
            report_path
        )

    _print_message(
        'Feature engineering completed. '
        'No Gold observation was overwritten or silently removed.',
        verbose=verbose,
    )

    return outputs
