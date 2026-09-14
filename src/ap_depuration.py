from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PUBLIC CONFIGURATION OBJECTS
# =============================================================================

@dataclass(frozen=True)
class DepurationPaths:
    """
    Output paths used by the depuration layer.

    Parameters
    ----------
    silver_dir : pathlib.Path
        Directory where Silver parquet datasets are stored.
    figures_dir : pathlib.Path
        Directory where data-quality figures are stored.
    reports_dir : pathlib.Path
        Directory where machine-readable quality reports are stored.
    """
    silver_dir: Path
    figures_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class DepurationRules:
    """
    Transparent default rules for automatic depuration.

    The defaults are deliberately conservative. This repository is intended to
    provide a reusable automatic analysis pipeline, not a forensic manual
    cleaning process tailored to one restaurant.

    Missing values
    --------------
    - Rows missing a REQUIRED KEY are dropped by default because the
      observation cannot be reliably identified or located in time.
    - Datetime columns are never statistically imputed.
    - Identifier/code columns are never statistically imputed.
    - Free-text/name/source columns are never statistically imputed.
    - Numeric non-key variables are median-imputed only when their missing
      fraction is at or below `max_missing_fraction_for_auto_imputation`.
    - Other categorical variables are filled with `categorical_missing_label`
      only when their missing fraction is at or below that same threshold.
    - Every automatic imputation creates `<column>__imputed`.

    Outliers
    --------
    - Outliers are detected using BOTH:
        1. Tukey IQR fences.
        2. Robust z-scores based on the median absolute deviation (MAD).
    - A value is flagged when either criterion considers it unusual.
    - By default outliers are KEPT and flagged, not deleted.
    - This is intentional: an unusually large table, ticket or sales day can
      be genuine business behaviour.
    - The console reports:
        * the IQR limits;
        * the value;
        * its absolute distance beyond the closest IQR fence;
        * that distance measured in IQR units;
        * its robust z-score;
        * whether one or both criteria flagged it.
    - Setting `outlier_action='drop'` makes the automatic pipeline remove
      flagged rows. This should be used cautiously.
    - Weather datasets are excluded from generic statistical outlier detection
      by default. Rainfall, sunshine and similar variables are naturally
      skewed/zero-inflated, so unusual values are not evidence of bad data.
      Weather quality is checked with physical/logical rules instead.
    - Structural event variables such as event duration/day number are also
      excluded from generic outlier detection.

    Duplicates
    ----------
    - Exact duplicated rows are dropped by default.
    - Non-exact repeated business keys are NOT automatically deleted. They are
      reported/flagged because repeated keys can be legitimate or can signal a
      master-data conflict.

    Dataset-specific invalid values
    --------------------------------
    - Clearly suspicious values are normally FLAGGED rather than silently
      changed.
    - Negative sales/ticket values are kept because they can represent refunds
      or corrections.
    - Physically suspicious weather values are kept and flagged.
    - Reservation lead times below zero and non-positive party sizes are kept
      and flagged.
    """

    # General
    drop_exact_duplicates: bool = True
    drop_rows_missing_required_keys: bool = True
    strip_string_whitespace: bool = True

    # Missing values
    auto_impute_low_missingness: bool = True
    max_missing_fraction_for_auto_imputation: float = 0.05
    numeric_imputation_strategy: str = 'median'
    categorical_imputation_strategy: str = 'constant'
    categorical_missing_label: str = 'UNKNOWN'

    # Outliers
    detect_outliers: bool = True
    outlier_action: str = 'flag'  # 'flag' or 'drop'
    iqr_multiplier: float = 1.5
    robust_z_threshold: float = 3.5
    min_rows_for_outlier_detection: int = 8
    min_unique_values_for_outlier_detection: int = 5
    max_outliers_to_print_per_column: int = 10

    # Dataset-specific thresholds
    large_group_people: int = 8
    payment_reconciliation_tolerance: float = 0.02
    invoice_reconciliation_tolerance: float = 0.02
    plausible_temperature_min_c: float = -60.0
    plausible_temperature_max_c: float = 60.0
    plausible_daily_hours_max: float = 24.0
    plausible_daily_seconds_max: float = 86400.0

    # Reporting
    save_quality_reports: bool = True
    save_figures: bool = True
    save_outlier_details: bool = True
    top_missing_columns_to_print: int = 12

    # Console
    # 'summary' keeps the terminal compact while preserving every CSV/JSON/plot.
    # 'detailed' restores the previous step-by-step console output.
    console_detail: str = 'summary'

    def validate(self) -> None:
        """Validate rule values before depuration starts."""
        if not 0 <= self.max_missing_fraction_for_auto_imputation <= 1:
            raise ValueError(
                'max_missing_fraction_for_auto_imputation must be between 0 and 1.'
            )

        if self.numeric_imputation_strategy not in {'median', 'mean', 'none'}:
            raise ValueError(
                "numeric_imputation_strategy must be 'median', 'mean' or 'none'."
            )

        if self.categorical_imputation_strategy not in {'constant', 'mode', 'none'}:
            raise ValueError(
                "categorical_imputation_strategy must be 'constant', 'mode' or 'none'."
            )

        if self.outlier_action not in {'flag', 'drop'}:
            raise ValueError(
                "outlier_action must be 'flag' or 'drop'."
            )

        if self.iqr_multiplier <= 0:
            raise ValueError(
                'iqr_multiplier must be positive.'
            )

        if self.robust_z_threshold <= 0:
            raise ValueError(
                'robust_z_threshold must be positive.'
            )

        if self.plausible_temperature_min_c >= self.plausible_temperature_max_c:
            raise ValueError(
                'plausible_temperature_min_c must be lower than '
                'plausible_temperature_max_c.'
            )

        if self.plausible_daily_hours_max <= 0:
            raise ValueError(
                'plausible_daily_hours_max must be positive.'
            )

        if self.plausible_daily_seconds_max <= 0:
            raise ValueError(
                'plausible_daily_seconds_max must be positive.'
            )

        if self.console_detail not in {'summary', 'detailed'}:
            raise ValueError(
                "console_detail must be 'summary' or 'detailed'."
            )


# =============================================================================
# DATASET-SPECIFIC KEYS AND NON-IMPUTABLE COLUMNS
# =============================================================================

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    'articulos': ('article_code',),
    'departamentos': ('department_code',),
    'eventos': ('date',),
    'facturas': ('ticket_id',),
    'festivos': ('date',),
    'menu': ('article_code',),
    'meteo_diaria': ('date',),
    'meteo_horaria': ('datetime',),
    'reservas': ('reservation_datetime',),
    'tickets': ('document_id', 'date'),
    'tips': ('document_id',),
    'total_articles': ('article_code',),
    'ventas': ('report_start', 'report_end', 'article_code'),
}


BUSINESS_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    'articulos': ('article_code',),
    'departamentos': ('department_code',),
    'eventos': ('event_id', 'date'),
    'facturas': ('ticket_id',),
    'festivos': ('date', 'holiday_name'),
    'menu': ('article_code',),
    'meteo_diaria': ('date',),
    'meteo_horaria': ('datetime',),
    'reservas': ('reference_code',),
    'tickets': ('document_id',),
    'tips': ('document_id',),
    'total_articles': ('report_start', 'report_end', 'article_code'),
    'ventas': ('report_start', 'report_end', 'article_code'),
}


SILVER_FILENAMES: dict[str, str] = {
    'articulos': 'articulos_silver.parquet',
    'departamentos': 'departamentos_silver.parquet',
    'eventos': 'eventos_silver.parquet',
    'facturas': 'facturas_silver.parquet',
    'festivos': 'festivos_silver.parquet',
    'menu': 'menu_silver.parquet',
    'meteo_diaria': 'meteo_diaria_silver.parquet',
    'meteo_horaria': 'meteo_horaria_silver.parquet',
    'reservas': 'reservas_silver.parquet',
    'tickets': 'tickets_silver.parquet',
    'tips': 'tips_silver.parquet',
    'total_articles': 'total_articles_silver.parquet',
    'ventas': 'ventas_silver.parquet',
}


NON_IMPUTABLE_NAME_PATTERNS = (
    'source_file',
    'raw_text',
    'name',
    'description',
    'article_name',
    'department_name',
    'holiday_name',
    'event_name',
    'reference',
    'reference_code',
    'lead_time',
)


STATISTICAL_OUTLIER_SKIP_DATASETS = {
    'meteo_diaria',
    'meteo_horaria',
}


STATISTICAL_OUTLIER_EXCLUDED_COLUMNS: dict[str, set[str]] = {
    'eventos': {
        'event_duration_days',
        'event_day_number',
        'event_intensity',
        'is_estimated',
    },
    'reservas': {
        'lead_time_hours',
        'lead_time_hours_raw',
    },
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
    if not verbose:
        return

    print(f'\n--- {title} ---')


def _print_message(
    message: str,
    level: str = 'INFO',
    verbose: bool = True,
) -> None:
    if verbose:
        print(f'[{level}] {message}')


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


def _normalize_text(
    value: Any,
) -> str:
    """Normalize text for comparisons without changing display columns."""
    if pd.isna(value):
        return ''

    text = unicodedata.normalize(
        'NFKD',
        str(value),
    )

    text = ''.join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    text = re.sub(
        r'\s+',
        ' ',
        text,
    )

    return text.strip().lower()


def _json_safe(
    value: Any,
) -> Any:
    """Convert pandas/numpy values into JSON-safe Python objects."""
    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
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

    if pd.isna(
        value
    ):
        return None

    return value


def _is_identifier_column(
    column: str,
) -> bool:
    """Return True for codes/ids that should not be treated as measurements."""
    normalized = column.lower()

    return (
        normalized == 'id'
        or normalized.endswith('_id')
        or normalized.endswith('_code')
        or normalized.startswith('id_')
        or normalized in {
            'ticket_id',
            'document_id',
            'reference_code',
        }
    )


def _is_non_imputable_text_column(
    column: str,
) -> bool:
    normalized = column.lower()

    return any(
        token in normalized
        for token in NON_IMPUTABLE_NAME_PATTERNS
    )


def _is_flag_column(
    column: str,
) -> bool:
    """Identify generated quality/imputation flags."""
    return (
        column.endswith('__outlier')
        or column.endswith('__imputed')
        or column.endswith('__invalid')
        or column.endswith('_suspicious')
        or column.endswith('__conflict')
        or column.startswith('quality_')
        or column.startswith('is_')
        or column.startswith('has_')
        or column.startswith('es_')
    )


def _to_numeric_if_possible(
    series: pd.Series,
    minimum_success_fraction: float = 0.95,
) -> pd.Series:
    """
    Convert a text/object Series to numeric only when conversion is reliable.

    This helps repair structured user tables passed directly to depuration
    without aggressively coercing genuine categorical variables.
    """
    if pd.api.types.is_numeric_dtype(
        series
    ):
        return series

    non_missing = series.dropna()

    if non_missing.empty:
        return series

    converted = pd.to_numeric(
        non_missing,
        errors='coerce',
    )

    success_fraction = converted.notna().mean()

    if success_fraction < minimum_success_fraction:
        return series

    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _coerce_known_column_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reassert common data types expected from ap_io.

    This makes depuration more robust when DO_INGEST=False and a user supplies
    already structured tables.
    """
    data = dataframe.copy()

    datetime_names = {
        'date',
        'datetime',
        'report_start',
        'report_end',
        'report_generated_on',
        'reservation_date',
        'reservation_datetime',
        'created_date',
        'created_datetime',
        'event_start',
        'event_end',
    }

    for column in data.columns:
        normalized = column.lower()

        if normalized in datetime_names:
            data[column] = pd.to_datetime(
                data[column],
                errors='coerce',
                dayfirst=True,
            )

        elif _is_identifier_column(
            normalized
        ):
            # Codes can be numeric or string identifiers. Keep strings if
            # numeric conversion is not highly reliable.
            data[column] = _to_numeric_if_possible(
                data[column]
            )

        elif normalized in {
            'units',
            'amount',
            'price',
            'people',
            'tip',
            'document_total',
            'document_amount',
            'receipt_count',
            'base',
            'vat',
            'total',
            'cash_amount',
            'card_amount',
            'event_intensity',
            'confidence',
        }:
            data[column] = pd.to_numeric(
                data[column],
                errors='coerce',
            )

    return data


# =============================================================================
# BASIC STRING / EMPTY-VALUE CLEANING
# =============================================================================

def standardize_empty_values(
    dataframe: pd.DataFrame,
    rules: DepurationRules,
) -> pd.DataFrame:
    """
    Standardize blank strings and common textual null markers to pd.NA.

    Display values are not lowercased or accent-stripped. The objective is to
    preserve human-readable business labels.
    """
    data = dataframe.copy()

    null_tokens = {
        '',
        'nan',
        'none',
        'null',
        'n/a',
        'na',
        '<na>',
    }

    for column in data.select_dtypes(
        include=[
            'object',
            'string',
        ]
    ).columns:
        if column == 'raw_text':
            continue

        series = data[column].astype(
            'string'
        )

        if rules.strip_string_whitespace:
            series = series.str.strip()

        normalized = series.str.lower()

        data[column] = series.mask(
            normalized.isin(
                null_tokens
            ),
            pd.NA,
        )

    return data


# =============================================================================
# DUPLICATES
# =============================================================================

def inspect_exact_duplicates(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Return a Boolean mask marking exact duplicated rows after the first."""
    return dataframe.duplicated(
        keep='first'
    )


def treat_exact_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Inspect and optionally remove exact duplicated rows.
    """
    data = dataframe.copy()

    duplicate_mask = inspect_exact_duplicates(
        data
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    summary = {
        'exact_duplicate_rows': duplicate_count,
        'action': (
            'dropped'
            if (
                duplicate_count > 0
                and rules.drop_exact_duplicates
            )
            else 'kept'
        ),
    }

    if duplicate_count == 0:
        _print_message(
            f'{dataset_name}: no exact duplicate rows found.',
            verbose=verbose,
        )

        return (
            data,
            summary,
        )

    _print_message(
        f'{dataset_name}: {duplicate_count:,} exact duplicated rows found '
        f'({duplicate_count / len(data):.2%} of rows).',
        level='WARNING',
        verbose=verbose,
    )

    if rules.drop_exact_duplicates:
        data = data.loc[
            ~duplicate_mask
        ].copy()

        _print_message(
            f'{dataset_name}: exact duplicates were DROPPED. '
            'This operation only removes rows that are identical across all columns.',
            verbose=verbose,
        )

    else:
        data['quality_exact_duplicate'] = (
            duplicate_mask
        )

        _print_message(
            f'{dataset_name}: exact duplicates were KEPT and flagged.',
            verbose=verbose,
        )

    return (
        data,
        summary,
    )


def inspect_business_key_duplicates(
    dataframe: pd.DataFrame,
    dataset_name: str,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Flag repeated business keys without deleting them automatically.

    Repeated business keys are different from exact duplicated rows. For
    example, the same article code with two different descriptions can signal
    a master-data change or conflict.
    """
    data = dataframe.copy()

    candidates = BUSINESS_KEY_CANDIDATES.get(
        dataset_name,
        (),
    )

    key_columns = [
        column
        for column in candidates
        if column in data.columns
    ]

    if not key_columns:
        return (
            data,
            {
                'business_key_columns': [],
                'repeated_business_key_rows': 0,
            },
        )

    complete_key_mask = (
        data[
            key_columns
        ]
        .notna()
        .all(
            axis=1
        )
    )

    duplicate_mask = pd.Series(
        False,
        index=data.index,
    )

    duplicate_mask.loc[
        complete_key_mask
    ] = (
        data.loc[
            complete_key_mask,
            key_columns,
        ]
        .duplicated(
            keep=False
        )
    )

    count = int(
        duplicate_mask.sum()
    )

    flag_column = (
        'quality_repeated_business_key'
    )

    data[
        flag_column
    ] = duplicate_mask

    if count:
        _print_message(
            f'{dataset_name}: {count:,} rows share the business key '
            f'{key_columns}. They are KEPT and flagged because repetition '
            'can be legitimate or can indicate conflicting master data.',
            level='WARNING',
            verbose=verbose,
        )

    return (
        data,
        {
            'business_key_columns': key_columns,
            'repeated_business_key_rows': count,
        },
    )


# =============================================================================
# MISSING VALUES
# =============================================================================

def missingness_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return missing-value counts and percentages for every column."""
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                'column',
                'dtype',
                'missing_count',
                'missing_fraction',
                'unique_non_null',
            ]
        )

    rows = []

    for column in dataframe.columns:
        rows.append(
            {
                'column': column,
                'dtype': str(
                    dataframe[column].dtype
                ),
                'missing_count': int(
                    dataframe[column].isna().sum()
                ),
                'missing_fraction': float(
                    dataframe[column].isna().mean()
                ),
                'unique_non_null': int(
                    dataframe[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                'missing_fraction',
                'missing_count',
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def print_missingness_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
    verbose: bool = True,
) -> None:
    """Print the most relevant missing-value information."""
    table = missingness_table(
        dataframe
    )

    missing = table[
        table['missing_count'] > 0
    ]

    if missing.empty:
        _print_message(
            f'{dataset_name}: no missing values.',
            verbose=verbose,
        )
        return

    total_missing = int(
        missing[
            'missing_count'
        ].sum()
    )

    _print_message(
        f'{dataset_name}: {total_missing:,} missing cells detected.',
        level='WARNING',
        verbose=verbose,
    )

    if not verbose:
        return

    print(
        missing[
            [
                'column',
                'dtype',
                'missing_count',
                'missing_fraction',
            ]
        ]
        .head(
            rules.top_missing_columns_to_print
        )
        .to_string(
            index=False,
            formatters={
                'missing_fraction': lambda value: f'{value:.1%}',
            },
        )
    )


def _required_keys_for(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return required keys that are actually present in the dataset."""
    expected = REQUIRED_KEYS.get(
        dataset_name,
        (),
    )

    return [
        column
        for column in expected
        if column in dataframe.columns
    ]


def treat_missing_required_keys(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Treat rows missing dataset-specific required keys.

    A required key is a field without which the observation cannot be reliably
    identified or placed in time.
    """
    data = dataframe.copy()

    required_keys = _required_keys_for(
        dataset_name,
        data,
    )

    if not required_keys:
        _print_message(
            f'{dataset_name}: no configured required keys were found in the table.',
            level='WARNING',
            verbose=verbose,
        )

        return (
            data,
            {
                'required_keys': [],
                'rows_missing_required_keys': 0,
                'action': 'none',
            },
        )

    missing_key_mask = (
        data[
            required_keys
        ]
        .isna()
        .any(
            axis=1
        )
    )

    count = int(
        missing_key_mask.sum()
    )

    if count == 0:
        _print_message(
            f'{dataset_name}: all rows contain required keys {required_keys}.',
            verbose=verbose,
        )

        return (
            data,
            {
                'required_keys': required_keys,
                'rows_missing_required_keys': 0,
                'action': 'none',
            },
        )

    _print_message(
        f'{dataset_name}: {count:,} rows are missing at least one required key '
        f'{required_keys}.',
        level='WARNING',
        verbose=verbose,
    )

    if rules.drop_rows_missing_required_keys:
        data = data.loc[
            ~missing_key_mask
        ].copy()

        action = 'dropped'

        _print_message(
            f'{dataset_name}: these rows were DROPPED because the observation '
            'cannot be reliably identified or located in time.',
            verbose=verbose,
        )

    else:
        data[
            'quality_missing_required_key'
        ] = missing_key_mask

        action = 'flagged'

        _print_message(
            f'{dataset_name}: these rows were KEPT and flagged.',
            verbose=verbose,
        )

    return (
        data,
        {
            'required_keys': required_keys,
            'rows_missing_required_keys': count,
            'action': action,
        },
    )


def _column_can_be_imputed(
    dataframe: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> tuple[
    bool,
    str,
]:
    """Decide whether automatic statistical imputation is appropriate."""
    required_keys = set(
        _required_keys_for(
            dataset_name,
            dataframe,
        )
    )

    if column in required_keys:
        return (
            False,
            'required_key',
        )

    if _is_identifier_column(
        column
    ):
        return (
            False,
            'identifier_or_code',
        )

    if pd.api.types.is_datetime64_any_dtype(
        dataframe[column]
    ):
        return (
            False,
            'datetime',
        )

    if _is_non_imputable_text_column(
        column
    ):
        return (
            False,
            'name_description_source_or_free_text',
        )

    if _is_flag_column(
        column
    ):
        return (
            False,
            'quality_or_boolean_flag',
        )

    return (
        True,
        'eligible',
    )


def impute_low_missingness(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    """
    Apply conservative automatic imputation.

    Only low-missingness, non-key, non-datetime, non-free-text fields are
    eligible.
    """
    data = dataframe.copy()

    actions: list[
        dict[str, Any]
    ] = []

    if not rules.auto_impute_low_missingness:
        _print_message(
            f'{dataset_name}: automatic imputation is disabled.',
            verbose=verbose,
        )

        return (
            data,
            actions,
        )

    for column in list(
        data.columns
    ):
        missing_count = int(
            data[column].isna().sum()
        )

        if missing_count == 0:
            continue

        fraction = (
            missing_count
            / len(data)
            if len(data)
            else 0.0
        )

        eligible, reason = (
            _column_can_be_imputed(
                data,
                column,
                dataset_name,
            )
        )

        if not eligible:
            actions.append(
                {
                    'column': column,
                    'missing_count': missing_count,
                    'missing_fraction': fraction,
                    'action': 'left_missing',
                    'reason': reason,
                }
            )

            _print_message(
                f'{dataset_name}.{column}: {missing_count:,} missing '
                f'({fraction:.1%}) -> LEFT MISSING ({reason}).',
                verbose=verbose,
            )

            continue

        if (
            fraction
            > rules.max_missing_fraction_for_auto_imputation
        ):
            actions.append(
                {
                    'column': column,
                    'missing_count': missing_count,
                    'missing_fraction': fraction,
                    'action': 'left_missing',
                    'reason': 'missing_fraction_above_threshold',
                }
            )

            _print_message(
                f'{dataset_name}.{column}: {missing_count:,} missing '
                f'({fraction:.1%}) -> LEFT MISSING because it exceeds the '
                f'{rules.max_missing_fraction_for_auto_imputation:.1%} '
                'automatic-imputation threshold.',
                level='WARNING',
                verbose=verbose,
            )

            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            strategy = (
                rules.numeric_imputation_strategy
            )

            if strategy == 'none':
                actions.append(
                    {
                        'column': column,
                        'missing_count': missing_count,
                        'missing_fraction': fraction,
                        'action': 'left_missing',
                        'reason': 'numeric_imputation_disabled',
                    }
                )

                continue

            if strategy == 'median':
                fill_value = (
                    data[column]
                    .median()
                )

            elif strategy == 'mean':
                fill_value = (
                    data[column]
                    .mean()
                )

            else:
                raise ValueError(
                    f'Unknown numeric imputation strategy: {strategy}'
                )

            if pd.isna(
                fill_value
            ):
                continue

            flag_column = (
                f'{column}__imputed'
            )

            data[
                flag_column
            ] = data[column].isna()

            data[column] = data[column].fillna(
                fill_value
            )

            actions.append(
                {
                    'column': column,
                    'missing_count': missing_count,
                    'missing_fraction': fraction,
                    'action': f'{strategy}_imputation',
                    'fill_value': _json_safe(
                        fill_value
                    ),
                    'flag_column': flag_column,
                }
            )

            _print_message(
                f'{dataset_name}.{column}: {missing_count:,} missing '
                f'({fraction:.1%}) -> {strategy.upper()} imputation '
                f'with {fill_value:.6g}. Flag created: {flag_column}.',
                verbose=verbose,
            )

            continue

        strategy = (
            rules.categorical_imputation_strategy
        )

        if strategy == 'none':
            actions.append(
                {
                    'column': column,
                    'missing_count': missing_count,
                    'missing_fraction': fraction,
                    'action': 'left_missing',
                    'reason': 'categorical_imputation_disabled',
                }
            )

            continue

        if strategy == 'constant':
            fill_value = (
                rules.categorical_missing_label
            )

        elif strategy == 'mode':
            mode = data[column].mode(
                dropna=True
            )

            if mode.empty:
                continue

            fill_value = (
                mode.iloc[0]
            )

        else:
            raise ValueError(
                f'Unknown categorical imputation strategy: {strategy}'
            )

        flag_column = (
            f'{column}__imputed'
        )

        data[
            flag_column
        ] = data[column].isna()

        data[column] = data[column].fillna(
            fill_value
        )

        actions.append(
            {
                'column': column,
                'missing_count': missing_count,
                'missing_fraction': fraction,
                'action': f'{strategy}_imputation',
                'fill_value': _json_safe(
                    fill_value
                ),
                'flag_column': flag_column,
            }
        )

        _print_message(
            f'{dataset_name}.{column}: {missing_count:,} missing '
            f'({fraction:.1%}) -> categorical {strategy.upper()} imputation '
            f'with {fill_value!r}. Flag created: {flag_column}.',
            verbose=verbose,
        )

    return (
        data,
        actions,
    )


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

def robust_z_scores(
    series: pd.Series,
) -> pd.Series:
    """
    Compute robust z-scores using median and median absolute deviation (MAD).

    Formula
    -------
    robust_z = 0.6745 * (x - median) / MAD
    """
    values = pd.to_numeric(
        series,
        errors='coerce',
    )

    median = values.median()

    mad = (
        values
        .sub(
            median
        )
        .abs()
        .median()
    )

    if (
        pd.isna(
            mad
        )
        or math.isclose(
            float(
                mad
            ),
            0.0,
        )
    ):
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    return (
        0.6745
        * (
            values
            - median
        )
        / mad
    )


def detect_numeric_outliers(
    series: pd.Series,
    rules: DepurationRules,
) -> pd.DataFrame:
    """
    Detect statistical outliers using Tukey IQR and robust z-score criteria.

    Returns one row per original observation with all diagnostic quantities.
    """
    values = pd.to_numeric(
        series,
        errors='coerce',
    )

    q1 = values.quantile(
        0.25
    )

    q3 = values.quantile(
        0.75
    )

    iqr = (
        q3
        - q1
    )

    if (
        pd.isna(
            iqr
        )
        or math.isclose(
            float(
                iqr
            ),
            0.0,
        )
    ):
        lower_fence = -np.inf
        upper_fence = np.inf

    else:
        lower_fence = (
            q1
            - rules.iqr_multiplier
            * iqr
        )

        upper_fence = (
            q3
            + rules.iqr_multiplier
            * iqr
        )

    robust_z = robust_z_scores(
        values
    )

    iqr_outlier = (
        (
            values
            < lower_fence
        )
        | (
            values
            > upper_fence
        )
    )

    robust_z_outlier = (
        robust_z.abs()
        > rules.robust_z_threshold
    )

    is_outlier = (
        iqr_outlier
        | robust_z_outlier
    ).fillna(
        False
    )

    distance_beyond_fence = pd.Series(
        0.0,
        index=values.index,
        dtype=float,
    )

    if np.isfinite(
        lower_fence
    ):
        low_mask = (
            values
            < lower_fence
        )

        distance_beyond_fence.loc[
            low_mask
        ] = (
            lower_fence
            - values.loc[
                low_mask
            ]
        )

    if np.isfinite(
        upper_fence
    ):
        high_mask = (
            values
            > upper_fence
        )

        distance_beyond_fence.loc[
            high_mask
        ] = (
            values.loc[
                high_mask
            ]
            - upper_fence
        )

    if (
        pd.isna(
            iqr
        )
        or math.isclose(
            float(
                iqr
            ),
            0.0,
        )
    ):
        distance_in_iqr = pd.Series(
            np.nan,
            index=values.index,
            dtype=float,
        )

    else:
        distance_in_iqr = (
            distance_beyond_fence
            / iqr
        )

    criteria_count = (
        iqr_outlier.astype(
            int
        )
        + robust_z_outlier.fillna(
            False
        ).astype(
            int
        )
    )

    severity = pd.Series(
        'none',
        index=values.index,
        dtype='string',
    )

    severity.loc[
        is_outlier
        & criteria_count.eq(
            1
        )
    ] = 'candidate'

    severity.loc[
        is_outlier
        & criteria_count.eq(
            2
        )
    ] = 'strong'

    return pd.DataFrame(
        {
            'value': values,
            'is_outlier': is_outlier,
            'severity': severity,
            'iqr_outlier': iqr_outlier.fillna(
                False
            ),
            'robust_z_outlier': robust_z_outlier.fillna(
                False
            ),
            'criteria_count': criteria_count,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'iqr_lower_fence': lower_fence,
            'iqr_upper_fence': upper_fence,
            'distance_beyond_iqr_fence': distance_beyond_fence,
            'distance_in_iqr_units': distance_in_iqr,
            'robust_z': robust_z,
        },
        index=series.index,
    )


def _eligible_numeric_outlier_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
) -> list[str]:
    """
    Select numeric measurement columns suitable for statistical outlier tests.

    Dataset semantics are considered before generic statistics:
    - weather is handled through physical/logical checks instead;
    - event structural/ordinal variables are not measurements whose large
      values imply poor data quality.

    IDs, codes, flags and near-constant columns are also excluded.
    """
    if dataset_name in STATISTICAL_OUTLIER_SKIP_DATASETS:
        return []

    excluded_columns = STATISTICAL_OUTLIER_EXCLUDED_COLUMNS.get(
        dataset_name,
        set(),
    )

    columns: list[str] = []

    for column in dataframe.select_dtypes(
        include='number'
    ).columns:
        if column in excluded_columns:
            continue

        if _is_identifier_column(
            column
        ):
            continue

        if _is_flag_column(
            column
        ):
            continue

        valid = pd.to_numeric(
            dataframe[column],
            errors='coerce',
        ).dropna()

        if (
            len(
                valid
            )
            < rules.min_rows_for_outlier_detection
        ):
            continue

        if (
            valid.nunique()
            < rules.min_unique_values_for_outlier_detection
        ):
            continue

        columns.append(
            column
        )

    return columns


def _print_outlier_examples(
    detail: pd.DataFrame,
    dataset_name: str,
    column: str,
    rules: DepurationRules,
    verbose: bool,
) -> None:
    """Print the most extreme outlier observations."""
    if not verbose:
        return

    outliers = detail[
        detail['is_outlier']
    ].copy()

    if outliers.empty:
        return

    outliers[
        '_sort_score'
    ] = (
        outliers[
            'distance_in_iqr_units'
        ]
        .fillna(
            0
        )
        .abs()
        + outliers[
            'robust_z'
        ]
        .fillna(
            0
        )
        .abs()
    )

    outliers = (
        outliers
        .sort_values(
            '_sort_score',
            ascending=False,
        )
        .head(
            rules.max_outliers_to_print_per_column
        )
    )

    for index, row in outliers.iterrows():
        print(
            f'    row={index} | '
            f'value={row["value"]:.6g} | '
            f'severity={row["severity"]} | '
            f'IQR_distance={row["distance_beyond_iqr_fence"]:.6g} | '
            f'IQR_units={row["distance_in_iqr_units"]:.3f} | '
            f'robust_z={row["robust_z"]:.3f} | '
            f'IQR={bool(row["iqr_outlier"])} | '
            f'robust_z_flag={bool(row["robust_z_outlier"])}'
        )


def flag_statistical_outliers(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
]:
    """
    Detect, report and flag/drop statistical outliers in numeric measurements.
    """
    data = dataframe.copy()

    summaries: list[
        dict[str, Any]
    ] = []

    details_by_column: dict[
        str,
        pd.DataFrame,
    ] = {}

    if not rules.detect_outliers:
        _print_message(
            f'{dataset_name}: statistical outlier detection is disabled.',
            verbose=verbose,
        )

        return (
            data,
            summaries,
            details_by_column,
        )

    numeric_columns = (
        _eligible_numeric_outlier_columns(
            data,
            dataset_name,
            rules,
        )
    )

    if not numeric_columns:
        if dataset_name in STATISTICAL_OUTLIER_SKIP_DATASETS:
            _print_message(
                f'{dataset_name}: generic statistical outlier detection is '
                'SKIPPED by design. Weather variables are validated with '
                'physical/logical rules instead.',
                verbose=verbose,
            )
        else:
            _print_message(
                f'{dataset_name}: no numeric measurement columns qualify for '
                'statistical outlier detection.',
                verbose=verbose,
            )

        return (
            data,
            summaries,
            details_by_column,
        )

    rows_to_drop = pd.Series(
        False,
        index=data.index,
    )

    for column in numeric_columns:
        detail = detect_numeric_outliers(
            data[column],
            rules,
        )

        detail.insert(
            0,
            'row_index',
            detail.index,
        )

        detail.insert(
            0,
            'column',
            column,
        )

        detail.insert(
            0,
            'dataset',
            dataset_name,
        )

        details_by_column[
            column
        ] = detail.copy()

        flag_column = (
            f'{column}__outlier'
        )

        data[
            flag_column
        ] = detail[
            'is_outlier'
        ].astype(
            bool
        )

        count = int(
            detail[
                'is_outlier'
            ].sum()
        )

        strong_count = int(
            (
                detail[
                    'severity'
                ]
                == 'strong'
            ).sum()
        )

        lower_fence = (
            detail[
                'iqr_lower_fence'
            ].iloc[
                0
            ]
        )

        upper_fence = (
            detail[
                'iqr_upper_fence'
            ].iloc[
                0
            ]
        )

        max_abs_robust_z = (
            detail[
                'robust_z'
            ]
            .abs()
            .max()
        )

        summary = {
            'column': column,
            'n_outliers': count,
            'n_strong_outliers': strong_count,
            'outlier_fraction': (
                count
                / len(data)
                if len(data)
                else 0.0
            ),
            'iqr_lower_fence': _json_safe(
                lower_fence
            ),
            'iqr_upper_fence': _json_safe(
                upper_fence
            ),
            'max_abs_robust_z': _json_safe(
                max_abs_robust_z
            ),
            'action': rules.outlier_action,
        }

        summaries.append(
            summary
        )

        if count == 0:
            _print_message(
                f'{dataset_name}.{column}: no statistical outliers detected.',
                verbose=verbose,
            )

            continue

        _print_message(
            f'{dataset_name}.{column}: {count:,} outliers detected '
            f'({summary["outlier_fraction"]:.2%}); '
            f'{strong_count:,} are flagged by BOTH criteria. '
            f'IQR fences=[{lower_fence:.6g}, {upper_fence:.6g}], '
            f'max |robust z|={max_abs_robust_z:.3f}.',
            level='OUTLIER',
            verbose=verbose,
        )

        _print_outlier_examples(
            detail=detail,
            dataset_name=dataset_name,
            column=column,
            rules=rules,
            verbose=verbose,
        )

        if rules.outlier_action == 'drop':
            rows_to_drop |= data[
                flag_column
            ]

            _print_message(
                f'{dataset_name}.{column}: configured action is DROP. '
                'Rows flagged as outliers will be removed after all numeric '
                'columns have been inspected.',
                level='WARNING',
                verbose=verbose,
            )

        else:
            _print_message(
                f'{dataset_name}.{column}: configured action is FLAG. '
                'Outliers are KEPT because unusual business observations can '
                'be genuine.',
                verbose=verbose,
            )

    if (
        rules.outlier_action == 'drop'
        and rows_to_drop.any()
    ):
        count = int(
            rows_to_drop.sum()
        )

        data = data.loc[
            ~rows_to_drop
        ].copy()

        _print_message(
            f'{dataset_name}: {count:,} rows removed because at least one '
            'numeric variable was flagged as an outlier.',
            level='WARNING',
            verbose=verbose,
        )

    return (
        data,
        summaries,
        details_by_column,
    )


# =============================================================================
# DATASET-SPECIFIC QUALITY RULES
# =============================================================================

def _normalize_reservation_status(
    value: Any,
) -> Any:
    """Map heterogeneous reservation statuses to broad reusable groups."""
    if pd.isna(
        value
    ):
        return pd.NA

    status = _normalize_text(
        value
    )

    no_show_tokens = (
        'no show',
        'no_show',
        'no presentado',
        'ausente',
    )

    cancelled_tokens = (
        'cancel',
        'anulad',
        'rechaz',
    )

    completed_tokens = (
        'complet',
        'finaliz',
        'sentad',
        'cerrad',
        'realiz',
    )

    pending_tokens = (
        'pend',
        'confirm',
        'reservad',
    )

    if any(
        token in status
        for token in no_show_tokens
    ):
        return 'no_show'

    if any(
        token in status
        for token in cancelled_tokens
    ):
        return 'cancelled'

    if any(
        token in status
        for token in completed_tokens
    ):
        return 'completed'

    if any(
        token in status
        for token in pending_tokens
    ):
        return 'pending'

    return re.sub(
        r'\s+',
        '_',
        status,
    )


def apply_reservation_rules(
    dataframe: pd.DataFrame,
    rules: DepurationRules,
) -> pd.DataFrame:
    """
    Add reservation-specific analytical and quality variables.

    Lead-time semantics
    -------------------
    The reservation export contains genuine walk-ins. For these observations,
    `created_datetime` is usually the moment the customer was entered into the
    system after arriving or being seated, so it can be a few minutes later
    than the nominal reservation/service slot.

    Therefore:

    - `lead_time_hours_raw` preserves the literal timestamp difference;
    - `created_after_reservation_time` records that literal situation;
    - `is_walk_in` identifies walk-ins from the origin field;
    - analytical `lead_time_hours` is set to 0 for walk-ins;
    - a negative lead time is considered invalid only for NON-walk-in records;
    - invalid non-walk-in analytical lead times are left missing rather than
      automatically imputed.

    Long positive lead times are not generic statistical outliers: advance
    reservations made weeks or months ahead can be perfectly valid.

    No suspicious source row is automatically deleted here.
    """
    data = dataframe.copy()

    if 'status' in data.columns:
        data[
            'status_grouped'
        ] = data[
            'status'
        ].map(
            _normalize_reservation_status
        )

    if 'people' in data.columns:
        data[
            'people__invalid'
        ] = (
            data[
                'people'
            ].notna()
            & (
                data[
                    'people'
                ]
                <= 0
            )
        )

        data[
            'is_large_group'
        ] = (
            data[
                'people'
            ]
            .fillna(
                0
            )
            >= rules.large_group_people
        )

    # Determine walk-ins before interpreting lead time.
    if 'origin' in data.columns:
        normalized_origin = (
            data[
                'origin'
            ]
            .astype(
                'string'
            )
            .map(
                _normalize_text
            )
        )

        data[
            'is_walk_in'
        ] = (
            normalized_origin
            .str.contains(
                r'walk|sin reserva|paso|puerta',
                regex=True,
                na=False,
            )
        )
    else:
        data[
            'is_walk_in'
        ] = False

    if {
        'created_datetime',
        'reservation_datetime',
    }.issubset(
        data.columns
    ):
        reservation_datetime = pd.to_datetime(
            data[
                'reservation_datetime'
            ],
            errors='coerce',
        )

        created_datetime = pd.to_datetime(
            data[
                'created_datetime'
            ],
            errors='coerce',
        )

        raw_lead_time_hours = (
            (
                reservation_datetime
                - created_datetime
            )
            .dt.total_seconds()
            / 3600.0
        )

        data[
            'lead_time_hours_raw'
        ] = raw_lead_time_hours

        data[
            'created_after_reservation_time'
        ] = (
            raw_lead_time_hours.notna()
            & (
                raw_lead_time_hours
                < 0
            )
        )

        data[
            'walk_in_created_after_slot'
        ] = (
            data[
                'is_walk_in'
            ]
            & data[
                'created_after_reservation_time'
            ]
        )

        # A walk-in has no advance-booking horizon by definition.
        analytical_lead_time = (
            raw_lead_time_hours
            .copy()
        )

        analytical_lead_time.loc[
            data[
                'is_walk_in'
            ]
            & analytical_lead_time.notna()
        ] = 0.0

        non_walk_in_negative = (
            ~data[
                'is_walk_in'
            ]
            & raw_lead_time_hours.notna()
            & (
                raw_lead_time_hours
                < 0
            )
        )

        # Keep the source calculation in lead_time_hours_raw, but do not expose
        # an impossible negative value as the analytical feature.
        analytical_lead_time.loc[
            non_walk_in_negative
        ] = np.nan

        data[
            'lead_time_hours'
        ] = analytical_lead_time

        data[
            'lead_time__invalid'
        ] = non_walk_in_negative

    return data


def apply_sales_rules(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add quality/context flags for article-sales datasets.

    Negative units/amounts are kept because they may represent returns,
    corrections or accounting adjustments.
    """
    data = dataframe.copy()

    if 'units' in data.columns:
        data[
            'units__negative'
        ] = (
            data[
                'units'
            ].notna()
            & (
                data[
                    'units'
                ]
                < 0
            )
        )

    if 'amount' in data.columns:
        data[
            'amount__negative'
        ] = (
            data[
                'amount'
            ].notna()
            & (
                data[
                    'amount'
                ]
                < 0
            )
        )

    if {
        'units',
        'amount',
    }.issubset(
        data.columns
    ):
        explicit_invitation = pd.Series(
            False,
            index=data.index,
        )

        invitation_columns = [
            column
            for column in data.columns
            if 'invit' in column.lower()
        ]

        for column in invitation_columns:
            explicit_invitation |= (
                data[
                    column
                ]
                .astype(
                    'string'
                )
                .map(
                    _normalize_text
                )
                .str.contains(
                    r'(^si$|^true$|invit)',
                    regex=True,
                    na=False,
                )
            )

        zero_amount_positive_units = (
            data[
                'units'
            ].fillna(
                0
            )
            > 0
        ) & (
            data[
                'amount'
            ]
            .fillna(
                0
            )
            .abs()
            <= 1e-12
        )

        data[
            'es_invitacion'
        ] = (
            explicit_invitation
            | zero_amount_positive_units
        )

    if {
        'report_start',
        'report_end',
    }.issubset(
        data.columns
    ):
        data[
            'report_days'
        ] = (
            pd.to_datetime(
                data[
                    'report_end'
                ],
                errors='coerce',
            )
            - pd.to_datetime(
                data[
                    'report_start'
                ],
                errors='coerce',
            )
        ).dt.days + 1

        data[
            'period__invalid'
        ] = (
            data[
                'report_days'
            ].notna()
            & (
                data[
                    'report_days'
                ]
                <= 0
            )
        )

    return data


def apply_ticket_rules(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add ticket-specific time and consistency flags."""
    data = dataframe.copy()

    if 'date' in data.columns:
        data[
            'date'
        ] = pd.to_datetime(
            data[
                'date'
            ],
            errors='coerce',
        ).dt.normalize()

        data[
            'day_of_week'
        ] = (
            data[
                'date'
            ].dt.day_name()
        )

        data[
            'week_of_year'
        ] = (
            data[
                'date'
            ]
            .dt.isocalendar()
            .week
            .astype(
                'Int64'
            )
        )

    if 'document_total' in data.columns:
        data[
            'document_total__negative'
        ] = (
            data[
                'document_total'
            ].notna()
            & (
                data[
                    'document_total'
                ]
                < 0
            )
        )

    if 'receipt_count' in data.columns:
        data[
            'receipt_count__negative'
        ] = (
            data[
                'receipt_count'
            ].notna()
            & (
                data[
                    'receipt_count'
                ]
                < 0
            )
        )

    return data


def apply_tip_rules(
    dataframe: pd.DataFrame,
    rules: DepurationRules,
) -> pd.DataFrame:
    """Add tip-specific consistency variables."""
    data = dataframe.copy()

    if 'tip' in data.columns:
        data[
            'tip__negative'
        ] = (
            data[
                'tip'
            ].notna()
            & (
                data[
                    'tip'
                ]
                < 0
            )
        )

    if {
        'document_amount',
        'tip',
        'document_total',
    }.issubset(
        data.columns
    ):
        data[
            'tip_reconciliation_difference'
        ] = (
            data[
                'document_amount'
            ]
            + data[
                'tip'
            ]
            - data[
                'document_total'
            ]
        )

        data[
            'tip_reconciliation__invalid'
        ] = (
            data[
                'tip_reconciliation_difference'
            ]
            .abs()
            > rules.payment_reconciliation_tolerance
        )

        denominator = (
            data[
                'document_amount'
            ]
            .abs()
            .replace(
                0,
                np.nan,
            )
        )

        data[
            'tip_ratio'
        ] = (
            data[
                'tip'
            ]
            / denominator
        )

    return data


def apply_invoice_rules(
    dataframe: pd.DataFrame,
    rules: DepurationRules,
) -> pd.DataFrame:
    """Add PDF invoice/ticket consistency checks."""
    data = dataframe.copy()

    for column in [
        'base',
        'vat',
        'total',
        'cash_amount',
        'card_amount',
    ]:
        if column in data.columns:
            data[
                f'{column}__negative'
            ] = (
                data[
                    column
                ].notna()
                & (
                    data[
                        column
                    ]
                    < 0
                )
            )

    if {
        'base',
        'vat',
        'total',
    }.issubset(
        data.columns
    ):
        available = (
            data[
                [
                    'base',
                    'vat',
                    'total',
                ]
            ]
            .notna()
            .all(
                axis=1
            )
        )

        data[
            'invoice_total_difference'
        ] = np.nan

        data.loc[
            available,
            'invoice_total_difference',
        ] = (
            data.loc[
                available,
                'base',
            ]
            + data.loc[
                available,
                'vat',
            ]
            - data.loc[
                available,
                'total',
            ]
        )

        data[
            'invoice_reconciliation__invalid'
        ] = (
            data[
                'invoice_total_difference'
            ]
            .abs()
            > rules.invoice_reconciliation_tolerance
        ).fillna(
            False
        )

    if {
        'cash_amount',
        'card_amount',
        'total',
    }.issubset(
        data.columns
    ):
        available = (
            data[
                [
                    'cash_amount',
                    'card_amount',
                    'total',
                ]
            ]
            .notna()
            .all(
                axis=1
            )
        )

        data[
            'payment_total_difference'
        ] = np.nan

        data.loc[
            available,
            'payment_total_difference',
        ] = (
            data.loc[
                available,
                'cash_amount',
            ]
            + data.loc[
                available,
                'card_amount',
            ]
            - data.loc[
                available,
                'total',
            ]
        )

        data[
            'payment_reconciliation__invalid'
        ] = (
            data[
                'payment_total_difference'
            ]
            .abs()
            > rules.payment_reconciliation_tolerance
        ).fillna(
            False
        )

        data[
            'pago_tarjeta'
        ] = (
            data[
                'card_amount'
            ]
            .fillna(
                0
            )
            > 0
        )

    if 'pdf_parse_error' in data.columns:
        data[
            'pdf_parse__invalid'
        ] = data[
            'pdf_parse_error'
        ].notna()

    return data


def apply_weather_rules(
    dataframe: pd.DataFrame,
    rules: DepurationRules,
) -> pd.DataFrame:
    """
    Add physically/logically motivated weather-quality flags.

    Generic IQR/MAD outlier detection is intentionally not used for weather.
    Rain, sunshine and precipitation duration are strongly skewed and often
    zero-inflated; a rare heavy-rain day can be completely valid.

    The rules below only flag values that violate broad physical/logical
    constraints. Values are preserved.
    """
    data = dataframe.copy()

    for column in data.columns:
        normalized = column.lower()

        if not pd.api.types.is_numeric_dtype(
            data[
                column
            ]
        ):
            continue

        values = pd.to_numeric(
            data[column],
            errors='coerce',
        )

        # Temperature: extremely broad physical plausibility bounds.
        if (
            'temperature' in normalized
            or normalized.startswith('temp')
            or '_temp' in normalized
        ):
            data[
                f'{column}__physically_suspicious'
            ] = (
                values.notna()
                & (
                    (
                        values
                        < rules.plausible_temperature_min_c
                    )
                    | (
                        values
                        > rules.plausible_temperature_max_c
                    )
                )
            )

        # Rain / precipitation amounts cannot be negative.
        if (
            'precip' in normalized
            or 'rain' in normalized
        ):
            data[
                f'{column}__negative'
            ] = (
                values.notna()
                & (
                    values
                    < 0
                )
            )

        # Wind speed cannot be negative.
        if 'wind' in normalized:
            data[
                f'{column}__negative'
            ] = (
                values.notna()
                & (
                    values
                    < 0
                )
            )

        # Sunshine duration cannot be negative.
        if 'sunshine' in normalized:
            data[
                f'{column}__negative'
            ] = (
                values.notna()
                & (
                    values
                    < 0
                )
            )

        # Daily duration variables expressed in hours must be within [0, 24].
        if (
            normalized in {
                'precipitation_hours',
                'sunshine_hours',
            }
        ):
            data[
                f'{column}__physically_suspicious'
            ] = (
                values.notna()
                & (
                    (
                        values
                        < 0
                    )
                    | (
                        values
                        > rules.plausible_daily_hours_max
                    )
                )
            )

        # Daily sunshine duration in seconds must be within [0, 86400].
        if normalized == 'sunshine_duration_s':
            data[
                f'{column}__physically_suspicious'
            ] = (
                values.notna()
                & (
                    (
                        values
                        < 0
                    )
                    | (
                        values
                        > rules.plausible_daily_seconds_max
                    )
                )
            )

    return data


def apply_master_data_rules(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Flag conflicting repeated codes in article/department/menu master data.
    """
    data = dataframe.copy()

    if dataset_name in {
        'articulos',
        'menu',
    }:
        code_column = (
            'article_code'
        )

        descriptive_candidates = [
            'article_name',
            'department_code',
            'price',
        ]

    elif dataset_name == 'departamentos':
        code_column = (
            'department_code'
        )

        descriptive_candidates = [
            'department_name',
            'department_short_name',
        ]

    else:
        return data

    if code_column not in data.columns:
        return data

    descriptive_columns = [
        column
        for column in descriptive_candidates
        if column in data.columns
    ]

    conflict = pd.Series(
        False,
        index=data.index,
    )

    for _, group in data.groupby(
        code_column,
        dropna=True,
    ):
        if len(
            group
        ) <= 1:
            continue

        has_conflict = any(
            group[
                column
            ].nunique(
                dropna=True
            )
            > 1
            for column in descriptive_columns
        )

        if has_conflict:
            conflict.loc[
                group.index
            ] = True

    data[
        f'{code_column}__conflict'
    ] = conflict

    return data


def apply_event_rules(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply semantic quality checks to the expanded event calendar.

    `expected_impact` is categorical in the current source and is deliberately
    preserved as text. Event duration/day-number variables are structural, not
    statistical measurements.
    """
    data = dataframe.copy()

    for column in [
        'date',
        'event_start',
        'event_end',
    ]:
        if column in data.columns:
            data[
                column
            ] = pd.to_datetime(
                data[
                    column
                ],
                errors='coerce',
                dayfirst=True,
            ).dt.normalize()

    categorical_columns = [
        'expected_impact',
        'demand_direction',
        'date_confidence',
        'event_category',
        'event_subcategory',
        'event_scope',
        'event_location',
        'proximity_la_roca',
        'probable_service_window',
        'customer_segment',
        'suggested_feature',
        'source_type',
        'modeling_notes',
    ]

    for column in categorical_columns:
        if column in data.columns:
            data[
                column
            ] = (
                data[
                    column
                ]
                .astype(
                    'string'
                )
                .str.strip()
            )

    if (
        'event_start' in data.columns
        and 'event_end' in data.columns
    ):
        data[
            'event_range__invalid'
        ] = (
            data[
                'event_start'
            ].notna()
            & data[
                'event_end'
            ].notna()
            & (
                data[
                    'event_end'
                ]
                < data[
                    'event_start'
                ]
            )
        )

    if 'event_duration_days' in data.columns:
        duration = pd.to_numeric(
            data[
                'event_duration_days'
            ],
            errors='coerce',
        )

        data[
            'event_duration_days__invalid'
        ] = (
            duration.notna()
            & (
                duration
                <= 0
            )
        )

    if 'event_day_number' in data.columns:
        day_number = pd.to_numeric(
            data[
                'event_day_number'
            ],
            errors='coerce',
        )

        invalid = (
            day_number.notna()
            & (
                day_number
                <= 0
            )
        )

        if 'event_duration_days' in data.columns:
            duration = pd.to_numeric(
                data[
                    'event_duration_days'
                ],
                errors='coerce',
            )

            invalid = (
                invalid
                | (
                    day_number.notna()
                    & duration.notna()
                    & (
                        day_number
                        > duration
                    )
                )
            )

        data[
            'event_day_number__invalid'
        ] = invalid

    return data


def apply_calendar_rules(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize date fields used by event/holiday tables."""
    data = dataframe.copy()

    if 'date' in data.columns:
        data[
            'date'
        ] = pd.to_datetime(
            data[
                'date'
            ],
            errors='coerce',
            dayfirst=True,
        ).dt.normalize()

    return data


def apply_dataset_specific_rules(
    dataframe: pd.DataFrame,
    dataset_name: str,
    rules: DepurationRules,
) -> pd.DataFrame:
    """
    Dispatch conservative quality rules for the currently supported datasets.
    """
    if dataset_name == 'reservas':
        return apply_reservation_rules(
            dataframe,
            rules,
        )

    if dataset_name in {
        'ventas',
        'total_articles',
    }:
        return apply_sales_rules(
            dataframe
        )

    if dataset_name == 'tickets':
        return apply_ticket_rules(
            dataframe
        )

    if dataset_name == 'tips':
        return apply_tip_rules(
            dataframe,
            rules,
        )

    if dataset_name == 'facturas':
        return apply_invoice_rules(
            dataframe,
            rules,
        )

    if dataset_name in {
        'meteo_diaria',
        'meteo_horaria',
    }:
        return apply_weather_rules(
            dataframe,
            rules,
        )

    if dataset_name in {
        'articulos',
        'menu',
        'departamentos',
    }:
        return apply_master_data_rules(
            dataframe,
            dataset_name,
        )

    if dataset_name == 'eventos':
        return apply_event_rules(
            dataframe
        )

    if dataset_name == 'festivos':
        return apply_calendar_rules(
            dataframe
        )

    return dataframe.copy()


# =============================================================================
# QUALITY-FLAG SUMMARIES
# =============================================================================

def _quality_flag_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return generated quality columns that can be summarized as Booleans."""
    columns = []

    for column in dataframe.columns:
        normalized = column.lower()

        if (
            normalized.startswith(
                'quality_'
            )
            or normalized.endswith(
                '__invalid'
            )
            or normalized.endswith(
                '__negative'
            )
            or normalized.endswith(
                '_suspicious'
            )
            or normalized.endswith(
                '__conflict'
            )
            or normalized.endswith(
                '__outlier'
            )
        ):
            columns.append(
                column
            )

    return columns


def quality_flag_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how many rows are affected by each Boolean quality flag."""
    rows = []

    for column in _quality_flag_columns(
        dataframe
    ):
        values = (
            dataframe[
                column
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

        count = int(
            values.sum()
        )

        rows.append(
            {
                'flag': column,
                'flagged_rows': count,
                'flagged_fraction': (
                    count
                    / len(dataframe)
                    if len(dataframe)
                    else 0.0
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            'flagged_rows',
            ascending=False,
        )
        .reset_index(
            drop=True
        )
        if rows
        else pd.DataFrame(
            columns=[
                'flag',
                'flagged_rows',
                'flagged_fraction',
            ]
        )
    )


def print_quality_flag_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    verbose: bool = True,
) -> None:
    """Print only quality flags with at least one affected row."""
    summary = quality_flag_summary(
        dataframe
    )

    if summary.empty:
        return

    affected = summary[
        summary[
            'flagged_rows'
        ]
        > 0
    ]

    if affected.empty:
        _print_message(
            f'{dataset_name}: no dataset-specific quality flags are active.',
            verbose=verbose,
        )
        return

    _print_message(
        f'{dataset_name}: dataset-specific quality flags detected:',
        level='WARNING',
        verbose=verbose,
    )

    if verbose:
        print(
            affected.to_string(
                index=False,
                formatters={
                    'flagged_fraction': lambda value: f'{value:.2%}',
                },
            )
        )


# =============================================================================
# FIGURES
# =============================================================================

def _safe_filename(
    value: str,
) -> str:
    """Create a filesystem-safe name."""
    return (
        re.sub(
            r'[^a-zA-Z0-9_-]+',
            '_',
            str(
                value
            ),
        )
        .strip(
            '_'
        )
    )


def plot_missingness(
    dataframe: pd.DataFrame,
    dataset_name: str,
    figures_dir: str | Path,
) -> Path | None:
    """Save a horizontal missing-value percentage chart."""
    table = missingness_table(
        dataframe
    )

    missing = table[
        table[
            'missing_count'
        ]
        > 0
    ].copy()

    if missing.empty:
        return None

    missing = (
        missing
        .sort_values(
            'missing_fraction',
            ascending=True,
        )
    )

    output_dir = _ensure_directory(
        figures_dir
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            max(
                4,
                0.32
                * len(
                    missing
                ),
            ),
        )
    )

    axis.barh(
        missing[
            'column'
        ],
        100
        * missing[
            'missing_fraction'
        ],
    )

    axis.set_xlabel(
        'Missing values (%)'
    )

    axis.set_title(
        f'{dataset_name} | Missing values'
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / (
            f'{_safe_filename(dataset_name)}'
            '__missingness.png'
        )
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
    )

    plt.close(
        figure
    )

    return output_path


def plot_numeric_distribution(
    series: pd.Series,
    dataset_name: str,
    column: str,
    outlier_mask: pd.Series,
    figures_dir: str | Path,
) -> list[Path]:
    """
    Save histogram and boxplot for one numeric quality variable.
    """
    values = pd.to_numeric(
        series,
        errors='coerce',
    ).dropna()

    if values.empty:
        return []

    output_dir = _ensure_directory(
        figures_dir
    )

    paths: list[
        Path
    ] = []

    unique_count = int(
        values.nunique()
    )

    bins = min(
        40,
        max(
            10,
            int(
                math.sqrt(
                    max(
                        unique_count,
                        1,
                    )
                )
            ),
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            4.5,
        )
    )

    axis.hist(
        values,
        bins=bins,
    )

    axis.set_title(
        f'{dataset_name} | {column} | Distribution'
    )

    axis.set_xlabel(
        column
    )

    axis.set_ylabel(
        'Count'
    )

    figure.tight_layout()

    histogram_path = (
        output_dir
        / (
            f'{_safe_filename(dataset_name)}'
            f'__{_safe_filename(column)}'
            '__hist.png'
        )
    )

    figure.savefig(
        histogram_path,
        dpi=150,
        bbox_inches='tight',
    )

    plt.close(
        figure
    )

    paths.append(
        histogram_path
    )

    figure, axis = plt.subplots(
        figsize=(
            8,
            2.8,
        )
    )

    axis.boxplot(
        values,
        vert=False,
    )

    axis.set_title(
        f'{dataset_name} | {column} | '
        f'{int(outlier_mask.fillna(False).sum())} flagged outliers'
    )

    axis.set_xlabel(
        column
    )

    figure.tight_layout()

    boxplot_path = (
        output_dir
        / (
            f'{_safe_filename(dataset_name)}'
            f'__{_safe_filename(column)}'
            '__boxplot.png'
        )
    )

    figure.savefig(
        boxplot_path,
        dpi=150,
        bbox_inches='tight',
    )

    plt.close(
        figure
    )

    paths.append(
        boxplot_path
    )

    return paths


# =============================================================================
# MACHINE-READABLE QUALITY REPORTS
# =============================================================================

def _save_dataframe_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Save a DataFrame to CSV with a stable UTF-8 encoding."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding='utf-8',
    )


def save_dataset_quality_report(
    dataset_name: str,
    dataframe_before: pd.DataFrame,
    dataframe_after: pd.DataFrame,
    duplicate_summary: dict[str, Any],
    business_key_summary: dict[str, Any],
    missing_key_summary: dict[str, Any],
    imputation_actions: list[dict[str, Any]],
    outlier_summaries: list[dict[str, Any]],
    outlier_details: dict[str, pd.DataFrame],
    rules: DepurationRules,
    paths: DepurationPaths,
) -> dict[str, Path]:
    """
    Save JSON/CSV quality artifacts for one dataset.
    """
    reports_dir = _ensure_directory(
        paths.reports_dir
    )

    saved_paths: dict[
        str,
        Path,
    ] = {}

    missing_path = (
        reports_dir
        / (
            f'{dataset_name}'
            '__missingness.csv'
        )
    )

    _save_dataframe_csv(
        missingness_table(
            dataframe_after
        ),
        missing_path,
    )

    saved_paths[
        'missingness'
    ] = missing_path

    flags_path = (
        reports_dir
        / (
            f'{dataset_name}'
            '__quality_flags.csv'
        )
    )

    _save_dataframe_csv(
        quality_flag_summary(
            dataframe_after
        ),
        flags_path,
    )

    saved_paths[
        'quality_flags'
    ] = flags_path

    if (
        rules.save_outlier_details
        and outlier_details
    ):
        combined_outliers = pd.concat(
            [
                detail[
                    detail[
                        'is_outlier'
                    ]
                ].copy()
                for detail in outlier_details.values()
                if detail[
                    'is_outlier'
                ].any()
            ],
            ignore_index=True,
        ) if any(
            detail[
                'is_outlier'
            ].any()
            for detail in outlier_details.values()
        ) else pd.DataFrame()

        if not combined_outliers.empty:
            outlier_path = (
                reports_dir
                / (
                    f'{dataset_name}'
                    '__outliers.csv'
                )
            )

            _save_dataframe_csv(
                combined_outliers,
                outlier_path,
            )

            saved_paths[
                'outliers'
            ] = outlier_path

    report = {
        'dataset': dataset_name,
        'rows_before': len(
            dataframe_before
        ),
        'rows_after': len(
            dataframe_after
        ),
        'columns_before': len(
            dataframe_before.columns
        ),
        'columns_after': len(
            dataframe_after.columns
        ),
        'duplicate_summary': duplicate_summary,
        'business_key_summary': business_key_summary,
        'missing_required_key_summary': missing_key_summary,
        'imputation_actions': imputation_actions,
        'outlier_summaries': outlier_summaries,
        'active_quality_flags': (
            quality_flag_summary(
                dataframe_after
            )
            .to_dict(
                orient='records'
            )
        ),
        'rules': asdict(
            rules
        ),
    }

    json_path = (
        reports_dir
        / (
            f'{dataset_name}'
            '__quality_report.json'
        )
    )

    with open(
        json_path,
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_safe,
        )

    saved_paths[
        'quality_report'
    ] = json_path

    return saved_paths


# =============================================================================
# SILVER PERSISTENCE
# =============================================================================

def save_silver_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    silver_dir: str | Path,
    verbose: bool = True,
) -> Path:
    """Save one depurated dataset as a Silver Parquet snapshot."""
    output_dir = _ensure_directory(
        silver_dir
    )

    filename = SILVER_FILENAMES.get(
        dataset_name,
        f'{dataset_name}_silver.parquet',
    )

    output_path = (
        output_dir
        / filename
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    _print_message(
        f'SILVER | {dataset_name}: '
        f'{len(dataframe):,} rows x {len(dataframe.columns):,} columns '
        f'-> {output_path}',
        verbose=verbose,
    )

    return output_path


def save_silver_datasets(
    datasets: dict[str, pd.DataFrame],
    silver_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Path]:
    """Save all depurated datasets."""
    saved_paths = {}

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        saved_paths[
            dataset_name
        ] = save_silver_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
            silver_dir=silver_dir,
            verbose=verbose,
        )

    return saved_paths



def _compact_dataset_console_summary(
    dataframe_before: pd.DataFrame,
    dataframe_after: pd.DataFrame,
    dataset_name: str,
    imputation_actions: list[dict[str, Any]],
    outlier_summaries: list[dict[str, Any]],
    verbose: bool = True,
) -> None:
    """Print one concise line per depurated dataset."""
    if not verbose:
        return

    rows_before = int(
        len(
            dataframe_before
        )
    )
    rows_after = int(
        len(
            dataframe_after
        )
    )
    rows_removed = (
        rows_before
        - rows_after
    )

    missing_cells = int(
        dataframe_after.isna().sum().sum()
    )

    imputed_columns = sum(
        str(
            action.get(
                'action',
                '',
            )
        ).endswith(
            '_imputation'
        )
        for action in imputation_actions
    )

    outlier_flags = sum(
        int(
            item.get(
                'n_outliers',
                0,
            )
        )
        for item in outlier_summaries
    )

    quality = quality_flag_summary(
        dataframe_after
    )

    active_quality_flags = int(
        (
            quality[
                'flagged_rows'
            ]
            > 0
        ).sum()
        if not quality.empty
        else 0
    )

    parts = [
        f'{dataset_name}: {rows_after:,} rows x {len(dataframe_after.columns):,} cols',
    ]

    if rows_removed:
        parts.append(
            f'rows removed={rows_removed:,}'
        )

    if missing_cells:
        parts.append(
            f'missing cells={missing_cells:,}'
        )

    if imputed_columns:
        parts.append(
            f'imputed cols={imputed_columns:,}'
        )

    if active_quality_flags:
        parts.append(
            f'active quality flags={active_quality_flags:,}'
        )

    if outlier_flags:
        parts.append(
            f'outlier flags={outlier_flags:,}'
        )

    print(
        '[INFO] '
        + ' | '.join(
            parts
        )
    )


def _compact_global_console_summary(
    summaries: dict[str, dict[str, Any]],
    summary_path: Path,
    verbose: bool = True,
) -> None:
    """Print only the global depuration totals in compact console mode."""
    if not verbose:
        return

    summary = depuration_summary_table(
        summaries
    )

    if summary.empty:
        print(
            '[INFO] Depuration completed: no datasets were processed.'
        )
        return

    print(
        '\n[INFO] Depuration summary: '
        f'{len(summary):,} datasets | '
        f'rows removed={int(summary["rows_removed"].sum()):,} | '
        f'imputed columns={int(summary["imputed_columns"].sum()):,} | '
        f'outlier flags={int(summary["statistical_outlier_flags"].sum()):,}.'
    )
    print(
        f'[INFO] Full depuration reports: {summary_path.parent}'
    )


# =============================================================================
# SINGLE-DATASET DEPURATION
# =============================================================================

def depurate_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    paths: DepurationPaths,
    rules: DepurationRules | None = None,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Depurate one standardized Bronze/structured dataset.

    Order of operations
    -------------------
    1. Reassert expected common data types.
    2. Standardize blank/missing string representations.
    3. Inspect/remove exact duplicates.
    4. Flag repeated business keys.
    5. Inspect missingness.
    6. Treat rows missing essential keys.
    7. Apply dataset-specific quality rules.
    8. Conservatively impute eligible low-missingness fields.
    9. Detect/report statistical outliers.
    10. Save plots and quality reports.
    11. Return the Silver-ready table.
    """
    rules = (
        rules
        if rules is not None
        else DepurationRules()
    )

    rules.validate()

    detailed_console = (
        verbose
        and rules.console_detail == 'detailed'
    )

    _print_header(
        f'DEPURATION | {dataset_name}',
        verbose=detailed_console,
    )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            f'{dataset_name} is not a pandas DataFrame.'
        )

    before = dataframe.copy(
        deep=True
    )

    data = _coerce_known_column_types(
        dataframe
    )

    data = standardize_empty_values(
        data,
        rules,
    )

    _print_message(
        f'{dataset_name}: starting with '
        f'{len(data):,} rows x {len(data.columns):,} columns.',
        verbose=detailed_console,
    )

    _print_subheader(
        '1. Exact duplicates',
        verbose=detailed_console,
    )

    data, duplicate_summary = (
        treat_exact_duplicates(
            dataframe=data,
            dataset_name=dataset_name,
            rules=rules,
            verbose=detailed_console,
        )
    )

    _print_subheader(
        '2. Repeated business keys',
        verbose=detailed_console,
    )

    data, business_key_summary = (
        inspect_business_key_duplicates(
            dataframe=data,
            dataset_name=dataset_name,
            verbose=detailed_console,
        )
    )

    _print_subheader(
        '3. Missing values before treatment',
        verbose=detailed_console,
    )

    print_missingness_summary(
        dataframe=data,
        dataset_name=dataset_name,
        rules=rules,
        verbose=detailed_console,
    )

    if rules.save_figures:
        plot_missingness(
            dataframe=data,
            dataset_name=dataset_name,
            figures_dir=paths.figures_dir,
        )

    _print_subheader(
        '4. Required keys',
        verbose=detailed_console,
    )

    data, missing_key_summary = (
        treat_missing_required_keys(
            dataframe=data,
            dataset_name=dataset_name,
            rules=rules,
            verbose=detailed_console,
        )
    )

    _print_subheader(
        '5. Dataset-specific quality rules',
        verbose=detailed_console,
    )

    data = apply_dataset_specific_rules(
        dataframe=data,
        dataset_name=dataset_name,
        rules=rules,
    )

    print_quality_flag_summary(
        dataframe=data,
        dataset_name=dataset_name,
        verbose=detailed_console,
    )

    _print_subheader(
        '6. Conservative missing-value treatment',
        verbose=detailed_console,
    )

    data, imputation_actions = (
        impute_low_missingness(
            dataframe=data,
            dataset_name=dataset_name,
            rules=rules,
            verbose=detailed_console,
        )
    )

    _print_subheader(
        '7. Statistical outliers',
        verbose=detailed_console,
    )

    data, outlier_summaries, outlier_details = (
        flag_statistical_outliers(
            dataframe=data,
            dataset_name=dataset_name,
            rules=rules,
            verbose=detailed_console,
        )
    )

    if rules.save_figures:
        for column, detail in outlier_details.items():
            if column not in data.columns:
                continue

            flag_column = (
                f'{column}__outlier'
            )

            if flag_column in data.columns:
                outlier_mask = data[
                    flag_column
                ]
            else:
                outlier_mask = pd.Series(
                    False,
                    index=data.index,
                )

            plot_numeric_distribution(
                series=data[
                    column
                ],
                dataset_name=dataset_name,
                column=column,
                outlier_mask=outlier_mask,
                figures_dir=paths.figures_dir,
            )

    _print_subheader(
        '8. Final quality summary',
        verbose=detailed_console,
    )

    print_missingness_summary(
        dataframe=data,
        dataset_name=dataset_name,
        rules=rules,
        verbose=detailed_console,
    )

    print_quality_flag_summary(
        dataframe=data,
        dataset_name=dataset_name,
        verbose=detailed_console,
    )

    report_paths = {}

    if rules.save_quality_reports:
        report_paths = (
            save_dataset_quality_report(
                dataset_name=dataset_name,
                dataframe_before=before,
                dataframe_after=data,
                duplicate_summary=duplicate_summary,
                business_key_summary=business_key_summary,
                missing_key_summary=missing_key_summary,
                imputation_actions=imputation_actions,
                outlier_summaries=outlier_summaries,
                outlier_details=outlier_details,
                rules=rules,
                paths=paths,
            )
        )

    summary = {
        'dataset': dataset_name,
        'rows_before': len(
            before
        ),
        'rows_after': len(
            data
        ),
        'columns_before': len(
            before.columns
        ),
        'columns_after': len(
            data.columns
        ),
        'duplicate_summary': duplicate_summary,
        'business_key_summary': business_key_summary,
        'missing_key_summary': missing_key_summary,
        'imputation_actions': imputation_actions,
        'outlier_summaries': outlier_summaries,
        'report_paths': {
            name: str(
                path
            )
            for name, path
            in report_paths.items()
        },
    }

    _print_message(
        f'{dataset_name}: depuration finished -> '
        f'{len(before):,} rows became {len(data):,} rows.',
        verbose=detailed_console,
    )

    if rules.console_detail == 'summary':
        _compact_dataset_console_summary(
            dataframe_before=before,
            dataframe_after=data,
            dataset_name=dataset_name,
            imputation_actions=imputation_actions,
            outlier_summaries=outlier_summaries,
            verbose=verbose,
        )

    return (
        data.reset_index(
            drop=True
        ),
        summary,
    )


# =============================================================================
# DEPURATION SUMMARY
# =============================================================================

def depuration_summary_table(
    summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a compact table summarizing all depurated datasets."""
    rows = []

    for dataset_name, summary in sorted(
        summaries.items()
    ):
        outlier_count = sum(
            int(
                item.get(
                    'n_outliers',
                    0,
                )
            )
            for item in summary.get(
                'outlier_summaries',
                []
            )
        )

        imputed_columns = sum(
            action.get(
                'action',
                '',
            ).endswith(
                '_imputation'
            )
            for action in summary.get(
                'imputation_actions',
                []
            )
        )

        rows.append(
            {
                'dataset': dataset_name,
                'rows_before': summary.get(
                    'rows_before',
                    0,
                ),
                'rows_after': summary.get(
                    'rows_after',
                    0,
                ),
                'rows_removed': (
                    summary.get(
                        'rows_before',
                        0,
                    )
                    - summary.get(
                        'rows_after',
                        0,
                    )
                ),
                'exact_duplicates': (
                    summary.get(
                        'duplicate_summary',
                        {}
                    )
                    .get(
                        'exact_duplicate_rows',
                        0,
                    )
                ),
                'missing_key_rows': (
                    summary.get(
                        'missing_key_summary',
                        {}
                    )
                    .get(
                        'rows_missing_required_keys',
                        0,
                    )
                ),
                'imputed_columns': int(
                    imputed_columns
                ),
                'statistical_outlier_flags': int(
                    outlier_count
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_depuration_summary(
    summaries: dict[str, dict[str, Any]],
) -> None:
    """Print the final multi-dataset depuration summary."""
    summary = depuration_summary_table(
        summaries
    )

    if summary.empty:
        print(
            'No datasets were depurated.'
        )
        return

    _print_header(
        'FINAL DEPURATION SUMMARY',
        verbose=True,
    )

    print(
        summary.to_string(
            index=False
        )
    )


# =============================================================================
# MASTER DEPURATION FUNCTION
# =============================================================================

def run_depuration(
    datasets: dict[str, pd.DataFrame],
    paths: DepurationPaths,
    rules: DepurationRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Master depuration entry point: structured/Bronze datasets -> Silver.

    This is the function the future `analysis_prediction.py` main file should
    normally call.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Structured datasets returned by `ap_io.run_ingestion()` or loaded from
        existing Bronze/user-provided structured tables.
    paths : DepurationPaths
        Silver, figure and report output folders.
    rules : DepurationRules, optional
        Transparent user-configurable cleaning policy. If omitted, conservative
        defaults are used.
    verbose : bool, default=True
        Print console progress. The amount of detail is controlled by
        `rules.console_detail`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Depurated Silver-ready datasets.

    Notes
    -----
    Gold integration intentionally does NOT happen here. Cleaning and
    integration are conceptually different operations. The future main program
    can still place both operations under a single `DO_DEPURATION=True` flag by
    calling `run_depuration()` followed by `run_integration()`.
    """
    rules = (
        rules
        if rules is not None
        else DepurationRules()
    )

    rules.validate()

    _ensure_directory(
        paths.silver_dir
    )

    _ensure_directory(
        paths.figures_dir
    )

    _ensure_directory(
        paths.reports_dir
    )

    _print_header(
        'BRONZE / STRUCTURED DATA -> DEPURATION -> SILVER',
        verbose=verbose,
    )

    if not datasets:
        raise ValueError(
            'No datasets were supplied to run_depuration().'
        )

    silver_datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        if dataframe is None:
            _print_message(
                f'{dataset_name}: skipped because dataset is None.',
                level='WARNING',
                verbose=verbose,
            )
            continue

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            _print_message(
                f'{dataset_name}: skipped because object is not a DataFrame.',
                level='WARNING',
                verbose=verbose,
            )
            continue

        depurated, summary = depurate_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
            paths=paths,
            rules=rules,
            verbose=verbose,
        )

        silver_datasets[
            dataset_name
        ] = depurated

        summaries[
            dataset_name
        ] = summary

        save_silver_dataset(
            dataframe=depurated,
            dataset_name=dataset_name,
            silver_dir=paths.silver_dir,
            verbose=(
                verbose
                and rules.console_detail == 'detailed'
            ),
        )

    summary_table = depuration_summary_table(
        summaries
    )

    summary_path = (
        _ensure_directory(
            paths.reports_dir
        )
        / 'depuration_summary.csv'
    )

    summary_table.to_csv(
        summary_path,
        index=False,
        encoding='utf-8',
    )

    if rules.console_detail == 'detailed':
        print_depuration_summary(
            summaries
        )

        _print_message(
            f'Depuration completed: {len(silver_datasets)} Silver datasets created. '
            f'Global summary saved to {summary_path}.',
            verbose=verbose,
        )

    else:
        _compact_global_console_summary(
            summaries=summaries,
            summary_path=summary_path,
            verbose=verbose,
        )

    return silver_datasets
