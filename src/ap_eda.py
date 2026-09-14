"""
Automatic exploratory data analysis for the Analysis Prediction framework.

This module is intentionally downstream from depuration and integration:

    Raw -> Bronze -> Silver -> Gold -> EDA

EDA does not clean, impute, drop or overwrite source observations. Its job is
to describe the data that survived the quality pipeline, identify useful
patterns, quantify coverage and produce reproducible diagnostics that can guide
feature engineering and modelling later.

The public entry point is `run_eda()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class EDAPaths:
    """Output locations used by the EDA module."""

    figures_dir: str | Path
    tables_dir: str | Path
    reports_dir: str | Path

    def __post_init__(self) -> None:
        self.figures_dir = Path(self.figures_dir)
        self.tables_dir = Path(self.tables_dir)
        self.reports_dir = Path(self.reports_dir)


@dataclass
class EDARules:
    """
    User-facing EDA behaviour.

    The defaults deliberately favour a compact, decision-oriented EDA instead
    of creating one histogram for every numeric column. That avoids mixing
    descriptive exploration with data-quality diagnostics and reduces the
    chance of misleading plots for repeated/expanded records.
    """

    analyze_silver: bool = True
    analyze_gold: bool = True

    clean_previous_outputs: bool = True
    save_figures: bool = True
    save_tables: bool = True
    save_report: bool = True

    # Generic profiling.
    top_n_categories: int = 15
    max_categories_for_frequency_table: int = 30
    min_non_null_for_correlation: int = 20
    max_target_correlations: int = 25

    # Daily Gold.
    daily_target_candidates: tuple[str, ...] = (
        'facturacion',
        'facturacion_tickets',
    )
    daily_activity_candidates: tuple[str, ...] = (
        'num_tickets',
        'num_tickets_unicos',
    )
    rolling_window_days: int = 7

    # Automatic operational diagnostics. These do not alter data; they only
    # flag recurring low-activity weekdays and near-perfect target associations
    # for later feature-design review.
    low_activity_weekday_threshold: float = 0.25
    min_days_for_weekday_diagnostic: int = 4
    possible_leakage_correlation_threshold: float = 0.995

    # Article-period Gold.
    article_target_candidates: tuple[str, ...] = (
        'units',
        'amount',
    )
    article_top_n: int = 20

    # Curated Silver analyses.
    analyze_reservations: bool = True
    analyze_unique_events: bool = True

    # Generic distributions are OFF by default. Curated plots below are safer
    # and more interpretable for this framework.
    generic_numeric_distributions: bool = False
    generic_numeric_max_columns: int = 8
    histogram_bins: int = 30

    figure_dpi: int = 150

    def validate(self) -> None:
        if not self.analyze_silver and not self.analyze_gold:
            raise ValueError(
                'At least one of analyze_silver/analyze_gold must be True.'
            )

        if self.top_n_categories < 1:
            raise ValueError('top_n_categories must be >= 1.')

        if self.max_categories_for_frequency_table < 2:
            raise ValueError(
                'max_categories_for_frequency_table must be >= 2.'
            )

        if self.min_non_null_for_correlation < 3:
            raise ValueError(
                'min_non_null_for_correlation must be >= 3.'
            )

        if self.max_target_correlations < 1:
            raise ValueError('max_target_correlations must be >= 1.')

        if self.rolling_window_days < 1:
            raise ValueError('rolling_window_days must be >= 1.')

        if not 0 <= self.low_activity_weekday_threshold <= 1:
            raise ValueError(
                'low_activity_weekday_threshold must be between 0 and 1.'
            )

        if self.min_days_for_weekday_diagnostic < 1:
            raise ValueError(
                'min_days_for_weekday_diagnostic must be >= 1.'
            )

        if not 0 < self.possible_leakage_correlation_threshold <= 1:
            raise ValueError(
                'possible_leakage_correlation_threshold must be in (0, 1].'
            )

        if self.article_top_n < 1:
            raise ValueError('article_top_n must be >= 1.')

        if self.generic_numeric_max_columns < 1:
            raise ValueError('generic_numeric_max_columns must be >= 1.')

        if self.histogram_bins < 5:
            raise ValueError('histogram_bins must be >= 5.')

        if self.figure_dpi < 72:
            raise ValueError('figure_dpi must be >= 72.')



# =============================================================================
# SAVED SILVER / GOLD LOADING
# =============================================================================


def load_saved_layer_datasets(
    directory: str | Path,
    layer: str,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load previously persisted Silver or Gold parquet datasets for EDA.

    This enables an EDA-only execution without rerunning ingestion, depuration
    or integration.
    """
    normalized_layer = str(layer).strip().lower()

    if normalized_layer not in {
        'silver',
        'gold',
    }:
        raise ValueError(
            "layer must be either 'silver' or 'gold'."
        )

    source_dir = Path(
        directory
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f'{normalized_layer.title()} directory not found: {source_dir}'
        )

    datasets: dict[str, pd.DataFrame] = {}

    for path in sorted(
        source_dir.glob(
            '*.parquet'
        )
    ):
        dataset_name = path.stem

        if (
            normalized_layer == 'silver'
            and dataset_name.endswith(
                '_silver'
            )
        ):
            dataset_name = dataset_name[
                :-len('_silver')
            ]

        datasets[
            dataset_name
        ] = pd.read_parquet(
            path
        )

        _print_message(
            f'Loaded {normalized_layer.upper()} for EDA | '
            f'{dataset_name}: '
            f'{len(datasets[dataset_name]):,} rows x '
            f'{len(datasets[dataset_name].columns):,} columns',
            verbose=verbose,
        )

    if not datasets:
        raise FileNotFoundError(
            f'No parquet datasets found in {source_dir}.'
        )

    return datasets


# =============================================================================
# SMALL UTILITIES
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
        print(f'[{level}] {message}')


def _ensure_directory(
    path: str | Path,
) -> Path:
    directory = Path(path)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    return directory


def _prepare_output_directories(
    paths: EDAPaths,
    rules: EDARules,
) -> None:
    """
    Prepare EDA-owned output directories.

    Cleaning these directories prevents stale figures from an older execution
    from being mistaken for current results. Only the three EDA directories
    supplied here are touched.
    """
    for directory in [
        paths.figures_dir,
        paths.tables_dir,
        paths.reports_dir,
    ]:
        directory = Path(directory)

        if (
            rules.clean_previous_outputs
            and directory.exists()
        ):
            shutil.rmtree(
                directory
            )

        _ensure_directory(
            directory
        )


def _json_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        if pd.isna(value):
            return None
        return value.item()

    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        if pd.isna(value):
            return None
        return str(
            pd.Timestamp(value)
        )

    if isinstance(
        value,
        Path,
    ):
        return str(value)

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
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if pd.isna(value):
        return None

    return value


def _slugify(
    value: str,
) -> str:
    text = str(value).strip().lower()
    text = re.sub(
        r'[^a-z0-9]+',
        '_',
        text,
    )
    return text.strip('_') or 'output'


def _first_existing(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _infer_expected_period_days_eda(
    series: pd.Series,
) -> int | None:
    """Infer the usual positive reporting-period length from observed data."""
    values = (
        _to_numeric(
            series
        )
        .dropna()
    )

    values = values.loc[
        values.gt(
            0
        )
    ]

    if values.empty:
        return None

    counts = (
        values
        .round()
        .astype(
            int
        )
        .value_counts()
    )

    max_count = counts.max()
    candidates = counts.loc[
        counts.eq(
            max_count
        )
    ].index

    if len(candidates) == 0:
        return None

    return int(
        max(candidates)
    )


def _to_datetime(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors='coerce',
    )


def _is_quality_flag(
    column: str,
) -> bool:
    return (
        column.startswith('quality_')
        or column.endswith('__invalid')
        or column.endswith('__negative')
        or column.endswith('__outlier')
        or column.endswith('__imputed')
        or column.endswith('__suspicious')
        or column.endswith('__master_conflict')
    )


def _is_identifier_like(
    column: str,
) -> bool:
    normalized = column.lower()

    exact = {
        'id',
        'article_code',
        'department_code',
        'document_id',
        'ticket_id',
        'event_id',
        'period_id',
        'reference',
    }

    return (
        normalized in exact
        or normalized.endswith('_id')
        or normalized.endswith('_code')
    )


def _candidate_date_column(
    dataframe: pd.DataFrame,
) -> str | None:
    preferred = (
        'date',
        'reservation_datetime',
        'reservation_date',
        'datetime',
        'report_start',
        'event_start',
        'event_date',
    )

    column = _first_existing(
        dataframe,
        preferred,
    )

    if column is not None:
        return column

    datetime_columns = [
        name
        for name in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(
            dataframe[name]
        )
    ]

    return (
        datetime_columns[0]
        if datetime_columns
        else None
    )


def _dataset_date_range(
    dataframe: pd.DataFrame,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    """
    Infer the temporal coverage of a dataset for EDA reporting.

    Row-level dates take precedence over report-period metadata. This matters
    for datasets such as tickets, which can contain both an actual transaction
    date and report_start/report_end fields. Period bounds are used only when
    no row-level date/datetime is available.
    """
    # Prefer an actual observation timestamp whenever one exists.
    for column in (
        'date',
        'reservation_datetime',
        'reservation_date',
        'datetime',
        'event_date',
    ):
        if column not in dataframe.columns:
            continue

        values = _to_datetime(
            dataframe[column]
        ).dropna()

        if not values.empty:
            return values.min(), values.max()

    observed: list[pd.Timestamp] = []

    # Period-based sources need both limits when no row-level date exists.
    if (
        'report_start' in dataframe.columns
        or 'report_end' in dataframe.columns
    ):
        for column in (
            'report_start',
            'report_end',
        ):
            if column not in dataframe.columns:
                continue

            values = _to_datetime(
                dataframe[column]
            ).dropna()

            if not values.empty:
                observed.extend(
                    [
                        values.min(),
                        values.max(),
                    ]
                )

        if observed:
            return min(observed), max(observed)

    # Event sources can also represent a start/end interval.
    if (
        'event_start' in dataframe.columns
        and 'event_end' in dataframe.columns
        and 'date' not in dataframe.columns
    ):
        for column in (
            'event_start',
            'event_end',
        ):
            values = _to_datetime(
                dataframe[column]
            ).dropna()

            if not values.empty:
                observed.extend(
                    [
                        values.min(),
                        values.max(),
                    ]
                )

        if observed:
            return min(observed), max(observed)

    date_column = _candidate_date_column(
        dataframe
    )

    if date_column is None:
        return pd.NaT, pd.NaT

    dates = _to_datetime(
        dataframe[
            date_column
        ]
    ).dropna()

    if dates.empty:
        return pd.NaT, pd.NaT

    return dates.min(), dates.max()


def _save_table(
    dataframe: pd.DataFrame,
    filename: str,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if not rules.save_tables:
        return None

    output_path = (
        Path(paths.tables_dir)
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding='utf-8',
    )

    return output_path


def _save_figure(
    figure: plt.Figure,
    filename: str,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if not rules.save_figures:
        plt.close(figure)
        return None

    output_path = (
        Path(paths.figures_dir)
        / filename
    )

    figure.savefig(
        output_path,
        dpi=rules.figure_dpi,
        bbox_inches='tight',
    )

    plt.close(figure)

    return output_path


# =============================================================================
# GENERIC PROFILING
# =============================================================================


def build_dataset_summary(
    datasets: dict[str, pd.DataFrame],
    layer: str,
) -> pd.DataFrame:
    """Build one compact profile row per dataset."""
    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        date_start, date_end = _dataset_date_range(
            dataframe
        )

        row_level_date = _first_existing(
            dataframe,
            (
                'date',
                'reservation_datetime',
                'reservation_date',
                'datetime',
                'event_date',
            ),
        )

        if row_level_date is not None:
            date_column = row_level_date
        elif (
            'report_start' in dataframe.columns
            or 'report_end' in dataframe.columns
        ):
            date_column = 'report_start..report_end'
        elif (
            'event_start' in dataframe.columns
            and 'event_end' in dataframe.columns
        ):
            date_column = 'event_start..event_end'
        else:
            date_column = _candidate_date_column(
                dataframe
            )

        missing_cells = int(
            dataframe.isna().sum().sum()
        )

        total_cells = (
            int(
                dataframe.shape[0]
                * dataframe.shape[1]
            )
        )

        rows.append(
            {
                'layer': layer,
                'dataset': dataset_name,
                'rows': len(dataframe),
                'columns': len(dataframe.columns),
                'missing_cells': missing_cells,
                'missing_fraction': (
                    missing_cells / total_cells
                    if total_cells
                    else np.nan
                ),
                'exact_duplicates': int(
                    dataframe.duplicated().sum()
                ),
                'date_column': date_column,
                'date_start': date_start,
                'date_end': date_end,
                'memory_mb': float(
                    dataframe.memory_usage(
                        deep=True
                    ).sum()
                    / 1024**2
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _column_role(
    dataframe: pd.DataFrame,
    column: str,
) -> str:
    series = dataframe[column]

    if _is_quality_flag(column):
        return 'quality_flag'

    if _is_identifier_like(column):
        return 'identifier'

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return 'datetime'

    if pd.api.types.is_bool_dtype(
        series
    ):
        return 'boolean'

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return 'numeric'

    return 'categorical_or_text'


def build_column_profile(
    datasets: dict[str, pd.DataFrame],
    layer: str,
) -> pd.DataFrame:
    """Build a transparent column-level EDA catalogue."""
    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        for column in dataframe.columns:
            series = dataframe[column]
            non_null = series.dropna()

            row: dict[str, Any] = {
                'layer': layer,
                'dataset': dataset_name,
                'column': column,
                'role': _column_role(
                    dataframe,
                    column,
                ),
                'dtype': str(
                    series.dtype
                ),
                'rows': len(series),
                'non_null_count': int(
                    series.notna().sum()
                ),
                'missing_count': int(
                    series.isna().sum()
                ),
                'missing_fraction': float(
                    series.isna().mean()
                ) if len(series) else np.nan,
                'unique_non_null': int(
                    non_null.nunique(
                        dropna=True
                    )
                ),
            }

            if (
                pd.api.types.is_numeric_dtype(
                    series
                )
                and not pd.api.types.is_bool_dtype(
                    series
                )
            ):
                values = _to_numeric(
                    series
                ).dropna()

                if not values.empty:
                    row.update(
                        {
                            'min': values.min(),
                            'q25': values.quantile(0.25),
                            'median': values.median(),
                            'mean': values.mean(),
                            'q75': values.quantile(0.75),
                            'max': values.max(),
                            'std': values.std(),
                        }
                    )

            rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_missingness_table(
    column_profile: pd.DataFrame,
) -> pd.DataFrame:
    if column_profile.empty:
        return pd.DataFrame()

    result = column_profile.loc[
        column_profile[
            'missing_count'
        ] > 0,
        [
            'layer',
            'dataset',
            'column',
            'role',
            'missing_count',
            'missing_fraction',
        ],
    ].copy()

    return result.sort_values(
        [
            'missing_fraction',
            'missing_count',
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )


def build_categorical_frequency_tables(
    datasets: dict[str, pd.DataFrame],
    layer: str,
    rules: EDARules,
) -> pd.DataFrame:
    """
    Build frequency rows only for reasonably low-cardinality categorical data.

    Free-text and ID-like columns are intentionally excluded.
    """
    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        for column in dataframe.columns:
            if _is_identifier_like(column):
                continue

            if _is_quality_flag(column):
                continue

            series = dataframe[column]

            if (
                pd.api.types.is_numeric_dtype(
                    series
                )
                and not pd.api.types.is_bool_dtype(
                    series
                )
            ):
                continue

            if pd.api.types.is_datetime64_any_dtype(
                series
            ):
                continue

            non_null = series.dropna()

            if non_null.empty:
                continue

            cardinality = int(
                non_null.nunique(
                    dropna=True
                )
            )

            if (
                cardinality
                > rules.max_categories_for_frequency_table
            ):
                continue

            counts = (
                non_null.astype('string')
                .value_counts(
                    dropna=False
                )
                .head(
                    rules.top_n_categories
                )
            )

            denominator = len(
                non_null
            )

            for value, count in counts.items():
                rows.append(
                    {
                        'layer': layer,
                        'dataset': dataset_name,
                        'column': column,
                        'value': value,
                        'count': int(count),
                        'fraction_non_null': (
                            count / denominator
                            if denominator
                            else np.nan
                        ),
                        'cardinality': cardinality,
                    }
                )

    return pd.DataFrame(
        rows
    )


def _plot_generic_numeric_distributions(
    datasets: dict[str, pd.DataFrame],
    layer: str,
    paths: EDAPaths,
    rules: EDARules,
) -> list[Path]:
    """
    Optional generic histograms.

    Disabled by default because the framework now favours curated EDA. This is
    kept as an opt-in diagnostic for users who explicitly want broad numeric
    exploration.
    """
    if not rules.generic_numeric_distributions:
        return []

    saved: list[Path] = []

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        candidates = [
            column
            for column in dataframe.select_dtypes(
                include='number'
            ).columns
            if not _is_quality_flag(column)
            and not _is_identifier_like(column)
        ]

        candidates = candidates[
            : rules.generic_numeric_max_columns
        ]

        for column in candidates:
            values = _to_numeric(
                dataframe[column]
            ).dropna()

            if values.nunique() < 2:
                continue

            figure, axis = plt.subplots(
                figsize=(9, 5)
            )

            axis.hist(
                values,
                bins=rules.histogram_bins,
            )
            axis.set_title(
                f'{layer} | {dataset_name} | {column} | Distribution'
            )
            axis.set_xlabel(
                column
            )
            axis.set_ylabel(
                'Count'
            )

            filename = (
                f'{_slugify(layer)}__'
                f'{_slugify(dataset_name)}__'
                f'{_slugify(column)}__distribution.png'
            )

            path = _save_figure(
                figure,
                filename,
                paths,
                rules,
            )

            if path is not None:
                saved.append(path)

    return saved


# =============================================================================
# DAILY GOLD EDA
# =============================================================================


def _daily_target_column(
    dataframe: pd.DataFrame,
    rules: EDARules,
) -> str | None:
    return _first_existing(
        dataframe,
        rules.daily_target_candidates,
    )


def _daily_activity_column(
    dataframe: pd.DataFrame,
    rules: EDARules,
) -> str | None:
    return _first_existing(
        dataframe,
        rules.daily_activity_candidates,
    )


def build_daily_gold_summary(
    dataframe: pd.DataFrame,
    rules: EDARules,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            'available': False,
        }

    if 'date' not in dataframe.columns:
        return {
            'available': False,
            'reason': 'date_column_missing',
        }

    dates = _to_datetime(
        dataframe['date']
    )

    valid_dates = dates.dropna()

    target = _daily_target_column(
        dataframe,
        rules,
    )

    activity = _daily_activity_column(
        dataframe,
        rules,
    )

    summary: dict[str, Any] = {
        'available': True,
        'rows': len(dataframe),
        'date_start': (
            valid_dates.min()
            if not valid_dates.empty
            else None
        ),
        'date_end': (
            valid_dates.max()
            if not valid_dates.empty
            else None
        ),
        'unique_dates': int(
            valid_dates.dt.normalize().nunique()
        ) if not valid_dates.empty else 0,
        'target_column': target,
        'activity_column': activity,
    }

    if not valid_dates.empty:
        full_range = pd.date_range(
            valid_dates.min().normalize(),
            valid_dates.max().normalize(),
            freq='D',
        )

        observed_dates = pd.DatetimeIndex(
            valid_dates.dt.normalize().unique()
        )

        missing_dates = full_range.difference(
            observed_dates
        )

        summary[
            'calendar_days_in_range'
        ] = len(full_range)
        summary[
            'missing_calendar_dates'
        ] = len(missing_dates)

    if target is not None:
        values = _to_numeric(
            dataframe[target]
        )
        valid = values.dropna()

        summary[
            'target_non_null'
        ] = int(
            valid.count()
        )
        summary[
            'target_missing'
        ] = int(
            values.isna().sum()
        )

        if not valid.empty:
            summary[
                'target_mean'
            ] = valid.mean()
            summary[
                'target_median'
            ] = valid.median()
            summary[
                'target_std'
            ] = valid.std()
            summary[
                'target_min'
            ] = valid.min()
            summary[
                'target_max'
            ] = valid.max()
            summary[
                'target_zero_days'
            ] = int(
                valid.eq(0).sum()
            )

    if activity is not None:
        values = _to_numeric(
            dataframe[activity]
        )

        summary[
            'days_with_positive_activity'
        ] = int(
            values.fillna(0).gt(0).sum()
        )
        summary[
            'days_without_positive_activity'
        ] = int(
            values.fillna(0).le(0).sum()
        )

    return summary


def build_daily_calendar_profiles(
    dataframe: pd.DataFrame,
    rules: EDARules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return target profiles by weekday and calendar month.

    Besides target statistics, the profiles quantify how often business
    activity is actually observed. This lets a general framework distinguish
    recurring low-activity patterns from arbitrary missing target values
    without hard-coding a specific restaurant's opening days.
    """
    target = _daily_target_column(
        dataframe,
        rules,
    )

    if (
        target is None
        or 'date' not in dataframe.columns
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    activity = _daily_activity_column(
        dataframe,
        rules,
    )

    columns = [
        'date',
        target,
    ]

    if (
        activity is not None
        and activity not in columns
    ):
        columns.append(
            activity
        )

    data = dataframe[
        columns
    ].copy()

    data['date'] = _to_datetime(
        data['date']
    )
    data[target] = _to_numeric(
        data[target]
    )

    if activity is not None:
        data[activity] = _to_numeric(
            data[activity]
        )

    data = data.dropna(
        subset=['date']
    )

    data['day_of_week_num'] = (
        data['date'].dt.dayofweek
    )
    data['day_of_week'] = (
        data['date'].dt.day_name()
    )
    data['year_month'] = (
        data['date'].dt.to_period('M').astype('string')
    )

    def _profile_group(
        group: pd.DataFrame,
    ) -> dict[str, Any]:
        values = group[
            target
        ].dropna()

        row: dict[str, Any] = {
            'count': int(
                values.count()
            ),
            'calendar_days': int(
                len(group)
            ),
            'target_observed_days': int(
                values.count()
            ),
            'target_missing_days': int(
                group[
                    target
                ].isna().sum()
            ),
            'target_observation_fraction': (
                float(
                    values.count()
                    / len(group)
                )
                if len(group)
                else np.nan
            ),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'minimum': values.min(),
            'maximum': values.max(),
        }

        if activity is not None:
            activity_values = (
                group[
                    activity
                ]
                .fillna(
                    0
                )
            )

            positive_days = int(
                activity_values.gt(
                    0
                ).sum()
            )

            row[
                'positive_activity_days'
            ] = positive_days
            row[
                'activity_fraction'
            ] = (
                positive_days
                / len(group)
                if len(group)
                else np.nan
            )

        return row

    weekday_rows: list[dict[str, Any]] = []

    for (
        weekday_num,
        weekday_name,
    ), group in data.groupby(
        [
            'day_of_week_num',
            'day_of_week',
        ],
        sort=True,
    ):
        row = {
            'day_of_week_num': weekday_num,
            'day_of_week': weekday_name,
            **_profile_group(
                group
            ),
        }

        if (
            activity is not None
            and row[
                'calendar_days'
            ]
            >= rules.min_days_for_weekday_diagnostic
        ):
            row[
                'candidate_low_activity_weekday'
            ] = bool(
                row[
                    'activity_fraction'
                ]
                < rules.low_activity_weekday_threshold
            )
        else:
            row[
                'candidate_low_activity_weekday'
            ] = False

        weekday_rows.append(
            row
        )

    weekday = pd.DataFrame(
        weekday_rows
    ).sort_values(
        'day_of_week_num'
    ).reset_index(
        drop=True
    )

    monthly_rows: list[dict[str, Any]] = []

    for year_month, group in data.groupby(
        'year_month',
        sort=True,
    ):
        row = {
            'year_month': year_month,
            **_profile_group(
                group
            ),
        }

        valid_dates = group[
            'date'
        ].dropna()

        if not valid_dates.empty:
            days_in_month = int(
                valid_dates.iloc[
                    0
                ].days_in_month
            )

            row[
                'calendar_days_in_month'
            ] = days_in_month
            row[
                'calendar_coverage_fraction'
            ] = (
                len(group)
                / days_in_month
                if days_in_month
                else np.nan
            )
            row[
                'is_partial_calendar_month'
            ] = bool(
                len(group)
                < days_in_month
            )

        monthly_rows.append(
            row
        )

    monthly = pd.DataFrame(
        monthly_rows
    ).sort_values(
        'year_month'
    ).reset_index(
        drop=True
    )

    weekday.insert(
        0,
        'target',
        target,
    )

    monthly.insert(
        0,
        'target',
        target,
    )

    return (
        weekday,
        monthly,
    )


def build_target_correlations(
    dataframe: pd.DataFrame,
    target: str,
    rules: EDARules,
) -> pd.DataFrame:
    """
    Calculate descriptive Pearson and Spearman associations with one target.

    These correlations are descriptive only. They are not automatic feature
    selection because some columns can be realized outcomes and therefore cause
    target leakage in a future prediction model.
    """
    if target not in dataframe.columns:
        return pd.DataFrame()

    numeric_columns = [
        column
        for column in dataframe.select_dtypes(
            include='number'
        ).columns
        if column != target
        and not _is_quality_flag(column)
        and not _is_identifier_like(column)
    ]

    rows: list[dict[str, Any]] = []

    target_values = _to_numeric(
        dataframe[target]
    )

    for column in numeric_columns:
        feature_values = _to_numeric(
            dataframe[column]
        )

        pair = pd.DataFrame(
            {
                'target': target_values,
                'feature': feature_values,
            }
        ).dropna()

        if (
            len(pair)
            < rules.min_non_null_for_correlation
            or pair['target'].nunique() < 2
            or pair['feature'].nunique() < 2
        ):
            continue

        pearson = pair[
            'target'
        ].corr(
            pair['feature'],
            method='pearson',
        )

        # Compute Spearman correlation without SciPy.
        # Spearman is simply Pearson correlation applied to the ranks.
        # This keeps ap_eda.py compatible with the project's lightweight
        # requirements and avoids an unnecessary scipy dependency.
        target_ranks = pair[
            'target'
        ].rank(
            method='average'
        )
        feature_ranks = pair[
            'feature'
        ].rank(
            method='average'
        )

        spearman = target_ranks.corr(
            feature_ranks,
            method='pearson',
        )

        max_abs_correlation = max(
            abs(pearson),
            abs(spearman),
        )

        rows.append(
            {
                'target': target,
                'feature': column,
                'n_pairs': len(pair),
                'pearson': pearson,
                'spearman': spearman,
                'abs_spearman': abs(spearman),
                'possible_target_leakage': bool(
                    max_abs_correlation
                    >= rules.possible_leakage_correlation_threshold
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    ).sort_values(
        'abs_spearman',
        ascending=False,
    ).head(
        rules.max_target_correlations
    ).reset_index(
        drop=True
    )

    return result


def _plot_daily_target_timeline(
    dataframe: pd.DataFrame,
    target: str,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    data = dataframe[
        [
            'date',
            target,
        ]
    ].copy()

    data['date'] = _to_datetime(
        data['date']
    )
    data[target] = _to_numeric(
        data[target]
    )
    data = data.dropna(
        subset=['date']
    ).sort_values(
        'date'
    )

    if data.empty:
        return None

    rolling = (
        data[target]
        .rolling(
            rules.rolling_window_days,
            min_periods=1,
            center=True,
        )
        .mean()
    )

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        data['date'],
        data[target],
        linewidth=0.9,
        alpha=0.55,
        label=target,
    )
    axis.plot(
        data['date'],
        rolling,
        linewidth=2.0,
        label=(
            f'{rules.rolling_window_days}-day rolling mean'
        ),
    )
    axis.set_title(
        f'Daily Gold | {target} | Time evolution'
    )
    axis.set_xlabel(
        'Date'
    )
    axis.set_ylabel(
        target
    )
    axis.legend()
    axis.grid(
        axis='y',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_daily__target_timeline.png',
        paths,
        rules,
    )


def _plot_profile_bars(
    dataframe: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if dataframe.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        dataframe[category_column].astype('string'),
        dataframe[value_column],
    )
    axis.set_title(
        title
    )
    axis.set_xlabel(
        xlabel
    )
    axis.set_ylabel(
        ylabel
    )
    axis.tick_params(
        axis='x',
        rotation=35,
    )
    axis.grid(
        axis='y',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        filename,
        paths,
        rules,
    )


def _plot_target_correlations(
    correlations: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if correlations.empty:
        return None

    ordered = correlations.sort_values(
        'spearman'
    )

    figure, axis = plt.subplots(
        figsize=(10, max(5, len(ordered) * 0.32))
    )

    axis.barh(
        ordered['feature'],
        ordered['spearman'],
        label='Spearman',
    )
    axis.scatter(
        ordered['pearson'],
        np.arange(len(ordered)),
        label='Pearson',
        zorder=3,
    )
    axis.axvline(
        0,
        linewidth=0.8,
    )
    axis.set_title(
        'Daily Gold | Descriptive correlations with target'
    )
    axis.set_xlabel(
        'Correlation coefficient'
    )
    axis.set_ylabel(
        'Feature'
    )
    axis.legend()
    axis.grid(
        axis='x',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_daily__target_correlations.png',
        paths,
        rules,
    )


def analyze_daily_gold(
    dataframe: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
    verbose: bool = True,
) -> dict[str, Any]:
    _print_subheader(
        'EDA | Daily Gold master',
        verbose=verbose,
    )

    summary = build_daily_gold_summary(
        dataframe,
        rules,
    )

    if not summary.get(
        'available',
        False,
    ):
        _print_message(
            'Daily Gold EDA skipped because a compatible daily table is not available.',
            level='WARNING',
            verbose=verbose,
        )
        return summary

    target = summary.get(
        'target_column'
    )

    weekday, monthly = build_daily_calendar_profiles(
        dataframe,
        rules,
    )

    if (
        not weekday.empty
        and 'candidate_low_activity_weekday' in weekday.columns
    ):
        summary[
            'candidate_low_activity_weekdays'
        ] = (
            weekday.loc[
                weekday[
                    'candidate_low_activity_weekday'
                ].fillna(
                    False
                ),
                'day_of_week',
            ]
            .astype(
                str
            )
            .tolist()
        )

    if (
        not monthly.empty
        and 'is_partial_calendar_month' in monthly.columns
    ):
        summary[
            'partial_calendar_months'
        ] = (
            monthly.loc[
                monthly[
                    'is_partial_calendar_month'
                ].fillna(
                    False
                ),
                'year_month',
            ]
            .astype(
                str
            )
            .tolist()
        )

    saved_tables: dict[str, str] = {}
    saved_figures: dict[str, str] = {}

    weekday_path = _save_table(
        weekday,
        'gold_daily__weekday_profile.csv',
        paths,
        rules,
    )

    monthly_path = _save_table(
        monthly,
        'gold_daily__monthly_profile.csv',
        paths,
        rules,
    )

    if weekday_path:
        saved_tables['weekday_profile'] = str(
            weekday_path
        )

    if monthly_path:
        saved_tables['monthly_profile'] = str(
            monthly_path
        )

    correlations = pd.DataFrame()

    if target is not None:
        correlations = build_target_correlations(
            dataframe,
            target,
            rules,
        )

        correlations_path = _save_table(
            correlations,
            'gold_daily__target_correlations.csv',
            paths,
            rules,
        )

        if correlations_path:
            saved_tables[
                'target_correlations'
            ] = str(
                correlations_path
            )

        timeline_path = _plot_daily_target_timeline(
            dataframe,
            target,
            paths,
            rules,
        )

        if timeline_path:
            saved_figures[
                'target_timeline'
            ] = str(
                timeline_path
            )

        correlation_figure = _plot_target_correlations(
            correlations,
            paths,
            rules,
        )

        if correlation_figure:
            saved_figures[
                'target_correlations'
            ] = str(
                correlation_figure
            )

    if not weekday.empty:
        weekday_figure = _plot_profile_bars(
            weekday,
            category_column='day_of_week',
            value_column='mean',
            title=(
                f'Daily Gold | Mean {target} by weekday'
            ),
            xlabel='Weekday',
            ylabel=f'Mean {target}',
            filename='gold_daily__weekday_pattern.png',
            paths=paths,
            rules=rules,
        )

        if weekday_figure:
            saved_figures[
                'weekday_pattern'
            ] = str(
                weekday_figure
            )

        if 'activity_fraction' in weekday.columns:
            activity_figure = _plot_profile_bars(
                weekday,
                category_column='day_of_week',
                value_column='activity_fraction',
                title='Daily Gold | Positive activity share by weekday',
                xlabel='Weekday',
                ylabel='Fraction of calendar days with positive activity',
                filename='gold_daily__weekday_activity_coverage.png',
                paths=paths,
                rules=rules,
            )

            if activity_figure:
                saved_figures[
                    'weekday_activity_coverage'
                ] = str(
                    activity_figure
                )

    if not monthly.empty:
        monthly_plot = monthly.copy()
        monthly_plot[
            'year_month_display'
        ] = monthly_plot[
            'year_month'
        ].astype(
            'string'
        )

        monthly_title = (
            f'Daily Gold | Mean {target} by month'
        )

        if 'is_partial_calendar_month' in monthly_plot.columns:
            partial = (
                monthly_plot[
                    'is_partial_calendar_month'
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
            )

            monthly_plot.loc[
                partial,
                'year_month_display',
            ] = (
                monthly_plot.loc[
                    partial,
                    'year_month_display',
                ]
                + '*'
            )

            if partial.any():
                monthly_title += ' | * partial calendar coverage'

        monthly_figure = _plot_profile_bars(
            monthly_plot,
            category_column='year_month_display',
            value_column='mean',
            title=monthly_title,
            xlabel='Month',
            ylabel=f'Mean {target}',
            filename='gold_daily__monthly_pattern.png',
            paths=paths,
            rules=rules,
        )

        if monthly_figure:
            saved_figures[
                'monthly_pattern'
            ] = str(
                monthly_figure
            )

    summary['saved_tables'] = saved_tables
    summary['saved_figures'] = saved_figures

    _print_message(
        f'Daily Gold EDA completed: {len(dataframe):,} rows. '
        f'Target={target!r}.',
        verbose=verbose,
    )

    return summary


# =============================================================================
# ARTICLE-PERIOD GOLD EDA
# =============================================================================


def build_article_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate period-level demand into one descriptive row per stable article.

    `article_code` is the entity key. Descriptive metadata such as department
    or name may legitimately change over time and therefore must not split one
    article into several EDA profiles.
    """
    if (
        dataframe.empty
        or 'article_code' not in dataframe.columns
    ):
        return pd.DataFrame()

    data = dataframe.copy()

    period_columns = [
        column
        for column in [
            'report_start',
            'report_end',
        ]
        if column in data.columns
    ]

    if len(period_columns) == 2:
        total_periods = int(
            data[
                period_columns
            ]
            .drop_duplicates()
            .shape[0]
        )
    elif 'period_id' in data.columns:
        total_periods = int(
            data['period_id'].nunique(
                dropna=True
            )
        )
    else:
        total_periods = np.nan

    aggregations: dict[str, tuple[str, Any]] = {}

    if 'units' in data.columns:
        aggregations.update(
            {
                'units_total': (
                    'units',
                    'sum',
                ),
                'units_mean_observed_period': (
                    'units',
                    'mean',
                ),
                'units_median_observed_period': (
                    'units',
                    'median',
                ),
                'units_std_observed_period': (
                    'units',
                    'std',
                ),
            }
        )

    if 'amount' in data.columns:
        aggregations[
            'amount_total'
        ] = (
            'amount',
            'sum',
        )

    if not aggregations:
        return pd.DataFrame()

    profile = (
        data.groupby(
            'article_code',
            dropna=False,
        )
        .agg(
            **aggregations
        )
        .reset_index()
    )

    if len(period_columns) == 2:
        observed = (
            data[
                [
                    'article_code',
                    *period_columns,
                ]
            ]
            .drop_duplicates()
            .groupby(
                'article_code',
                dropna=False,
            )
            .size()
            .rename(
                'periods_observed'
            )
            .reset_index()
        )
    elif 'period_id' in data.columns:
        observed = (
            data.groupby(
                'article_code',
                dropna=False,
            )[
                'period_id'
            ]
            .nunique(
                dropna=True
            )
            .rename(
                'periods_observed'
            )
            .reset_index()
        )
    else:
        observed = (
            data.groupby(
                'article_code',
                dropna=False,
            )
            .size()
            .rename(
                'periods_observed'
            )
            .reset_index()
        )

    profile = profile.merge(
        observed,
        on='article_code',
        how='left',
        validate='one_to_one',
    )

    # Descriptive metadata is summarized without becoming part of the entity
    # key. Latest and modal values are both retained where useful.
    sort_columns = [
        column
        for column in [
            'report_end',
            'report_start',
        ]
        if column in data.columns
    ]

    metadata_rows: list[dict[str, Any]] = []

    for article_code, group in data.groupby(
        'article_code',
        dropna=False,
        sort=False,
    ):
        ordered = (
            group.sort_values(
                sort_columns
            )
            if sort_columns
            else group
        )

        row: dict[str, Any] = {
            'article_code': article_code,
        }

        for column in [
            'article_name',
            'department_code',
            'department_name',
        ]:
            if column not in group.columns:
                continue

            values = (
                group[
                    column
                ]
                .dropna()
            )

            unique_count = int(
                values.nunique(
                    dropna=True
                )
            )

            latest_values = (
                ordered[
                    column
                ]
                .dropna()
            )

            mode_values = (
                values.mode(
                    dropna=True
                )
            )

            latest = (
                latest_values.iloc[
                    -1
                ]
                if not latest_values.empty
                else pd.NA
            )

            mode = (
                mode_values.iloc[
                    0
                ]
                if not mode_values.empty
                else pd.NA
            )

            if column == 'article_name':
                row[
                    'article_name'
                ] = latest
                row[
                    'article_name_mode'
                ] = mode
                row[
                    'article_name_values'
                ] = unique_count
                row[
                    'article_name_changed'
                ] = bool(
                    unique_count > 1
                )
            elif column == 'department_code':
                row[
                    'department_code'
                ] = latest
                row[
                    'department_code_mode'
                ] = mode
                row[
                    'departments_observed'
                ] = unique_count
                row[
                    'department_changed'
                ] = bool(
                    unique_count > 1
                )
            elif column == 'department_name':
                row[
                    'department_name'
                ] = latest
                row[
                    'department_name_mode'
                ] = mode

        metadata_rows.append(
            row
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    if not metadata.empty:
        profile = profile.merge(
            metadata,
            on='article_code',
            how='left',
            validate='one_to_one',
        )

    if (
        isinstance(
            total_periods,
            int,
        )
        and total_periods > 0
    ):
        profile[
            'period_coverage_fraction'
        ] = (
            profile[
                'periods_observed'
            ]
            / total_periods
        )
    else:
        profile[
            'period_coverage_fraction'
        ] = np.nan

    if 'units_total' in profile.columns:
        profile = profile.sort_values(
            'units_total',
            ascending=False,
        )

        total_units = profile[
            'units_total'
        ].sum(
            min_count=1
        )

        if pd.notna(total_units) and total_units != 0:
            profile[
                'units_share'
            ] = (
                profile[
                    'units_total'
                ]
                / total_units
            )
            profile[
                'units_cumulative_share'
            ] = profile[
                'units_share'
            ].cumsum()

    return profile.reset_index(
        drop=True
    )


def build_article_period_summary(
    dataframe: pd.DataFrame,
    article_profile: pd.DataFrame,
    period_totals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            'available': False,
        }

    summary: dict[str, Any] = {
        'available': True,
        'rows': len(dataframe),
        'articles': int(
            dataframe[
                'article_code'
            ].nunique(
                dropna=True
            )
        ) if 'article_code' in dataframe.columns else None,
        'article_profile_rows': int(
            len(
                article_profile
            )
        ),
    }

    if {
        'report_start',
        'report_end',
    }.issubset(
        dataframe.columns
    ):
        periods = dataframe[
            [
                'report_start',
                'report_end',
            ]
        ].drop_duplicates()

        summary['periods'] = len(
            periods
        )
        summary['date_start'] = _to_datetime(
            periods['report_start']
        ).min()
        summary['date_end'] = _to_datetime(
            periods['report_end']
        ).max()

    elif 'period_id' in dataframe.columns:
        summary['periods'] = int(
            dataframe['period_id'].nunique(
                dropna=True
            )
        )

    if period_totals is None:
        period_totals = build_article_period_totals(
            dataframe
        )

    if (
        period_totals is not None
        and not period_totals.empty
        and 'period_days' in period_totals.columns
    ):
        period_days = _to_numeric(
            period_totals[
                'period_days'
            ]
        ).dropna()

        if not period_days.empty:
            expected_period_days = _infer_expected_period_days_eda(
                period_days
            )

            summary[
                'expected_period_days'
            ] = expected_period_days
            summary[
                'period_days_min'
            ] = period_days.min()
            summary[
                'period_days_max'
            ] = period_days.max()
            summary[
                'period_days_median'
            ] = period_days.median()

    if (
        period_totals is not None
        and not period_totals.empty
        and 'period_is_complete' in period_totals.columns
    ):
        values = (
            period_totals[
                'period_is_complete'
            ]
            .astype(
                'boolean'
            )
        )

        summary[
            'complete_period_fraction'
        ] = values.mean()
        summary[
            'complete_periods'
        ] = int(
            values.fillna(
                False
            ).sum()
        )
        summary[
            'incomplete_periods'
        ] = int(
            values.fillna(
                False
            ).eq(
                False
            ).sum()
        )

    if (
        not article_profile.empty
        and 'units_share' in article_profile.columns
    ):
        summary[
            'top_10_articles_units_share'
        ] = article_profile[
            'units_share'
        ].head(10).sum()
        summary[
            'top_20_articles_units_share'
        ] = article_profile[
            'units_share'
        ].head(20).sum()

    if (
        not article_profile.empty
        and 'period_coverage_fraction' in article_profile.columns
    ):
        coverage = article_profile[
            'period_coverage_fraction'
        ]

        summary[
            'articles_observed_in_lt_25pct_periods'
        ] = int(
            coverage.lt(0.25).sum()
        )
        summary[
            'articles_observed_in_lt_50pct_periods'
        ] = int(
            coverage.lt(0.50).sum()
        )
        summary[
            'median_article_period_coverage'
        ] = coverage.median()

    if (
        not article_profile.empty
        and 'department_changed' in article_profile.columns
    ):
        summary[
            'articles_with_department_changes'
        ] = int(
            article_profile[
                'department_changed'
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
            .sum()
        )

    return summary


def _article_label_series(
    dataframe: pd.DataFrame,
) -> pd.Series:
    if 'article_name' in dataframe.columns:
        name = dataframe[
            'article_name'
        ].astype(
            'string'
        )

        if 'article_code' in dataframe.columns:
            code = dataframe[
                'article_code'
            ].astype(
                'string'
            )

            return name.fillna(
                code
            )

        return name

    if 'article_code' in dataframe.columns:
        return dataframe[
            'article_code'
        ].astype(
            'string'
        )

    return pd.Series(
        range(len(dataframe)),
        index=dataframe.index,
        dtype='string',
    )


def _plot_article_top_units(
    article_profile: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if (
        article_profile.empty
        or 'units_total' not in article_profile.columns
    ):
        return None

    top = article_profile.head(
        rules.article_top_n
    ).copy()

    top['article_label'] = _article_label_series(
        top
    )

    top = top.sort_values(
        'units_total'
    )

    figure, axis = plt.subplots(
        figsize=(10, max(5, len(top) * 0.34))
    )

    axis.barh(
        top['article_label'],
        top['units_total'],
    )
    axis.set_title(
        f'Article-period Gold | Top {len(top)} articles by units'
    )
    axis.set_xlabel(
        'Total units'
    )
    axis.set_ylabel(
        'Article'
    )
    axis.grid(
        axis='x',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_articles__top_units.png',
        paths,
        rules,
    )


def _plot_article_pareto(
    article_profile: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if (
        article_profile.empty
        or 'units_cumulative_share' not in article_profile.columns
    ):
        return None

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(
        1,
        len(article_profile) + 1,
    )

    axis.plot(
        x,
        article_profile[
            'units_cumulative_share'
        ],
    )
    axis.axhline(
        0.8,
        linewidth=1.0,
        linestyle='--',
    )
    axis.set_title(
        'Article-period Gold | Demand concentration (Pareto)'
    )
    axis.set_xlabel(
        'Articles ranked by total units'
    )
    axis.set_ylabel(
        'Cumulative share of units'
    )
    axis.set_ylim(
        0,
        1.02,
    )
    axis.grid(
        axis='y',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_articles__pareto_units.png',
        paths,
        rules,
    )


def _plot_article_period_coverage(
    article_profile: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if (
        article_profile.empty
        or 'period_coverage_fraction' not in article_profile.columns
    ):
        return None

    values = article_profile[
        'period_coverage_fraction'
    ].dropna()

    if values.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        values,
        bins=np.linspace(
            0,
            1,
            11,
        ),
    )
    axis.set_title(
        'Article-period Gold | Article temporal coverage'
    )
    axis.set_xlabel(
        'Fraction of report periods in which article is observed'
    )
    axis.set_ylabel(
        'Articles'
    )
    axis.set_xlim(
        0,
        1,
    )

    return _save_figure(
        figure,
        'gold_articles__period_coverage.png',
        paths,
        rules,
    )


def build_article_period_totals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate article rows to one row per reporting period.

    Raw totals are preserved, while per-day rates and a data-driven period
    completeness flag are added when duration information is available.
    """
    if (
        dataframe.empty
        or 'units' not in dataframe.columns
    ):
        return pd.DataFrame()

    data = dataframe.copy()

    period_columns = [
        column
        for column in [
            'report_start',
            'report_end',
        ]
        if column in data.columns
    ]

    if not period_columns:
        if 'period_id' not in data.columns:
            return pd.DataFrame()
        period_columns = [
            'period_id'
        ]

    if (
        'period_days' not in data.columns
        and {
            'report_start',
            'report_end',
        }.issubset(
            data.columns
        )
    ):
        data[
            'period_days'
        ] = (
            _to_datetime(
                data[
                    'report_end'
                ]
            )
            - _to_datetime(
                data[
                    'report_start'
                ]
            )
        ).dt.days + 1

    aggregations: dict[str, tuple[str, Any]] = {
        'units_total': (
            'units',
            'sum',
        ),
        'articles_observed': (
            'article_code',
            'nunique',
        ) if 'article_code' in data.columns else (
            'units',
            'count',
        ),
    }

    if 'amount' in data.columns:
        aggregations[
            'amount_total'
        ] = (
            'amount',
            'sum',
        )

    if 'period_days' in data.columns:
        aggregations[
            'period_days'
        ] = (
            'period_days',
            'first',
        )

    completeness_column = _first_existing(
        data,
        (
            'period_is_complete',
            'period_is_complete_week',
        ),
    )

    if completeness_column is not None:
        aggregations[
            'period_is_complete'
        ] = (
            completeness_column,
            'first',
        )

    result = (
        data.groupby(
            period_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            **aggregations
        )
    )

    if 'period_days' in result.columns:
        period_days = _to_numeric(
            result[
                'period_days'
            ]
        )

        result[
            'units_per_day'
        ] = (
            _to_numeric(
                result[
                    'units_total'
                ]
            )
            / period_days.replace(
                0,
                np.nan,
            )
        )

        if 'amount_total' in result.columns:
            result[
                'amount_per_day'
            ] = (
                _to_numeric(
                    result[
                        'amount_total'
                    ]
                )
                / period_days.replace(
                    0,
                    np.nan,
                )
            )

        if 'period_is_complete' not in result.columns:
            expected_days = _infer_expected_period_days_eda(
                period_days
            )

            if expected_days is not None:
                result[
                    'period_is_complete'
                ] = period_days.eq(
                    expected_days
                )

    if 'report_start' in result.columns:
        result = result.sort_values(
            'report_start'
        )

    return result.reset_index(
        drop=True
    )


def _plot_article_period_totals(
    period_totals: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if (
        period_totals.empty
        or 'units_total' not in period_totals.columns
    ):
        return None

    x_column = _first_existing(
        period_totals,
        (
            'report_start',
            'period_id',
        ),
    )

    if x_column is None:
        return None

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        period_totals[x_column],
        period_totals['units_total'],
        marker='o',
        linewidth=1.2,
    )

    incomplete_count = 0

    if 'period_is_complete' in period_totals.columns:
        complete = (
            period_totals[
                'period_is_complete'
            ]
            .astype(
                'boolean'
            )
            .fillna(
                False
            )
        )

        incomplete = ~complete
        incomplete_count = int(
            incomplete.sum()
        )

        if incomplete_count:
            axis.scatter(
                period_totals.loc[
                    incomplete,
                    x_column,
                ],
                period_totals.loc[
                    incomplete,
                    'units_total',
                ],
                marker='x',
                s=70,
                label='Incomplete / non-standard period',
                zorder=3,
            )
            axis.legend()

    title = (
        'Article-period Gold | Total units by report period'
    )

    if incomplete_count:
        title += (
            f' | {incomplete_count} incomplete/non-standard period(s) marked'
        )

    axis.set_title(
        title
    )
    axis.set_xlabel(
        'Report period'
    )
    axis.set_ylabel(
        'Total units'
    )
    axis.tick_params(
        axis='x',
        rotation=35,
    )
    axis.grid(
        axis='y',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_articles__period_units.png',
        paths,
        rules,
    )


def _plot_article_period_rate(
    period_totals: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    """Plot units/day so periods of different lengths remain comparable."""
    if (
        period_totals.empty
        or 'units_per_day' not in period_totals.columns
    ):
        return None

    x_column = _first_existing(
        period_totals,
        (
            'report_start',
            'period_id',
        ),
    )

    if x_column is None:
        return None

    values = _to_numeric(
        period_totals[
            'units_per_day'
        ]
    )

    if values.dropna().empty:
        return None

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        period_totals[
            x_column
        ],
        values,
        marker='o',
        linewidth=1.2,
    )
    axis.set_title(
        'Article-period Gold | Units per day by report period'
    )
    axis.set_xlabel(
        'Report period'
    )
    axis.set_ylabel(
        'Units per day'
    )
    axis.tick_params(
        axis='x',
        rotation=35,
    )
    axis.grid(
        axis='y',
        alpha=0.25,
    )

    return _save_figure(
        figure,
        'gold_articles__period_units_per_day.png',
        paths,
        rules,
    )


def analyze_article_period_gold(
    dataframe: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
    verbose: bool = True,
) -> dict[str, Any]:
    _print_subheader(
        'EDA | Article-period Gold master',
        verbose=verbose,
    )

    if dataframe.empty:
        return {
            'available': False,
        }

    profile = build_article_profile(
        dataframe
    )

    period_totals = build_article_period_totals(
        dataframe
    )

    summary = build_article_period_summary(
        dataframe,
        profile,
        period_totals,
    )

    saved_tables: dict[str, str] = {}
    saved_figures: dict[str, str] = {}

    profile_path = _save_table(
        profile,
        'gold_articles__article_profile.csv',
        paths,
        rules,
    )

    period_path = _save_table(
        period_totals,
        'gold_articles__period_totals.csv',
        paths,
        rules,
    )

    if profile_path:
        saved_tables[
            'article_profile'
        ] = str(
            profile_path
        )

    if period_path:
        saved_tables[
            'period_totals'
        ] = str(
            period_path
        )

    for key, plot_path in {
        'top_units': _plot_article_top_units(
            profile,
            paths,
            rules,
        ),
        'pareto_units': _plot_article_pareto(
            profile,
            paths,
            rules,
        ),
        'period_coverage': _plot_article_period_coverage(
            profile,
            paths,
            rules,
        ),
        'period_units': _plot_article_period_totals(
            period_totals,
            paths,
            rules,
        ),
        'period_units_per_day': _plot_article_period_rate(
            period_totals,
            paths,
            rules,
        ),
    }.items():
        if plot_path:
            saved_figures[
                key
            ] = str(
                plot_path
            )

    summary[
        'saved_tables'
    ] = saved_tables
    summary[
        'saved_figures'
    ] = saved_figures

    _print_message(
        'Article-period Gold EDA completed: '
        f'{summary.get("periods", "?")} periods | '
        f'{summary.get("articles", "?")} articles.',
        verbose=verbose,
    )

    return summary


# =============================================================================
# RESERVATION SILVER EDA
# =============================================================================


def build_reservation_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            'available': False,
        }

    summary: dict[str, Any] = {
        'available': True,
        'rows': len(dataframe),
    }

    if 'status_grouped' in dataframe.columns:
        statuses = (
            dataframe[
                'status_grouped'
            ]
            .astype('string')
            .value_counts(
                dropna=False
            )
        )

        summary[
            'status_counts'
        ] = statuses.to_dict()

        if 'no_show' in statuses.index:
            summary[
                'no_show_fraction'
            ] = (
                statuses.get(
                    'no_show',
                    0,
                )
                / len(dataframe)
            )

    if 'is_walk_in' in dataframe.columns:
        walk_in = (
            dataframe[
                'is_walk_in'
            ]
            .astype(
                'boolean'
            )
            .fillna(
                False
            )
        )

        summary[
            'walk_in_count'
        ] = int(
            walk_in.sum()
        )
        summary[
            'walk_in_fraction'
        ] = walk_in.mean()

    if 'people' in dataframe.columns:
        people = _to_numeric(
            dataframe['people']
        ).dropna()

        if not people.empty:
            summary[
                'people_median'
            ] = people.median()
            summary[
                'people_mean'
            ] = people.mean()
            summary[
                'people_max'
            ] = people.max()

    if 'lead_time_hours' in dataframe.columns:
        lead = _to_numeric(
            dataframe['lead_time_hours']
        ).dropna()

        if not lead.empty:
            summary[
                'lead_time_hours_median'
            ] = lead.median()
            summary[
                'lead_time_hours_mean'
            ] = lead.mean()
            summary[
                'lead_time_hours_p90'
            ] = lead.quantile(0.90)
            summary[
                'lead_time_hours_p99'
            ] = lead.quantile(0.99)
            summary[
                'lead_time_hours_max'
            ] = lead.max()

    if 'lead_time__invalid' in dataframe.columns:
        invalid = (
            dataframe[
                'lead_time__invalid'
            ]
            .astype(
                'boolean'
            )
            .fillna(
                False
            )
        )

        summary[
            'lead_time_invalid_count'
        ] = int(
            invalid.sum()
        )
        summary[
            'lead_time_invalid_fraction'
        ] = invalid.mean()

    return summary


def build_reservation_status_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if 'status_grouped' not in dataframe.columns:
        return pd.DataFrame()

    counts = (
        dataframe[
            'status_grouped'
        ]
        .astype('string')
        .value_counts(
            dropna=False
        )
        .rename_axis(
            'status_grouped'
        )
        .reset_index(
            name='count'
        )
    )

    counts[
        'fraction'
    ] = counts[
        'count'
    ] / len(dataframe)

    return counts


def _plot_reservation_status(
    status_table: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if status_table.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        status_table[
            'status_grouped'
        ].astype('string'),
        status_table[
            'count'
        ],
    )
    axis.set_title(
        'Reservations Silver | Status distribution'
    )
    axis.set_xlabel(
        'Reservation status'
    )
    axis.set_ylabel(
        'Reservations'
    )
    axis.tick_params(
        axis='x',
        rotation=30,
    )

    return _save_figure(
        figure,
        'silver_reservations__status_distribution.png',
        paths,
        rules,
    )


def _plot_reservation_lead_time(
    dataframe: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if 'lead_time_hours' not in dataframe.columns:
        return None

    values = _to_numeric(
        dataframe['lead_time_hours']
    ).dropna()

    if values.empty:
        return None

    # A few legitimate bookings can be made months in advance. The plot is
    # clipped only for visual readability; no data are modified and full-range
    # statistics remain in the CSV/JSON summaries.
    upper = values.quantile(
        0.99
    )

    visible = values.loc[
        values <= upper
    ]

    if visible.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        visible,
        bins=rules.histogram_bins,
    )
    axis.set_title(
        'Reservations Silver | Lead time distribution (<= p99 for display)'
    )
    axis.set_xlabel(
        'Lead time (hours)'
    )
    axis.set_ylabel(
        'Reservations'
    )

    return _save_figure(
        figure,
        'silver_reservations__lead_time.png',
        paths,
        rules,
    )


def analyze_reservations(
    dataframe: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
    verbose: bool = True,
) -> dict[str, Any]:
    _print_subheader(
        'EDA | Reservations Silver',
        verbose=verbose,
    )

    summary = build_reservation_summary(
        dataframe
    )

    if not summary.get(
        'available',
        False,
    ):
        return summary

    status_table = build_reservation_status_table(
        dataframe
    )

    saved_tables: dict[str, str] = {}
    saved_figures: dict[str, str] = {}

    status_path = _save_table(
        status_table,
        'silver_reservations__status_profile.csv',
        paths,
        rules,
    )

    if status_path:
        saved_tables[
            'status_profile'
        ] = str(
            status_path
        )

    for key, plot_path in {
        'status_distribution': _plot_reservation_status(
            status_table,
            paths,
            rules,
        ),
        'lead_time': _plot_reservation_lead_time(
            dataframe,
            paths,
            rules,
        ),
    }.items():
        if plot_path:
            saved_figures[
                key
            ] = str(
                plot_path
            )

    summary[
        'saved_tables'
    ] = saved_tables
    summary[
        'saved_figures'
    ] = saved_figures

    return summary


# =============================================================================
# EVENT SILVER EDA
# =============================================================================


def build_unique_event_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse event-day Silver rows back to one row per event for event-level EDA.

    This is deliberate: histograms of `event_duration_days` or
    `event_day_number` on the expanded event-day table over-weight long events.
    The EDA module therefore analyses event duration/intensity at unique-event
    granularity whenever `event_id` is available.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if 'event_id' not in dataframe.columns:
        return dataframe.copy()

    useful_columns = [
        column
        for column in [
            'event_id',
            'event_name',
            'event_category',
            'event_subcategory',
            'event_scope',
            'event_location',
            'expected_impact',
            'event_intensity',
            'demand_direction',
            'date_confidence',
            'event_start',
            'event_end',
            'event_duration_days',
        ]
        if column in dataframe.columns
    ]

    data = dataframe[
        useful_columns
    ].copy()

    if 'event_start' not in data.columns:
        date_column = _first_existing(
            dataframe,
            (
                'date',
                'event_date',
            ),
        )

        if date_column is not None:
            starts = (
                dataframe.groupby(
                    'event_id',
                    dropna=False,
                )[date_column]
                .min()
                .rename(
                    'event_start'
                )
            )
            ends = (
                dataframe.groupby(
                    'event_id',
                    dropna=False,
                )[date_column]
                .max()
                .rename(
                    'event_end'
                )
            )

            data = data.merge(
                starts,
                on='event_id',
                how='left',
            )
            data = data.merge(
                ends,
                on='event_id',
                how='left',
            )

    # For metadata columns, the event-day expansion should repeat the same
    # information. Keeping the first row is therefore appropriate for EDA; the
    # quality layer has already checked semantic consistency.
    unique = (
        data.sort_values(
            'event_id'
        )
        .drop_duplicates(
            subset=['event_id'],
            keep='first',
        )
        .reset_index(
            drop=True
        )
    )

    if (
        'event_duration_days' not in unique.columns
        and {
            'event_start',
            'event_end',
        }.issubset(
            unique.columns
        )
    ):
        start = _to_datetime(
            unique['event_start']
        )
        end = _to_datetime(
            unique['event_end']
        )

        unique[
            'event_duration_days'
        ] = (
            end
            - start
        ).dt.days + 1

    return unique


def build_event_summary(
    unique_events: pd.DataFrame,
) -> dict[str, Any]:
    if unique_events.empty:
        return {
            'available': False,
        }

    summary: dict[str, Any] = {
        'available': True,
        'unique_events': len(unique_events),
    }

    if 'event_duration_days' in unique_events.columns:
        duration = _to_numeric(
            unique_events[
                'event_duration_days'
            ]
        ).dropna()

        if not duration.empty:
            summary[
                'duration_days_median'
            ] = duration.median()
            summary[
                'duration_days_mean'
            ] = duration.mean()
            summary[
                'duration_days_max'
            ] = duration.max()
            summary[
                'events_over_7_days'
            ] = int(
                duration.gt(7).sum()
            )
            summary[
                'events_over_30_days'
            ] = int(
                duration.gt(30).sum()
            )

    if 'event_intensity' in unique_events.columns:
        intensity = _to_numeric(
            unique_events[
                'event_intensity'
            ]
        ).dropna()

        summary[
            'intensity_counts'
        ] = intensity.value_counts().sort_index().to_dict()

    return summary


def _plot_unique_event_duration(
    unique_events: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if 'event_duration_days' not in unique_events.columns:
        return None

    duration = _to_numeric(
        unique_events[
            'event_duration_days'
        ]
    ).dropna()

    if duration.empty:
        return None

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        duration,
        bins=min(
            rules.histogram_bins,
            max(5, int(duration.nunique())),
        ),
    )
    axis.set_title(
        'Unique events | Event duration distribution'
    )
    axis.set_xlabel(
        'Event duration (days)'
    )
    axis.set_ylabel(
        'Unique events'
    )

    return _save_figure(
        figure,
        'silver_events__unique_event_duration.png',
        paths,
        rules,
    )


def _plot_unique_event_intensity(
    unique_events: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if 'event_intensity' not in unique_events.columns:
        return None

    intensity = _to_numeric(
        unique_events[
            'event_intensity'
        ]
    ).dropna()

    if intensity.empty:
        return None

    counts = intensity.value_counts().sort_index()

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        counts.index.astype('string'),
        counts.values,
    )
    axis.set_title(
        'Unique events | Suggested event intensity'
    )
    axis.set_xlabel(
        'Event intensity'
    )
    axis.set_ylabel(
        'Unique events'
    )

    return _save_figure(
        figure,
        'silver_events__unique_event_intensity.png',
        paths,
        rules,
    )


def analyze_events(
    dataframe: pd.DataFrame,
    paths: EDAPaths,
    rules: EDARules,
    verbose: bool = True,
) -> dict[str, Any]:
    _print_subheader(
        'EDA | Events Silver',
        verbose=verbose,
    )

    unique_events = build_unique_event_table(
        dataframe
    )

    summary = build_event_summary(
        unique_events
    )

    if not summary.get(
        'available',
        False,
    ):
        return summary

    saved_tables: dict[str, str] = {}
    saved_figures: dict[str, str] = {}

    table_path = _save_table(
        unique_events,
        'silver_events__unique_events.csv',
        paths,
        rules,
    )

    if table_path:
        saved_tables[
            'unique_events'
        ] = str(
            table_path
        )

    for key, plot_path in {
        'duration': _plot_unique_event_duration(
            unique_events,
            paths,
            rules,
        ),
        'intensity': _plot_unique_event_intensity(
            unique_events,
            paths,
            rules,
        ),
    }.items():
        if plot_path:
            saved_figures[
                key
            ] = str(
                plot_path
            )

    summary[
        'saved_tables'
    ] = saved_tables
    summary[
        'saved_figures'
    ] = saved_figures

    return summary


# =============================================================================
# EDA REPORT
# =============================================================================


def _build_framework_diagnostics(
    daily_summary: dict[str, Any] | None,
    article_summary: dict[str, Any] | None,
    reservation_summary: dict[str, Any] | None,
    event_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Create factual EDA diagnostics, not modelling prescriptions.

    These messages deliberately avoid choosing algorithms. Model selection is a
    later responsibility of the modelling module.
    """
    diagnostics: list[dict[str, Any]] = []

    if daily_summary and daily_summary.get('available'):
        missing_dates = daily_summary.get(
            'missing_calendar_dates'
        )

        diagnostics.append(
            {
                'area': 'daily_gold',
                'diagnostic': 'calendar_continuity',
                'value': missing_dates,
                'message': (
                    'Daily Gold calendar is continuous.'
                    if missing_dates == 0
                    else (
                        f'Daily Gold contains {missing_dates} missing calendar dates.'
                    )
                ),
            }
        )

        low_activity_weekdays = daily_summary.get(
            'candidate_low_activity_weekdays',
            [],
        )

        if low_activity_weekdays:
            diagnostics.append(
                {
                    'area': 'daily_gold',
                    'diagnostic': 'candidate_low_activity_weekdays',
                    'value': low_activity_weekdays,
                    'message': (
                        'Recurring low-activity weekdays were detected from '
                        'observed business activity. They are candidates for '
                        'closure/operating-schedule features, not automatically '
                        'assumed closures.'
                    ),
                }
            )

        partial_months = daily_summary.get(
            'partial_calendar_months',
            [],
        )

        if partial_months:
            diagnostics.append(
                {
                    'area': 'daily_gold',
                    'diagnostic': 'partial_calendar_months',
                    'value': partial_months,
                    'message': (
                        'At least one calendar month is only partially covered '
                        'by the available date range; monthly means should be '
                        'interpreted with their coverage statistics.'
                    ),
                }
            )

    if article_summary and article_summary.get('available'):
        coverage = article_summary.get(
            'median_article_period_coverage'
        )

        if coverage is not None and pd.notna(coverage):
            diagnostics.append(
                {
                    'area': 'article_period_gold',
                    'diagnostic': 'median_article_period_coverage',
                    'value': coverage,
                    'message': (
                        'Article temporal coverage has been quantified. '
                        'Low-coverage articles should be treated separately '
                        'during later demand-modelling design.'
                    ),
                }
            )

        profile_rows = article_summary.get(
            'article_profile_rows'
        )
        articles = article_summary.get(
            'articles'
        )

        if (
            profile_rows is not None
            and articles is not None
        ):
            diagnostics.append(
                {
                    'area': 'article_period_gold',
                    'diagnostic': 'stable_article_profile_key',
                    'value': {
                        'article_codes': articles,
                        'profile_rows': profile_rows,
                    },
                    'message': (
                        'Article profiling uses the stable article code as its '
                        'entity key; mutable department/name metadata does not '
                        'split one article into several profiles.'
                    ),
                }
            )

        share = article_summary.get(
            'top_20_articles_units_share'
        )

        if share is not None and pd.notna(share):
            diagnostics.append(
                {
                    'area': 'article_period_gold',
                    'diagnostic': 'top_20_units_share',
                    'value': share,
                    'message': (
                        'Demand concentration among the top articles has been '
                        'quantified for later segmentation.'
                    ),
                }
            )

    if reservation_summary and reservation_summary.get('available'):
        invalid_fraction = reservation_summary.get(
            'lead_time_invalid_fraction'
        )

        if invalid_fraction is not None:
            diagnostics.append(
                {
                    'area': 'reservations',
                    'diagnostic': 'lead_time_invalid_fraction',
                    'value': invalid_fraction,
                    'message': (
                        'Semantically invalid analytical lead times remain '
                        'explicitly missing and traceable.'
                    ),
                }
            )

    if event_summary and event_summary.get('available'):
        diagnostics.append(
            {
                'area': 'events',
                'diagnostic': 'unique_event_granularity',
                'value': event_summary.get(
                    'unique_events'
                ),
                'message': (
                    'Event duration/intensity EDA is computed on unique events, '
                    'not on expanded event-day rows.'
                ),
            }
        )

    return diagnostics


def save_eda_report(
    report: dict[str, Any],
    paths: EDAPaths,
    rules: EDARules,
) -> Path | None:
    if not rules.save_report:
        return None

    output_path = (
        Path(paths.reports_dir)
        / 'eda_report.json'
    )

    with open(
        output_path,
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(
            _json_safe(
                report
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# =============================================================================
# MASTER EDA FUNCTION
# =============================================================================


def run_eda(
    silver_datasets: dict[str, pd.DataFrame] | None,
    gold_datasets: dict[str, pd.DataFrame] | None,
    paths: EDAPaths,
    rules: EDARules | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Master Silver/Gold -> EDA entry point.

    Parameters
    ----------
    silver_datasets
        Clean source-specific datasets returned by `run_depuration()`.
    gold_datasets
        Integrated analytical datasets returned by `run_integration()`.
    paths
        EDA-owned output directories.
    rules
        User-facing EDA behaviour. Defaults to `EDARules()`.
    verbose
        Print compact progress information.

    Returns
    -------
    dict
        Machine-readable EDA report that can later feed feature-engineering or
        modelling diagnostics.

    Notes
    -----
    This function never mutates the supplied Silver or Gold DataFrames.
    """
    rules = (
        rules
        if rules is not None
        else EDARules()
    )

    rules.validate()

    silver_datasets = (
        silver_datasets
        if silver_datasets is not None
        else {}
    )

    gold_datasets = (
        gold_datasets
        if gold_datasets is not None
        else {}
    )

    if (
        not silver_datasets
        and not gold_datasets
    ):
        raise ValueError(
            'run_eda() received neither Silver nor Gold datasets.'
        )

    _prepare_output_directories(
        paths,
        rules,
    )

    _print_header(
        'SILVER / GOLD -> EXPLORATORY DATA ANALYSIS',
        verbose=verbose,
    )

    dataset_summaries: list[pd.DataFrame] = []
    column_profiles: list[pd.DataFrame] = []
    category_tables: list[pd.DataFrame] = []

    if (
        rules.analyze_silver
        and silver_datasets
    ):
        dataset_summaries.append(
            build_dataset_summary(
                silver_datasets,
                layer='silver',
            )
        )
        column_profiles.append(
            build_column_profile(
                silver_datasets,
                layer='silver',
            )
        )
        category_tables.append(
            build_categorical_frequency_tables(
                silver_datasets,
                layer='silver',
                rules=rules,
            )
        )

    if (
        rules.analyze_gold
        and gold_datasets
    ):
        dataset_summaries.append(
            build_dataset_summary(
                gold_datasets,
                layer='gold',
            )
        )
        column_profiles.append(
            build_column_profile(
                gold_datasets,
                layer='gold',
            )
        )
        category_tables.append(
            build_categorical_frequency_tables(
                gold_datasets,
                layer='gold',
                rules=rules,
            )
        )

    dataset_summary = (
        pd.concat(
            dataset_summaries,
            ignore_index=True,
        )
        if dataset_summaries
        else pd.DataFrame()
    )

    column_profile = (
        pd.concat(
            column_profiles,
            ignore_index=True,
        )
        if column_profiles
        else pd.DataFrame()
    )

    categorical_frequencies = (
        pd.concat(
            [
                table
                for table in category_tables
                if not table.empty
            ],
            ignore_index=True,
        )
        if any(
            not table.empty
            for table in category_tables
        )
        else pd.DataFrame()
    )

    missingness = build_missingness_table(
        column_profile
    )

    generic_paths: list[Path] = []

    if rules.analyze_silver:
        generic_paths.extend(
            _plot_generic_numeric_distributions(
                silver_datasets,
                layer='silver',
                paths=paths,
                rules=rules,
            )
        )

    if rules.analyze_gold:
        generic_paths.extend(
            _plot_generic_numeric_distributions(
                gold_datasets,
                layer='gold',
                paths=paths,
                rules=rules,
            )
        )

    saved_general_tables: dict[str, str] = {}

    for key, dataframe, filename in [
        (
            'dataset_summary',
            dataset_summary,
            'eda_dataset_summary.csv',
        ),
        (
            'column_profile',
            column_profile,
            'eda_column_profile.csv',
        ),
        (
            'missingness',
            missingness,
            'eda_missingness.csv',
        ),
        (
            'categorical_frequencies',
            categorical_frequencies,
            'eda_categorical_frequencies.csv',
        ),
    ]:
        path = _save_table(
            dataframe,
            filename,
            paths,
            rules,
        )

        if path:
            saved_general_tables[
                key
            ] = str(
                path
            )

    daily_summary: dict[str, Any] | None = None
    article_summary: dict[str, Any] | None = None
    reservation_summary: dict[str, Any] | None = None
    event_summary: dict[str, Any] | None = None

    daily_name = 'tabla_maestra_diaria'
    article_name = 'tabla_maestra_semanal_articulos'

    if (
        rules.analyze_gold
        and daily_name in gold_datasets
    ):
        daily_summary = analyze_daily_gold(
            gold_datasets[
                daily_name
            ].copy(),
            paths=paths,
            rules=rules,
            verbose=verbose,
        )

    if (
        rules.analyze_gold
        and article_name in gold_datasets
    ):
        article_summary = analyze_article_period_gold(
            gold_datasets[
                article_name
            ].copy(),
            paths=paths,
            rules=rules,
            verbose=verbose,
        )

    if (
        rules.analyze_silver
        and rules.analyze_reservations
        and 'reservas' in silver_datasets
    ):
        reservation_summary = analyze_reservations(
            silver_datasets[
                'reservas'
            ].copy(),
            paths=paths,
            rules=rules,
            verbose=verbose,
        )

    if (
        rules.analyze_silver
        and rules.analyze_unique_events
        and 'eventos' in silver_datasets
    ):
        event_summary = analyze_events(
            silver_datasets[
                'eventos'
            ].copy(),
            paths=paths,
            rules=rules,
            verbose=verbose,
        )

    diagnostics = _build_framework_diagnostics(
        daily_summary=daily_summary,
        article_summary=article_summary,
        reservation_summary=reservation_summary,
        event_summary=event_summary,
    )

    report: dict[str, Any] = {
        'rules': asdict(
            rules
        ),
        'silver_datasets_analyzed': (
            sorted(
                silver_datasets.keys()
            )
            if rules.analyze_silver
            else []
        ),
        'gold_datasets_analyzed': (
            sorted(
                gold_datasets.keys()
            )
            if rules.analyze_gold
            else []
        ),
        'general_tables': saved_general_tables,
        'generic_distribution_figures': [
            str(path)
            for path in generic_paths
        ],
        'daily_gold': daily_summary,
        'article_period_gold': article_summary,
        'reservations': reservation_summary,
        'events': event_summary,
        'diagnostics': diagnostics,
    }

    report_path = save_eda_report(
        report,
        paths,
        rules,
    )

    if report_path:
        report[
            'report_path'
        ] = str(
            report_path
        )

    if verbose and not dataset_summary.empty:
        print(
            '\nEDA DATASET SUMMARY'
        )
        print(
            dataset_summary[
                [
                    'layer',
                    'dataset',
                    'rows',
                    'columns',
                    'missing_cells',
                    'date_start',
                    'date_end',
                ]
            ].to_string(
                index=False
            )
        )

    _print_message(
        'EDA completed. No input Silver/Gold observation was modified.',
        verbose=verbose,
    )

    return report
