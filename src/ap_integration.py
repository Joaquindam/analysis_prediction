from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PUBLIC CONFIGURATION OBJECTS
# =============================================================================

@dataclass(frozen=True)
class IntegrationPaths:
    """
    Output paths used by the Gold integration layer.

    Parameters
    ----------
    gold_dir : pathlib.Path
        Directory where Gold parquet datasets are stored.
    reports_dir : pathlib.Path
        Directory where integration/provenance reports are stored.
    """
    gold_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class IntegrationRules:
    """
    Transparent rules for automatic Silver -> Gold integration.

    Philosophy
    ----------
    The integration layer must combine already-clean Silver datasets without
    inventing information.

    Important consequences:
    - Weekly article sales are NEVER disaggregated into invented daily sales.
    - Daily tables are merged only after each source has been reduced to one
      row per date.
    - Repeated dates inside a source are aggregated before merging, preventing
      accidental row multiplication.
    - The daily calendar is anchored, when possible, to internal business
      activity (tickets, invoices or reservations), not to external weather or
      event files that may cover a longer period.
    - Missing daily activity counts can be filled with zero when a calendar day
      exists but no records from that source are present.
    - Missing continuous measurements such as weather are NOT automatically
      filled with zero.
    - Ticket-derived revenue is preferred as the canonical historical revenue
      when available; PDF-invoice revenue is retained separately for auditing.
    """

    build_daily_master: bool = True
    build_weekly_article_master: bool = True

    # Daily calendar
    create_continuous_daily_calendar: bool = True
    anchor_daily_calendar_to_business_activity: bool = True

    # Missing values after joins
    fill_missing_activity_counts_with_zero: bool = True
    fill_missing_boolean_context_with_false: bool = True

    # Canonical historical revenue
    prefer_ticket_revenue: bool = True

    # Article-period master
    keep_invitations: bool = True
    aggregate_duplicate_article_period_rows: bool = True

    # Reporting
    save_integration_report: bool = True
    save_column_provenance: bool = True

    # Console
    # 'summary' keeps the terminal compact while preserving all persisted
    # Gold datasets and integration reports.
    # 'detailed' restores the previous source-by-source output.
    console_detail: str = 'summary'

    def validate(self) -> None:
        """Validate configuration values before integration starts."""
        for field_name in [
            'build_daily_master',
            'build_weekly_article_master',
            'create_continuous_daily_calendar',
            'anchor_daily_calendar_to_business_activity',
            'fill_missing_activity_counts_with_zero',
            'fill_missing_boolean_context_with_false',
            'prefer_ticket_revenue',
            'keep_invitations',
            'aggregate_duplicate_article_period_rows',
            'save_integration_report',
            'save_column_provenance',
        ]:
            value = getattr(self, field_name)

            if not isinstance(value, bool):
                raise TypeError(
                    f'{field_name} must be True or False.'
                )

        if self.console_detail not in {'summary', 'detailed'}:
            raise ValueError(
                "console_detail must be 'summary' or 'detailed'."
            )


# =============================================================================
# CONSTANTS
# =============================================================================

GOLD_FILENAMES = {
    'tabla_maestra_diaria': 'tabla_maestra_diaria.parquet',
    'tabla_maestra_semanal_articulos': 'tabla_maestra_semanal_articulos.parquet',
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
        print(f'[{level}] {message}')


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _ensure_directory(
    directory: str | Path,
) -> Path:
    path = Path(directory)
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def _json_safe(
    value: Any,
) -> Any:
    """Convert pandas/numpy objects to JSON-compatible values."""
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    if isinstance(value, Path):
        return str(value)

    if pd.isna(value):
        return None

    return value


def _first_existing(
    dataframe: pd.DataFrame,
    candidates: list[str] | tuple[str, ...],
) -> str | None:
    """Return the first candidate column present in a DataFrame."""
    return next(
        (
            column
            for column in candidates
            if column in dataframe.columns
        ),
        None,
    )


def _to_normalized_date(
    series: pd.Series,
) -> pd.Series:
    """Convert values to midnight-normalized pandas datetimes."""
    return pd.to_datetime(
        series,
        errors='coerce',
        dayfirst=True,
    ).dt.normalize()


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _consistent_non_null_value(
    series: pd.Series,
) -> Any:
    """
    Return the unique non-null value when a group is internally consistent.

    If a group contains conflicting non-null metadata, pd.NA is returned. This
    is intentionally conservative: mutable descriptive metadata must never
    become part of a business key or silently split one analytical entity.
    """
    values = (
        series
        .dropna()
        .drop_duplicates()
    )

    if len(values) == 1:
        return values.iloc[0]

    return pd.NA


def _infer_expected_period_days(
    series: pd.Series,
) -> int | None:
    """
    Infer the usual report-period length from the data.

    The modal positive duration is used. If several durations tie for the
    highest frequency, the longest is preferred because truncated first/last
    reporting windows are commonly shorter than the normal reporting period.
    """
    values = (
        _safe_numeric(series)
        .dropna()
    )

    values = values.loc[
        values.gt(0)
    ]

    if values.empty:
        return None

    counts = (
        values
        .round()
        .astype(int)
        .value_counts()
    )

    max_count = counts.max()
    candidates = counts.loc[
        counts.eq(max_count)
    ].index

    if len(candidates) == 0:
        return None

    return int(
        max(candidates)
    )


def _join_unique_strings(
    series: pd.Series,
) -> Any:
    """
    Join unique non-null text values in deterministic order.

    Returns pd.NA when no usable values are present.
    """
    values = sorted(
        {
            str(value).strip()
            for value in series.dropna()
            if str(value).strip()
        }
    )

    if not values:
        return pd.NA

    return ' | '.join(values)


def _assert_unique_key(
    dataframe: pd.DataFrame,
    key: str | list[str],
    dataset_label: str,
) -> None:
    """Fail loudly if an integration output contains duplicate merge keys."""
    keys = [key] if isinstance(key, str) else key

    if dataframe.empty:
        return

    duplicated = dataframe.duplicated(
        subset=keys,
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            f'{dataset_label} is not unique by key {keys}. '
            f'{int(duplicated.sum())} duplicated rows remain. '
            'Integration was stopped to avoid accidental row multiplication.'
        )


def _merge_daily_one_to_one(
    left: pd.DataFrame | None,
    right: pd.DataFrame | None,
    source_name: str,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Left-merge one daily source onto the already defined calendar backbone.

    Once the Gold calendar has been created, contextual or longer-history
    sources must not extend it. Dates outside the selected analytical window
    are reported and ignored rather than silently added.
    """
    if right is None or right.empty:
        return left

    _assert_unique_key(
        right,
        key='date',
        dataset_label=source_name,
    )

    if left is None or left.empty:
        return right.copy()

    _assert_unique_key(
        left,
        key='date',
        dataset_label='current daily master',
    )

    left_dates = set(
        pd.to_datetime(
            left['date'],
            errors='coerce',
        )
        .dropna()
        .dt.normalize()
    )

    right_dates = set(
        pd.to_datetime(
            right['date'],
            errors='coerce',
        )
        .dropna()
        .dt.normalize()
    )

    outside_dates = right_dates.difference(
        left_dates
    )

    if outside_dates:
        _print_message(
            f'Daily merge: {source_name} -> '
            f'{len(outside_dates):,} source dates fall outside the selected '
            'Gold calendar and were ignored.',
            level='WARNING',
            verbose=verbose,
        )

    before_rows = len(left)

    merged = left.merge(
        right,
        on='date',
        how='left',
        validate='one_to_one',
    )

    _print_message(
        f'Daily merge: {source_name} | '
        f'{before_rows:,} master dates -> {len(merged):,} dates.',
        verbose=verbose,
    )

    return merged


# =============================================================================
# SOURCE DATE RANGES / COVERAGE
# =============================================================================

def _extract_source_date_range(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Infer the usable temporal range of a Silver dataset.

    Actual row-level observation dates take precedence over report-period
    metadata. Period bounds are used only when a source has no row-level date.
    This prevents a report envelope from artificially extending the apparent
    coverage of transactional datasets.
    """
    row_level_candidates = {
        'reservas': (
            'reservation_datetime',
            'reservation_date',
            'date',
        ),
        'meteo_horaria': (
            'datetime',
            'date',
        ),
    }.get(
        dataset_name,
        (
            'date',
            'datetime',
            'reservation_datetime',
            'reservation_date',
            'event_date',
        ),
    )

    for column in row_level_candidates:
        if column not in dataframe.columns:
            continue

        values = pd.to_datetime(
            dataframe[column],
            errors='coerce',
        ).dropna()

        if not values.empty:
            return (
                values.min().normalize(),
                values.max().normalize(),
            )

    observed: list[pd.Timestamp] = []

    for column in (
        'report_start',
        'report_end',
        'event_start',
        'event_end',
    ):
        if column not in dataframe.columns:
            continue

        values = pd.to_datetime(
            dataframe[column],
            errors='coerce',
        ).dropna()

        if not values.empty:
            observed.extend(
                [
                    values.min().normalize(),
                    values.max().normalize(),
                ]
            )

    if not observed:
        return None, None

    return min(observed), max(observed)


def build_source_coverage_report(
    silver_datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarize dataset availability and temporal coverage before integration.
    """
    rows = []

    for dataset_name, dataframe in sorted(
        silver_datasets.items()
    ):
        if not isinstance(dataframe, pd.DataFrame):
            continue

        start, end = _extract_source_date_range(
            dataset_name,
            dataframe,
        )

        rows.append(
            {
                'dataset': dataset_name,
                'rows': len(dataframe),
                'columns': len(dataframe.columns),
                'date_start': start,
                'date_end': end,
                'missing_cells': int(
                    dataframe.isna().sum().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# DAILY AGGREGATORS
# =============================================================================

def aggregate_tickets_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate ticket-level Silver data to one row per day.

    Expected current schema from ap_io/ap_depuration:
    - date
    - document_id
    - document_total
    - receipt_count
    """
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    rows = []

    for current_date, group in data.groupby(
        'date',
        sort=True,
    ):
        row = {
            'date': current_date,
            'num_tickets': len(group),
        }

        if 'document_id' in group.columns:
            row['num_tickets_unicos'] = int(
                group['document_id'].nunique(
                    dropna=True
                )
            )

        if 'document_total' in group.columns:
            values = _safe_numeric(
                group['document_total']
            )

            row['facturacion_tickets'] = values.sum(
                min_count=1
            )
            row['ticket_medio'] = values.mean()
            row['ticket_mediano'] = values.median()
            row['ticket_maximo'] = values.max()
            row['ticket_minimo'] = values.min()

        if 'receipt_count' in group.columns:
            values = _safe_numeric(
                group['receipt_count']
            )

            row['num_comprobantes'] = values.sum(
                min_count=1
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily tickets',
    )

    return result


def aggregate_invoices_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate parsed PDF invoices/tickets to one row per day.

    PDF-derived revenue remains separate from POS ticket revenue to make
    reconciliation/auditing possible.
    """
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    rows = []

    for current_date, group in data.groupby(
        'date',
        sort=True,
    ):
        row = {
            'date': current_date,
            'num_facturas_pdf': len(group),
        }

        if 'ticket_id' in group.columns:
            row['num_facturas_pdf_unicas'] = int(
                group['ticket_id'].nunique(
                    dropna=True
                )
            )

        for source_column, output_prefix in [
            ('total', 'facturacion_facturas_pdf'),
            ('base', 'base_facturas_pdf'),
            ('vat', 'iva_facturas_pdf'),
            ('cash_amount', 'efectivo_facturas_pdf'),
            ('card_amount', 'tarjeta_facturas_pdf'),
        ]:
            if source_column in group.columns:
                values = _safe_numeric(
                    group[source_column]
                )

                row[output_prefix] = values.sum(
                    min_count=1
                )

        if 'invoice_reconciliation__invalid' in group.columns:
            row['facturas_pdf_no_reconciliadas'] = int(
                group[
                    'invoice_reconciliation__invalid'
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        if 'payment_reconciliation__invalid' in group.columns:
            row['pagos_pdf_no_reconciliados'] = int(
                group[
                    'payment_reconciliation__invalid'
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily PDF invoices',
    )

    return result


def aggregate_reservations_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate reservation-level Silver data to one row per reservation date.

    Historical Gold contains both total reservations and status outcomes. These
    realized outcomes are useful for retrospective analysis but must not later
    be used as future features unless they are genuinely known at prediction
    time.
    """
    datetime_column = _first_existing(
        dataframe,
        (
            'reservation_datetime',
            'reservation_date',
            'date',
        ),
    )

    if datetime_column is None:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data[datetime_column]
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    if 'people' in data.columns:
        data['people'] = _safe_numeric(
            data['people']
        )

    rows = []

    standard_statuses = (
        'completed',
        'cancelled',
        'no_show',
        'pending',
    )

    for current_date, group in data.groupby(
        'date',
        sort=True,
    ):
        row = {
            'date': current_date,
            'reservas_total': len(group),
        }

        if 'people' in group.columns:
            people_values = _safe_numeric(
                group[
                    'people'
                ]
            ).dropna()

            if people_values.empty:
                row[
                    'comensales_reservados_total'
                ] = np.nan
                row[
                    'tamano_grupo_medio'
                ] = np.nan
                row[
                    'tamano_grupo_mediano'
                ] = np.nan
                row[
                    'tamano_grupo_maximo'
                ] = np.nan
            else:
                row[
                    'comensales_reservados_total'
                ] = people_values.sum()
                row[
                    'tamano_grupo_medio'
                ] = people_values.mean()
                row[
                    'tamano_grupo_mediano'
                ] = people_values.median()
                row[
                    'tamano_grupo_maximo'
                ] = people_values.max()

        if 'status_grouped' in group.columns:
            for status in standard_statuses:
                status_mask = (
                    group['status_grouped']
                    .eq(status)
                )

                row[
                    f'reservas_{status}'
                ] = int(
                    status_mask.sum()
                )

                if 'people' in group.columns:
                    status_people = _safe_numeric(
                        group.loc[
                            status_mask,
                            'people',
                        ]
                    )

                    if not status_mask.any():
                        # Zero reservations of a status means zero people in
                        # that status. This is structural zero, not missing.
                        row[
                            f'comensales_{status}'
                        ] = 0.0
                    else:
                        # If reservations exist but every party size is
                        # unknown, preserve NaN rather than inventing zero.
                        row[
                            f'comensales_{status}'
                        ] = status_people.sum(
                            min_count=1
                        )

            denominator = row[
                'reservas_total'
            ]

            row['no_show_rate'] = (
                row.get(
                    'reservas_no_show',
                    0,
                )
                / denominator
                if denominator
                else np.nan
            )

            row['cancel_rate'] = (
                row.get(
                    'reservas_cancelled',
                    0,
                )
                / denominator
                if denominator
                else np.nan
            )

        if 'lead_time_hours' in group.columns:
            lead = _safe_numeric(
                group['lead_time_hours']
            ).dropna()

            if lead.empty:
                row[
                    'lead_time_horas_medio'
                ] = np.nan
                row[
                    'lead_time_horas_mediano'
                ] = np.nan
            else:
                row[
                    'lead_time_horas_medio'
                ] = lead.mean()
                row[
                    'lead_time_horas_mediano'
                ] = lead.median()

        if 'is_walk_in' in group.columns:
            row['reservas_walk_in'] = int(
                group[
                    'is_walk_in'
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        if 'is_large_group' in group.columns:
            row['reservas_grupo_grande'] = int(
                group[
                    'is_large_group'
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        if 'shift' in group.columns:
            shifts = (
                group['shift']
                .dropna()
                .astype('string')
                .str.strip()
            )

            for shift_name, count in (
                shifts.value_counts()
                .to_dict()
                .items()
            ):
                safe_shift = (
                    str(shift_name)
                    .lower()
                    .replace(' ', '_')
                )

                row[
                    f'reservas_turno_{safe_shift}'
                ] = int(count)

        rows.append(row)

    result = pd.DataFrame(rows)

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily reservations',
    )

    return result


def aggregate_tips_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate tips to one row per day only when an actual date exists.

    The current POS tip summary may only provide report-level metadata. In that
    case this function returns an empty DataFrame rather than inventing a daily
    date from a period.
    """
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    tip_column = _first_existing(
        data,
        (
            'tip',
            'tip_amount',
        ),
    )

    if tip_column is None:
        return pd.DataFrame()

    data['_tip_value'] = _safe_numeric(
        data[tip_column]
    )

    result = (
        data.groupby(
            'date',
            as_index=False,
        )
        .agg(
            propina_total=(
                '_tip_value',
                'sum',
            ),
            propina_media=(
                '_tip_value',
                'mean',
            ),
            tickets_con_propina=(
                '_tip_value',
                lambda values: int(
                    (
                        values.fillna(0)
                        > 0
                    ).sum()
                ),
            ),
        )
    )

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily tips',
    )

    return result


def aggregate_weather_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure daily weather has exactly one row per date.

    If duplicate dates remain, variables are aggregated with a name-aware
    conservative rule:
    - precipitation/sunshine/duration -> sum;
    - max-like variables -> max;
    - min-like variables -> min;
    - other numeric variables -> mean.
    """
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    numeric_columns = [
        column
        for column in data.select_dtypes(
            include='number'
        ).columns
        if not (
            column.endswith('__outlier')
            or column.endswith('__imputed')
            or column.endswith('__invalid')
            or column.endswith('__negative')
            or column.endswith('__suspicious')
        )
    ]

    aggregations = {}

    for column in numeric_columns:
        normalized = column.lower()

        if (
            'precip' in normalized
            or 'sunshine' in normalized
            or 'duration' in normalized
        ):
            aggregations[column] = 'sum'

        elif (
            'max' in normalized
            or 'gust' in normalized
        ):
            aggregations[column] = 'max'

        elif 'min' in normalized:
            aggregations[column] = 'min'

        else:
            aggregations[column] = 'mean'

    if aggregations:
        result = (
            data.groupby(
                'date',
                as_index=False,
            )
            .agg(
                aggregations
            )
        )
    else:
        result = (
            data[
                ['date']
            ]
            .drop_duplicates()
            .sort_values(
                'date'
            )
            .reset_index(
                drop=True
            )
        )

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily weather',
    )

    return result


def aggregate_holidays_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate holiday data to one row per date."""
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    name_column = _first_existing(
        data,
        (
            'holiday_name',
            'name',
        ),
    )

    rows = []

    for current_date, group in data.groupby(
        'date',
        sort=True,
    ):
        row = {
            'date': current_date,
            'es_festivo': True,
            'num_festivos': len(group),
        }

        if name_column is not None:
            row['festivo_nombre'] = (
                _join_unique_strings(
                    group[name_column]
                )
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily holidays',
    )

    return result


def aggregate_events_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate scheduled event information to one row per date."""
    if 'date' not in dataframe.columns:
        return pd.DataFrame()

    data = dataframe.copy()

    data['date'] = _to_normalized_date(
        data['date']
    )

    data = data.dropna(
        subset=['date']
    )

    if data.empty:
        return pd.DataFrame()

    name_column = _first_existing(
        data,
        (
            'event_name',
            'name',
        ),
    )

    category_column = _first_existing(
        data,
        (
            'event_category',
            'category',
        ),
    )

    intensity_column = _first_existing(
        data,
        (
            'event_intensity',
            'expected_impact',
        ),
    )

    confidence_column = _first_existing(
        data,
        (
            'confidence',
        ),
    )

    rows = []

    for current_date, group in data.groupby(
        'date',
        sort=True,
    ):
        row = {
            'date': current_date,
            'tiene_evento': True,
            'num_eventos': len(group),
        }

        if name_column is not None:
            row['eventos_nombres'] = (
                _join_unique_strings(
                    group[name_column]
                )
            )

        if category_column is not None:
            row['eventos_categorias'] = (
                _join_unique_strings(
                    group[category_column]
                )
            )

        if intensity_column is not None:
            values = _safe_numeric(
                group[intensity_column]
            )

            row['event_intensity_mean'] = (
                values.mean()
            )
            row['event_intensity_max'] = (
                values.max()
            )

        if confidence_column is not None:
            values = _safe_numeric(
                group[confidence_column]
            )

            row['event_confidence_mean'] = (
                values.mean()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    _assert_unique_key(
        result,
        key='date',
        dataset_label='daily events',
    )

    return result


# =============================================================================
# DAILY CALENDAR
# =============================================================================

def _select_daily_calendar_anchor_sources(
    daily_frames: dict[str, pd.DataFrame],
    rules: IntegrationRules,
) -> list[str]:
    """
    Select the source(s) that define the daily analytical window.

    The choice is semantic rather than restaurant-specific. When business
    anchoring is enabled, the framework anchors the calendar to the source
    used for canonical revenue whenever possible. If that source is absent,
    it falls back to another transactional source and finally to reservations.

    External/context sources such as weather, holidays and events never define
    the business observation window when business anchoring is enabled.
    """
    available = {
        name
        for name, frame
        in daily_frames.items()
        if (
            frame is not None
            and not frame.empty
            and 'date' in frame.columns
        )
    }

    if not available:
        return []

    if not rules.anchor_daily_calendar_to_business_activity:
        return sorted(
            available
        )

    if rules.prefer_ticket_revenue:
        priority = (
            'tickets',
            'facturas',
            'reservas',
        )
    else:
        priority = (
            'facturas',
            'tickets',
            'reservas',
        )

    for source_name in priority:
        if source_name in available:
            return [
                source_name
            ]

    # Generic fallback for deployments whose normalized source set differs
    # from the currently supported hospitality sources. Prefer non-context
    # sources before allowing an external source to anchor the calendar.
    context_sources = {
        'meteo_diaria',
        'meteo_horaria',
        'festivos',
        'eventos',
    }

    non_context = sorted(
        available.difference(
            context_sources
        )
    )

    if non_context:
        return [
            non_context[0]
        ]

    return [
        sorted(
            available
        )[0]
    ]


def _date_bounds_from_daily_frames(
    daily_frames: dict[str, pd.DataFrame],
    rules: IntegrationRules,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    """
    Determine calendar bounds from the selected analytical anchor source(s).
    """
    selected_sources = _select_daily_calendar_anchor_sources(
        daily_frames=daily_frames,
        rules=rules,
    )

    dates = []

    for name in selected_sources:
        frame = daily_frames[
            name
        ]

        valid_dates = pd.to_datetime(
            frame['date'],
            errors='coerce',
        ).dropna()

        if valid_dates.empty:
            continue

        dates.extend(
            [
                valid_dates.min().normalize(),
                valid_dates.max().normalize(),
            ]
        )

    if not dates:
        return None, None, selected_sources

    return (
        min(dates),
        max(dates),
        selected_sources,
    )


def build_daily_calendar(
    daily_frames: dict[str, pd.DataFrame],
    rules: IntegrationRules,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create the date backbone used by the daily Gold master.
    """
    start, end, anchor_sources = (
        _date_bounds_from_daily_frames(
            daily_frames=daily_frames,
            rules=rules,
        )
    )

    if start is None or end is None:
        return pd.DataFrame(
            columns=['date']
        )

    if rules.create_continuous_daily_calendar:
        dates = pd.date_range(
            start=start,
            end=end,
            freq='D',
        )

        calendar = pd.DataFrame(
            {
                'date': dates,
            }
        )

        _print_message(
            f'Daily calendar created from {start.date()} to {end.date()} '
            f'({len(calendar):,} consecutive days). '
            f'Anchor source(s): {anchor_sources}.',
            verbose=verbose,
        )

        return calendar

    observed_dates = sorted(
        set().union(
            *[
                set(
                    pd.to_datetime(
                        daily_frames[name]['date'],
                        errors='coerce',
                    )
                    .dropna()
                    .dt.normalize()
                )
                for name in anchor_sources
                if (
                    name in daily_frames
                    and not daily_frames[name].empty
                    and 'date' in daily_frames[name].columns
                )
            ]
        )
        if anchor_sources
        else set()
    )

    calendar = pd.DataFrame(
        {
            'date': observed_dates,
        }
    )

    _print_message(
        f'Daily calendar contains {len(calendar):,} observed anchor dates '
        f'only. Anchor source(s): {anchor_sources}.',
        verbose=verbose,
    )

    return calendar


# =============================================================================
# DAILY MASTER
# =============================================================================

def _add_calendar_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar variables known directly from the date."""
    data = dataframe.copy()

    if data.empty:
        return data

    date = pd.to_datetime(
        data['date'],
        errors='coerce',
    )

    iso = date.dt.isocalendar()

    data['year'] = date.dt.year.astype(
        'Int64'
    )
    data['month'] = date.dt.month.astype(
        'Int64'
    )
    data['quarter'] = date.dt.quarter.astype(
        'Int64'
    )
    data['day_of_month'] = date.dt.day.astype(
        'Int64'
    )
    data['day_of_week_num'] = date.dt.dayofweek.astype(
        'Int64'
    )
    data['day_of_week'] = date.dt.day_name()
    data['week_of_year'] = iso.week.astype(
        'Int64'
    )
    data['is_weekend'] = (
        date.dt.dayofweek
        .isin(
            [
                5,
                6,
            ]
        )
    )
    data['is_month_start'] = date.dt.is_month_start
    data['is_month_end'] = date.dt.is_month_end

    # Cyclical variables are useful later for predictive models and are purely
    # calendar-derived, therefore known in advance.
    week_number = (
        data[
            'week_of_year'
        ]
        .astype(
            'Float64'
        )
    )

    data['week_sin'] = np.sin(
        2
        * np.pi
        * week_number
        / 52.0
    )

    data['week_cos'] = np.cos(
        2
        * np.pi
        * week_number
        / 52.0
    )

    return data


def _fill_daily_structural_missing_values(
    dataframe: pd.DataFrame,
    rules: IntegrationRules,
) -> pd.DataFrame:
    """
    Fill only structural missing values introduced by outer joins.

    Examples
    --------
    If 10 June exists in the calendar but no holiday row exists:
        es_festivo = False
        num_festivos = 0

    Weather is NOT filled with zero, because missing temperature does not mean
    zero degrees.
    """
    data = dataframe.copy()

    if rules.fill_missing_activity_counts_with_zero:
        count_columns = [
            column
            for column in data.columns
            if (
                column.startswith('num_')
                or column.startswith('reservas_')
                or column.startswith('comensales_')
                or column.startswith('tickets_con_')
                or column.startswith('facturas_pdf_no_')
                or column.startswith('pagos_pdf_no_')
            )
            and not (
                column.endswith('_medio')
                or column.endswith('_mediano')
                or column.endswith('_maximo')
                or column.endswith('_minimo')
                or column.endswith('_rate')
            )
        ]

        for column in count_columns:
            original = data[
                column
            ]

            numeric = pd.to_numeric(
                original,
                errors='coerce',
            )

            # Coerce only when every existing non-null value is genuinely
            # numeric. This allows nullable/object-backed count columns to be
            # handled without silently converting arbitrary text to zero.
            can_be_numeric = (
                int(original.notna().sum())
                == int(numeric.notna().sum())
            )

            if can_be_numeric:
                data[column] = numeric.fillna(
                    0
                )

    if rules.fill_missing_boolean_context_with_false:
        for column in [
            'es_festivo',
            'tiene_evento',
        ]:
            if column in data.columns:
                data[column] = (
                    data[column]
                    .astype(
                        'boolean'
                    )
                    .fillna(
                        False
                    )
                    .astype(
                        bool
                    )
                )

    return data


def _create_canonical_revenue(
    dataframe: pd.DataFrame,
    rules: IntegrationRules,
) -> pd.DataFrame:
    """
    Create a canonical historical revenue column without discarding sources.

    POS ticket revenue and parsed PDF revenue are retained separately.
    """
    data = dataframe.copy()

    ticket_column = (
        'facturacion_tickets'
        if 'facturacion_tickets' in data.columns
        else None
    )

    invoice_column = (
        'facturacion_facturas_pdf'
        if 'facturacion_facturas_pdf' in data.columns
        else None
    )

    if (
        ticket_column is None
        and invoice_column is None
    ):
        return data

    data['facturacion'] = np.nan
    data['facturacion_source'] = pd.NA

    if (
        rules.prefer_ticket_revenue
        and ticket_column is not None
    ):
        ticket_available = data[
            ticket_column
        ].notna()

        data.loc[
            ticket_available,
            'facturacion',
        ] = data.loc[
            ticket_available,
            ticket_column,
        ]

        data.loc[
            ticket_available,
            'facturacion_source',
        ] = 'tickets'

        if invoice_column is not None:
            fallback = (
                data[
                    'facturacion'
                ].isna()
                & data[
                    invoice_column
                ].notna()
            )

            data.loc[
                fallback,
                'facturacion',
            ] = data.loc[
                fallback,
                invoice_column,
            ]

            data.loc[
                fallback,
                'facturacion_source',
            ] = 'pdf_facturas'

    elif invoice_column is not None:
        invoice_available = data[
            invoice_column
        ].notna()

        data.loc[
            invoice_available,
            'facturacion',
        ] = data.loc[
            invoice_available,
            invoice_column,
        ]

        data.loc[
            invoice_available,
            'facturacion_source',
        ] = 'pdf_facturas'

        if ticket_column is not None:
            fallback = (
                data[
                    'facturacion'
                ].isna()
                & data[
                    ticket_column
                ].notna()
            )

            data.loc[
                fallback,
                'facturacion',
            ] = data.loc[
                fallback,
                ticket_column,
            ]

            data.loc[
                fallback,
                'facturacion_source',
            ] = 'tickets'

    return data


def build_daily_master(
    silver_datasets: dict[str, pd.DataFrame],
    rules: IntegrationRules | None = None,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Build one historical Gold row per calendar day.

    Sources currently supported
    ---------------------------
    - tickets
    - facturas
    - reservas
    - tips, only if a true date exists
    - meteo_diaria
    - festivos
    - eventos

    Notes
    -----
    `ventas` is intentionally NOT merged into the daily table because its
    source granularity is weekly/periodic. Daily article sales are never
    invented.
    """
    rules = (
        rules
        if rules is not None
        else IntegrationRules()
    )

    rules.validate()

    _print_subheader(
        'Build daily Gold master',
        verbose=verbose,
    )

    aggregators = {
        'tickets': aggregate_tickets_daily,
        'facturas': aggregate_invoices_daily,
        'reservas': aggregate_reservations_daily,
        'tips': aggregate_tips_daily,
        'meteo_diaria': aggregate_weather_daily,
        'festivos': aggregate_holidays_daily,
        'eventos': aggregate_events_daily,
    }

    daily_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    source_summaries = []

    for source_name, aggregator in aggregators.items():
        if source_name not in silver_datasets:
            _print_message(
                f'Daily integration: {source_name} not available -> skipped.',
                verbose=verbose,
            )
            continue

        source = silver_datasets[
            source_name
        ]

        frame = aggregator(
            source
        )

        if frame.empty:
            _print_message(
                f'Daily integration: {source_name} cannot produce daily rows '
                'with the current schema -> skipped without inventing dates.',
                level='WARNING',
                verbose=verbose,
            )
            continue

        daily_frames[
            source_name
        ] = frame

        source_summaries.append(
            {
                'source': source_name,
                'silver_rows': len(source),
                'daily_rows': len(frame),
                'date_start': frame['date'].min(),
                'date_end': frame['date'].max(),
            }
        )

        _print_message(
            f'Daily integration: {source_name} -> '
            f'{len(frame):,} unique daily rows '
            f'[{frame["date"].min().date()} .. {frame["date"].max().date()}].',
            verbose=verbose,
        )

    if not daily_frames:
        return (
            pd.DataFrame(),
            {
                'sources': source_summaries,
                'rows': 0,
                'columns': 0,
            },
        )

    master = build_daily_calendar(
        daily_frames=daily_frames,
        rules=rules,
        verbose=verbose,
    )

    for source_name in [
        'tickets',
        'facturas',
        'reservas',
        'tips',
        'meteo_diaria',
        'festivos',
        'eventos',
    ]:
        if source_name not in daily_frames:
            continue

        master = _merge_daily_one_to_one(
            left=master,
            right=daily_frames[source_name],
            source_name=source_name,
            verbose=verbose,
        )

    if master is None:
        master = pd.DataFrame()

    master = (
        master
        .sort_values(
            'date'
        )
        .reset_index(
            drop=True
        )
    )

    master = _add_calendar_features(
        master
    )

    master = _create_canonical_revenue(
        master,
        rules,
    )

    master = _fill_daily_structural_missing_values(
        master,
        rules,
    )

    _assert_unique_key(
        master,
        key='date',
        dataset_label='final daily Gold master',
    )

    summary = {
        'sources': source_summaries,
        'rows': len(master),
        'columns': len(master.columns),
        'date_start': (
            master['date'].min()
            if not master.empty
            else None
        ),
        'date_end': (
            master['date'].max()
            if not master.empty
            else None
        ),
    }

    _print_message(
        f'Daily Gold master created: {len(master):,} rows x '
        f'{len(master.columns):,} columns.',
        verbose=verbose,
    )

    return (
        master,
        summary,
    )


# =============================================================================
# WEEKLY / PERIODIC ARTICLE MASTER
# =============================================================================

def _safe_master_lookup(
    dataframe: pd.DataFrame,
    key: str,
    value_columns: list[str],
    dataset_name: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a one-row-per-key lookup without silently trusting conflicting rows.

    For each requested value column, the first non-null value is retained only
    when all non-null values for the same key agree. Conflicting fields are
    returned as missing and a `<column>__master_conflict` flag is created.
    """
    if key not in dataframe.columns:
        return pd.DataFrame()

    available_values = [
        column
        for column in value_columns
        if (
            column in dataframe.columns
            and dataframe[column].notna().any()
        )
    ]

    if not available_values:
        return pd.DataFrame()

    rows = []

    for key_value, group in dataframe.groupby(
        key,
        dropna=True,
        sort=False,
    ):
        row = {
            key: key_value,
        }

        for column in available_values:
            values = (
                group[column]
                .dropna()
                .drop_duplicates()
            )

            conflict = (
                len(values)
                > 1
            )

            row[
                f'{column}__master_conflict'
            ] = conflict

            if len(values) == 1:
                row[column] = (
                    values.iloc[0]
                )
            else:
                row[column] = pd.NA

        rows.append(row)

    lookup = pd.DataFrame(rows)

    conflict_columns = [
        column
        for column in lookup.columns
        if column.endswith(
            '__master_conflict'
        )
    ]

    conflict_rows = int(
        lookup[
            conflict_columns
        ]
        .any(
            axis=1
        )
        .sum()
    ) if conflict_columns else 0

    if conflict_rows:
        _print_message(
            f'{dataset_name}: {conflict_rows:,} master keys contain conflicting '
            'metadata. Conflicting fields are left missing during Gold '
            'enrichment rather than silently choosing one value.',
            level='WARNING',
            verbose=verbose,
        )

    _assert_unique_key(
        lookup,
        key=key,
        dataset_label=f'{dataset_name} master lookup',
    )

    return lookup


def _coalesce_columns(
    dataframe: pd.DataFrame,
    primary: str,
    secondary: str,
    output: str,
) -> pd.DataFrame:
    """Coalesce two columns without overwriting non-null primary information."""
    data = dataframe.copy()

    if primary in data.columns:
        result = data[
            primary
        ].copy()

    else:
        result = pd.Series(
            pd.NA,
            index=data.index,
        )

    if secondary in data.columns:
        result = result.fillna(
            data[
                secondary
            ]
        )

    data[
        output
    ] = result

    return data


def enrich_article_period_master(
    dataframe: pd.DataFrame,
    silver_datasets: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enrich article-period sales with conflict-safe master metadata.
    """
    data = dataframe.copy()

    if (
        'articulos' in silver_datasets
        and 'article_code' in data.columns
    ):
        article_lookup = _safe_master_lookup(
            dataframe=silver_datasets['articulos'],
            key='article_code',
            value_columns=[
                'article_name',
                'department_code',
                'price',
            ],
            dataset_name='articulos',
            verbose=verbose,
        )

        if not article_lookup.empty:
            article_lookup = article_lookup.rename(
                columns={
                    column: (
                        f'{column}__articles_master'
                        if column != 'article_code'
                        else column
                    )
                    for column in article_lookup.columns
                }
            )

            data = data.merge(
                article_lookup,
                on='article_code',
                how='left',
                validate='many_to_one',
            )

            data = _coalesce_columns(
                data,
                primary='article_name',
                secondary='article_name__articles_master',
                output='article_name',
            )

            data = _coalesce_columns(
                data,
                primary='department_code',
                secondary='department_code__articles_master',
                output='department_code',
            )

            for column in [
                'article_name',
                'department_code',
            ]:
                conflict_column = (
                    f'{column}__period_conflict'
                )

                if (
                    conflict_column in data.columns
                    and column in data.columns
                ):
                    data.loc[
                        data[
                            conflict_column
                        ]
                        .fillna(
                            False
                        )
                        .astype(
                            bool
                        ),
                        column,
                    ] = pd.NA

    if (
        'departamentos' in silver_datasets
        and 'department_code' in data.columns
    ):
        department_lookup = _safe_master_lookup(
            dataframe=silver_datasets['departamentos'],
            key='department_code',
            value_columns=[
                'department_name',
                'department_short_name',
            ],
            dataset_name='departamentos',
            verbose=verbose,
        )

        if not department_lookup.empty:
            department_lookup = department_lookup.rename(
                columns={
                    column: (
                        f'{column}__departments_master'
                        if column != 'department_code'
                        else column
                    )
                    for column in department_lookup.columns
                }
            )

            data = data.merge(
                department_lookup,
                on='department_code',
                how='left',
                validate='many_to_one',
            )

            data = _coalesce_columns(
                data,
                primary='department_name',
                secondary='department_name__departments_master',
                output='department_name',
            )

    return data


def build_weekly_article_master(
    silver_datasets: dict[str, pd.DataFrame],
    rules: IntegrationRules | None = None,
    verbose: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Build the Gold article-period table used by future demand models.

    Despite the filename 'semanal', the source period length is preserved in
    `period_days`. This is important because some reports can be incomplete or
    cover a non-standard number of days.

    One row represents:
        report period x article

    No daily product sales are invented.
    """
    rules = (
        rules
        if rules is not None
        else IntegrationRules()
    )

    rules.validate()

    _print_subheader(
        'Build article-period Gold master',
        verbose=verbose,
    )

    if 'ventas' not in silver_datasets:
        _print_message(
            'Article-period Gold master: `ventas` is not available -> skipped.',
            level='WARNING',
            verbose=verbose,
        )

        return (
            pd.DataFrame(),
            {
                'rows': 0,
                'columns': 0,
                'reason': 'ventas_not_available',
            },
        )

    data = silver_datasets[
        'ventas'
    ].copy()

    required_columns = [
        'report_start',
        'report_end',
        'article_code',
        'units',
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_required:
        raise ValueError(
            'Cannot build article-period Gold master. '
            f'Missing required columns: {missing_required}'
        )

    data['report_start'] = _to_normalized_date(
        data['report_start']
    )

    data['report_end'] = _to_normalized_date(
        data['report_end']
    )

    data['units'] = _safe_numeric(
        data['units']
    )

    if 'amount' in data.columns:
        data['amount'] = _safe_numeric(
            data['amount']
        )

    if (
        not rules.keep_invitations
        and 'es_invitacion' in data.columns
    ):
        before = len(data)

        data = data.loc[
            ~data[
                'es_invitacion'
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

        _print_message(
            f'Article-period master: {before - len(data):,} invitation rows '
            'excluded by configuration.',
            verbose=verbose,
        )

    data['period_days'] = (
        data[
            'report_end'
        ]
        - data[
            'report_start'
        ]
    ).dt.days + 1

    expected_period_days = _infer_expected_period_days(
        data[
            'period_days'
        ]
    )

    if expected_period_days is not None:
        data[
            'period_is_complete'
        ] = (
            data[
                'period_days'
            ]
            .eq(
                expected_period_days
            )
        )
    else:
        data[
            'period_is_complete'
        ] = pd.Series(
            pd.NA,
            index=data.index,
            dtype='boolean',
        )

    # Backward-compatible field for consumers that explicitly care about
    # calendar weeks. The general completeness flag above is inferred from
    # the source rather than hard-coded to seven days.
    data['period_is_complete_week'] = (
        data[
            'period_days'
        ]
        .eq(
            7
        )
    )

    duplicate_key_columns = [
        'report_start',
        'report_end',
        'article_code',
    ]

    # Mutable descriptive metadata (article name/department) is deliberately
    # NOT part of the analytical key. The stable entity key is article_code.
    group_columns = duplicate_key_columns.copy()

    if rules.aggregate_duplicate_article_period_rows:
        duplicated = data.duplicated(
            subset=duplicate_key_columns,
            keep=False,
        )

        duplicate_rows = int(
            duplicated.sum()
        )

        if duplicate_rows:
            _print_message(
                f'Article-period integration: {duplicate_rows:,} rows share '
                f'the key {duplicate_key_columns}. They will be aggregated '
                'before Gold creation.',
                level='WARNING',
                verbose=verbose,
            )

        aggregations: dict[str, Any] = {
            'units': 'sum',
            'period_days': 'first',
            'period_is_complete': 'first',
            'period_is_complete_week': 'first',
        }

        if 'amount' in data.columns:
            aggregations['amount'] = 'sum'

        if 'es_invitacion' in data.columns:
            aggregations['es_invitacion'] = 'any'

        metadata_columns = [
            column
            for column in [
                'article_name',
                'department_code',
                'department_name',
            ]
            if column in data.columns
        ]

        for column in metadata_columns:
            aggregations[
                column
            ] = _consistent_non_null_value

        # Keep quality flags conservatively with ANY.
        for column in data.columns:
            if (
                column.endswith('__outlier')
                or column.endswith('__invalid')
                or column.endswith('__negative')
                or column.startswith('quality_')
            ):
                if column not in aggregations:
                    aggregations[column] = 'any'

        master = (
            data.groupby(
                group_columns,
                dropna=False,
                as_index=False,
            )
            .agg(
                aggregations
            )
        )

        # Metadata conflicts are reported explicitly rather than creating
        # multiple analytical rows for the same article-period key.
        for column in metadata_columns:
            conflicts = (
                data.groupby(
                    group_columns,
                    dropna=False,
                )[column]
                .nunique(
                    dropna=True
                )
                .gt(
                    1
                )
                .rename(
                    f'{column}__period_conflict'
                )
                .reset_index()
            )

            if conflicts[
                f'{column}__period_conflict'
            ].any():
                master = master.merge(
                    conflicts,
                    on=group_columns,
                    how='left',
                    validate='one_to_one',
                )

                _print_message(
                    f'Article-period integration: conflicting {column!r} '
                    'metadata detected within at least one stable '
                    'article-period key. Conflicts are flagged and the '
                    'descriptive value is left missing.',
                    level='WARNING',
                    verbose=verbose,
                )

    else:
        master = data.copy()

    master = enrich_article_period_master(
        dataframe=master,
        silver_datasets=silver_datasets,
        verbose=verbose,
    )

    master['period_id'] = (
        master[
            'report_start'
        ].dt.strftime(
            '%Y-%m-%d'
        )
        + '__'
        + master[
            'report_end'
        ].dt.strftime(
            '%Y-%m-%d'
        )
    )

    master = (
        master
        .sort_values(
            [
                'report_start',
                'report_end',
                'article_code',
            ]
        )
        .reset_index(
            drop=True
        )
    )

    _assert_unique_key(
        master,
        key=[
            'report_start',
            'report_end',
            'article_code',
        ],
        dataset_label='final article-period Gold master',
    )

    summary = {
        'rows': len(master),
        'columns': len(master.columns),
        'periods': int(
            master[
                [
                    'report_start',
                    'report_end',
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        'articles': int(
            master[
                'article_code'
            ]
            .nunique(
                dropna=True
            )
        ),
        'expected_period_days': expected_period_days,
        'complete_period_rows': int(
            master[
                'period_is_complete'
            ]
            .fillna(False)
            .sum()
        ),
        'complete_week_rows': int(
            master[
                'period_is_complete_week'
            ]
            .fillna(False)
            .sum()
        ),
        'period_days_min': _json_safe(
            master[
                'period_days'
            ].min()
        ),
        'period_days_max': _json_safe(
            master[
                'period_days'
            ].max()
        ),
        'invitations_kept': rules.keep_invitations,
    }

    _print_message(
        f'Article-period Gold master created: '
        f'{len(master):,} rows | '
        f'{summary["periods"]:,} periods | '
        f'{summary["articles"]:,} articles.',
        verbose=verbose,
    )

    return (
        master,
        summary,
    )


# =============================================================================
# COLUMN PROVENANCE
# =============================================================================

def build_column_provenance(
    gold_datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a readable provenance catalogue for major Gold columns.

    The catalogue is intentionally rule-based and transparent; it is not used
    to transform data.
    """
    rows = []

    for gold_name, dataframe in gold_datasets.items():
        for column in dataframe.columns:
            source = 'derived'
            temporal_availability = 'depends_on_source'
            description = ''

            if column == 'date':
                source = 'calendar'
                temporal_availability = 'known_in_advance'
                description = 'Calendar date.'

            elif column in {
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
                source = 'calendar'
                temporal_availability = 'known_in_advance'
                description = 'Calendar-derived feature.'

            elif column.startswith(
                (
                    'num_tickets',
                    'facturacion',
                    'ticket_',
                    'num_comprobantes',
                )
            ):
                source = 'tickets_or_invoices'
                temporal_availability = 'realized_after_service'
                description = 'Historical realized sales/ticket information.'

            elif column.startswith(
                (
                    'reservas_',
                    'comensales_',
                    'tamano_grupo_',
                    'lead_time_',
                    'no_show_rate',
                    'cancel_rate',
                )
            ):
                source = 'reservas'
                if (
                    'no_show' in column
                    or 'cancel' in column
                    or 'completed' in column
                ):
                    temporal_availability = 'realized_outcome'
                else:
                    temporal_availability = 'mixed_check_prediction_cutoff'
                description = 'Reservation-derived historical information.'

            elif column in {
                'es_festivo',
                'num_festivos',
                'festivo_nombre',
            }:
                source = 'festivos'
                temporal_availability = 'known_in_advance'
                description = 'Holiday/calendar context.'

            elif (
                column.startswith('event_')
                or column.startswith('eventos_')
                or column in {
                    'tiene_evento',
                    'num_eventos',
                }
            ):
                source = 'eventos'
                temporal_availability = 'known_in_advance_if_event_is_scheduled'
                description = 'Scheduled-event context.'

            elif gold_name == 'tabla_maestra_semanal_articulos':
                if column in {
                    'units',
                    'amount',
                    'es_invitacion',
                }:
                    source = 'ventas'
                    temporal_availability = 'realized_after_period'
                    description = 'Observed article-period outcome.'
                elif column in {
                    'report_start',
                    'report_end',
                    'period_days',
                    'period_is_complete_week',
                    'period_id',
                }:
                    source = 'ventas_period_metadata'
                    temporal_availability = 'known_from_report_period'
                    description = 'Article-sales report period metadata.'
                elif (
                    'article' in column
                    or 'department' in column
                    or column == 'price'
                ):
                    source = 'master_data'
                    temporal_availability = 'known_in_advance'
                    description = 'Article/department master metadata.'

            rows.append(
                {
                    'gold_dataset': gold_name,
                    'column': column,
                    'source': source,
                    'temporal_availability': temporal_availability,
                    'description': description,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# INTEGRATION REPORTS
# =============================================================================

def save_integration_reports(
    silver_datasets: dict[str, pd.DataFrame],
    gold_datasets: dict[str, pd.DataFrame],
    daily_summary: dict[str, Any] | None,
    weekly_summary: dict[str, Any] | None,
    rules: IntegrationRules,
    paths: IntegrationPaths,
) -> dict[str, Path]:
    """Save integration coverage, provenance and global JSON reports."""
    reports_dir = _ensure_directory(
        paths.reports_dir
    )

    saved = {}

    coverage = build_source_coverage_report(
        silver_datasets
    )

    coverage_path = (
        reports_dir
        / 'integration_source_coverage.csv'
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding='utf-8',
    )

    saved['source_coverage'] = (
        coverage_path
    )

    if rules.save_column_provenance:
        provenance = build_column_provenance(
            gold_datasets
        )

        provenance_path = (
            reports_dir
            / 'gold_column_provenance.csv'
        )

        provenance.to_csv(
            provenance_path,
            index=False,
            encoding='utf-8',
        )

        saved['column_provenance'] = (
            provenance_path
        )

    if rules.save_integration_report:
        report = {
            'rules': asdict(
                rules
            ),
            'silver_datasets_available': sorted(
                silver_datasets.keys()
            ),
            'gold_datasets_created': {
                name: {
                    'rows': len(dataframe),
                    'columns': len(dataframe.columns),
                }
                for name, dataframe
                in gold_datasets.items()
            },
            'daily_summary': daily_summary,
            'article_period_summary': weekly_summary,
        }

        report_path = (
            reports_dir
            / 'integration_report.json'
        )

        with open(
            report_path,
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

        saved['integration_report'] = (
            report_path
        )

    return saved


# =============================================================================
# GOLD PERSISTENCE
# =============================================================================

def save_gold_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    gold_dir: str | Path,
    verbose: bool = True,
) -> Path:
    """Save one Gold dataset as Parquet."""
    output_dir = _ensure_directory(
        gold_dir
    )

    filename = GOLD_FILENAMES.get(
        dataset_name,
        f'{dataset_name}.parquet',
    )

    output_path = (
        output_dir
        / filename
    )

    try:
        dataframe.to_parquet(
            output_path,
            index=False,
        )

    except ImportError as exc:
        raise ImportError(
            'Saving Gold parquet files requires pyarrow or fastparquet. '
            'The repository requirements should include pyarrow. '
            'Install with: python -m pip install pyarrow'
        ) from exc

    _print_message(
        f'GOLD | {dataset_name}: '
        f'{len(dataframe):,} rows x '
        f'{len(dataframe.columns):,} columns '
        f'-> {output_path}',
        verbose=verbose,
    )

    return output_path


def save_gold_datasets(
    datasets: dict[str, pd.DataFrame],
    gold_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Path]:
    """Persist all created Gold datasets."""
    paths = {}

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        paths[
            dataset_name
        ] = save_gold_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
            gold_dir=gold_dir,
            verbose=verbose,
        )

    return paths



def _compact_source_coverage_console(
    coverage: pd.DataFrame,
    verbose: bool = True,
) -> None:
    """Print one compact line for Silver source coverage."""
    if not verbose or coverage.empty:
        return

    total_rows = int(
        coverage[
            'rows'
        ].sum()
    )
    total_missing = int(
        coverage[
            'missing_cells'
        ].sum()
    )

    date_starts = pd.to_datetime(
        coverage[
            'date_start'
        ],
        errors='coerce',
    )
    date_ends = pd.to_datetime(
        coverage[
            'date_end'
        ],
        errors='coerce',
    )

    valid_start = date_starts.dropna()
    valid_end = date_ends.dropna()

    date_text = ''

    if (
        not valid_start.empty
        and not valid_end.empty
    ):
        date_text = (
            f' | overall coverage='
            f'{valid_start.min().date()} -> '
            f'{valid_end.max().date()}'
        )

    print(
        '[INFO] Silver inputs: '
        f'{len(coverage):,} datasets | '
        f'{total_rows:,} total rows | '
        f'{total_missing:,} missing cells'
        f'{date_text}.'
    )


def _compact_daily_gold_console(
    summary: dict[str, Any] | None,
    verbose: bool = True,
) -> None:
    """Print one concise line for the daily Gold table."""
    if not verbose or not summary:
        return

    rows = int(
        summary.get(
            'rows',
            0,
        )
    )
    columns = int(
        summary.get(
            'columns',
            0,
        )
    )
    sources = [
        item.get(
            'source'
        )
        for item in summary.get(
            'sources',
            []
        )
        if item.get(
            'source'
        )
    ]

    start = summary.get(
        'date_start'
    )
    end = summary.get(
        'date_end'
    )

    date_text = ''

    if (
        start is not None
        and end is not None
        and not pd.isna(
            start
        )
        and not pd.isna(
            end
        )
    ):
        date_text = (
            f' | {pd.Timestamp(start).date()}'
            f' -> {pd.Timestamp(end).date()}'
        )

    source_text = (
        ', '.join(
            sources
        )
        if sources
        else 'none'
    )

    print(
        '[INFO] Daily Gold: '
        f'{rows:,} rows x {columns:,} cols'
        f'{date_text} | sources={source_text}.'
    )


def _compact_article_gold_console(
    summary: dict[str, Any] | None,
    verbose: bool = True,
) -> None:
    """Print one concise line for the article-period Gold table."""
    if not verbose or not summary:
        return

    rows = int(
        summary.get(
            'rows',
            0,
        )
    )
    columns = int(
        summary.get(
            'columns',
            0,
        )
    )
    periods = int(
        summary.get(
            'periods',
            0,
        )
    )
    articles = int(
        summary.get(
            'articles',
            0,
        )
    )
    expected_days = summary.get(
        'expected_period_days'
    )

    period_text = (
        f' | expected period={int(expected_days)} days'
        if expected_days is not None
        and not pd.isna(
            expected_days
        )
        else ''
    )

    print(
        '[INFO] Article-period Gold: '
        f'{rows:,} rows x {columns:,} cols | '
        f'{periods:,} periods | '
        f'{articles:,} articles'
        f'{period_text}.'
    )


# =============================================================================
# MASTER INTEGRATION FUNCTION
# =============================================================================

def run_integration(
    silver_datasets: dict[str, pd.DataFrame],
    paths: IntegrationPaths,
    rules: IntegrationRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Master Silver -> Gold integration entry point.

    This is the function that the future `analysis_prediction.py` main file
    should normally call immediately after `run_depuration()`.

    Current Gold outputs
    --------------------
    1. `tabla_maestra_diaria`
       One row per calendar day, integrating compatible daily sources.

    2. `tabla_maestra_semanal_articulos`
       One row per article and sales-report period. The original weekly/periodic
       granularity is preserved.

    Notes
    -----
    Gold integration is intentionally implemented in its own module because
    data cleaning and multi-source integration are conceptually different.
    The future main can nevertheless keep both under a single
    `DO_DEPURATION=True` user flag:

        if DO_DEPURATION:
            silver = run_depuration(...)
            gold = run_integration(...)

    This preserves a simple user interface while keeping the code modular.

    Console
    -------
    `rules.console_detail='summary'` is the default and prints only compact
    Silver coverage plus one line per Gold output. Setting it to 'detailed'
    restores the previous source-by-source merge diagnostics.
    """
    rules = (
        rules
        if rules is not None
        else IntegrationRules()
    )

    rules.validate()

    detailed_console = (
        verbose
        and rules.console_detail == 'detailed'
    )

    _ensure_directory(
        paths.gold_dir
    )

    _ensure_directory(
        paths.reports_dir
    )

    _print_header(
        'SILVER -> INTEGRATION -> GOLD',
        verbose=verbose,
    )

    if not silver_datasets:
        raise ValueError(
            'No Silver datasets were supplied to run_integration().'
        )

    coverage = build_source_coverage_report(
        silver_datasets
    )

    if rules.console_detail == 'detailed':
        if verbose and not coverage.empty:
            print(
                '\nSILVER SOURCE COVERAGE'
            )
            print(
                coverage.to_string(
                    index=False
                )
            )

    else:
        _compact_source_coverage_console(
            coverage=coverage,
            verbose=verbose,
        )

    gold_datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    daily_summary: dict[
        str,
        Any,
    ] | None = None

    weekly_summary: dict[
        str,
        Any,
    ] | None = None

    if rules.build_daily_master:
        daily_master, daily_summary = (
            build_daily_master(
                silver_datasets=silver_datasets,
                rules=rules,
                verbose=detailed_console,
            )
        )

        if not daily_master.empty:
            gold_datasets[
                'tabla_maestra_diaria'
            ] = daily_master

        else:
            _print_message(
                'Daily Gold master was not created because no compatible '
                'daily Silver sources were available.',
                level='WARNING',
                verbose=verbose,
            )

    if rules.build_weekly_article_master:
        weekly_master, weekly_summary = (
            build_weekly_article_master(
                silver_datasets=silver_datasets,
                rules=rules,
                verbose=detailed_console,
            )
        )

        if not weekly_master.empty:
            gold_datasets[
                'tabla_maestra_semanal_articulos'
            ] = weekly_master

        else:
            _print_message(
                'Article-period Gold master was not created.',
                level='WARNING',
                verbose=verbose,
            )

    if not gold_datasets:
        raise RuntimeError(
            'Integration completed without creating any Gold dataset.'
        )

    save_gold_datasets(
        datasets=gold_datasets,
        gold_dir=paths.gold_dir,
        verbose=detailed_console,
    )

    report_paths = save_integration_reports(
        silver_datasets=silver_datasets,
        gold_datasets=gold_datasets,
        daily_summary=daily_summary,
        weekly_summary=weekly_summary,
        rules=rules,
        paths=paths,
    )

    if rules.console_detail == 'detailed':
        _print_message(
            f'Integration completed: {len(gold_datasets)} Gold datasets created.',
            verbose=verbose,
        )

        if report_paths:
            _print_message(
                'Integration reports: '
                + ', '.join(
                    str(path)
                    for path
                    in report_paths.values()
                ),
                verbose=verbose,
            )

    else:
        if daily_summary is not None:
            _compact_daily_gold_console(
                summary=daily_summary,
                verbose=verbose,
            )

        if weekly_summary is not None:
            _compact_article_gold_console(
                summary=weekly_summary,
                verbose=verbose,
            )

        _print_message(
            f'Integration completed: {len(gold_datasets)} Gold datasets created '
            f'| full reports: {paths.reports_dir}',
            verbose=verbose,
        )

    return gold_datasets
