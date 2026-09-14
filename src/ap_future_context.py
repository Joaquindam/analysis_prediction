"""
Future context builder for operational hospitality forecasts.

The module creates one daily row for a future forecast horizon using only
information that can legitimately be known before the target period:

* scheduled events;
* public holidays;
* optional weather forecasts supplied by the caller.

It deliberately does NOT use future observed weather, transactions,
reservations, customer counts or other same-period outcomes.

The resulting frame is designed to be passed directly as
`future_daily_context` to `run_operational_forecast`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FutureContextPaths:
    tables_dir: Path
    reports_dir: Path


@dataclass
class FutureContextRules:
    """
    General rules for building known future daily context.

    Column aliases make the builder tolerant to raw-like and Silver-like
    schemas instead of coupling it to one restaurant export.
    """

    event_date_candidates: tuple[str, ...] = (
        'date',
        'fecha',
        'event_date',
        'fecha_evento',
    )
    event_start_candidates: tuple[str, ...] = (
        'event_start',
        'start_date',
        'fecha_inicio',
        'date_start',
    )
    event_end_candidates: tuple[str, ...] = (
        'event_end',
        'end_date',
        'fecha_fin',
        'date_end',
    )
    event_id_candidates: tuple[str, ...] = (
        'event_id',
        'id_evento',
        'evento_id',
        'id',
    )
    event_name_candidates: tuple[str, ...] = (
        'event_name',
        'nombre_evento',
        'evento_nombre',
        'name',
    )
    event_intensity_candidates: tuple[str, ...] = (
        'event_intensity',
        'event_intensity_score',
        'intensidad_sugerida',
        'intensidad',
        'intensity',
    )

    holiday_date_candidates: tuple[str, ...] = (
        'date',
        'fecha',
        'holiday_date',
        'fecha_festivo',
    )
    holiday_flag_candidates: tuple[str, ...] = (
        'es_festivo',
        'is_holiday',
        'holiday_flag',
    )
    holiday_name_candidates: tuple[str, ...] = (
        'festivo_nombre',
        'holiday_name',
        'nombre_festivo',
        'name',
    )

    weather_date_candidates: tuple[str, ...] = (
        'date',
        'fecha',
        'time',
        'datetime',
    )
    weather_column_map: dict[str, str] = field(
        default_factory=dict
    )

    include_events: bool = True
    include_holidays: bool = True
    include_weather_forecast: bool = True

    save_tables: bool = True
    save_report: bool = True

    def validate(self) -> None:
        if not (
            self.include_events
            or self.include_holidays
            or self.include_weather_forecast
        ):
            raise ValueError(
                'At least one future-context source must be enabled.'
            )


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def _to_datetime(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors='coerce',
    )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors='coerce',
    )


def _safe_bool(
    series: pd.Series,
) -> pd.Series:
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


def _resolve_column(
    dataframe: pd.DataFrame | None,
    candidates: tuple[str, ...],
) -> str | None:
    if dataframe is None:
        return None

    lookup = {
        str(
            column
        ).strip().lower(): str(
            column
        )
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[
                key
            ]

    return None


def _json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): _json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_safe(
                item
            )
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

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if not isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
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


def _ensure_dirs(
    paths: FutureContextPaths,
) -> tuple[Path, Path]:
    tables_dir = Path(
        paths.tables_dir
    )
    reports_dir = Path(
        paths.reports_dir
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return tables_dir, reports_dir


# =============================================================================
# FORECAST HORIZON
# =============================================================================


def _article_period_table(
    article_gold: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        'report_start',
        'report_end',
    }

    missing = required.difference(
        article_gold.columns
    )

    if missing:
        raise KeyError(
            'Article Gold is missing required period columns: '
            f'{sorted(missing)}'
        )

    periods = article_gold[
        [
            'report_start',
            'report_end',
        ]
        + [
            column
            for column in [
                'period_days',
                'period_is_complete',
            ]
            if column in article_gold.columns
        ]
    ].copy()

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

    if 'period_is_complete' in periods.columns:
        periods[
            'period_is_complete'
        ] = _safe_bool(
            periods[
                'period_is_complete'
            ]
        )

    aggregation: dict[str, Any] = {}

    if 'period_days' in periods.columns:
        aggregation[
            'period_days'
        ] = 'first'

    if 'period_is_complete' in periods.columns:
        aggregation[
            'period_is_complete'
        ] = 'all'

    if aggregation:
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
                aggregation
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


def infer_next_forecast_period(
    article_gold: pd.DataFrame,
) -> dict[str, Any]:
    """
    Infer the next complete reporting period from historical cadence.

    Explicitly incomplete trailing periods are not used as cadence anchors.
    """
    periods = _article_period_table(
        article_gold
    )

    if periods.empty:
        raise ValueError(
            'Article Gold contains no report periods.'
        )

    complete = periods.copy()

    if 'period_is_complete' in complete.columns:
        complete_mask = _safe_bool(
            complete[
                'period_is_complete'
            ]
        ).fillna(
            False
        )

        if complete_mask.any():
            complete = complete.loc[
                complete_mask
            ].copy()

    if complete.empty:
        raise ValueError(
            'No historical period is available to infer forecast cadence.'
        )

    if 'period_days' in complete.columns:
        duration = _to_numeric(
            complete[
                'period_days'
            ]
        )
    else:
        duration = (
            complete[
                'report_end'
            ]
            - complete[
                'report_start'
            ]
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
            'Could not infer a positive report-period duration.'
        )

    duration_mode = duration.mode()

    expected_days = int(
        duration_mode.iloc[
            0
        ]
        if not duration_mode.empty
        else round(
            float(
                duration.median()
            )
        )
    )

    starts = (
        complete[
            'report_start'
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    start_deltas = starts.diff().dt.days
    start_deltas = start_deltas.loc[
        start_deltas.gt(
            0
        )
    ]

    if start_deltas.empty:
        cadence_days = expected_days
    else:
        cadence_mode = start_deltas.mode()

        cadence_days = int(
            cadence_mode.iloc[
                0
            ]
            if not cadence_mode.empty
            else round(
                float(
                    start_deltas.median()
                )
            )
        )

    cadence_days = max(
        1,
        cadence_days,
    )

    latest_observed_end = pd.Timestamp(
        periods[
            'report_end'
        ].max()
    )

    latest_complete = complete.sort_values(
        [
            'report_start',
            'report_end',
        ]
    ).iloc[
        -1
    ]

    future_start = pd.Timestamp(
        latest_complete[
            'report_start'
        ]
    )

    while future_start <= latest_observed_end:
        future_start += pd.Timedelta(
            days=cadence_days
        )

    future_end = (
        future_start
        + pd.Timedelta(
            days=expected_days - 1
        )
    )

    return {
        'forecast_start': future_start,
        'forecast_end': future_end,
        'period_days': expected_days,
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


def _resolve_horizon(
    article_gold: pd.DataFrame | None,
    forecast_start: str | pd.Timestamp | None,
    forecast_end: str | pd.Timestamp | None,
) -> dict[str, Any]:
    if (
        forecast_start is not None
        or forecast_end is not None
    ):
        if (
            forecast_start is None
            or forecast_end is None
        ):
            raise ValueError(
                'forecast_start and forecast_end must be supplied together.'
            )

        start = pd.Timestamp(
            forecast_start
        )
        end = pd.Timestamp(
            forecast_end
        )

        if end < start:
            raise ValueError(
                'forecast_end cannot be earlier than forecast_start.'
            )

        return {
            'forecast_start': start,
            'forecast_end': end,
            'period_days': int(
                (
                    end
                    - start
                ).days
                + 1
            ),
            'cadence_days': None,
            'latest_observed_end': None,
            'latest_complete_start': None,
            'latest_complete_end': None,
            'horizon_source': 'explicit',
        }

    if article_gold is None:
        raise ValueError(
            'Provide article_gold or an explicit forecast_start/forecast_end.'
        )

    horizon = infer_next_forecast_period(
        article_gold
    )
    horizon[
        'horizon_source'
    ] = 'article_gold_cadence'

    return horizon


# =============================================================================
# EVENTS
# =============================================================================


def _expand_events(
    events: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rules: FutureContextRules,
) -> pd.DataFrame:
    columns = [
        'date',
        'event_id',
        'event_name',
        'event_intensity',
    ]

    if (
        events is None
        or events.empty
        or not rules.include_events
    ):
        return pd.DataFrame(
            columns=columns
        )

    data = events.copy()

    date_column = _resolve_column(
        data,
        rules.event_date_candidates,
    )
    start_column = _resolve_column(
        data,
        rules.event_start_candidates,
    )
    end_column = _resolve_column(
        data,
        rules.event_end_candidates,
    )
    id_column = _resolve_column(
        data,
        rules.event_id_candidates,
    )
    name_column = _resolve_column(
        data,
        rules.event_name_candidates,
    )
    intensity_column = _resolve_column(
        data,
        rules.event_intensity_candidates,
    )

    rows: list[dict[str, Any]] = []

    if date_column is not None:
        event_dates = _to_datetime(
            data[
                date_column
            ]
        )

        for row_position, (
            index,
            row,
        ) in enumerate(
            data.iterrows()
        ):
            date = event_dates.loc[
                index
            ]

            if pd.isna(
                date
            ):
                continue

            date = pd.Timestamp(
                date
            ).normalize()

            if not (
                start
                <= date
                <= end
            ):
                continue

            rows.append(
                {
                    'date': date,
                    'event_id': (
                        row[
                            id_column
                        ]
                        if id_column is not None
                        else f'event_{row_position}'
                    ),
                    'event_name': (
                        row[
                            name_column
                        ]
                        if name_column is not None
                        else pd.NA
                    ),
                    'event_intensity': (
                        row[
                            intensity_column
                        ]
                        if intensity_column is not None
                        else np.nan
                    ),
                }
            )

    elif start_column is not None:
        starts = _to_datetime(
            data[
                start_column
            ]
        )

        if end_column is not None:
            ends = _to_datetime(
                data[
                    end_column
                ]
            )
        else:
            ends = starts.copy()

        for row_position, (
            index,
            row,
        ) in enumerate(
            data.iterrows()
        ):
            event_start = starts.loc[
                index
            ]
            event_end = ends.loc[
                index
            ]

            if pd.isna(
                event_start
            ):
                continue

            event_start = pd.Timestamp(
                event_start
            ).normalize()

            if pd.isna(
                event_end
            ):
                event_end = event_start
            else:
                event_end = pd.Timestamp(
                    event_end
                ).normalize()

            if event_end < event_start:
                event_end = event_start

            overlap_start = max(
                start,
                event_start,
            )
            overlap_end = min(
                end,
                event_end,
            )

            if overlap_end < overlap_start:
                continue

            for date in pd.date_range(
                overlap_start,
                overlap_end,
                freq='D',
            ):
                rows.append(
                    {
                        'date': date,
                        'event_id': (
                            row[
                                id_column
                            ]
                            if id_column is not None
                            else f'event_{row_position}'
                        ),
                        'event_name': (
                            row[
                                name_column
                            ]
                            if name_column is not None
                            else pd.NA
                        ),
                        'event_intensity': (
                            row[
                                intensity_column
                            ]
                            if intensity_column is not None
                            else np.nan
                        ),
                    }
                )

    else:
        raise KeyError(
            'Events require either a daily date column or a start-date column. '
            f'Available columns: {list(data.columns)}'
        )

    expanded = pd.DataFrame(
        rows,
        columns=columns,
    )

    if expanded.empty:
        return expanded

    expanded[
        'date'
    ] = _to_datetime(
        expanded[
            'date'
        ]
    )
    expanded[
        'event_intensity'
    ] = _to_numeric(
        expanded[
            'event_intensity'
        ]
    )

    # If Silver is already expanded daily, this prevents duplicate source rows
    # from counting the same event twice on one day.
    expanded = expanded.drop_duplicates(
        subset=[
            'date',
            'event_id',
        ],
        keep='first',
    )

    return expanded.sort_values(
        [
            'date',
            'event_id',
        ]
    ).reset_index(
        drop=True
    )


def _aggregate_events(
    expanded_events: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    if expanded_events.empty:
        result = calendar.copy()
        result[
            'tiene_evento'
        ] = False
        result[
            'num_eventos'
        ] = 0
        result[
            'event_intensity_mean'
        ] = 0.0
        result[
            'event_intensity_max'
        ] = 0.0

        return result

    grouped = (
        expanded_events
        .groupby(
            'date',
            as_index=False,
        )
        .agg(
            num_eventos=(
                'event_id',
                'nunique',
            ),
            event_intensity_mean=(
                'event_intensity',
                'mean',
            ),
            event_intensity_max=(
                'event_intensity',
                'max',
            ),
        )
    )

    grouped[
        'tiene_evento'
    ] = grouped[
        'num_eventos'
    ].gt(
        0
    )

    result = calendar.merge(
        grouped,
        on='date',
        how='left',
        validate='one_to_one',
    )

    result[
        'num_eventos'
    ] = _to_numeric(
        result[
            'num_eventos'
        ]
    ).fillna(
        0
    ).astype(
        int
    )
    result[
        'tiene_evento'
    ] = result[
        'num_eventos'
    ].gt(
        0
    )

    no_event = ~result[
        'tiene_evento'
    ]

    for column in [
        'event_intensity_mean',
        'event_intensity_max',
    ]:
        result.loc[
            no_event,
            column,
        ] = 0.0

    return result


# =============================================================================
# HOLIDAYS
# =============================================================================


def _prepare_holidays(
    holidays: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rules: FutureContextRules,
) -> pd.DataFrame:
    columns = [
        'date',
        'holiday_name',
    ]

    if (
        holidays is None
        or holidays.empty
        or not rules.include_holidays
    ):
        return pd.DataFrame(
            columns=columns
        )

    data = holidays.copy()

    date_column = _resolve_column(
        data,
        rules.holiday_date_candidates,
    )

    if date_column is None:
        raise KeyError(
            'Holiday data requires a date column. '
            f'Available columns: {list(data.columns)}'
        )

    flag_column = _resolve_column(
        data,
        rules.holiday_flag_candidates,
    )
    name_column = _resolve_column(
        data,
        rules.holiday_name_candidates,
    )

    data[
        '__date'
    ] = _to_datetime(
        data[
            date_column
        ]
    ).dt.normalize()

    data = data.loc[
        data[
            '__date'
        ].between(
            start,
            end,
            inclusive='both',
        )
    ].copy()

    if flag_column is not None:
        holiday_mask = _safe_bool(
            data[
                flag_column
            ]
        ).fillna(
            False
        )

        data = data.loc[
            holiday_mask
        ].copy()

    result = pd.DataFrame(
        {
            'date': data[
                '__date'
            ],
            'holiday_name': (
                data[
                    name_column
                ]
                if name_column is not None
                else pd.Series(
                    pd.NA,
                    index=data.index,
                    dtype='string',
                )
            ),
        }
    )

    return result.drop_duplicates(
        subset=[
            'date',
            'holiday_name',
        ]
    ).sort_values(
        'date'
    ).reset_index(
        drop=True
    )


def _aggregate_holidays(
    holidays: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    if holidays.empty:
        result = calendar.copy()
        result[
            'es_festivo'
        ] = False
        result[
            'num_festivos'
        ] = 0

        return result

    grouped = (
        holidays
        .groupby(
            'date',
            as_index=False,
        )
        .size()
        .rename(
            columns={
                'size': 'num_festivos',
            }
        )
    )

    result = calendar.merge(
        grouped,
        on='date',
        how='left',
        validate='one_to_one',
    )

    result[
        'num_festivos'
    ] = _to_numeric(
        result[
            'num_festivos'
        ]
    ).fillna(
        0
    ).astype(
        int
    )
    result[
        'es_festivo'
    ] = result[
        'num_festivos'
    ].gt(
        0
    )

    return result


# =============================================================================
# WEATHER FORECAST
# =============================================================================


WEATHER_TOKENS = (
    'temperature',
    'temperatura',
    'apparent',
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


def _is_weather_column(
    column: str,
) -> bool:
    name = str(
        column
    ).lower()

    return any(
        token in name
        for token in WEATHER_TOKENS
    )


def _prepare_weather_forecast(
    weather_forecast: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rules: FutureContextRules,
) -> tuple[pd.DataFrame, list[str]]:
    if (
        weather_forecast is None
        or weather_forecast.empty
        or not rules.include_weather_forecast
    ):
        return pd.DataFrame(
            columns=[
                'date',
            ]
        ), []

    data = weather_forecast.copy()

    date_column = _resolve_column(
        data,
        rules.weather_date_candidates,
    )

    if date_column is None:
        raise KeyError(
            'Weather forecast requires a date/time column. '
            f'Available columns: {list(data.columns)}'
        )

    data[
        'date'
    ] = _to_datetime(
        data[
            date_column
        ]
    ).dt.normalize()

    data = data.loc[
        data[
            'date'
        ].between(
            start,
            end,
            inclusive='both',
        )
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                'date',
            ]
        ), []

    rename_map = {
        source: target
        for source, target in rules.weather_column_map.items()
        if source in data.columns
    }

    if rename_map:
        data = data.rename(
            columns=rename_map
        )

    weather_columns = [
        column
        for column in data.columns
        if column != 'date'
        and column != date_column
        and _is_weather_column(
            column
        )
    ]

    if not weather_columns:
        return pd.DataFrame(
            columns=[
                'date',
            ]
        ), []

    aggregations: dict[str, str] = {}

    for column in weather_columns:
        if pd.api.types.is_numeric_dtype(
            data[
                column
            ]
        ):
            name = str(
                column
            ).lower()

            if any(
                token in name
                for token in (
                    'precipitation',
                    'precipitacion',
                    'rain',
                    'lluvia',
                    'snow',
                    'nieve',
                )
            ):
                aggregations[
                    column
                ] = 'sum'
            else:
                aggregations[
                    column
                ] = 'mean'

        elif pd.api.types.is_bool_dtype(
            data[
                column
            ]
        ):
            aggregations[
                column
            ] = 'max'

    if not aggregations:
        return pd.DataFrame(
            columns=[
                'date',
            ]
        ), []

    daily = (
        data
        .groupby(
            'date',
            as_index=False,
        )
        .agg(
            aggregations
        )
    )

    return daily, list(
        aggregations
    )


# =============================================================================
# MASTER BUILDER
# =============================================================================


def build_future_daily_context(
    events: pd.DataFrame | None,
    holidays: pd.DataFrame | None,
    paths: FutureContextPaths,
    article_gold: pd.DataFrame | None = None,
    forecast_start: str | pd.Timestamp | None = None,
    forecast_end: str | pd.Timestamp | None = None,
    weather_forecast: pd.DataFrame | None = None,
    rules: FutureContextRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build a leakage-safe future daily context table.

    Use either:
      * `article_gold` to infer the next aligned report period, or
      * explicit `forecast_start` + `forecast_end`.

    `weather_forecast` must be an actual forecast available at prediction time.
    Never pass subsequently observed weather to reconstruct past forecasts.
    """
    rules = (
        FutureContextRules()
        if rules is None
        else rules
    )
    rules.validate()

    horizon = _resolve_horizon(
        article_gold=article_gold,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    start = pd.Timestamp(
        horizon[
            'forecast_start'
        ]
    ).normalize()
    end = pd.Timestamp(
        horizon[
            'forecast_end'
        ]
    ).normalize()

    calendar = pd.DataFrame(
        {
            'date': pd.date_range(
                start,
                end,
                freq='D',
            )
        }
    )

    expanded_events = _expand_events(
        events=events,
        start=start,
        end=end,
        rules=rules,
    )

    future_context = _aggregate_events(
        expanded_events=expanded_events,
        calendar=calendar,
    )

    future_holidays = _prepare_holidays(
        holidays=holidays,
        start=start,
        end=end,
        rules=rules,
    )

    future_context = _aggregate_holidays(
        holidays=future_holidays,
        calendar=future_context,
    )

    weather_daily, weather_columns = _prepare_weather_forecast(
        weather_forecast=weather_forecast,
        start=start,
        end=end,
        rules=rules,
    )

    if not weather_daily.empty and weather_columns:
        future_context = future_context.merge(
            weather_daily,
            on='date',
            how='left',
            validate='one_to_one',
        )

    future_context = future_context.sort_values(
        'date'
    ).reset_index(
        drop=True
    )

    expected_days = len(
        calendar
    )

    weather_days = (
        int(
            weather_daily[
                'date'
            ].nunique()
        )
        if not weather_daily.empty
        else 0
    )

    report = {
        'rules': asdict(
            rules
        ),
        'forecast_horizon': horizon,
        'context_rows': int(
            len(
                future_context
            )
        ),
        'expected_days': int(
            expected_days
        ),
        'daily_coverage_fraction': (
            float(
                len(
                    future_context
                )
                / expected_days
            )
            if expected_days
            else None
        ),
        'events': {
            'expanded_event_day_rows': int(
                len(
                    expanded_events
                )
            ),
            'unique_events': int(
                expanded_events[
                    'event_id'
                ].nunique(
                    dropna=True
                )
                if not expanded_events.empty
                else 0
            ),
            'days_with_events': int(
                future_context[
                    'tiene_evento'
                ].sum()
            ),
            'max_simultaneous_events': int(
                future_context[
                    'num_eventos'
                ].max()
                if not future_context.empty
                else 0
            ),
        },
        'holidays': {
            'holiday_rows': int(
                len(
                    future_holidays
                )
            ),
            'holiday_days': int(
                future_context[
                    'es_festivo'
                ].sum()
            ),
        },
        'weather_forecast': {
            'supplied': weather_forecast is not None,
            'weather_columns': weather_columns,
            'days_available': weather_days,
            'coverage_fraction': (
                float(
                    weather_days
                    / expected_days
                )
                if expected_days
                else None
            ),
        },
        'methodology': {
            'events_and_holidays': (
                'Scheduled event and public-holiday information is treated as '
                'known in advance.'
            ),
            'weather': (
                'Weather is included only when supplied explicitly as a '
                'forecast available at prediction time.'
            ),
            'prohibited_future_information': (
                'Observed future weather, future sales, tickets, customer '
                'counts and post-cutoff reservation outcomes are not created '
                'or inferred by this module.'
            ),
        },
        'warnings': [],
    }

    if weather_forecast is None:
        report[
            'warnings'
        ].append(
            'No weather forecast was supplied. Event and holiday context is '
            'still complete, but weather-dependent features remain unavailable.'
        )

    elif weather_days < expected_days:
        report[
            'warnings'
        ].append(
            'Weather forecast coverage does not span the complete forecast '
            'period.'
        )

    tables_dir, reports_dir = _ensure_dirs(
        paths
    )

    if rules.save_tables:
        future_context.to_csv(
            tables_dir
            / 'future_daily_context.csv',
            index=False,
        )

        expanded_events.to_csv(
            tables_dir
            / 'future_events_expanded.csv',
            index=False,
        )

        future_holidays.to_csv(
            tables_dir
            / 'future_holidays.csv',
            index=False,
        )

    if rules.save_report:
        with (
            reports_dir
            / 'future_context_report.json'
        ).open(
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(
                _json_safe(
                    report
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

    if verbose:
        print()
        print(
            '=' * 96
        )
        print(
            'KNOWN FUTURE CONTEXT'
        )
        print(
            '=' * 96
        )
        print(
            f'[INFO] Forecast period: '
            f'{start.date()} -> {end.date()} '
            f'({expected_days} days).'
        )
        print(
            f'[INFO] Daily context rows: '
            f'{len(future_context)}/{expected_days}.'
        )
        print(
            f'[INFO] Events: '
            f'{report["events"]["unique_events"]} unique | '
            f'{report["events"]["days_with_events"]} days with events | '
            f'max {report["events"]["max_simultaneous_events"]} simultaneous.'
        )
        print(
            f'[INFO] Holidays: '
            f'{report["holidays"]["holiday_days"]} holiday days.'
        )
        print(
            f'[INFO] Weather forecast: '
            f'{weather_days}/{expected_days} days | '
            f'{len(weather_columns)} usable columns.'
        )

        for warning in report[
            'warnings'
        ]:
            print(
                f'[WARNING] {warning}'
            )

        print()
        print(
            'DAILY FUTURE CONTEXT'
        )
        print(
            future_context.to_string(
                index=False
            )
        )

        if not expanded_events.empty:
            preview_columns = [
                column
                for column in [
                    'date',
                    'event_id',
                    'event_name',
                    'event_intensity',
                ]
                if column in expanded_events.columns
            ]

            print()
            print(
                'EVENTS IN FORECAST HORIZON'
            )
            print(
                expanded_events[
                    preview_columns
                ]
                .drop_duplicates()
                .to_string(
                    index=False
                )
            )

    return {
        'future_daily_context': future_context,
        'future_events_expanded': expanded_events,
        'future_holidays': future_holidays,
    }
