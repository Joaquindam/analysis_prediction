"""
Article-universe audit and forecast-scope selection.

This module answers a separate question from demand modeling:

    Which catalogue articles should receive an operational forecast?

The logic is intentionally conservative. It prefers explicit active/inactive
metadata when available. If no authoritative status signal exists, it does NOT
silently declare an article inactive just because it has not sold recently.
Instead it keeps historically observed articles forecastable and flags stale
items for review.

That distinction matters in hospitality because a missing article-period row
can mean zero demand, temporary unavailability, menu rotation or missing data.

The module is designed to be called from the project main pipeline later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ArticleScopePaths:
    tables_dir: Path
    reports_dir: Path


@dataclass
class ArticleScopeRules:
    article_code_column: str = 'article_code'
    article_name_candidates: tuple[str, ...] = (
        'article_name',
        'name',
        'descripcion',
    )

    complete_period_column: str = 'period_is_complete'
    period_start_column: str = 'report_start'
    period_end_column: str = 'report_end'
    units_column: str = 'units'

    recent_periods_short: int = 4
    recent_periods_long: int = 8

    active_flag_candidates: tuple[str, ...] = (
        'is_active',
        'active',
        'activo',
        'habilitado',
        'enabled',
        'en_carta',
        'in_menu',
    )
    status_candidates: tuple[str, ...] = (
        'status',
        'estado',
        'article_status',
        'estado_articulo',
    )
    valid_from_candidates: tuple[str, ...] = (
        'valid_from',
        'start_date',
        'fecha_inicio',
        'fecha_alta',
        'active_from',
    )
    valid_to_candidates: tuple[str, ...] = (
        'valid_to',
        'end_date',
        'fecha_fin',
        'fecha_baja',
        'active_to',
    )

    active_status_tokens: tuple[str, ...] = (
        'active',
        'activo',
        'enabled',
        'habilitado',
        'current',
        'vigente',
    )
    inactive_status_tokens: tuple[str, ...] = (
        'inactive',
        'inactivo',
        'disabled',
        'deshabilitado',
        'retired',
        'retirado',
        'baja',
    )

    # A menu export that contains virtually the entire article master is not
    # treated as evidence that every row is currently active.
    menu_informative_max_master_coverage: float = 0.95

    # Conservative deployment policy:
    # explicit inactive -> exclude;
    # otherwise keep articles with demand history, even if stale.
    policy: str = 'conservative_observed_history'

    save_tables: bool = True
    save_report: bool = True

    def validate(self) -> None:
        if self.recent_periods_short < 1:
            raise ValueError(
                'recent_periods_short must be >= 1.'
            )

        if self.recent_periods_long < self.recent_periods_short:
            raise ValueError(
                'recent_periods_long must be >= recent_periods_short.'
            )

        if not 0 < self.menu_informative_max_master_coverage <= 1:
            raise ValueError(
                'menu_informative_max_master_coverage must be in (0, 1].'
            )

        if self.policy not in {
            'conservative_observed_history',
            'recent_or_explicit',
        }:
            raise ValueError(
                "policy must be 'conservative_observed_history' or "
                "'recent_or_explicit'."
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
        'active': True,
        'activo': True,
        'enabled': True,
        'habilitado': True,
        'false': False,
        '0': False,
        'no': False,
        'n': False,
        'inactive': False,
        'inactivo': False,
        'disabled': False,
        'deshabilitado': False,
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
    paths: ArticleScopePaths,
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


def _normalise_code(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors='coerce',
    )

    if numeric.notna().all():
        return numeric.astype(
            'Int64'
        )

    return series.astype(
        'string'
    )


def _source_article_table(
    dataframe: pd.DataFrame | None,
    source_name: str,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(
            columns=[
                rules.article_code_column,
                f'in_{source_name}',
                f'article_name_{source_name}',
            ]
        )

    code = rules.article_code_column

    if code not in dataframe.columns:
        raise KeyError(
            f'{source_name} is missing required article code column '
            f'{code!r}.'
        )

    data = dataframe.copy()
    data[
        code
    ] = _normalise_code(
        data[
            code
        ]
    )

    name_column = _resolve_column(
        data,
        rules.article_name_candidates,
    )

    columns = [
        code,
    ]

    if name_column is not None:
        columns.append(
            name_column
        )

    result = data[
        columns
    ].dropna(
        subset=[
            code,
        ]
    ).drop_duplicates(
        subset=[
            code,
        ],
        keep='last',
    )

    result[
        f'in_{source_name}'
    ] = True

    if name_column is not None:
        result = result.rename(
            columns={
                name_column:
                    f'article_name_{source_name}',
            }
        )
    else:
        result[
            f'article_name_{source_name}'
        ] = pd.NA

    return result[
        [
            code,
            f'in_{source_name}',
            f'article_name_{source_name}',
        ]
    ].reset_index(
        drop=True
    )


# =============================================================================
# EXPLICIT ACTIVE / INACTIVE METADATA
# =============================================================================


def _extract_explicit_status(
    dataframe: pd.DataFrame | None,
    source_name: str,
    forecast_start: pd.Timestamp,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    code = rules.article_code_column

    columns = [
        code,
        f'explicit_active_{source_name}',
        f'explicit_inactive_{source_name}',
        f'explicit_signal_{source_name}',
    ]

    if dataframe is None or dataframe.empty:
        return pd.DataFrame(
            columns=columns
        )

    if code not in dataframe.columns:
        return pd.DataFrame(
            columns=columns
        )

    data = dataframe.copy()
    data[
        code
    ] = _normalise_code(
        data[
            code
        ]
    )

    active_flag_column = _resolve_column(
        data,
        rules.active_flag_candidates,
    )
    status_column = _resolve_column(
        data,
        rules.status_candidates,
    )
    valid_from_column = _resolve_column(
        data,
        rules.valid_from_candidates,
    )
    valid_to_column = _resolve_column(
        data,
        rules.valid_to_candidates,
    )

    active = pd.Series(
        False,
        index=data.index,
        dtype=bool,
    )
    inactive = pd.Series(
        False,
        index=data.index,
        dtype=bool,
    )
    has_signal = pd.Series(
        False,
        index=data.index,
        dtype=bool,
    )

    if active_flag_column is not None:
        parsed = _safe_bool(
            data[
                active_flag_column
            ]
        )
        active |= parsed.eq(
            True
        ).fillna(
            False
        )
        inactive |= parsed.eq(
            False
        ).fillna(
            False
        )
        has_signal |= parsed.notna()

    if status_column is not None:
        status = (
            data[
                status_column
            ]
            .astype('string')
            .str.strip()
            .str.lower()
        )

        active_tokens = {
            token.lower()
            for token in rules.active_status_tokens
        }
        inactive_tokens = {
            token.lower()
            for token in rules.inactive_status_tokens
        }

        active |= status.isin(
            active_tokens
        )
        inactive |= status.isin(
            inactive_tokens
        )
        has_signal |= status.isin(
            active_tokens
            | inactive_tokens
        )

    if valid_from_column is not None:
        valid_from = _to_datetime(
            data[
                valid_from_column
            ]
        )
        not_started = (
            valid_from.notna()
            & valid_from.gt(
                forecast_start
            )
        )
        inactive |= not_started
        has_signal |= valid_from.notna()

    if valid_to_column is not None:
        valid_to = _to_datetime(
            data[
                valid_to_column
            ]
        )
        expired = (
            valid_to.notna()
            & valid_to.lt(
                forecast_start
            )
        )
        inactive |= expired
        has_signal |= valid_to.notna()

        if valid_from_column is None:
            active |= (
                valid_to.isna()
                | valid_to.ge(
                    forecast_start
                )
            ) & has_signal

    result = pd.DataFrame(
        {
            code: data[
                code
            ],
            f'explicit_active_{source_name}':
                active,
            f'explicit_inactive_{source_name}':
                inactive,
            f'explicit_signal_{source_name}':
                has_signal,
        }
    )

    result = (
        result
        .dropna(
            subset=[
                code,
            ]
        )
        .groupby(
            code,
            as_index=False,
            dropna=False,
        )
        .agg(
            {
                f'explicit_active_{source_name}':
                    'max',
                f'explicit_inactive_{source_name}':
                    'max',
                f'explicit_signal_{source_name}':
                    'max',
            }
        )
    )

    return result


# =============================================================================
# DEMAND-HISTORY ACTIVITY
# =============================================================================


def _complete_history(
    article_gold: pd.DataFrame,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    data = article_gold.copy()

    if rules.complete_period_column in data.columns:
        complete = _safe_bool(
            data[
                rules.complete_period_column
            ]
        ).fillna(
            False
        )

        if complete.any():
            data = data.loc[
                complete
            ].copy()

    return data


def _period_reference(
    article_gold: pd.DataFrame,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    data = _complete_history(
        article_gold,
        rules,
    )

    start_col = rules.period_start_column
    end_col = rules.period_end_column

    for column in [
        start_col,
        end_col,
    ]:
        if column not in data.columns:
            raise KeyError(
                f'Article Gold is missing required period column {column!r}.'
            )

        data[
            column
        ] = _to_datetime(
            data[
                column
            ]
        )

    periods = (
        data[
            [
                start_col,
                end_col,
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                start_col,
                end_col,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    periods[
        'scope_period_index'
    ] = np.arange(
        len(
            periods
        ),
        dtype='int64',
    )

    return periods


def _history_profile(
    article_gold: pd.DataFrame,
    rules: ArticleScopeRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = rules.article_code_column
    start_col = rules.period_start_column
    end_col = rules.period_end_column

    if code not in article_gold.columns:
        raise KeyError(
            f'Article Gold is missing article code column {code!r}.'
        )

    periods = _period_reference(
        article_gold,
        rules,
    )

    history = _complete_history(
        article_gold,
        rules,
    ).copy()

    history[
        code
    ] = _normalise_code(
        history[
            code
        ]
    )
    history[
        start_col
    ] = _to_datetime(
        history[
            start_col
        ]
    )
    history[
        end_col
    ] = _to_datetime(
        history[
            end_col
        ]
    )

    history = history.merge(
        periods,
        on=[
            start_col,
            end_col,
        ],
        how='left',
        validate='many_to_one',
    )

    name_column = _resolve_column(
        history,
        rules.article_name_candidates,
    )

    if name_column is None:
        history[
            '__article_name_history'
        ] = pd.NA
        name_column = '__article_name_history'

    grouped = history.groupby(
        code,
        dropna=False,
    )

    profile = grouped.agg(
        periods_observed=(
            'scope_period_index',
            'nunique',
        ),
        first_observed_period_index=(
            'scope_period_index',
            'min',
        ),
        last_observed_period_index=(
            'scope_period_index',
            'max',
        ),
        first_observed_period_start=(
            start_col,
            'min',
        ),
        last_observed_period_start=(
            start_col,
            'max',
        ),
        last_observed_period_end=(
            end_col,
            'max',
        ),
        article_name_history=(
            name_column,
            'last',
        ),
    ).reset_index()

    if rules.units_column in history.columns:
        unit_totals = (
            history
            .assign(
                __units=_to_numeric(
                    history[
                        rules.units_column
                    ]
                )
            )
            .groupby(
                code,
                as_index=False,
                dropna=False,
            )[
                '__units'
            ]
            .sum(
                min_count=1
            )
            .rename(
                columns={
                    '__units': 'units_total_history',
                }
            )
        )

        profile = profile.merge(
            unit_totals,
            on=code,
            how='left',
            validate='one_to_one',
        )
    else:
        profile[
            'units_total_history'
        ] = np.nan

    latest_period_index = (
        int(
            periods[
                'scope_period_index'
            ].max()
        )
        if not periods.empty
        else -1
    )

    profile[
        'periods_since_last_observed'
    ] = (
        latest_period_index
        - profile[
            'last_observed_period_index'
        ]
    ).astype(
        'Int64'
    )

    profile[
        f'observed_last_{rules.recent_periods_short}_periods'
    ] = profile[
        'periods_since_last_observed'
    ].lt(
        rules.recent_periods_short
    )

    profile[
        f'observed_last_{rules.recent_periods_long}_periods'
    ] = profile[
        'periods_since_last_observed'
    ].lt(
        rules.recent_periods_long
    )

    profile[
        'observed_latest_complete_period'
    ] = profile[
        'periods_since_last_observed'
    ].eq(
        0
    )

    profile[
        'observed_in_complete_history'
    ] = True

    return profile, periods


def _all_history_codes(
    article_gold: pd.DataFrame,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    code = rules.article_code_column

    data = article_gold.copy()
    data[
        code
    ] = _normalise_code(
        data[
            code
        ]
    )

    name_column = _resolve_column(
        data,
        rules.article_name_candidates,
    )

    columns = [
        code,
    ]

    if name_column is not None:
        columns.append(
            name_column
        )

    result = (
        data[
            columns
        ]
        .dropna(
            subset=[
                code,
            ]
        )
        .drop_duplicates(
            subset=[
                code,
            ],
            keep='last',
        )
    )

    result[
        'observed_in_any_history'
    ] = True

    if name_column is not None:
        result = result.rename(
            columns={
                name_column:
                    'article_name_any_history',
            }
        )
    else:
        result[
            'article_name_any_history'
        ] = pd.NA

    return result.reset_index(
        drop=True
    )


# =============================================================================
# AUDIT + SCOPE DECISION
# =============================================================================


def _combine_catalogues(
    article_master: pd.DataFrame | None,
    menu: pd.DataFrame | None,
    article_gold: pd.DataFrame,
    rules: ArticleScopeRules,
) -> pd.DataFrame:
    code = rules.article_code_column

    sources = [
        _source_article_table(
            article_master,
            'article_master',
            rules,
        ),
        _source_article_table(
            menu,
            'menu',
            rules,
        ),
        _all_history_codes(
            article_gold,
            rules,
        ),
    ]

    union_codes = pd.concat(
        [
            frame[
                [
                    code,
                ]
            ]
            for frame in sources
            if not frame.empty
        ],
        ignore_index=True,
    ).drop_duplicates()

    audit = union_codes.copy()

    for frame in sources:
        if not frame.empty:
            audit = audit.merge(
                frame,
                on=code,
                how='left',
                validate='one_to_one',
            )

    for column in [
        'in_article_master',
        'in_menu',
        'observed_in_any_history',
    ]:
        if column not in audit.columns:
            audit[
                column
            ] = False
        else:
            audit[
                column
            ] = _safe_bool(
                audit[
                    column
                ]
            ).fillna(
                False
            ).astype(
                bool
            )

    return audit


def _source_capabilities(
    article_master: pd.DataFrame | None,
    menu: pd.DataFrame | None,
    rules: ArticleScopeRules,
) -> dict[str, Any]:
    code = rules.article_code_column

    master_codes = (
        set(
            _normalise_code(
                article_master[
                    code
                ]
            ).dropna().tolist()
        )
        if (
            article_master is not None
            and not article_master.empty
            and code in article_master.columns
        )
        else set()
    )

    menu_codes = (
        set(
            _normalise_code(
                menu[
                    code
                ]
            ).dropna().tolist()
        )
        if (
            menu is not None
            and not menu.empty
            and code in menu.columns
        )
        else set()
    )

    menu_master_coverage = (
        len(
            menu_codes
            & master_codes
        )
        / len(
            master_codes
        )
        if master_codes
        else None
    )

    menu_is_informative_subset = bool(
        master_codes
        and menu_codes
        and menu_master_coverage
        < rules.menu_informative_max_master_coverage
    )

    def status_columns(
        dataframe: pd.DataFrame | None,
    ) -> dict[str, str | None]:
        return {
            'active_flag': _resolve_column(
                dataframe,
                rules.active_flag_candidates,
            ),
            'status': _resolve_column(
                dataframe,
                rules.status_candidates,
            ),
            'valid_from': _resolve_column(
                dataframe,
                rules.valid_from_candidates,
            ),
            'valid_to': _resolve_column(
                dataframe,
                rules.valid_to_candidates,
            ),
        }

    return {
        'article_master_rows': int(
            len(
                article_master
            )
            if article_master is not None
            else 0
        ),
        'article_master_unique_codes': int(
            len(
                master_codes
            )
        ),
        'menu_rows': int(
            len(
                menu
            )
            if menu is not None
            else 0
        ),
        'menu_unique_codes': int(
            len(
                menu_codes
            )
        ),
        'menu_master_code_coverage': menu_master_coverage,
        'menu_is_informative_subset':
            menu_is_informative_subset,
        'article_master_status_columns':
            status_columns(
                article_master
            ),
        'menu_status_columns':
            status_columns(
                menu
            ),
    }


def audit_article_scope(
    article_master: pd.DataFrame | None,
    menu: pd.DataFrame | None,
    article_gold: pd.DataFrame,
    forecast_start: str | pd.Timestamp,
    paths: ArticleScopePaths,
    rules: ArticleScopeRules | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Audit catalogue activity and produce a conservative forecast-eligibility
    recommendation.

    The output does not modify Gold, features or model selection.
    """
    rules = (
        ArticleScopeRules()
        if rules is None
        else rules
    )
    rules.validate()

    forecast_start = pd.Timestamp(
        forecast_start
    )

    code = rules.article_code_column

    audit = _combine_catalogues(
        article_master=article_master,
        menu=menu,
        article_gold=article_gold,
        rules=rules,
    )

    history_profile, periods = _history_profile(
        article_gold,
        rules,
    )

    audit = audit.merge(
        history_profile,
        on=code,
        how='left',
        validate='one_to_one',
    )

    for source_name, source in [
        (
            'article_master',
            article_master,
        ),
        (
            'menu',
            menu,
        ),
    ]:
        explicit = _extract_explicit_status(
            dataframe=source,
            source_name=source_name,
            forecast_start=forecast_start,
            rules=rules,
        )

        if not explicit.empty:
            audit = audit.merge(
                explicit,
                on=code,
                how='left',
                validate='one_to_one',
            )

    for source_name in [
        'article_master',
        'menu',
    ]:
        for prefix in [
            'explicit_active',
            'explicit_inactive',
            'explicit_signal',
        ]:
            column = f'{prefix}_{source_name}'

            if column not in audit.columns:
                audit[
                    column
                ] = False
            else:
                audit[
                    column
                ] = audit[
                    column
                ].fillna(
                    False
                ).astype(
                    bool
                )

    audit[
        'explicit_active_any'
    ] = (
        audit[
            'explicit_active_article_master'
        ]
        | audit[
            'explicit_active_menu'
        ]
    )
    audit[
        'explicit_inactive_any'
    ] = (
        audit[
            'explicit_inactive_article_master'
        ]
        | audit[
            'explicit_inactive_menu'
        ]
    )
    audit[
        'explicit_status_conflict'
    ] = (
        audit[
            'explicit_active_any'
        ]
        & audit[
            'explicit_inactive_any'
        ]
    )
    audit[
        'explicit_signal_available'
    ] = (
        audit[
            'explicit_signal_article_master'
        ]
        | audit[
            'explicit_signal_menu'
        ]
    )

    audit[
        'observed_in_complete_history'
    ] = _safe_bool(
        audit[
            'observed_in_complete_history'
        ]
    ).fillna(
        False
    ).astype(
        bool
    )

    capabilities = _source_capabilities(
        article_master=article_master,
        menu=menu,
        rules=rules,
    )

    menu_informative = bool(
        capabilities[
            'menu_is_informative_subset'
        ]
    )

    recent_short = (
        f'observed_last_{rules.recent_periods_short}_periods'
    )
    recent_long = (
        f'observed_last_{rules.recent_periods_long}_periods'
    )

    for column in [
        recent_short,
        recent_long,
        'observed_latest_complete_period',
    ]:
        if column not in audit.columns:
            audit[
                column
            ] = False
        else:
            audit[
                column
            ] = audit[
                column
            ].fillna(
                False
            ).astype(
                bool
            )

    conditions = [
        audit[
            'explicit_status_conflict'
        ],
        audit[
            'explicit_inactive_any'
        ],
        audit[
            'explicit_active_any'
        ],
        (
            audit[
                'observed_in_any_history'
            ]
            & audit[
                recent_short
            ]
        ),
        (
            audit[
                'observed_in_any_history'
            ]
            & audit[
                recent_long
            ]
        ),
        audit[
            'observed_in_any_history'
        ],
    ]

    choices = [
        'explicit_status_conflict',
        'explicit_inactive',
        'explicit_active',
        'recent_observed',
        'moderately_recent_observed',
        'stale_observed',
    ]

    audit[
        'activity_status'
    ] = np.select(
        conditions,
        choices,
        default='catalogue_only_unobserved',
    )

    audit[
        'manual_review_required'
    ] = (
        audit[
            'explicit_status_conflict'
        ]
        | audit[
            'activity_status'
        ].eq(
            'stale_observed'
        )
    )

    if rules.policy == 'conservative_observed_history':
        eligible = (
            audit[
                'observed_in_any_history'
            ]
            & ~audit[
                'explicit_inactive_any'
            ]
            & ~audit[
                'explicit_status_conflict'
            ]
        )

        reason = np.select(
            [
                audit[
                    'explicit_status_conflict'
                ],
                audit[
                    'explicit_inactive_any'
                ],
                ~audit[
                    'observed_in_any_history'
                ],
                audit[
                    'explicit_active_any'
                ],
                (
                    menu_informative
                    & audit[
                        'in_menu'
                    ]
                ),
                audit[
                    recent_long
                ],
            ],
            [
                'excluded_explicit_status_conflict',
                'excluded_explicit_inactive',
                'excluded_no_demand_history',
                'included_explicit_active_with_history',
                'included_in_informative_menu_with_history',
                'included_recent_demand_history',
            ],
            default='included_observed_history_conservative',
        )

    else:
        eligible = (
            audit[
                'observed_in_any_history'
            ]
            & ~audit[
                'explicit_inactive_any'
            ]
            & ~audit[
                'explicit_status_conflict'
            ]
            & (
                audit[
                    'explicit_active_any'
                ]
                | audit[
                    recent_long
                ]
                | (
                    menu_informative
                    & audit[
                        'in_menu'
                    ]
                )
            )
        )

        reason = np.select(
            [
                audit[
                    'explicit_status_conflict'
                ],
                audit[
                    'explicit_inactive_any'
                ],
                ~audit[
                    'observed_in_any_history'
                ],
                audit[
                    'explicit_active_any'
                ],
                (
                    menu_informative
                    & audit[
                        'in_menu'
                    ]
                ),
                audit[
                    recent_long
                ],
            ],
            [
                'excluded_explicit_status_conflict',
                'excluded_explicit_inactive',
                'excluded_no_demand_history',
                'included_explicit_active_with_history',
                'included_in_informative_menu_with_history',
                'included_recent_demand_history',
            ],
            default='excluded_stale_without_authoritative_active_signal',
        )

    audit[
        'forecast_eligible_recommended'
    ] = eligible.astype(
        bool
    )
    audit[
        'forecast_scope_reason'
    ] = reason

    name_candidates = [
        column
        for column in [
            'article_name_history',
            'article_name_any_history',
            'article_name_menu',
            'article_name_article_master',
        ]
        if column in audit.columns
    ]

    if name_candidates:
        canonical_name = audit[
            name_candidates[
                0
            ]
        ].copy()

        for column in name_candidates[
            1:
        ]:
            canonical_name = canonical_name.fillna(
                audit[
                    column
                ]
            )

        audit.insert(
            1,
            'article_name',
            canonical_name,
        )

    audit = audit.sort_values(
        [
            'forecast_eligible_recommended',
            'observed_in_any_history',
            'periods_since_last_observed',
            code,
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        na_position='last',
    ).reset_index(
        drop=True
    )

    eligible_table = audit.loc[
        audit[
            'forecast_eligible_recommended'
        ]
    ].copy()

    summary_rows = [
        {
            'metric': 'catalogue_union_articles',
            'value': int(
                len(
                    audit
                )
            ),
        },
        {
            'metric': 'article_master_unique_codes',
            'value': int(
                capabilities[
                    'article_master_unique_codes'
                ]
            ),
        },
        {
            'metric': 'menu_unique_codes',
            'value': int(
                capabilities[
                    'menu_unique_codes'
                ]
            ),
        },
        {
            'metric': 'observed_in_demand_history',
            'value': int(
                audit[
                    'observed_in_any_history'
                ].sum()
            ),
        },
        {
            'metric': 'observed_in_complete_history',
            'value': int(
                audit[
                    'observed_in_complete_history'
                ].sum()
            ),
        },
        {
            'metric':
                f'observed_last_{rules.recent_periods_short}_periods',
            'value': int(
                audit[
                    recent_short
                ].sum()
            ),
        },
        {
            'metric':
                f'observed_last_{rules.recent_periods_long}_periods',
            'value': int(
                audit[
                    recent_long
                ].sum()
            ),
        },
        {
            'metric': 'explicit_active',
            'value': int(
                audit[
                    'explicit_active_any'
                ].sum()
            ),
        },
        {
            'metric': 'explicit_inactive',
            'value': int(
                audit[
                    'explicit_inactive_any'
                ].sum()
            ),
        },
        {
            'metric': 'manual_review_required',
            'value': int(
                audit[
                    'manual_review_required'
                ].sum()
            ),
        },
        {
            'metric': 'forecast_eligible_recommended',
            'value': int(
                audit[
                    'forecast_eligible_recommended'
                ].sum()
            ),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    authoritative_status_available = bool(
        audit[
            'explicit_signal_available'
        ].any()
    )

    if authoritative_status_available:
        recommendation = (
            'Explicit active/inactive metadata exists. Use it as the primary '
            'scope signal, intersected with articles supported by the current '
            'demand model.'
        )
    elif menu_informative:
        recommendation = (
            'No explicit active flag exists, but menu membership is a proper '
            'subset of the article master and can be used as a secondary '
            'current-catalogue signal. Stale articles should still be reviewed.'
        )
    else:
        recommendation = (
            'No authoritative active/inactive signal is available. Keep '
            'historically observed articles forecastable under the conservative '
            'policy and flag stale articles for manual/business review rather '
            'than silently excluding them.'
        )

    report = {
        'rules': asdict(
            rules
        ),
        'forecast_start': forecast_start,
        'period_reference': {
            'complete_periods': int(
                len(
                    periods
                )
            ),
            'first_complete_period': (
                periods[
                    rules.period_start_column
                ].min()
                if not periods.empty
                else None
            ),
            'last_complete_period': (
                periods[
                    rules.period_end_column
                ].max()
                if not periods.empty
                else None
            ),
        },
        'source_capabilities': capabilities,
        'authoritative_status_available':
            authoritative_status_available,
        'recommendation': recommendation,
        'activity_status_counts': (
            audit[
                'activity_status'
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ),
        'scope_reason_counts': (
            audit[
                'forecast_scope_reason'
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ),
        'summary': summary.set_index(
            'metric'
        )[
            'value'
        ].to_dict(),
        'methodology': {
            'missing_article_period_semantics': (
                'Absence from a historical article-period is not interpreted '
                'as zero demand or definitive inactivity.'
            ),
            'stale_articles': (
                'Recency is used as a review signal, not as an automatic '
                'inactive label under the conservative policy.'
            ),
            'catalogue_only_articles': (
                'Articles with no demand history are not forecast by the '
                'current article-demand architecture by default.'
            ),
        },
    }

    tables_dir, reports_dir = _ensure_dirs(
        paths
    )

    if rules.save_tables:
        audit.to_csv(
            tables_dir
            / 'article_scope_audit.csv',
            index=False,
        )
        eligible_table.to_csv(
            tables_dir
            / 'article_scope_eligible.csv',
            index=False,
        )
        summary.to_csv(
            tables_dir
            / 'article_scope_summary.csv',
            index=False,
        )

    if rules.save_report:
        with (
            reports_dir
            / 'article_scope_report.json'
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
            'ARTICLE FORECAST SCOPE AUDIT'
        )
        print(
            '=' * 96
        )
        print(
            f'[INFO] Article master: '
            f'{capabilities["article_master_unique_codes"]:,} unique codes.'
        )
        print(
            f'[INFO] Menu: '
            f'{capabilities["menu_unique_codes"]:,} unique codes.'
        )

        coverage = capabilities[
            'menu_master_code_coverage'
        ]

        if coverage is not None:
            print(
                f'[INFO] Menu coverage of article master: '
                f'{coverage:.1%} | '
                f'informative subset={menu_informative}.'
            )

        print(
            f'[INFO] Demand history: '
            f'{int(audit["observed_in_any_history"].sum()):,} '
            f'observed articles.'
        )
        print(
            f'[INFO] Recent activity: '
            f'{int(audit[recent_short].sum()):,} observed in last '
            f'{rules.recent_periods_short} complete periods | '
            f'{int(audit[recent_long].sum()):,} in last '
            f'{rules.recent_periods_long}.'
        )
        print(
            f'[INFO] Explicit active/inactive signal available: '
            f'{authoritative_status_available}.'
        )
        print(
            f'[INFO] Recommended forecast universe: '
            f'{int(audit["forecast_eligible_recommended"].sum()):,} articles.'
        )
        print(
            f'[INFO] Manual-review flags: '
            f'{int(audit["manual_review_required"].sum()):,}.'
        )
        print(
            f'[INFO] Recommendation: {recommendation}'
        )

        print()
        print(
            'ACTIVITY STATUS'
        )
        print(
            audit[
                'activity_status'
            ]
            .value_counts(
                dropna=False
            )
            .rename_axis(
                'activity_status'
            )
            .reset_index(
                name='article_count',
            )
            .to_string(
                index=False
            )
        )

        stale = audit.loc[
            audit[
                'activity_status'
            ].eq(
                'stale_observed'
            )
        ].copy()

        if not stale.empty:
            preview_columns = [
                column
                for column in [
                    code,
                    'article_name',
                    'periods_observed',
                    'last_observed_period_end',
                    'periods_since_last_observed',
                    'forecast_eligible_recommended',
                ]
                if column in stale.columns
            ]

            print()
            print(
                'STALE OBSERVED ARTICLES | REVIEW SAMPLE'
            )
            print(
                stale[
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
        'article_scope_audit': audit,
        'article_scope_eligible': eligible_table,
        'article_scope_summary': summary,
    }
