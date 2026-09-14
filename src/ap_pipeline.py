from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .ap_article_scope import audit_article_scope
from .ap_config import (
    PipelineConfig,
    prepare_repository_directories,
    print_configuration,
    should_load_bronze,
    validate_runtime_inputs,
)
from .ap_depuration import run_depuration
from .ap_eda import run_eda
from .ap_features import run_feature_engineering
from .ap_future_context import build_future_daily_context
from .ap_integration import run_integration
from .ap_io import (
    load_bronze_datasets,
    load_structured_user_data,
    run_ingestion,
)
from .ap_modeling import run_modeling_foundation
from .ap_operational import run_operational_forecast


# =============================================================================
# PERSISTED-LAYER HELPERS
# =============================================================================


def _load_parquet_directory(
    directory: Path,
    layer: str,
) -> dict[str, pd.DataFrame]:
    if not directory.exists():
        raise FileNotFoundError(
            f'{layer} directory does not exist: {directory}'
        )

    outputs: dict[str, pd.DataFrame] = {}

    for path in sorted(
        directory.glob(
            '*.parquet'
        )
    ):
        stem = path.stem

        if (
            layer == 'silver'
            and stem.endswith(
                '_silver'
            )
        ):
            key = stem[
                :-len(
                    '_silver'
                )
            ]

        elif (
            layer == 'bronze'
            and stem.endswith(
                '_raw'
            )
        ):
            key = stem[
                :-len(
                    '_raw'
                )
            ]

        else:
            key = stem

        outputs[
            key
        ] = pd.read_parquet(
            path
        )

    if not outputs:
        raise FileNotFoundError(
            f'No persisted {layer} parquet files found in {directory}.'
        )

    return outputs


def _load_feature_datasets(
    feature_dir: Path,
) -> dict[str, pd.DataFrame]:
    return _load_parquet_directory(
        feature_dir,
        layer='features',
    )


def _first_dataset(
    datasets: dict[str, pd.DataFrame],
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> pd.DataFrame | None:
    for candidate in candidates:
        if candidate in datasets:
            return datasets[
                candidate
            ]

    if required:
        raise KeyError(
            'None of the required datasets are available: '
            + ', '.join(
                candidates
            )
        )

    return None


def _read_optional_table(
    path: Path | None,
) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None

    suffix = path.suffix.lower()

    if suffix == '.parquet':
        return pd.read_parquet(
            path
        )

    if suffix == '.csv':
        return pd.read_csv(
            path,
            sep=None,
            engine='python',
        )

    if suffix in {
        '.xlsx',
        '.xls',
    }:
        return pd.read_excel(
            path
        )

    raise ValueError(
        'Unsupported future-weather forecast format: '
        f'{path}'
    )


# =============================================================================
# PIPELINE
# =============================================================================


def run_pipeline(
    config: PipelineConfig,
) -> dict[str, Any]:
    """
    Execute the configured pipeline while preserving stage independence.

    Persisted Silver/Gold/Features/model artifacts are reused when the
    corresponding upstream stage is disabled.
    """
    prepare_repository_directories(
        config
    )

    warnings = validate_runtime_inputs(
        config
    )

    print_configuration(
        config
    )

    for warning in warnings:
        print(
            f'[WARNING] {warning}'
        )

    outputs: dict[str, Any] = {}

    source_datasets: dict[str, pd.DataFrame] = {}
    silver_datasets: dict[str, pd.DataFrame] = {}
    gold_datasets: dict[str, pd.DataFrame] = {}
    feature_datasets: dict[str, pd.DataFrame] = {}
    modeling_outputs: dict[str, pd.DataFrame] = {}

    # -----------------------------------------------------------------
    # RAW / BRONZE -> SILVER -> GOLD
    # -----------------------------------------------------------------

    if config.run.do_ingest:
        source_datasets = run_ingestion(
            paths=config.ingestion_paths,
            recursive_raw_scan=config.run.recursive_raw_scan,
            strict=config.run.strict_ingestion,
            verbose=config.run.verbose,
        )
        outputs[
            'ingestion'
        ] = source_datasets

    elif config.run.do_depuration:
        if should_load_bronze(
            config
        ):
            source_datasets = load_bronze_datasets(
                bronze_dir=config.paths.bronze_dir,
                verbose=config.run.verbose,
            )

        else:
            source_datasets = load_structured_user_data(
                input_paths=config.paths.structured_input_paths,
                verbose=config.run.verbose,
            )

    if config.run.do_depuration:
        silver_datasets = run_depuration(
            datasets=source_datasets,
            paths=config.depuration_paths,
            rules=config.depuration,
            verbose=config.run.verbose,
        )
        outputs[
            'silver'
        ] = silver_datasets

        gold_datasets = run_integration(
            silver_datasets=silver_datasets,
            paths=config.integration_paths,
            rules=config.integration,
            verbose=config.run.verbose,
        )
        outputs[
            'gold'
        ] = gold_datasets

    # -----------------------------------------------------------------
    # EDA
    # -----------------------------------------------------------------

    if config.run.do_eda:
        if (
            config.eda.analyze_silver
            and not silver_datasets
        ):
            silver_datasets = _load_parquet_directory(
                config.paths.silver_dir,
                layer='silver',
            )

        if (
            config.eda.analyze_gold
            and not gold_datasets
        ):
            gold_datasets = _load_parquet_directory(
                config.paths.gold_dir,
                layer='gold',
            )

        outputs[
            'eda'
        ] = run_eda(
            silver_datasets=(
                silver_datasets
                if config.eda.analyze_silver
                else {}
            ),
            gold_datasets=(
                gold_datasets
                if config.eda.analyze_gold
                else {}
            ),
            paths=config.eda_paths,
            rules=config.eda,
            verbose=config.run.verbose,
        )

    # -----------------------------------------------------------------
    # FEATURES
    # -----------------------------------------------------------------

    if config.run.do_features:
        if not gold_datasets:
            gold_datasets = _load_parquet_directory(
                config.paths.gold_dir,
                layer='gold',
            )

        feature_datasets = run_feature_engineering(
            gold_datasets=gold_datasets,
            paths=config.feature_paths,
            rules=config.features,
            verbose=config.run.verbose,
        )
        outputs[
            'features'
        ] = feature_datasets

    elif (
        config.run.do_modeling
        or config.run.do_operational
    ):
        feature_datasets = _load_feature_datasets(
            config.paths.features_dir
        )

    # -----------------------------------------------------------------
    # MODELING
    # -----------------------------------------------------------------

    feature_catalogue_path = (
        config.paths.feature_reports_dir
        / 'feature_catalogue.csv'
    )

    if config.run.do_modeling:
        modeling_outputs = run_modeling_foundation(
            feature_datasets=feature_datasets,
            feature_catalogue=feature_catalogue_path,
            paths=config.modeling_paths,
            rules=config.modeling,
            verbose=config.run.verbose,
        )
        outputs[
            'modeling'
        ] = modeling_outputs

    # -----------------------------------------------------------------
    # OPERATIONAL FORECAST
    # -----------------------------------------------------------------

    if config.run.do_operational:
        if not silver_datasets:
            silver_datasets = _load_parquet_directory(
                config.paths.silver_dir,
                layer='silver',
            )

        if not gold_datasets:
            gold_datasets = _load_parquet_directory(
                config.paths.gold_dir,
                layer='gold',
            )

        if not feature_datasets:
            feature_datasets = _load_feature_datasets(
                config.paths.features_dir
            )

        article_gold = _first_dataset(
            gold_datasets,
            (
                'tabla_maestra_semanal_articulos',
                'article_period',
                'weekly_articles',
            ),
        )

        daily_gold = _first_dataset(
            gold_datasets,
            (
                'tabla_maestra_diaria',
                'daily',
            ),
            required=False,
        )

        events = _first_dataset(
            silver_datasets,
            (
                'eventos',
                'events',
            ),
            required=False,
        )

        holidays = _first_dataset(
            silver_datasets,
            (
                'festivos',
                'holidays',
            ),
            required=False,
        )

        article_master = _first_dataset(
            silver_datasets,
            (
                'articulos',
                'articles',
            ),
            required=False,
        )

        menu = _first_dataset(
            silver_datasets,
            (
                'menu',
                'carta',
            ),
            required=False,
        )

        weather_forecast = _read_optional_table(
            config.paths.future_weather_forecast_path
        )

        future_context_outputs = build_future_daily_context(
            events=events,
            holidays=holidays,
            article_gold=article_gold,
            weather_forecast=weather_forecast,
            paths=config.future_context_paths,
            rules=config.future_context,
            verbose=config.run.verbose,
        )
        outputs[
            'future_context'
        ] = future_context_outputs

        future_daily_context = future_context_outputs[
            'future_daily_context'
        ]

        if future_daily_context.empty:
            raise RuntimeError(
                'Future context produced no dates for the operational horizon.'
            )

        forecast_start = pd.to_datetime(
            future_daily_context[
                'date'
            ],
            errors='coerce',
        ).min()

        article_scope_outputs = audit_article_scope(
            article_master=article_master,
            menu=menu,
            article_gold=article_gold,
            forecast_start=forecast_start,
            paths=config.article_scope_paths,
            rules=config.article_scope,
            verbose=config.run.verbose,
        )
        outputs[
            'article_scope'
        ] = article_scope_outputs

        if modeling_outputs:
            feature_selection = modeling_outputs[
                'feature_selection'
            ]
            frozen_mapping = modeling_outputs[
                'frozen_adaptive_mapping'
            ]
        else:
            feature_selection = (
                config.paths.modeling_tables_dir
                / 'feature_selection.csv'
            )
            frozen_mapping = (
                config.paths.modeling_tables_dir
                / 'frozen_adaptive_mapping.csv'
            )

        modeling_report_path = (
            config.paths.modeling_reports_dir
            / 'modeling_development_report.json'
        )

        historical_article_features = _first_dataset(
            feature_datasets,
            (
                'features_article_period',
            ),
        )

        operational_outputs = run_operational_forecast(
            article_gold=article_gold,
            daily_gold=daily_gold,
            historical_feature_table=historical_article_features,
            feature_selection=feature_selection,
            frozen_adaptive_mapping=frozen_mapping,
            modeling_report=(
                modeling_report_path
                if modeling_report_path.exists()
                else None
            ),
            future_daily_context=future_daily_context,
            article_scope_audit=article_scope_outputs[
                'article_scope_audit'
            ],
            paths=config.operational_paths,
            feature_rules=config.features,
            modeling_rules=config.modeling,
            operational_rules=config.operational,
            verbose=config.run.verbose,
        )
        outputs[
            'operational'
        ] = operational_outputs

    return outputs
