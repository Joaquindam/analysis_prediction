from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ap_article_scope import ArticleScopePaths, ArticleScopeRules
from .ap_depuration import DepurationPaths, DepurationRules
from .ap_eda import EDAPaths, EDARules
from .ap_features import FeaturePaths, FeatureRules
from .ap_future_context import FutureContextPaths, FutureContextRules
from .ap_integration import IntegrationPaths, IntegrationRules
from .ap_io import IngestionPaths
from .ap_modeling import ModelingPaths, ModelingRules
from .ap_operational import OperationalPaths, OperationalRules


# =============================================================================
# EXECUTION OPTIONS
# =============================================================================


@dataclass(frozen=True)
class RunOptions:
    """Controls which pipeline stages are executed."""

    do_ingest: bool = True
    do_depuration: bool = True
    do_eda: bool = True
    do_features: bool = True
    do_modeling: bool = True
    do_operational: bool = True

    verbose: bool = True
    strict_ingestion: bool = False
    recursive_raw_scan: bool = False

    source_when_ingest_disabled: str = 'auto'

    def validate(self) -> None:
        boolean_fields = {
            'do_ingest': self.do_ingest,
            'do_depuration': self.do_depuration,
            'do_eda': self.do_eda,
            'do_features': self.do_features,
            'do_modeling': self.do_modeling,
            'do_operational': self.do_operational,
            'verbose': self.verbose,
            'strict_ingestion': self.strict_ingestion,
            'recursive_raw_scan': self.recursive_raw_scan,
        }

        for name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f'{name} must be True or False.'
                )

        allowed_sources = {
            'auto',
            'bronze',
            'structured',
        }

        if self.source_when_ingest_disabled not in allowed_sources:
            raise ValueError(
                'source_when_ingest_disabled must be one of: '
                + ', '.join(
                    sorted(
                        allowed_sources
                    )
                )
            )

        downstream_after_depuration = (
            self.do_eda
            or self.do_features
            or self.do_modeling
            or self.do_operational
        )

        if (
            self.do_ingest
            and not self.do_depuration
            and downstream_after_depuration
        ):
            raise ValueError(
                'DO_INGEST=True + DO_DEPURATION=False cannot feed downstream '
                'stages: new Bronze data would coexist with stale Silver/Gold. '
                'Enable depuration or disable downstream stages.'
            )

        if (
            self.do_depuration
            and not self.do_features
            and (
                self.do_modeling
                or self.do_operational
            )
        ):
            raise ValueError(
                'A run that rebuilds Silver/Gold cannot skip feature '
                'engineering and then model/forecast using stale features.'
            )

        if (
            self.do_features
            and not self.do_modeling
            and self.do_operational
        ):
            raise ValueError(
                'A run that rebuilds features cannot skip modeling and then '
                'forecast with stale frozen model artifacts.'
            )


# =============================================================================
# PROJECT PATHS
# =============================================================================


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path

    raw_dir: Path
    weekly_sales_dir: Path | None
    pdf_tickets_dir: Path | None
    future_weather_forecast_path: Path | None

    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    features_dir: Path

    results_dir: Path

    depuration_figures_dir: Path
    depuration_reports_dir: Path
    integration_reports_dir: Path

    eda_figures_dir: Path
    eda_tables_dir: Path
    eda_reports_dir: Path

    feature_reports_dir: Path

    modeling_tables_dir: Path
    modeling_reports_dir: Path
    modeling_predictions_dir: Path

    future_context_tables_dir: Path
    future_context_reports_dir: Path

    article_scope_tables_dir: Path
    article_scope_reports_dir: Path

    operational_forecasts_dir: Path
    operational_reports_dir: Path

    structured_input_paths: dict[str, Path] = field(
        default_factory=dict
    )


# =============================================================================
# COMPLETE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    run: RunOptions
    paths: ProjectPaths

    depuration: DepurationRules
    integration: IntegrationRules
    eda: EDARules
    features: FeatureRules
    modeling: ModelingRules
    future_context: FutureContextRules
    article_scope: ArticleScopeRules
    operational: OperationalRules

    @property
    def ingestion_paths(self) -> IngestionPaths:
        return IngestionPaths(
            raw_dir=self.paths.raw_dir,
            bronze_dir=self.paths.bronze_dir,
            weekly_sales_dir=self.paths.weekly_sales_dir,
            pdf_tickets_dir=self.paths.pdf_tickets_dir,
        )

    @property
    def depuration_paths(self) -> DepurationPaths:
        return DepurationPaths(
            silver_dir=self.paths.silver_dir,
            figures_dir=self.paths.depuration_figures_dir,
            reports_dir=self.paths.depuration_reports_dir,
        )

    @property
    def integration_paths(self) -> IntegrationPaths:
        return IntegrationPaths(
            gold_dir=self.paths.gold_dir,
            reports_dir=self.paths.integration_reports_dir,
        )

    @property
    def eda_paths(self) -> EDAPaths:
        return EDAPaths(
            figures_dir=self.paths.eda_figures_dir,
            tables_dir=self.paths.eda_tables_dir,
            reports_dir=self.paths.eda_reports_dir,
        )

    @property
    def feature_paths(self) -> FeaturePaths:
        return FeaturePaths(
            feature_dir=self.paths.features_dir,
            reports_dir=self.paths.feature_reports_dir,
        )

    @property
    def modeling_paths(self) -> ModelingPaths:
        return ModelingPaths(
            tables_dir=self.paths.modeling_tables_dir,
            reports_dir=self.paths.modeling_reports_dir,
            predictions_dir=self.paths.modeling_predictions_dir,
        )

    @property
    def future_context_paths(self) -> FutureContextPaths:
        return FutureContextPaths(
            tables_dir=self.paths.future_context_tables_dir,
            reports_dir=self.paths.future_context_reports_dir,
        )

    @property
    def article_scope_paths(self) -> ArticleScopePaths:
        return ArticleScopePaths(
            tables_dir=self.paths.article_scope_tables_dir,
            reports_dir=self.paths.article_scope_reports_dir,
        )

    @property
    def operational_paths(self) -> OperationalPaths:
        return OperationalPaths(
            forecasts_dir=self.paths.operational_forecasts_dir,
            reports_dir=self.paths.operational_reports_dir,
        )

    def validate(self) -> None:
        self.run.validate()
        self.depuration.validate()
        self.integration.validate()
        self.eda.validate()
        self.features.validate()
        self.modeling.validate()
        self.future_context.validate()
        self.article_scope.validate()
        self.operational.validate()

        if not isinstance(
            self.paths.structured_input_paths,
            dict,
        ):
            raise TypeError(
                'structured_input_paths must be a dictionary.'
            )

        for dataset_name, path in (
            self.paths.structured_input_paths.items()
        ):
            if not isinstance(
                dataset_name,
                str,
            ):
                raise TypeError(
                    'Every structured input key must be a string.'
                )

            if not isinstance(
                path,
                Path,
            ):
                raise TypeError(
                    'Every structured input path must be a pathlib.Path.'
                )


# =============================================================================
# PATH CONSTRUCTION
# =============================================================================


def build_project_paths(
    project_root: str | Path,
    raw_dir: str | Path | None = None,
    weekly_sales_dir: str | Path | None = None,
    pdf_tickets_dir: str | Path | None = None,
    future_weather_forecast_path: str | Path | None = None,
    structured_input_paths: dict[str, str | Path] | None = None,
) -> ProjectPaths:
    root = Path(
        project_root
    ).expanduser().resolve()

    data_dir = (
        root
        / 'data'
    )
    results_dir = (
        root
        / 'results'
    )

    resolved_raw_dir = (
        Path(
            raw_dir
        ).expanduser().resolve()
        if raw_dir is not None
        else data_dir / 'raw'
    )

    resolved_weekly_sales_dir = (
        Path(
            weekly_sales_dir
        ).expanduser().resolve()
        if weekly_sales_dir is not None
        else resolved_raw_dir / 'ventas_semanal'
    )

    resolved_pdf_tickets_dir = (
        Path(
            pdf_tickets_dir
        ).expanduser().resolve()
        if pdf_tickets_dir is not None
        else resolved_raw_dir / 'pdfs'
    )

    weather_path = (
        Path(
            future_weather_forecast_path
        ).expanduser().resolve()
        if future_weather_forecast_path is not None
        else None
    )

    structured = {
        str(
            dataset_name
        ): Path(
            path
        ).expanduser().resolve()
        for dataset_name, path in (
            structured_input_paths
            or {}
        ).items()
    }

    return ProjectPaths(
        project_root=root,
        raw_dir=resolved_raw_dir,
        weekly_sales_dir=resolved_weekly_sales_dir,
        pdf_tickets_dir=resolved_pdf_tickets_dir,
        future_weather_forecast_path=weather_path,
        bronze_dir=data_dir / 'bronze',
        silver_dir=data_dir / 'silver',
        gold_dir=data_dir / 'gold',
        features_dir=data_dir / 'features',
        results_dir=results_dir,
        depuration_figures_dir=results_dir / 'depuration_figures',
        depuration_reports_dir=results_dir / 'depuration_reports',
        integration_reports_dir=results_dir / 'integration_reports',
        eda_figures_dir=results_dir / 'eda_figures',
        eda_tables_dir=results_dir / 'eda_tables',
        eda_reports_dir=results_dir / 'eda_reports',
        feature_reports_dir=results_dir / 'feature_reports',
        modeling_tables_dir=results_dir / 'modeling_tables',
        modeling_reports_dir=results_dir / 'modeling_reports',
        modeling_predictions_dir=results_dir / 'modeling_predictions',
        future_context_tables_dir=results_dir / 'future_context_tables',
        future_context_reports_dir=results_dir / 'future_context_reports',
        article_scope_tables_dir=results_dir / 'article_scope_tables',
        article_scope_reports_dir=results_dir / 'article_scope_reports',
        operational_forecasts_dir=results_dir / 'operational_forecasts',
        operational_reports_dir=results_dir / 'operational_reports',
        structured_input_paths=structured,
    )


# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================


def build_pipeline_config(
    project_root: str | Path,
    *,
    do_ingest: bool = True,
    do_depuration: bool = True,
    do_eda: bool = True,
    do_features: bool = True,
    do_modeling: bool = True,
    do_operational: bool = True,
    raw_dir: str | Path | None = None,
    weekly_sales_dir: str | Path | None = None,
    pdf_tickets_dir: str | Path | None = None,
    future_weather_forecast_path: str | Path | None = None,
    structured_input_paths: dict[str, str | Path] | None = None,
    source_when_ingest_disabled: str = 'auto',
    verbose: bool = True,
    strict_ingestion: bool = False,
    recursive_raw_scan: bool = False,
    depuration_rules: DepurationRules | None = None,
    integration_rules: IntegrationRules | None = None,
    eda_rules: EDARules | None = None,
    feature_rules: FeatureRules | None = None,
    modeling_rules: ModelingRules | None = None,
    future_context_rules: FutureContextRules | None = None,
    article_scope_rules: ArticleScopeRules | None = None,
    operational_rules: OperationalRules | None = None,
) -> PipelineConfig:
    paths = build_project_paths(
        project_root=project_root,
        raw_dir=raw_dir,
        weekly_sales_dir=weekly_sales_dir,
        pdf_tickets_dir=pdf_tickets_dir,
        future_weather_forecast_path=future_weather_forecast_path,
        structured_input_paths=structured_input_paths,
    )

    run = RunOptions(
        do_ingest=do_ingest,
        do_depuration=do_depuration,
        do_eda=do_eda,
        do_features=do_features,
        do_modeling=do_modeling,
        do_operational=do_operational,
        verbose=verbose,
        strict_ingestion=strict_ingestion,
        recursive_raw_scan=recursive_raw_scan,
        source_when_ingest_disabled=source_when_ingest_disabled,
    )

    config = PipelineConfig(
        run=run,
        paths=paths,
        depuration=(
            depuration_rules
            if depuration_rules is not None
            else DepurationRules()
        ),
        integration=(
            integration_rules
            if integration_rules is not None
            else IntegrationRules()
        ),
        eda=(
            eda_rules
            if eda_rules is not None
            else EDARules()
        ),
        features=(
            feature_rules
            if feature_rules is not None
            else FeatureRules()
        ),
        modeling=(
            modeling_rules
            if modeling_rules is not None
            else ModelingRules()
        ),
        future_context=(
            future_context_rules
            if future_context_rules is not None
            else FutureContextRules()
        ),
        article_scope=(
            article_scope_rules
            if article_scope_rules is not None
            else ArticleScopeRules()
        ),
        operational=(
            operational_rules
            if operational_rules is not None
            else OperationalRules()
        ),
    )

    config.validate()

    return config


# =============================================================================
# DIRECTORY PREPARATION
# =============================================================================


def prepare_repository_directories(
    config: PipelineConfig,
    create_raw_input_directories: bool = True,
) -> None:
    config.validate()

    output_directories = [
        config.paths.bronze_dir,
        config.paths.silver_dir,
        config.paths.gold_dir,
        config.paths.features_dir,
        config.paths.depuration_figures_dir,
        config.paths.depuration_reports_dir,
        config.paths.integration_reports_dir,
        config.paths.eda_figures_dir,
        config.paths.eda_tables_dir,
        config.paths.eda_reports_dir,
        config.paths.feature_reports_dir,
        config.paths.modeling_tables_dir,
        config.paths.modeling_reports_dir,
        config.paths.modeling_predictions_dir,
        config.paths.future_context_tables_dir,
        config.paths.future_context_reports_dir,
        config.paths.article_scope_tables_dir,
        config.paths.article_scope_reports_dir,
        config.paths.operational_forecasts_dir,
        config.paths.operational_reports_dir,
    ]

    for directory in output_directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if create_raw_input_directories:
        for directory in [
            config.paths.raw_dir,
            config.paths.weekly_sales_dir,
            config.paths.pdf_tickets_dir,
        ]:
            if directory is not None:
                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )


# =============================================================================
# RUNTIME VALIDATION
# =============================================================================


def _parquet_available(
    directory: Path,
) -> bool:
    return (
        directory.exists()
        and any(
            directory.glob(
                '*.parquet'
            )
        )
    )


def _require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f'{description} not found: {path}'
        )


def validate_runtime_inputs(
    config: PipelineConfig,
) -> list[str]:
    config.validate()

    warnings: list[str] = []

    if config.run.do_ingest:
        if not config.paths.raw_dir.exists():
            raise FileNotFoundError(
                'DO_INGEST=True but raw_dir does not exist: '
                f'{config.paths.raw_dir}'
            )

        raw_files = [
            path
            for path in config.paths.raw_dir.rglob(
                '*'
            )
            if path.is_file()
        ]

        if not raw_files:
            raise FileNotFoundError(
                'DO_INGEST=True but no source files were found below '
                f'{config.paths.raw_dir}.'
            )

    elif config.run.do_depuration:
        mode = config.run.source_when_ingest_disabled

        bronze_available = _parquet_available(
            config.paths.bronze_dir
        )
        structured_available = bool(
            config.paths.structured_input_paths
        )

        if (
            mode == 'bronze'
            and not bronze_available
        ):
            raise FileNotFoundError(
                'Bronze mode selected but no Bronze parquet files exist.'
            )

        if (
            mode == 'structured'
            and not structured_available
        ):
            raise ValueError(
                'Structured mode selected but structured_input_paths is empty.'
            )

        if (
            mode == 'auto'
            and not bronze_available
            and not structured_available
        ):
            raise FileNotFoundError(
                'DO_INGEST=False + DO_DEPURATION=True requires existing Bronze '
                'or structured inputs.'
            )

    needs_saved_silver = (
        not config.run.do_depuration
        and (
            (
                config.run.do_eda
                and config.eda.analyze_silver
            )
            or config.run.do_operational
        )
    )

    if (
        needs_saved_silver
        and not _parquet_available(
            config.paths.silver_dir
        )
    ):
        raise FileNotFoundError(
            'This execution mode requires persisted Silver parquet files.'
        )

    needs_saved_gold = (
        not config.run.do_depuration
        and (
            (
                config.run.do_eda
                and config.eda.analyze_gold
            )
            or config.run.do_features
            or config.run.do_operational
        )
    )

    if (
        needs_saved_gold
        and not _parquet_available(
            config.paths.gold_dir
        )
    ):
        raise FileNotFoundError(
            'This execution mode requires persisted Gold parquet files.'
        )

    needs_saved_features = (
        not config.run.do_features
        and (
            config.run.do_modeling
            or config.run.do_operational
        )
    )

    if (
        needs_saved_features
        and not _parquet_available(
            config.paths.features_dir
        )
    ):
        raise FileNotFoundError(
            'Modeling/operational mode requires persisted feature parquet '
            f'files in {config.paths.features_dir}.'
        )

    if config.run.do_modeling:
        if not config.run.do_features:
            _require_file(
                config.paths.feature_reports_dir
                / 'feature_catalogue.csv',
                'Feature catalogue',
            )

    if (
        config.run.do_operational
        and not config.run.do_modeling
    ):
        for path, description in [
            (
                config.paths.modeling_tables_dir
                / 'feature_selection.csv',
                'Frozen feature selection',
            ),
            (
                config.paths.modeling_tables_dir
                / 'frozen_adaptive_mapping.csv',
                'Frozen adaptive mapping',
            ),
            (
                config.paths.modeling_reports_dir
                / 'modeling_development_report.json',
                'Modeling report',
            ),
        ]:
            _require_file(
                path,
                description,
            )

    weather_path = (
        config.paths.future_weather_forecast_path
    )

    if (
        weather_path is not None
        and not weather_path.exists()
    ):
        warnings.append(
            'Configured future weather forecast does not exist and will be '
            f'ignored: {weather_path}'
        )

    return warnings


# =============================================================================
# SOURCE DECISION HELPERS
# =============================================================================


def should_load_bronze(
    config: PipelineConfig,
) -> bool:
    if config.run.do_ingest:
        return False

    mode = (
        config.run.source_when_ingest_disabled
    )

    if mode == 'bronze':
        return True

    if mode == 'structured':
        return False

    return _parquet_available(
        config.paths.bronze_dir
    )


def should_load_structured_inputs(
    config: PipelineConfig,
) -> bool:
    if config.run.do_ingest:
        return False

    if (
        config.run.source_when_ingest_disabled
        == 'structured'
    ):
        return True

    if (
        config.run.source_when_ingest_disabled
        == 'bronze'
    ):
        return False

    return not should_load_bronze(
        config
    )


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================


def configuration_summary(
    config: PipelineConfig,
) -> dict[str, Any]:
    config.validate()

    return {
        'run': asdict(
            config.run
        ),
        'paths': {
            key: (
                {
                    name: str(
                        path
                    )
                    for name, path in value.items()
                }
                if isinstance(
                    value,
                    dict,
                )
                else (
                    str(
                        value
                    )
                    if value is not None
                    else None
                )
            )
            for key, value in vars(
                config.paths
            ).items()
        },
        'depuration_rules': asdict(
            config.depuration
        ),
        'integration_rules': asdict(
            config.integration
        ),
        'eda_rules': asdict(
            config.eda
        ),
        'feature_rules': asdict(
            config.features
        ),
        'modeling_rules': asdict(
            config.modeling
        ),
        'future_context_rules': asdict(
            config.future_context
        ),
        'article_scope_rules': asdict(
            config.article_scope
        ),
        'operational_rules': asdict(
            config.operational
        ),
    }


def print_configuration(
    config: PipelineConfig,
) -> None:
    summary = configuration_summary(
        config
    )

    line = '=' * 96
    print(
        f'\n{line}'
    )
    print(
        'ANALYSIS PREDICTION | ACTIVE CONFIGURATION'
    )
    print(
        line
    )

    print(
        '\nEXECUTION'
    )

    for label, key in [
        (
            'DO_INGEST',
            'do_ingest',
        ),
        (
            'DO_DEPURATION',
            'do_depuration',
        ),
        (
            'DO_EDA',
            'do_eda',
        ),
        (
            'DO_FEATURES',
            'do_features',
        ),
        (
            'DO_MODELING',
            'do_modeling',
        ),
        (
            'DO_OPERATIONAL',
            'do_operational',
        ),
    ]:
        print(
            f'  {label:<26}= '
            f'{summary["run"][key]}'
        )

    print(
        f'  {"source if no ingestion":<26}= '
        f'{summary["run"]["source_when_ingest_disabled"]}'
    )

    print(
        '\nCORE DATA PATHS'
    )

    for label, key in [
        (
            'Raw',
            'raw_dir',
        ),
        (
            'Bronze',
            'bronze_dir',
        ),
        (
            'Silver',
            'silver_dir',
        ),
        (
            'Gold',
            'gold_dir',
        ),
        (
            'Features',
            'features_dir',
        ),
    ]:
        print(
            f'  {label:<26}= '
            f'{summary["paths"][key]}'
        )

    print(
        '\nMODELING / OPERATIONAL'
    )
    print(
        f'  final test evaluation      = '
        f'{config.modeling.evaluate_final_test}'
    )
    print(
        f'  adaptive selector          = '
        f'{config.modeling.run_adaptive_regime_selector}'
    )
    print(
        f'  article-scope policy       = '
        f'{config.article_scope.policy}'
    )
    print(
        f'  operational article scope  = '
        f'{config.operational.article_scope}'
    )
    print(
        f'  future weather forecast    = '
        f'{summary["paths"]["future_weather_forecast_path"]}'
    )
