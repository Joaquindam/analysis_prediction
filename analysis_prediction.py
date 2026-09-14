from pathlib import Path

from src.ap_config import (
    ArticleScopeRules,
    DepurationRules,
    EDARules,
    FeatureRules,
    FutureContextRules,
    IntegrationRules,
    ModelingRules,
    OperationalRules,
    build_pipeline_config,
)
from src.ap_pipeline import run_pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parent


# Pipeline stages
DO_INGEST = True
DO_DEPURATION = True
DO_EDA = True
DO_FEATURES = True
DO_MODELING = True
DO_OPERATIONAL = True


# Raw-data locations
RAW_DIR = (
    PROJECT_ROOT
    / 'data'
    / 'raw'
)
WEEKLY_SALES_DIR = (
    RAW_DIR
    / 'ventas_semanal'
)
PDF_TICKETS_DIR = (
    RAW_DIR
    / 'pdfs'
)


# Optional real weather forecast available at prediction time.
# Keep None when no legitimate forecast snapshot is available.
FUTURE_WEATHER_FORECAST_PATH = None


# Used only when ingestion is deliberately bypassed.
STRUCTURED_INPUT_PATHS = {}
SOURCE_WHEN_INGEST_DISABLED = 'auto'


# Console / ingestion behaviour
VERBOSE = True
STRICT_INGESTION = False
RECURSIVE_RAW_SCAN = False


DEPURATION_RULES = DepurationRules(
    drop_exact_duplicates=True,
    drop_rows_missing_required_keys=True,

    auto_impute_low_missingness=True,
    max_missing_fraction_for_auto_imputation=0.05,
    numeric_imputation_strategy='median',
    categorical_imputation_strategy='constant',
    categorical_missing_label='UNKNOWN',

    detect_outliers=True,
    outlier_action='flag',
    iqr_multiplier=1.5,
    robust_z_threshold=3.5,
    min_rows_for_outlier_detection=8,
    min_unique_values_for_outlier_detection=5,
    max_outliers_to_print_per_column=10,

    large_group_people=8,
    payment_reconciliation_tolerance=0.02,
    invoice_reconciliation_tolerance=0.02,
    plausible_temperature_min_c=-60.0,
    plausible_temperature_max_c=60.0,

    save_quality_reports=True,
    save_figures=True,
    save_outlier_details=True,
)


INTEGRATION_RULES = IntegrationRules(
    build_daily_master=True,
    build_weekly_article_master=True,

    create_continuous_daily_calendar=True,
    anchor_daily_calendar_to_business_activity=True,

    fill_missing_activity_counts_with_zero=True,
    fill_missing_boolean_context_with_false=True,

    prefer_ticket_revenue=True,

    keep_invitations=False,
    aggregate_duplicate_article_period_rows=True,

    save_integration_report=True,
    save_column_provenance=True,
)


EDA_RULES = EDARules(
    analyze_silver=True,
    analyze_gold=True,

    clean_previous_outputs=True,
    save_figures=True,
    save_tables=True,
    save_report=True,

    analyze_reservations=True,
    analyze_unique_events=True,
    generic_numeric_distributions=False,
)


FEATURE_RULES = FeatureRules(
    build_daily_features=True,
    build_article_period_features=True,

    include_calendar_features=True,
    include_cyclical_calendar_features=True,
    include_current_known_context=True,
    aggregate_daily_context_to_article_period=True,

    include_weather_context=True,

    require_complete_article_period_for_training=True,
    exclude_negative_targets_from_training=True,
)


MODELING_RULES = ModelingRules(
    run_ml_candidates=True,
    use_hist_gradient_boosting_poisson=True,
    use_random_forest=True,

    run_adaptive_regime_selector=True,
    adaptive_min_prior_regime_rows=20,
    adaptive_min_prior_global_rows=50,
    adaptive_initial_baseline_kind='expanding',

    # The final holdout has already been formally opened and the architecture
    # frozen. True reproduces the definitive TFM evaluation.
    evaluate_final_test=True,

    save_tables=True,
    save_predictions=True,
    save_report=True,
)


FUTURE_CONTEXT_RULES = FutureContextRules(
    include_events=True,
    include_holidays=True,
    include_weather_forecast=True,

    save_tables=True,
    save_report=True,
)


ARTICLE_SCOPE_RULES = ArticleScopeRules(
    recent_periods_short=4,
    recent_periods_long=8,

    # Do not silently remove stale historical products when no authoritative
    # active/inactive catalogue signal exists.
    policy='conservative_observed_history',

    save_tables=True,
    save_report=True,
)


OPERATIONAL_RULES = OperationalRules(
    article_scope='all_observed',

    round_prediction_to_units=True,
    abc_threshold_a=0.80,
    abc_threshold_b=0.95,

    save_forecast=True,
    save_summary=True,
    save_report=True,
)


CONFIG = build_pipeline_config(
    project_root=PROJECT_ROOT,

    do_ingest=DO_INGEST,
    do_depuration=DO_DEPURATION,
    do_eda=DO_EDA,
    do_features=DO_FEATURES,
    do_modeling=DO_MODELING,
    do_operational=DO_OPERATIONAL,

    raw_dir=RAW_DIR,
    weekly_sales_dir=WEEKLY_SALES_DIR,
    pdf_tickets_dir=PDF_TICKETS_DIR,
    future_weather_forecast_path=FUTURE_WEATHER_FORECAST_PATH,

    structured_input_paths=STRUCTURED_INPUT_PATHS,
    source_when_ingest_disabled=SOURCE_WHEN_INGEST_DISABLED,

    verbose=VERBOSE,
    strict_ingestion=STRICT_INGESTION,
    recursive_raw_scan=RECURSIVE_RAW_SCAN,

    depuration_rules=DEPURATION_RULES,
    integration_rules=INTEGRATION_RULES,
    eda_rules=EDA_RULES,
    feature_rules=FEATURE_RULES,
    modeling_rules=MODELING_RULES,
    future_context_rules=FUTURE_CONTEXT_RULES,
    article_scope_rules=ARTICLE_SCOPE_RULES,
    operational_rules=OPERATIONAL_RULES,
)


# =============================================================================
# MAIN
# =============================================================================


def main():
    return run_pipeline(
        CONFIG
    )


if __name__ == '__main__':
    main()
