# Analysis & Prediction

Framework modular de análisis, modelado temporal y apoyo a la decisión para pequeños negocios hosteleros con históricos escasos y heterogéneos.

El objetivo de este repositorio es transformar datos operativos procedentes de distintas fuentes como ventas por artículo, tickets, reservas, facturas, eventos, festivos y meteorología, en predicciones útiles para la planificación del siguiente periodo de operación.

La parte desarrollada en este pipeline se centra en predecir la demanda de cada artículo en el siguiente periodo de reporte, expresada en unidades previstas por artículo.

El sistema no convierte directamente artículos en ingredientes. Para generar necesidades de compra a nivel de ingrediente sería necesario disponer de una relación explícita receta/BOM entre cada artículo y sus ingredientes.


## 1. Objetivo del proyecto

La pregunta principal que guía este desarrollo es:

> **¿Puede un pipeline automático, basado en validación temporal y selección adaptativa de modelos, proporcionar predicciones operativas útiles en pequeños negocios hosteleros con históricos escasos y heterogéneos?**

El framework ha sido diseñado para:

- integrar fuentes con distintas granularidades

- conservar trazabilidad entre `Raw`, `Bronze`, `Silver`, `Gold` y `Features`

- evitar leakage temporal

- comparar modelos complejos con baselines causales simples

- aislar un holdout temporal final

- adaptar el modelo al régimen histórico de demanda de cada artículo

- incorporar información futura conocida en el momento de predicción

- generar una salida operativa directamente interpretable por negocio.


## 2. Arquitectura general

El pipeline completo es:

```text
Raw
 ↓
Ingestion
 ↓
Bronze
 ↓
Depuration
 ↓
Silver
 ↓
Integration
 ↓
Gold
 ↓
EDA
 ↓
Feature engineering
 ↓
Temporal modeling
 ↓
Future context
 ↓
Article scope
 ↓
Operational forecast
```

La ejecución completa se coordina desde:

`analysis_prediction.py`

El main delega la lógica de ejecución en:

`src/ap_pipeline.py`

y la configuración central del proyecto se define mediante:

`src/ap_config.py`


## 3. Estructura del repositorio

Una estructura esperada del proyecto es:

```text
analysis_prediction/
│
├── analysis_prediction.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── ap_config.py
│   ├── ap_pipeline.py
│   ├── ap_io.py
│   ├── ap_depuration.py
│   ├── ap_integration.py
│   ├── ap_eda.py
│   ├── ap_features.py
│   ├── ap_modeling.py
│   ├── ap_future_context.py
│   ├── ap_article_scope.py
│   └── ap_operational.py
│
├── data/
│   ├── raw/
│   │   ├── ventas_semanal/
│   │   ├── pdfs/
│   │   └── ...
│   │
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── features/
│
└── results/
    ├── depuration_figures/
    ├── depuration_reports/
    ├── integration_reports/
    ├── eda_figures/
    ├── eda_tables/
    ├── eda_reports/
    ├── feature_reports/
    ├── modeling_tables/
    ├── modeling_reports/
    ├── modeling_predictions/
    ├── future_context_tables/
    ├── future_context_reports/
    ├── article_scope_tables/
    ├── article_scope_reports/
    ├── operational_forecasts/
    └── operational_reports/
```

Los antiguos scripts `test_*.py` se utilizaron durante el desarrollo y validación modular, pero ya no forman parte de la ejecución normal del proyecto.


## 4. Instalación

Se recomienda trabajar dentro de un entorno virtual.


### Windows / PowerShell

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```


### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El fichero `requirements.txt` contiene únicamente las dependencias directas necesarias para el pipeline actual, evitando incluir todo el entorno Jupyter o paquetes auxiliares no utilizados por la ejecución principal.


## 5. Dependencias principales

El pipeline utiliza principalmente:

- numpy y pandas para tratamiento de datos

- pyarrow para persistencia Parquet

- openpyxl para .xlsx

- python-calamine para determinados .xls antiguos

- PyMuPDF para extracción de información desde PDFs

- matplotlib para figuras

- scipy para cálculos estadísticos utilizados durante el análisis

- scikit-learn para preprocessing y modelos predictivos.


## 6. Fuentes de datos

El framework trabaja con fuentes heterogéneas. En la ejecución actual se utilizan, entre otras:

- maestro de artículos

- carta / menú

- departamentos de venta

- ventas por artículo y periodo

- total acumulado de artículos

- tickets

- propinas

- reservas

- facturas extraídas desde PDF

- meteorología

- festivos

- eventos externos.

La ingestión normaliza nombres, tipos y metadatos técnicos, pero evita realizar limpieza semántica prematura.


## 7. Capas de datos


### Raw

Ficheros originales suministrados al proyecto.


### Bronze

Versión estandarizada técnicamente de las fuentes:

- nombres de columnas normalizados

- tipos básicos corregidos

- fechas interpretadas

- metadatos de procedencia conservados.

No se eliminan observaciones únicamente por resultar inusuales.


### Silver

Datos depurados por fuente.

La depuración incluye:

- detección de duplicados

- validación de claves

- tratamiento conservador de valores ausentes

- reglas lógicas específicas por dataset

- flags de calidad

- detección de outliers.

Los outliers se marcan por defecto, no se eliminan automáticamente.


### Gold

Se generan dos tablas maestras principales:

`data/gold/tabla_maestra_diaria.parquet`

`data/gold/tabla_maestra_semanal_articulos.parquet`

La primera trabaja a granularidad diaria.

La segunda utiliza la granularidad:

periodo de reporte × artículo

Las ventas semanales no se expanden artificialmente a días.


## 8. Exploratory Data Analysis

ap_eda.py analiza Silver y Gold sin modificar las observaciones de entrada.

El EDA incluye, entre otros:

- cobertura temporal

- distribuciones

- valores ausentes

- relaciones con targets

- concentración de demanda

- actividad por artículo

- análisis de reservas

- eventos únicos

- detección de periodos parciales

- señales potencialmente asociadas a leakage.

Los resultados se guardan en:

`results/eda_figures/`

`results/eda_tables/`

`results/eda_reports/`


## 9. Feature engineering

ap_features.py genera variables de manera explícitamente leakage-aware.

Principio fundamental:

Una feature segura por defecto debe ser conocida antes del periodo objetivo o derivarse exclusivamente de observaciones pasadas.

Se generan, entre otras:

- variables de calendario

- variables cíclicas

- lags temporales

- medias móviles

- medias históricas expanding

- actividad histórica del artículo

- cobertura histórica

- contexto de eventos/festivos

- variables meteorológicas históricas

- régimen histórico de demanda.

Las features con riesgo temporal se conservan para auditabilidad, pero no se seleccionan automáticamente para modelado.

Outputs:

`data/features/features_daily.parquet`

`data/features/features_article_period.parquet`

`results/feature_reports/feature_catalogue.csv`

`results/feature_reports/feature_report.json`


### Política sobre paneles dispersos

La ausencia de un artículo en un periodo histórico no se interpreta automáticamente como demanda cero.

Podría significar, por ejemplo:

- producto temporalmente no disponible

- rotación de carta

- producto retirado

- ausencia real de demanda

- falta de observación.

Sin una semántica explícita de disponibilidad, convertir esas ausencias en ceros introduciría una hipótesis no demostrada.


## 10. Modelado temporal

El target principal es:

`units`

es decir, unidades previstas de cada artículo para un periodo futuro.

El modelado utiliza separación estrictamente temporal.


### Baselines

Se comparan modelos sencillos y causales:

- Naive lag 1

- Rolling mean 4

- Historical expanding mean.


### Machine Learning

Se incluyen:

- HistGradientBoostingRegressor con pérdida Poisson

- RandomForestRegressor.

El preprocessing se ajusta únicamente con los datos de entrenamiento de cada fold.


### Validación

La evaluación sigue esta secuencia:

```text
histórico
├── Development
│   └── expanding-window cross-validation
│
└── Final temporal holdout
```

El holdout final se separa antes de realizar la selección de modelos.


## 11. Selector adaptativo por régimen

El framework no obliga a que todos los artículos utilicen el mismo modelo.

Cada artículo-periodo se clasifica usando exclusivamente información histórica anterior en uno de los siguientes regímenes:

- `cold_start`
- `frequent`
- `regular`
- `intermittent`
- `sparse`

El selector adaptativo aprende qué modelo utilizar para cada régimen utilizando únicamente evidencia out-of-fold anterior.

En la ejecución final, el mapping congelado fue:

```text
cold_start   -> HistGradientBoosting Poisson
frequent     -> Historical expanding mean
intermittent -> HistGradientBoosting Poisson
regular      -> Historical expanding mean
sparse       -> Historical expanding mean
```

Este mapping se congela antes de evaluar el holdout final.


## 12. Resultados del experimento final

En la ejecución final del dataset del proyecto:


### Development

4,507 filas
33 periodos

El selector adaptativo obtuvo aproximadamente:

WAPE = 0.279
MAE  = 5.155
RMSE = 9.979
Bias = -0.048


### Final temporal holdout

789 filas
6 periodos

Resultado del modelo primario congelado:

WAPE = 0.288
MAE  = 5.335
RMSE = 9.365
Bias = -0.010

Los resultados de otros modelos sobre el holdout final se consideran únicamente diagnósticos y no deben utilizarse para volver a seleccionar el modelo después de haber abierto el test.


## 13. Contexto futuro

ap_future_context.py genera contexto diario para el horizonte de predicción utilizando únicamente información que puede conocerse antes del periodo futuro.

Actualmente admite:

- eventos programados

- festivos

previsiones meteorológicas opcionales.

Nunca debe sustituirse una previsión meteorológica histórica por el tiempo realmente observado posteriormente, ya que eso introduciría leakage.

Si no existe una previsión meteorológica legítimamente disponible, el pipeline continúa sin ella.


## 14. Scope de artículos

ap_article_scope.py determina qué artículos pueden recibir una predicción operativa.

La lógica prioriza:

- señales explícitas de activo/inactivo

- información de vigencia temporal

- pertenencia informativa a carta

- histórico observado

recencia de demanda como señal de revisión.

En el dataset actual:

405 artículos en maestro
405 artículos en menú
251 artículos con histórico de demanda

Como no existe una señal fiable de activo/inactivo y el menú cubre el 100 % del maestro, el pipeline adopta una política conservadora:

251 artículos forecastables
55 artículos stale marcados para revisión manual

Los artículos antiguos no se eliminan automáticamente.


## 15. Forecast operativo

ap_operational.py transforma la arquitectura congelada en una previsión utilizable por negocio.

Durante deployment:

- el mapping de selección de modelos permanece congelado

- los modelos pueden reajustarse con todo el histórico ya conocido

- el antiguo holdout puede utilizarse para refit una vez terminada la evaluación experimental

no se modifica la regla de selección utilizando el rendimiento del test.

La salida incluye:

- unidades previstas

- unidades redondeadas

- ranking

- régimen de demanda

- modelo utilizado

- clase ABC

- estado de actividad

flag de revisión manual.

Outputs principales:

`results/operational_forecasts/next_period_article_forecast.csv`

`results/operational_forecasts/next_period_forecast_summary.csv`

`results/operational_forecasts/next_period_model_usage.csv`

`results/operational_forecasts/next_period_regime_usage.csv`

`results/operational_reports/operational_forecast_report.json`

En la ejecución final del proyecto, el periodo inferido automáticamente fue:

2026-07-13 -> 2026-07-19

con una previsión total aproximada de:

2,731.4 unidades

para 251 artículos.


## 16. Ejecución desde el main

La ejecución se controla mediante seis flags en `analysis_prediction.py`:

```python
DO_INGEST = True
DO_DEPURATION = True
DO_EDA = True
DO_FEATURES = True
DO_MODELING = True
DO_OPERATIONAL = True
```


### Pipeline completo

```python
DO_INGEST = True
DO_DEPURATION = True
DO_EDA = True
DO_FEATURES = True
DO_MODELING = True
DO_OPERATIONAL = True
```

Ejecutar:

```bash
python analysis_prediction.py
```


### Solo modelado y forecast usando capas persistidas

```python
DO_INGEST = False
DO_DEPURATION = False
DO_EDA = False
DO_FEATURES = False
DO_MODELING = True
DO_OPERATIONAL = True
```


### Solo nuevo forecast operativo

Útil cuando ya existen Silver, Gold, Features y artefactos del modelado:

```python
DO_INGEST = False
DO_DEPURATION = False
DO_EDA = False
DO_FEATURES = False
DO_MODELING = False
DO_OPERATIONAL = True
```

En este modo se reutilizan los artefactos existentes y se reconstruyen:

future context
→ article scope
→ next-period operational forecast


### Gold -> Features -> Modeling -> Operational

```python
DO_INGEST = False
DO_DEPURATION = False
DO_EDA = False
DO_FEATURES = True
DO_MODELING = True
DO_OPERATIONAL = True
```


## 17. Validación de combinaciones de ejecución

`ap_config.py` impide combinaciones que podrían mezclar capas nuevas con artefactos antiguos.

Por ejemplo, no se permite reconstruir Gold y después modelar utilizando Features antiguas:

```python
DO_DEPURATION = True
DO_FEATURES = False
DO_MODELING = True
```

Tampoco se permite reconstruir Features y generar un forecast utilizando artefactos de modelado desactualizados:

```python
DO_FEATURES = True
DO_MODELING = False
DO_OPERATIONAL = True
```

Estas restricciones evitan inconsistencias silenciosas entre capas.


## 18. Meteorología futura

La ruta de una previsión meteorológica legítima puede configurarse mediante:

`FUTURE_WEATHER_FORECAST_PATH = None`

Cuando sea None, el forecast se genera sin meteorología futura.

Si se proporciona una fuente válida, debe representar información que estuviese disponible en el instante de predicción.


## 19. Principios metodológicos

El proyecto sigue varias reglas de diseño:


### No leakage

Nunca se utiliza información que no estaría disponible en el instante de predicción.


### Validación temporal

No se utiliza una partición aleatoria convencional para seleccionar el modelo principal.


### Baselines obligatorios

Un modelo de Machine Learning solo se considera útil si aporta valor frente a reglas históricas simples.


### Holdout final aislado

El test final se abre una sola vez tras congelar la decisión de modelado.


### No eliminación automática de observaciones raras

Los outliers de negocio pueden representar actividad real.


### Ausencia no equivale a cero

Una fila artículo-periodo ausente no se convierte automáticamente en demanda nula.


### Complejidad no implica superioridad

El framework permite que un baseline simple sea seleccionado cuando generaliza mejor que un modelo ML.


## 20. Interpretación de métricas

Las métricas principales son:


### MAE

Error absoluto medio por predicción artículo-periodo.


### RMSE

Penaliza con mayor intensidad los errores grandes.


### WAPE

$$
\mathrm{WAPE}=\frac{\sum |y-\hat y|}{\sum |y|}
$$

Es la métrica principal de comparación global.


### Bias

$$
\mathrm{Bias}=\frac{\sum(\hat y-y)}{\sum y}
$$

Permite saber si el sistema tiende a sobrepredecir o infrapredecir volumen total.


### Top-K overlap

Mide la capacidad de recuperar los artículos de mayor demanda.


## 21. Limitaciones actuales

El framework presenta varias limitaciones que deben mantenerse explícitas:

- el histórico de ventas por artículo es relativamente corto

- el panel artículo-periodo es disperso

- no existe una semántica histórica fiable de disponibilidad de cada artículo

- no existe actualmente una variable autoritativa de producto activo/inactivo

- las reservas actuales no pueden utilizarse como feature segura sin snapshots históricos equivalentes

- la meteorología observada futura no puede utilizarse como sustituto de una previsión

- la predicción se realiza a nivel de artículo, no de ingrediente

- los artículos completamente nuevos requieren una estrategia de true cold-start distinta

el comportamiento de un único establecimiento no implica generalización inmediata a toda la hostelería.


## 22. Reproducibilidad

Los resultados principales del pipeline se persisten en CSV, JSON y Parquet.

Para reproducir el experimento desde las fuentes originales:

- crear y activar el entorno virtual

- instalar `requirements.txt`

- colocar las fuentes Raw en la estructura configurada

- activar todos los flags del pipeline

ejecutar:

```bash
python analysis_prediction.py
```

Para una congelación completa de todas las dependencias transitivas del entorno concreto utilizado en una máquina, puede generarse adicionalmente:

```bash
pip freeze > requirements-lock.txt
```

`requirements.txt` se mantiene deliberadamente como lista limpia de dependencias directas del proyecto.


## 23. Salida final esperada

Una ejecución completa termina con una sección similar a:

```text
FROZEN MODEL -> NEXT-PERIOD OPERATIONAL FORECAST

Forecast period: 2026-07-13 -> 2026-07-19
Reporting cadence: 7 days
Deployment refit: 5,296 eligible historical rows
Forecasted articles: 251
Predicted total units: ~2,731
Frozen predictors: 41
```

y genera el ranking operativo de artículos para el siguiente periodo.


## 24. Estado del proyecto

La arquitectura principal se considera cerrada a nivel funcional:

ap_io
ap_depuration
ap_integration
ap_eda
ap_features
ap_modeling
ap_future_context
ap_article_scope
ap_operational
ap_config
ap_pipeline

El pipeline completo ha sido ejecutado end-to-end desde `analysis_prediction.py`.

Los siguientes trabajos naturales corresponden principalmente a documentación, memoria del TFM, análisis de resultados y eventual incorporación de nuevas fuentes de datos o de una relación artículo → receta/ingrediente.
