# ITESM_MNA_2026_TC5061_Experimentacion_MLFlow_CarlosMartinez

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Experimentación con MLflow para predecir el **rating de IMDb** de películas y series a partir de
sus metadatos (duración, votos, año, tipo de título, clasificación y géneros) con un
`RandomForestRegressor`. TC5061 — Operaciones de aprendizaje automático, Semana 4.

Autor: Carlos David Martínez Calva (A01056864)

## 1. Preparar el ambiente

Requiere conda y Python 3.14. Desde esta carpeta:

```bash
conda create -n TC5061_mlflow python=3.14 -y
conda activate TC5061_mlflow
pip install -r requirements.txt     # incluye `-e .` para instalar el paquete imdb_rating
```

El dataset original ([Kaggle: payamamanat/imbd-dataset](https://www.kaggle.com/datasets/payamamanat/imbd-dataset))
ya está incluido en `data/raw/IMBD.csv`.

## 2. Entrenar

Con los valores por defecto (no requiere argumentos):

```bash
python -m imdb_rating.modeling.train
```

Cambiando hiperparámetros sin editar código:

```bash
python -m imdb_rating.modeling.train --n-estimators 500 --max-depth 20 --min-samples-leaf 3 --max-features 0.5 --seed 42
```

Si `data/processed/imdb_clean.csv` no existe, el script lo genera a partir del CSV original
(también se puede generar aparte con `make data`).

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--n-estimators` | 300 | Número de árboles del bosque |
| `--max-depth` | 15 | Profundidad máxima de cada árbol; `0` = sin límite |
| `--min-samples-leaf` | 5 | Mínimo de observaciones por hoja (más alto = más regularización) |
| `--max-features` | 1.0 | Variables evaluadas por división: `sqrt`, `log2` o fracción en (0, 1] |
| `--seed` | 42 | Semilla para el split, la validación cruzada y el modelo |
| `--test-size` | 0.2 | Proporción del conjunto de prueba |
| `--cv-folds` | 5 | Particiones de validación cruzada sobre entrenamiento |
| `--data-path` | `data/processed/imdb_clean.csv` | CSV limpio de entrada |
| `--experiment-name` | `IMDB_Rating_RF_Experimentacion` | Experimento de MLflow |
| `--run-name` | (automático) | Nombre de la corrida |

`python -m imdb_rating.modeling.train --help` muestra la misma ayuda.

## 3. Qué se registra en MLflow

Cada ejecución crea una corrida en el experimento `IMDB_Rating_RF_Experimentacion`:

| Tipo | Contenido |
|---|---|
| Parámetros | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `seed`, `test_size`, `cv_folds`, `n_train`, `n_test` |
| Métricas | `test_mae`, `test_rmse`, `test_r2`, `train_mae`, `cv_mae_mean`, `cv_mae_std` |
| Artefactos | `real_vs_predicho.png`, `residuos.png`, `feature_importance.png`, `metrics.json` |
| Modelo | Pipeline completo (preprocesamiento + Random Forest) con firma y ejemplo de entrada |

Por defecto el tracking es local: metadatos en `mlflow.db` (SQLite) y artefactos en `mlruns/`.
Para usar otro servidor, define `MLFLOW_TRACKING_URI` en el ambiente o en un archivo `.env`.

## 4. Correr las configuraciones del experimento y compararlas

```bash
make experiments    # 6 corridas con configuraciones distintas
make mlflow-ui      # abre http://127.0.0.1:8080
```

En la UI selecciona el experimento, marca las corridas y da clic en **Compare**.

## 5. Reproducibilidad

- La semilla (`--seed`) se aplica a `random`, `numpy`, `train_test_split`, `KFold` y
  `RandomForestRegressor`. Dos ejecuciones con los mismos argumentos producen las mismas métricas.
- Con `n_jobs=-1` el orden en que se promedian los árboles varía entre hilos, lo que genera
  diferencias del orden de 1e-15 en las predicciones individuales, sin efecto en las métricas.
- Las versiones de las dependencias están fijadas en `requirements.txt`.
- `make test` verifica la limpieza, el imputador y que la misma semilla dé las mismas predicciones.

## 6. Organización

```
├── Makefile           <- make data | train | experiments | mlflow-ui | test | lint
├── data/raw           <- IMBD.csv original (versionado)
├── data/processed     <- dataset limpio (se regenera; no se versiona)
├── notebooks          <- solo exploración de datos (EDA)
├── reports/figures    <- figuras para el reporte
├── tests              <- pruebas con pytest
└── imdb_rating
    ├── config.py      <- rutas, nombre del experimento y reglas de limpieza
    ├── dataset.py     <- carga y limpieza determinista del CSV
    ├── features.py    <- split estratificado, imputador por grupo y preprocesamiento
    ├── plots.py       <- gráficas que se registran como artefactos
    └── modeling
        └── train.py   <- CLI de entrenamiento instrumentada con MLflow
```
