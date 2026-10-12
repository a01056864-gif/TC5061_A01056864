import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "IMBD.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "imdb_clean.csv"

# MLflow: por defecto un tracking store local (SQLite) en la raíz del proyecto.
# Se puede sobrescribir con la variable de entorno MLFLOW_TRACKING_URI (p. ej. en .env).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlruns"
EXPERIMENT_NAME = "IMDB_Rating_RF_Experimentacion"

# Limpieza: reglas deterministas definidas por dominio (ver notebook S03, Paso 3)
TOPE_DURACION = 300  # minutos; títulos más largos son miniseries con duración total
CERT_VALIDAS = [
    "TV-MA",
    "TV-14",
    "TV-PG",
    "TV-Y7",
    "TV-Y",
    "TV-G",
    "R",
    "PG-13",
    "PG",
    "G",
    "Not Rated",
]
CERT_EQUIVALENTES = {"Unrated": "Not Rated", "TV-Y7-FV": "TV-Y7"}
TIPOS_VALIDOS = ["Película", "Serie", "TV Special", "TV Movie", "Video"]

# Columnas del modelo (las de género gen_* se detectan a partir de los datos)
OBJETIVO = "rating"
COLS_NUMERICAS = ["duracion_min", "log10_votos", "anio_inicio"]
COLS_BINARIAS_BASE = ["en_emision", "duracion_es_total"]
COLS_CATEGORICAS = ["tipo_titulo", "certificado"]
PREFIJO_GENERO = "gen_"
# Categoría omitida en el one-hot de cada variable de COLS_CATEGORICAS
CATEGORIAS_REFERENCIA = ["Película", "Sin clasificación"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
