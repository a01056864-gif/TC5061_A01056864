"""Entrena un RandomForestRegressor para predecir el rating de IMDb y registra la corrida en MLflow.

Uso:
    python -m imdb_rating.modeling.train --help
    python -m imdb_rating.modeling.train --n-estimators 300 --max-depth 15 --seed 42
"""

import json
from pathlib import Path
import random
import tempfile

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
import typer

from imdb_rating.config import (
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_FILE,
    PROJ_ROOT,
    RAW_DATA_FILE,
)
from imdb_rating.dataset import build_dataset
from imdb_rating.features import build_preprocessor, columnas_binarias, split_data
from imdb_rating.plots import plot_feature_importance, plot_real_vs_predicho, plot_residuos

app = typer.Typer()

SKOPS_TRUSTED_TYPES = ["imdb_rating.features.ImputadorMedianaPorGrupo", "sklearn.tree._tree.Tree"]


def set_seed(seed: int) -> None:
    """Fija las semillas globales; los objetos de sklearn reciben la semilla explícitamente."""
    random.seed(seed)
    np.random.seed(seed)


def parse_max_features(valor: str) -> str | float:
    """'sqrt' / 'log2' se dejan como texto; un número se interpreta como fracción de variables."""
    if valor in ("sqrt", "log2"):
        return valor
    fraccion = float(valor)
    if not 0 < fraccion <= 1:
        raise typer.BadParameter("max_features debe ser 'sqrt', 'log2' o un número en (0, 1].")
    return fraccion


def load_data(path: Path) -> pd.DataFrame:
    """Lee el dataset limpio; si no existe, lo genera a partir del CSV original."""
    if not path.exists():
        logger.info(f"{path} no existe; se genera desde {RAW_DATA_FILE}")
        return build_dataset(RAW_DATA_FILE, path)
    return pd.read_csv(path)


def build_model(
    cols_binarias: list[str],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    seed: int,
) -> Pipeline:
    """Preprocesamiento + RandomForestRegressor en un solo Pipeline."""
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preproceso", build_preprocessor(cols_binarias)), ("modelo", modelo)])


def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": root_mean_squared_error(y_real, y_pred),
        "r2": r2_score(y_real, y_pred),
    }


def evaluate(pipe: Pipeline, X_train, y_train, X_test, y_test, cv_folds: int, seed: int):
    """Validación cruzada sobre entrenamiento, ajuste final y métricas en prueba."""
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_mae = -scores["test_score"]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test = calcular_metricas(y_test, y_pred)
    metricas = {
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_r2": test["r2"],
        "train_mae": mean_absolute_error(y_train, pipe.predict(X_train)),
        "cv_mae_mean": cv_mae.mean(),
        "cv_mae_std": cv_mae.std(),
    }
    return metricas, y_pred


def save_artifacts(pipe: Pipeline, y_test, y_pred, metricas: dict, out_dir: Path) -> None:
    """Genera gráficas y metrics.json en out_dir para registrarlos como artefactos."""
    nombres = pipe.named_steps["preproceso"].get_feature_names_out()
    nombres = pd.Index(nombres).str.replace(r"^(num|cat|bin)__", "", regex=True)
    importancias = pd.Series(pipe.named_steps["modelo"].feature_importances_, index=nombres)

    plot_real_vs_predicho(y_test, y_pred, out_dir / "real_vs_predicho.png")
    plot_residuos(y_test, y_pred, out_dir / "residuos.png")
    plot_feature_importance(importancias, out_dir / "feature_importance.png")
    (out_dir / "metrics.json").write_text(json.dumps(metricas, indent=2))


def get_or_create_experiment(name: str) -> str:
    """Reutiliza el experimento si existe; si no, lo crea con artefactos dentro del proyecto."""
    experimento = mlflow.get_experiment_by_name(name)
    if experimento is not None:
        return experimento.experiment_id
    return mlflow.create_experiment(name, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri())


@app.command()
def main(
    n_estimators: int = typer.Option(300, min=1, help="Número de árboles del bosque."),
    max_depth: int = typer.Option(
        15, min=0, help="Profundidad máxima de cada árbol; 0 = sin límite."
    ),
    min_samples_leaf: int = typer.Option(
        5, min=1, help="Mínimo de observaciones por hoja (más alto = más regularización)."
    ),
    max_features: str = typer.Option(
        "1.0", help="Variables evaluadas por división: 'sqrt', 'log2' o fracción en (0, 1]."
    ),
    seed: int = typer.Option(42, help="Semilla para split, validación cruzada y modelo."),
    test_size: float = typer.Option(0.2, min=0.05, max=0.5, help="Proporción de prueba."),
    cv_folds: int = typer.Option(5, min=2, help="Particiones de validación cruzada."),
    data_path: Path = typer.Option(PROCESSED_DATA_FILE, help="CSV limpio de entrada."),
    experiment_name: str = typer.Option(EXPERIMENT_NAME, help="Experimento de MLflow."),
    run_name: str = typer.Option(None, help="Nombre de la corrida (opcional)."),
):
    """Entrena el modelo con los hiperparámetros indicados y registra la corrida en MLflow."""
    set_seed(seed)
    profundidad = max_depth or None
    max_features_valor = parse_max_features(max_features)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
    pipe = build_model(
        columnas_binarias(df),
        n_estimators,
        profundidad,
        min_samples_leaf,
        max_features_valor,
        seed,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(experiment_name))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"model_type": "RandomForestRegressor", "dataset": data_path.name})
        params = {
            "n_estimators": n_estimators,
            "max_depth": profundidad,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features_valor,
            "seed": seed,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        for nombre, valor in params.items():
            mlflow.log_param(nombre, valor)

        metricas, y_pred = evaluate(pipe, X_train, y_train, X_test, y_test, cv_folds, seed)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        with tempfile.TemporaryDirectory() as tmp:
            save_artifacts(pipe, y_test, y_pred, metricas, Path(tmp))
            for archivo in sorted(Path(tmp).iterdir()):
                mlflow.log_artifact(str(archivo))

        ejemplo = X_train.head(5)
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(ejemplo, pipe.predict(ejemplo)),
            input_example=ejemplo,
            # skops solo carga tipos declarados como confiables: nuestro imputador y los árboles
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            # el imputador vive en este paquete; se empaqueta para poder cargar el modelo fuera
            code_paths=[str(PROJ_ROOT / "imdb_rating")],
        )

        logger.info(" | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
        logger.success(f"Run {run.info.run_id} registrado en '{experiment_name}'")


if __name__ == "__main__":
    app()
