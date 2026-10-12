"""Gráficas de diagnóstico que se registran como artefactos en MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: el script corre desde la terminal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_real_vs_predicho(y_real, y_pred, path: Path) -> Path:
    """Dispersión real vs predicho; un modelo perfecto pone todo sobre la diagonal."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_real, y_pred, alpha=0.3, s=10)
    ax.plot([1, 10], [1, 10], "k--", lw=1, label="Predicción perfecta")
    ax.set(
        xlim=(1, 10),
        ylim=(1, 10),
        xlabel="Rating real",
        ylabel="Rating predicho",
        title="Real vs predicho (prueba)",
    )
    ax.legend(loc="upper left")
    return _guardar(fig, path)


def plot_residuos(y_real, y_pred, path: Path) -> Path:
    """Histograma de residuos (real - predicho)."""
    residuos = np.asarray(y_real) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuos, bins=50)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(
        xlabel="Residuo (real - predicho)",
        ylabel="Frecuencia",
        title="Distribución de residuos (prueba)",
    )
    return _guardar(fig, path)


def plot_feature_importance(importancias: pd.Series, path: Path, top: int = 20) -> Path:
    """Barras horizontales con las variables más importantes del Random Forest."""
    datos = importancias.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    datos.plot.barh(ax=ax)
    ax.set(xlabel="Importancia (reducción de impureza)", title=f"Top {top} variables")
    return _guardar(fig, path)


def _guardar(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
