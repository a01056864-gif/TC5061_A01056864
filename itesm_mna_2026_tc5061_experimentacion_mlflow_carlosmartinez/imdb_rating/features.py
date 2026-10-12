"""Separación de variables y pipeline de preprocesamiento.

Todo lo que aprende de los datos (medianas, escalado, categorías) se ajusta solo con
entrenamiento dentro del Pipeline, y se aplica igual a prueba y a datos nuevos.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imdb_rating.config import (
    CATEGORIAS_REFERENCIA,
    COLS_BINARIAS_BASE,
    COLS_CATEGORICAS,
    COLS_NUMERICAS,
    OBJETIVO,
    PREFIJO_GENERO,
)


class ImputadorMedianaPorGrupo(BaseEstimator, TransformerMixin):
    """Imputa una columna numérica con la mediana de su grupo, aprendida solo en fit().

    Si un grupo no se vio en entrenamiento (o no tiene valores), usa la mediana global.
    """

    def __init__(self, columna="duracion_min", grupo="tipo_titulo"):
        self.columna = columna
        self.grupo = grupo

    def fit(self, X, y=None):
        self.medianas_ = X.groupby(self.grupo)[self.columna].median().to_dict()
        self.mediana_global_ = X[self.columna].median()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def get_feature_names_out(self, input_features=None):
        # No agrega ni quita columnas: necesario para que el Pipeline reporte los nombres finales
        return self.feature_names_in_

    def transform(self, X):
        X = X.copy()
        relleno = X[self.grupo].map(self.medianas_).fillna(self.mediana_global_)
        X[self.columna] = X[self.columna].fillna(relleno)
        return X


def columnas_binarias(df: pd.DataFrame) -> list[str]:
    """Indicadores fijos más las columnas de género presentes en los datos."""
    return COLS_BINARIAS_BASE + [c for c in df.columns if c.startswith(PREFIJO_GENERO)]


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """División train/test estratificada por tipo de título."""
    X = df.drop(columns=OBJETIVO)
    y = df[OBJETIVO]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=X["tipo_titulo"]
    )


def build_preprocessor(cols_binarias: list[str]) -> Pipeline:
    """Imputación por grupo -> escalado de numéricas / one-hot de categóricas / binarias."""
    columnas = ColumnTransformer(
        [
            ("num", StandardScaler(), COLS_NUMERICAS),
            (
                "cat",
                OneHotEncoder(
                    drop=CATEGORIAS_REFERENCIA, handle_unknown="ignore", sparse_output=False
                ),
                COLS_CATEGORICAS,
            ),
            ("bin", "passthrough", cols_binarias),
        ]
    )
    return Pipeline(
        [
            ("imputar_duracion", ImputadorMedianaPorGrupo("duracion_min", "tipo_titulo")),
            ("columnas", columnas),
        ]
    )
