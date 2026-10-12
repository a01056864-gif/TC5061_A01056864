import numpy as np
import pandas as pd
import pytest

from imdb_rating.config import RAW_DATA_FILE
from imdb_rating.dataset import convertir_tipos, limpiar
from imdb_rating.features import ImputadorMedianaPorGrupo, columnas_binarias, split_data
from imdb_rating.modeling.train import build_model, parse_max_features


@pytest.fixture(scope="module")
def datos_limpios():
    return limpiar(pd.read_csv(RAW_DATA_FILE))


def test_convertir_tipos_extrae_numeros_y_tipo():
    raw = pd.DataFrame(
        {
            "duration": ["46 min", np.nan],
            "votes": ["145,893", "10"],
            "year": ["(2016– )", "(2019 TV Movie)"],
        }
    )
    out = convertir_tipos(raw)
    assert out["duracion_min"].iloc[0] == 46
    assert np.isnan(out["duracion_min"].iloc[1])  # no inventa faltantes
    assert out["votos"].iloc[0] == 145893
    assert list(out["tipo_titulo"]) == ["Serie", "TV Movie"]
    assert out["en_emision"].iloc[0]


def test_limpiar_sin_faltantes_en_objetivo_ni_duplicados(datos_limpios):
    assert datos_limpios["rating"].notna().all()
    assert datos_limpios["duracion_min"].max() <= 300


def test_imputador_usa_mediana_de_entrenamiento():
    train = pd.DataFrame({"duracion_min": [20.0, 40.0, 100.0], "tipo_titulo": ["S", "S", "P"]})
    test = pd.DataFrame({"duracion_min": [np.nan, np.nan], "tipo_titulo": ["S", "Nuevo"]})
    out = ImputadorMedianaPorGrupo().fit(train).transform(test)
    assert list(out["duracion_min"]) == [30.0, 40.0]  # grupo S y mediana global


def test_parse_max_features():
    assert parse_max_features("sqrt") == "sqrt"
    assert parse_max_features("0.5") == 0.5


def test_misma_semilla_mismas_predicciones(datos_limpios):
    X_train, X_test, y_train, _ = split_data(datos_limpios, 0.2, seed=7)
    preds = [
        build_model(columnas_binarias(datos_limpios), 10, 5, 5, 1.0, seed=7)
        .fit(X_train, y_train)
        .predict(X_test)
        for _ in range(2)
    ]
    # Con n_jobs=-1 los árboles se suman en distinto orden entre hilos: diferencias ~1e-15
    # por redondeo de punto flotante, sin efecto en las métricas
    np.testing.assert_allclose(preds[0], preds[1], rtol=0, atol=1e-12)
