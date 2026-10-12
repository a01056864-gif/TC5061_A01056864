"""Carga y limpieza determinista del dataset IMDB.

Solo se aplican reglas que no aprenden de los datos (convertir tipos, eliminar filas inválidas,
recortar con un tope de dominio, mapear categorías con listas fijas). Lo que calcula estadísticos
(imputación, escalado) vive en el pipeline de features.py para evitar fuga de información.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from imdb_rating.config import (
    CERT_EQUIVALENTES,
    CERT_VALIDAS,
    COLS_BINARIAS_BASE,
    COLS_CATEGORICAS,
    COLS_NUMERICAS,
    OBJETIVO,
    PREFIJO_GENERO,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TIPOS_VALIDOS,
    TOPE_DURACION,
)

app = typer.Typer()


def load_raw(path: Path) -> pd.DataFrame:
    """Lee el CSV original sin modificarlo."""
    return pd.read_csv(path)


def convertir_tipos(data: pd.DataFrame) -> pd.DataFrame:
    """Convierte duration, votes y year a variables numéricas/categóricas.

    No imputa ni elimina filas: los faltantes del original siguen como NaN.
    """
    out = data.copy()

    # duration: '46 min' -> 46.0 (float para admitir NaN)
    out["duracion_min"] = out["duration"].str.extract(r"(\d+)", expand=False).astype(float)

    # votes: '145,893' -> 145893.0
    out["votos"] = out["votes"].str.replace(",", "", regex=False).astype(float)

    # year: se extraen el año de inicio y, si existe, el año de fin
    anios = out["year"].str.extract(r"(\d{4})(?:–(\d{4}))?")
    out["anio_inicio"] = anios[0].astype(float)
    out["anio_fin"] = anios[1].astype(float)

    # Un guion en el periodo indica serie; si no hay año de cierre sigue en emisión
    out["es_serie"] = out["year"].str.contains("–", na=False)
    out["en_emision"] = out["year"].str.contains(r"–\s*\)", na=False)

    # Tipo de título: etiqueta textual dentro del paréntesis (TV Special, Video, ...)
    etiqueta = out["year"].str.extract(r"\(\d{0,4}\s*([A-Za-z][A-Za-z ]+)\)", expand=False)
    etiqueta = etiqueta.where(~etiqueta.str.fullmatch(r"[IVXLC]+", na=False))  # números romanos
    out["tipo_titulo"] = np.select(
        [out["es_serie"], etiqueta.notna()],
        ["Serie", etiqueta],
        default="Película",
    )
    return out


def limpiar(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza del notebook S03 y devuelve el dataset de modelado."""
    out = convertir_tipos(data)

    # Una fila por título: descarta episodios individuales que repiten title + year
    out = out.drop_duplicates(subset=["title", "year"], keep="first")

    # Sin variable objetivo (títulos no estrenados) o sin género: no se pueden usar
    out = out.dropna(subset=[OBJETIVO, "genre"])

    # Outliers de duración: miniseries con duración total; se recortan y se marca el caso
    out["duracion_es_total"] = (out["duracion_min"] > TOPE_DURACION).astype(int)
    out["duracion_min"] = out["duracion_min"].clip(upper=TOPE_DURACION)

    # votos tiene cola larga a la derecha
    out["log10_votos"] = np.log10(out["votos"])

    # Categorías poco frecuentes se agrupan con listas fijas
    cert = out["certificate"].replace(CERT_EQUIVALENTES)
    out["certificado"] = np.where(
        cert.isna(), "Sin clasificación", np.where(cert.isin(CERT_VALIDAS), cert, "Otra")
    )
    out["tipo_titulo"] = out["tipo_titulo"].where(out["tipo_titulo"].isin(TIPOS_VALIDOS), "Otro")

    # Géneros: una columna binaria por género (multi-hot)
    generos = out["genre"].str.get_dummies(sep=", ").add_prefix(PREFIJO_GENERO)
    out = pd.concat([out, generos], axis=1)

    out["en_emision"] = out["en_emision"].astype(int)
    columnas = (
        COLS_NUMERICAS
        + COLS_BINARIAS_BASE
        + generos.columns.tolist()
        + COLS_CATEGORICAS
        + [OBJETIVO]
    )
    return out[columnas].reset_index(drop=True)


def build_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Lee el CSV crudo, lo limpia y guarda el resultado en data/processed."""
    raw = load_raw(input_path)
    clean = limpiar(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    logger.info(f"{len(raw):,} filas originales -> {len(clean):,} filas limpias")
    return clean


@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_FILE, help="CSV original de IMDB."),
    output_path: Path = typer.Option(PROCESSED_DATA_FILE, help="Destino del CSV limpio."),
):
    """Genera el dataset limpio a partir del CSV original."""
    build_dataset(input_path, output_path)
    logger.success(f"Dataset limpio guardado en {output_path}")


if __name__ == "__main__":
    app()
