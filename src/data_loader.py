from pathlib import Path
import pandas as pd


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Columnas oficiales usadas en los archivos processed.*.data de UCI
UCI_COLUMNS = [
    "age",       # edad
    "sex",       # sexo: 1 = hombre, 0 = mujer
    "cp",        # tipo de dolor de pecho
    "trestbps",  # presión arterial en reposo
    "chol",      # colesterol sérico
    "fbs",       # glucosa en ayunas > 120 mg/dl
    "restecg",   # resultado electrocardiográfico en reposo
    "thalach",   # frecuencia cardíaca máxima alcanzada
    "exang",     # angina inducida por ejercicio
    "oldpeak",   # depresión ST inducida por ejercicio
    "slope",     # pendiente del segmento ST
    "ca",        # número de vasos principales coloreados por fluoroscopía
    "thal",      # resultado de prueba thal
    "num"        # diagnóstico original: 0, 1, 2, 3, 4
]


def load_uci_raw(filename="processed.cleveland.data", source="cleveland"):
    """
    Carga un archivo procesado de UCI Heart Disease.

    Parámetros:
        filename: nombre del archivo dentro de data/raw/uci/
        source: nombre de la fuente, por ejemplo 'cleveland'

    Retorna:
        DataFrame con variables predictoras, columna target y columna source.
    """

    path = RAW_DIR / "uci" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}. "
            "Verifica que lo hayas descargado en data/raw/uci/."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=UCI_COLUMNS,
        na_values="?"
    )

    # Convertimos la variable original num a clasificación binaria:
    # 0 = ausencia de enfermedad cardíaca
    # 1 = presencia de enfermedad cardíaca
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)

    # Ya no usaremos num directamente
    df = df.drop(columns=["num"])

    # Guardamos la fuente del dataset
    df["source"] = source

    return df


def load_cleveland():
    """
    Carga el dataset principal Cleveland.
    """
    return load_uci_raw(
        filename="processed.cleveland.data",
        source="cleveland"
    )


def load_external_uci_datasets():
    """
    Carga datasets externos de UCI para experimentos de generalización.
    Requiere que los archivos estén en data/raw/uci/.
    """

    files = [
        ("processed.hungarian.data", "hungarian"),
        ("processed.switzerland.data", "switzerland"),
        ("processed.va.data", "va_long_beach")
    ]

    dataframes = []

    for filename, source in files:
        path = RAW_DIR / "uci" / filename

        if path.exists():
            df = load_uci_raw(filename=filename, source=source)
            dataframes.append(df)
        else:
            print(f"Advertencia: no se encontró {path}. Se omitirá.")

    if not dataframes:
        raise FileNotFoundError(
            "No se encontró ningún dataset externo de UCI en data/raw/uci/."
        )

    return pd.concat(dataframes, ignore_index=True)


def load_github_small_dataset():
    """
    Carga el dataset pequeño del repositorio de GitHub.
    """

    path = RAW_DIR / "github" / "problemas_del_corazon.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}. "
            "Verifica que esté guardado en data/raw/github/."
        )

    df = pd.read_csv(path)

    return df


def save_processed(df, filename):
    """
    Guarda un DataFrame procesado en data/processed/.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)

    return output_path