# Motor de scoring extraído de 02_produccion_scoring.py
import numpy as np
import pandas as pd
import cloudpickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTEFACTO_PATH = BASE_DIR / "artefacto_pipeline.pkl"
DATA_PATH = Path(__file__).resolve().parent / "data" / "demo_03_riesgos.csv"

with open(ARTEFACTO_PATH, "rb") as f:
    artefacto = cloudpickle.load(f)

pipe_pd = artefacto["pipe_pd"]
pipe_ead = artefacto["pipe_ead"]
pipe_lgd = artefacto["pipe_lgd"]

INT_COLS = ["num_lineas_credito", "num_derogatorios"]


def normaliza_num_cuotas(valor):
    if pd.isna(valor):
        return valor
    if isinstance(valor, str):
        texto = valor.strip()
        if texto.endswith("months"):
            return f" {texto}"
        return texto
    try:
        return f" {int(valor)} months"
    except (TypeError, ValueError):
        return valor


# --- Preprocesado y armonización ---
def prepara_datos(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        c for c in ["id_prestamo", "descripcion", "empleo"] if c in df.columns
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if "antigüedad_empleo" in df.columns:
        df["antigüedad_empleo"] = df["antigüedad_empleo"].fillna("desconocido")

    num_cols_df = df.select_dtypes(include="number").columns.tolist()
    if num_cols_df:
        df[num_cols_df] = df[num_cols_df].fillna(0)

    int_cols_present = [c for c in INT_COLS if c in df.columns]
    if int_cols_present:
        df[int_cols_present] = (
            df[int_cols_present]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if "num_cuotas" in df.columns:
        df["num_cuotas"] = df["num_cuotas"].apply(normaliza_num_cuotas).astype(str)

    for col in [
        "ingresos_verificados",
        "vivienda",
        "finalidad",
        "rating",
        "antigüedad_empleo",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "vivienda" in df.columns:
        df["vivienda"] = df["vivienda"].replace(["ANY", "NONE", "OTHER"], "MORTGAGE")
    if "finalidad" in df.columns:
        df["finalidad"] = df["finalidad"].replace(
            {
                "debt_consolidation": "debt_consolidation",
                "credit_card": "credit_card",
                "home_improvement": "home_improvement",
                "major_purchase": "major_purchase",
                "small_business": "small_business",
                "car": "other",
                "moving": "other",
                "vacation": "other",
                "medical": "other",
                "wedding": "other",
                "renewable_energy": "other",
                "house": "other",
                "educational": "other",
                "other": "other",
            }
        )

    df = df.infer_objects(copy=False)

    # Clips
    if "dti" in df.columns:
        df["dti"] = np.clip(df["dti"], 0, 100)
    if "num_hipotecas" in df.columns:
        df["num_hipotecas"] = np.clip(df["num_hipotecas"], 0, 7)
    if "porc_uso_revolving" in df.columns:
        df["porc_uso_revolving"] = np.clip(df["porc_uso_revolving"], 0, 100)
    return df


# --- Motor de scoring ---
def scoring_df(df: pd.DataFrame) -> pd.DataFrame:
    df = prepara_datos(df.copy())
    ids = df["id_cliente"].values if "id_cliente" in df.columns else df.index.values
    score_pd = pipe_pd.predict_proba(df)[:, 1]
    score_ead = np.clip(pipe_ead.predict(df), 0, 1)
    score_lgd = np.clip(pipe_lgd.predict(df), 0, 1)
    df_resultado = pd.DataFrame(
        {
            "id_cliente": ids,
            "score_pd": score_pd,
            "score_ead": score_ead,
            "score_lgd": score_lgd,
            "perdida_esperada": score_pd * score_ead * score_lgd,
        }
    )
    return df_resultado


def calcular_scoring(datos: dict) -> dict:
    """Calcula scoring para un único registro dado como dict.
    Reutiliza scoring_df y la lógica de preprocesado ya existente.
    """
    principal = float(datos["principal"])
    num_cuotas = int(datos["num_cuotas"])
    tipo_interes = float(datos["tipo_interes"])

    # Cuota francesa
    if tipo_interes == 0:
        imp_cuota = principal / num_cuotas
    else:
        i = tipo_interes / 12 / 100
        imp_cuota = (
            principal * (i * (1 + i) ** num_cuotas) / ((1 + i) ** num_cuotas - 1)
        )

    df_entrada = pd.DataFrame([{**datos, "imp_cuota": imp_cuota}])
    df_hist = pd.read_csv(DATA_PATH)
    df_merged = df_entrada.merge(
        df_hist, on="id_cliente", how="left", suffixes=("", "_hist")
    )

    # Rellenar nulos numéricos
    num_cols = df_merged.select_dtypes(include="number").columns.tolist()
    df_merged[num_cols] = df_merged[num_cols].fillna(0)

    for col in [
        "ingresos_verificados",
        "vivienda",
        "finalidad",
        "rating",
        "antigüedad_empleo",
    ]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna("desconocido").astype(str)

    # Formatear num_cuotas como espera el pipeline
    df_merged["num_cuotas"] = df_merged["num_cuotas"].apply(
        lambda x: (
            f" {int(x)} months" if not str(x).strip().endswith("months") else str(x)
        )
    )
    df_merged = df_merged.infer_objects(copy=False)

    df_resultado = scoring_df(df_merged)
    return df_resultado.iloc[0].to_dict()
