from fastapi import FastAPI, HTTPException
from .schemas import LoteEntrada, ScoringSalida, RegistroEntrada
from .scoring import scoring_df, calcular_scoring
import pandas as pd
from pathlib import Path

app = FastAPI(title="API Scoring Riesgos")

# Ruta al historial de clientes
DATA_PATH = Path(__file__).parent / "data" / "demo_03_riesgos.csv"


def calcular_cuota_francesa(principal, num_cuotas, tipo_interes):
    """Calcula la cuota usando el sistema francés."""
    if tipo_interes == 0:
        return principal / num_cuotas
    i = tipo_interes / 12 / 100
    cuota = principal * (i * (1 + i) ** num_cuotas) / ((1 + i) ** num_cuotas - 1)
    return cuota


def formatea_num_cuotas(valor):
    return f" {int(valor)} months"


@app.post("/predict", response_model=list[ScoringSalida])
def predict(lote: LoteEntrada):
    # Convertir entrada a DataFrame
    df_entrada = lote.to_df()

    # Leer historial de clientes
    try:
        df_hist = pd.read_csv(DATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo historial: {e}")

    # Calcular imp_cuota para cada registro
    df_entrada["imp_cuota"] = df_entrada.apply(
        lambda row: calcular_cuota_francesa(
            row["principal"], row["num_cuotas"], row["tipo_interes"]
        ),
        axis=1,
    )

    # Unir con historial por id_cliente (añade columnas necesarias para el modelo)
    df_merged = df_entrada.merge(
        df_hist, on="id_cliente", how="left", suffixes=("", "_hist")
    )

    # Si alguna columna crítica falta tras el merge, lanzar error claro
    columnas_necesarias = [
        "porc_uso_revolving",
        "ingresos_verificados",
        "antigüedad_empleo",
        "num_derogatorios",
        "dti",
        "ingresos",
        "rating",
        "num_lineas_credito",
    ]
    faltantes = [col for col in columnas_necesarias if col not in df_merged.columns]
    if faltantes:
        raise HTTPException(
            status_code=500, detail=f"Faltan columnas tras el merge: {faltantes}"
        )

    # Rellenar nulos y asegurar tipos
    for col in columnas_necesarias:
        if df_merged[col].dtype.kind in "biufc":
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce").fillna(0)
        else:
            df_merged[col] = df_merged[col].fillna("desconocido")

    df_merged["principal"] = pd.to_numeric(
        df_merged["principal"], errors="coerce"
    ).fillna(0)
    df_merged["tipo_interes"] = pd.to_numeric(
        df_merged["tipo_interes"], errors="coerce"
    ).fillna(0)
    df_merged["num_cuotas"] = (
        pd.to_numeric(df_merged["num_cuotas"], errors="coerce").fillna(0).astype(int)
    )
    df_merged["num_cuotas"] = df_merged["num_cuotas"].apply(formatea_num_cuotas)

    for col in [
        "ingresos_verificados",
        "vivienda",
        "finalidad",
        "rating",
        "antigüedad_empleo",
    ]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna("desconocido").astype(str)

    df_merged = df_merged.infer_objects(copy=False)

    # Scoring
    df_resultado = scoring_df(df_merged)
    return df_resultado.to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# HELPERS ALTERNATIVES
# =========================
_CUOTAS_VALIDAS = [12, 24, 36, 48, 60]
_RIESGO_SCORE = {"bajo": 0, "medio": 1, "alto": 2}


def _clasificar_nivel_riesgo(pe_euros: float) -> str:
    if pe_euros < 1000:
        return "bajo"
    elif pe_euros < 5000:
        return "medio"
    return "alto"


def _generar_combinaciones(principal_orig: float, cuotas_orig: int) -> list:
    """Genera todas las combinaciones válidas de principal y cuotas."""
    principals = []
    p = principal_orig - 5000
    while p >= 5000:
        principals.append(p)
        p -= 5000

    cuotas_vars = [c for c in _CUOTAS_VALIDAS if c > cuotas_orig]

    combos = set()
    for p in principals:
        combos.add((p, cuotas_orig))
    for c in cuotas_vars:
        combos.add((principal_orig, c))
    for p in principals:
        for c in cuotas_vars:
            combos.add((p, c))
    return list(combos)


@app.post("/alternatives")
def alternatives(caso: RegistroEntrada):
    caso_dict = caso.model_dump()
    principal_orig = float(caso_dict["principal"])
    cuotas_orig = int(caso_dict["num_cuotas"])

    # Scoring del escenario original
    try:
        scoring_orig = calcular_scoring(caso_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring original: {e}")

    pe_orig = scoring_orig["perdida_esperada"] * principal_orig
    nivel_orig = _clasificar_nivel_riesgo(pe_orig)

    escenario_original = {
        "input": caso_dict,
        "scoring": {**scoring_orig, "pe": pe_orig, "nivel_riesgo": nivel_orig},
        "ranking_score": 0.0,
        "impacto_cambio": 0.0,
        "es_mejor": False,
    }

    # Generar y evaluar alternativas
    alternativas = []
    for p_alt, c_alt in _generar_combinaciones(principal_orig, cuotas_orig):
        caso_alt = {**caso_dict, "principal": p_alt, "num_cuotas": c_alt}
        try:
            scoring_alt = calcular_scoring(caso_alt)
        except Exception:
            continue

        pe_alt = scoring_alt["perdida_esperada"] * p_alt
        nivel_alt = _clasificar_nivel_riesgo(pe_alt)

        # Filtrar: eliminar alto y pe >= pe_orig
        if nivel_alt == "alto" or pe_alt >= pe_orig:
            continue

        pct_reduccion = (principal_orig - p_alt) / principal_orig

        alternativas.append(
            {
                "input": caso_alt,
                "scoring": {**scoring_alt, "pe": pe_alt, "nivel_riesgo": nivel_alt},
                "ranking_score": pct_reduccion,
                "impacto_cambio": pct_reduccion,
                "es_mejor": False,
            }
        )

    # Ordenar: mayor principal primero (menor reducción), empate por menor PE
    alternativas.sort(key=lambda x: (-x["input"]["principal"], x["scoring"]["pe"]))
    if alternativas:
        alternativas[0]["es_mejor"] = True

    return {
        "escenario_original": escenario_original,
        "alternativas": alternativas,
        "mejor_alternativa": alternativas[0] if alternativas else None,
        "activar_chat": nivel_orig == "alto",
    }
