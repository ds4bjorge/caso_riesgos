import os
import re
import json
import requests
import openai
from pathlib import Path

# =========================
# CARGA DE .env
# =========================
_BASE = Path(__file__).parent
_CANDIDATOS = [
    _BASE / ".env",  # app/.env
    _BASE.parent / ".env",  # 07_despliegue/.env
    _BASE.parent.parent / ".env",  # raíz del proyecto
]
for _env_path in _CANDIDATOS:
    if _env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=_env_path, override=False)
        except ImportError:
            pass
        break

# =========================
# CONSTANTES
# =========================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL", "https://caso-riesgos-api.onrender.com")


# =========================
# CLIENTE MCP (HTTP)
# =========================
def _llamar_mcp_tool(caso: dict):
    """Llama a POST /alternatives con el caso y devuelve la respuesta JSON o None."""
    try:
        url = API_BASE_URL.rstrip("/") + "/alternatives"
        r = requests.post(url, json=caso, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _extraer_alternativas_texto(mcp_response) -> str:
    """Convierte la respuesta de /alternatives en texto legible."""
    if not mcp_response:
        return "No se pudieron obtener alternativas del sistema."
    alternativas = mcp_response.get("alternativas", [])
    if not alternativas:
        return "El sistema no encontró alternativas viables para este caso."
    _SEMAFORO = {"bajo": "🟢", "medio": "🟠", "alto": "🔴"}
    lineas = []
    for i, alt in enumerate(alternativas[:5], 1):
        inp = alt.get("input", {})
        scoring = alt.get("scoring", {})
        nivel = str(scoring.get("nivel_riesgo", "")).lower()
        semaforo = _SEMAFORO.get(nivel, "⚪")
        pe = scoring.get("pe", 0)
        principal = inp.get("principal", 0)
        cuotas = inp.get("num_cuotas", 0)
        es_mejor = alt.get("es_mejor", False)
        marca = " ★ MEJOR OPCIÓN" if es_mejor else ""
        lineas.append(
            f"Alternativa {i}{marca}: "
            f"Principal {principal:,.0f} € | {cuotas} cuotas | "
            f"{semaforo} {nivel.capitalize()} | PE: {pe:,.2f} €"
        )
    return "\n".join(lineas)


# =========================
# DETECCIÓN DE INTENCIÓN
# =========================
def normalizar_texto(texto: str) -> str:
    return " ".join(texto.lower().split())


def es_peticion_nueva_simulacion(pregunta_usuario: str) -> bool:
    texto = normalizar_texto(pregunta_usuario)

    # Whitelist: frases que NO son nuevas simulaciones
    _WHITELIST = [
        "si pide menos",
        "que alternativa recomiendas",
        "qué alternativa recomiendas",
        "hay alguna opcion viable",
        "hay alguna opción viable",
        "por que sale rojo",
        "por qué sale rojo",
        "si se reduce el importe",
        "menos importe",
        "importe menor",
    ]
    for frase in _WHITELIST:
        if frase in texto:
            return False

    # Importe exacto (número con puntos/comas de miles) o plazo exacto
    tiene_importe = bool(re.search(r"\b\d{1,3}(?:[.,]\d{3})*\b", texto))
    tiene_plazo = bool(re.search(r"\b\d+\s*(?:mes(?:es)?|cuotas?)\b", texto))

    # Palabras que indican petición de escenario
    _PALABRAS_ESCENARIO = [
        "podría pedir",
        "puedo pedir",
        "simula",
        "simular",
        "que pasa si pido",
        "y a ",
    ]
    tiene_escenario = any(p in texto for p in _PALABRAS_ESCENARIO)

    return (tiene_importe or tiene_plazo) and tiene_escenario


# =========================
# CONSTRUCCIÓN DE MENSAJES
# =========================
def construir_instrucciones() -> str:
    return (
        "Eres un analista de riesgo de crédito experto. "
        "Respondes siempre en español de España, con tono profesional y directo.\n\n"
        "REGLAS ESTRICTAS:\n"
        "- No recalculas ni estimas scoring, pérdidas esperadas ni alternativas por tu cuenta.\n"
        "- Usas exclusivamente las alternativas ya calculadas por el sistema que se te pasan en el contexto.\n"
        "- Si el usuario hace una pregunta general sobre el riesgo o las alternativas, "
        "respondes con las alternativas ya calculadas.\n"
        "- Si el usuario pide una simulación exacta con un importe o plazo concreto distinto a los que ya tienes, "
        "le indicas que debe relanzar la simulación desde el formulario.\n\n"
        "FORMATO DE RESPUESTA:\n"
        "Responde siempre con este formato ejecutivo:\n"
        "1. Una frase de conclusión directa.\n"
        "2. Tres bloques HTML con estos títulos exactos:\n"
        "   - <u><strong>Diagnóstico</strong></u>\n"
        "   - <u><strong>Alternativas</strong></u>\n"
        "   - <u><strong>Siguiente paso</strong></u>\n"
        "3. Cada bloque con bullets HTML breves (máx. 3 bullets).\n"
        "4. Usa negrita (<strong>) para importes, cuotas y niveles de riesgo.\n"
        "5. Usa semáforos: 🟢 bajo, 🟠 medio, 🔴 alto.\n"
        "6. Muestra un máximo de 3 alternativas."
    )


def _clasificar_nivel_riesgo(pe_euros: float) -> str:
    if pe_euros < 1000:
        return "bajo"
    elif pe_euros < 5000:
        return "medio"
    return "alto"


def construir_input_usuario(
    pregunta_usuario: str, caso: dict, scoring_result: dict
) -> str:
    mcp_response = _llamar_mcp_tool(caso)
    alternativas_texto = _extraer_alternativas_texto(mcp_response)

    principal = float(caso.get("principal", 1))
    pe_ratio = scoring_result.get("perdida_esperada", 0)
    pe_euros = pe_ratio * principal
    nivel_riesgo = _clasificar_nivel_riesgo(pe_euros)
    _SEMAFORO = {"bajo": "🟢", "medio": "🟠", "alto": "🔴"}

    contexto = {
        "caso": caso,
        "resultado_scoring_original": {
            "score_pd": scoring_result.get("score_pd", 0),
            "score_ead": scoring_result.get("score_ead", 0),
            "score_lgd": scoring_result.get("score_lgd", 0),
            "perdida_esperada_ratio": pe_ratio,
            "perdida_esperada_euros": round(pe_euros, 2),
            "nivel_riesgo": nivel_riesgo,
            "semaforo": _SEMAFORO[nivel_riesgo],
            "principal": principal,
        },
        "alternativas_calculadas": alternativas_texto,
        "tarea": (
            "Explica al usuario el diagnóstico de su caso usando los valores "
            "ya calculados (perdida_esperada_euros y nivel_riesgo del resultado_scoring_original) "
            "y muestra las alternativas calculadas. NO recalcules nada."
        ),
        "reglas_presentacion": [
            "Usa siempre perdida_esperada_euros (en €) y nivel_riesgo del resultado_scoring_original para el diagnóstico.",
            "Usa semáforos con icono: 🟢 bajo, 🟠 medio, 🔴 alto.",
            "Muestra un máximo de 3 alternativas.",
            "Escribe en negrita los importes, cuotas y niveles de riesgo.",
        ],
        "peticion_usuario": pregunta_usuario,
    }
    return json.dumps(contexto, ensure_ascii=False, indent=2)


# =========================
# RESPUESTAS FALLBACK
# =========================
def construir_fallback(caso: dict, scoring_result: dict, motivo: str) -> str:
    pe = scoring_result.get("perdida_esperada", 0) * caso.get("principal", 0)
    return (
        "<b>No se pudo obtener la explicación del agente.</b>"
        "<ul>"
        "<li><u><strong>Diagnóstico</strong></u><ul>"
        f"<li>Pérdida esperada calculada: <strong>{pe:,.2f} €</strong></li>"
        "<li>Nivel de riesgo: <strong>Alto</strong> 🔴</li>"
        "</ul></li>"
        "<li><u><strong>Alternativas</strong></u><ul>"
        "<li>No disponibles en este momento.</li>"
        "</ul></li>"
        "<li><u><strong>Siguiente paso</strong></u><ul>"
        f"<li>Motivo técnico: {motivo}</li>"
        "<li>Comprueba la conexión con el servicio y vuelve a intentarlo.</li>"
        "</ul></li>"
        "</ul>"
    )


def construir_respuesta_recalculo_requerido(caso: dict) -> str:
    return (
        "<b>Para validar ese escenario exacto es necesario relanzar la simulación.</b>"
        "<ul>"
        "<li><u><strong>Diagnóstico</strong></u><ul>"
        "<li>Has pedido una simulación con parámetros concretos distintos a los actuales.</li>"
        f"<li>Préstamo actual: <strong>{caso.get('principal', 0):,.0f} €</strong> "
        f"a <strong>{caso.get('num_cuotas', 0)} cuotas</strong>.</li>"
        "</ul></li>"
        "<li><u><strong>Alternativas</strong></u><ul>"
        "<li>Usa el formulario lateral para ajustar el importe o el plazo.</li>"
        "<li>Pulsa <strong>Calcular riesgo</strong> para obtener el scoring exacto.</li>"
        "</ul></li>"
        "<li><u><strong>Siguiente paso</strong></u><ul>"
        "<li>Modifica los parámetros en el panel lateral y recalcula.</li>"
        "</ul></li>"
        "</ul>"
    )


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def solicitar_explicacion_riesgo(
    pregunta_usuario: str,
    caso: dict,
    scoring_result: dict,
    previous_response_id=None,
    history: list = None,
) -> dict:
    if history is None:
        history = []

    if es_peticion_nueva_simulacion(pregunta_usuario):
        return {
            "ok": True,
            "message": construir_respuesta_recalculo_requerido(caso),
            "response_id": None,
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "message": construir_fallback(
                caso, scoring_result, "OPENAI_API_KEY no configurada."
            ),
            "response_id": None,
        }

    try:
        client = openai.OpenAI(api_key=api_key)
        input_usuario = construir_input_usuario(pregunta_usuario, caso, scoring_result)

        messages = [{"role": "system", "content": construir_instrucciones()}]
        messages.extend(history)
        messages.append({"role": "user", "content": input_usuario})

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        texto = response.choices[0].message.content
        return {"ok": True, "message": texto, "response_id": None}

    except Exception as e:
        return {
            "ok": False,
            "message": construir_fallback(caso, scoring_result, str(e)),
            "response_id": None,
        }
