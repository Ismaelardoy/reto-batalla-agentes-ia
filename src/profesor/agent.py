"""
Agente Profesor IA.

Responsabilidades:
    1. Leer el temario y generar una pregunta pedagógicamente relevante.
    2. Evaluar la respuesta del Alumno con criterios objetivos.

Usa `with_structured_output` de LangChain para garantizar que el LLM
siempre devuelva datos que cumplan los esquemas Pydantic definidos.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.config.llm_factory import get_llm
from src.graph.state import BattleState
from src.shared.schemas import EvaluacionProfesor, PreguntaProfesor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts del Profesor
# ---------------------------------------------------------------------------

_PROMPT_GENERAR_PREGUNTA = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un profesor universitario experto en la materia descrita en el temario. "
                "Tu objetivo es formular UNA sola pregunta que evalúe el nivel de comprensión "
                "profunda del alumno sobre los conceptos más importantes del temario.\n\n"
                "La pregunta debe:\n"
                "- Requerir razonamiento, no solo memorización.\n"
                "- Ser clara y sin ambigüedades.\n"
                "- Estar directamente relacionada con el temario provisto.\n\n"
                "Responde ÚNICAMENTE con el JSON estructurado solicitado."
            ),
        ),
        (
            "human",
            "TEMARIO:\n{temario}\n\nGenera una pregunta de evaluación para este temario.",
        ),
    ]
)

_PROMPT_EVALUAR_RESPUESTA = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un profesor universitario riguroso e imparcial. "
                "Tu tarea es evaluar la respuesta de un alumno basándote EXCLUSIVAMENTE "
                "en el temario provisto y la pregunta formulada.\n\n"
                "Criterios de evaluación:\n"
                "- Precisión técnica y conceptual (0-4 puntos)\n"
                "- Completitud de la respuesta (0-3 puntos)\n"
                "- Claridad y coherencia en la explicación (0-3 puntos)\n\n"
                "Responde ÚNICAMENTE con el JSON estructurado solicitado."
            ),
        ),
        (
            "human",
            (
                "TEMARIO:\n{temario}\n\n"
                "PREGUNTA FORMULADA:\n{pregunta}\n\n"
                "RESPUESTA DEL ALUMNO:\n{respuesta}\n\n"
                "Evalúa la respuesta del alumno con nota del 0.0 al 10.0."
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# Nodos del grafo (funciones puras que reciben y devuelven estado)
# ---------------------------------------------------------------------------


def generar_pregunta(state: BattleState) -> BattleState:
    """
    Nodo 1: El Profesor lee el temario y formula una pregunta estructurada.

    Args:
        state: Estado actual del grafo con el campo `temario` obligatorio.

    Returns:
        Diccionario parcial del estado con la `pregunta` generada.
    """
    logger.info("🎓 Profesor: Generando pregunta sobre el temario...")

    llm = get_llm(temperature=0.8)
    structured_llm = llm.with_structured_output(PreguntaProfesor)

    chain = _PROMPT_GENERAR_PREGUNTA | structured_llm
    pregunta: PreguntaProfesor = chain.invoke({"temario": state["temario"]})

    logger.info("✅ Profesor: Pregunta generada -> %s", pregunta.pregunta)

    return {"pregunta": pregunta}


def evaluar_respuesta(state: BattleState) -> BattleState:
    """
    Nodo 3: El Profesor evalúa la respuesta del Alumno.

    Args:
        state: Estado actual con `temario`, `pregunta` y `respuesta` disponibles.

    Returns:
        Diccionario parcial del estado con la `evaluacion` emitida.
    """
    logger.info("📝 Profesor: Evaluando la respuesta del alumno...")

    llm = get_llm(temperature=0.2)  # Temperatura baja para evaluaciones consistentes
    structured_llm = llm.with_structured_output(EvaluacionProfesor)

    chain = _PROMPT_EVALUAR_RESPUESTA | structured_llm
    evaluacion: EvaluacionProfesor = chain.invoke(
        {
            "temario": state["temario"],
            "pregunta": state["pregunta"].pregunta,  # type: ignore[union-attr]
            "respuesta": state["respuesta"].respuesta,  # type: ignore[union-attr]
        }
    )

    logger.info("✅ Profesor: Evaluación completada -> Nota: %.1f/10", evaluacion.nota)

    return {"evaluacion": evaluacion}
