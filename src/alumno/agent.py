"""
Agente Alumno IA con sistema RAG simulado (Mock Retriever).

El Alumno recibe la pregunta del Profesor y:
    1. Consulta el Mock Retriever para obtener contexto relevante.
    2. Usa el contexto recuperado y el LLM para formular una respuesta.

El Mock Retriever simula el comportamiento de un sistema de recuperación
vectorial real (ej. FAISS, ChromaDB). En producción, se reemplazaría
el retriever sin tocar la lógica del agente (Principio Open/Closed).
"""

import logging
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from src.config.llm_factory import get_llm
from src.graph.state import BattleState
from src.shared.schemas import RespuestaAlumno

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock Retriever (simulación del sistema RAG)
# ---------------------------------------------------------------------------


@dataclass
class DocumentoRecuperado:
    """Representa un fragmento de texto recuperado por el retriever."""

    id: str
    contenido: str
    relevancia: float = 1.0


@dataclass
class MockRetriever:
    """
    Retriever simulado que devuelve documentos predefinidos.

    Principio Open/Closed: Esta clase puede reemplazarse por un
    retriever real (FAISS, ChromaDB, Pinecone) sin modificar el
    resto del agente, simplemente implementando el mismo contrato.
    """

    _knowledge_base: list[DocumentoRecuperado] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Inicializa la base de conocimiento con fragmentos de ejemplo."""
        self._knowledge_base = [
            DocumentoRecuperado(
                id="doc_001",
                contenido=(
                    "La Inteligencia Artificial (IA) es la simulación de procesos de inteligencia "
                    "humana por parte de sistemas informáticos. Incluye el aprendizaje automático "
                    "(Machine Learning), el procesamiento del lenguaje natural (NLP) y la visión por "
                    "computadora. Los sistemas de IA modernos utilizan redes neuronales profundas "
                    "(Deep Learning) para aprender patrones complejos a partir de grandes datasets."
                ),
                relevancia=0.95,
            ),
            DocumentoRecuperado(
                id="doc_002",
                contenido=(
                    "Los Modelos de Lenguaje Grande (LLMs) son redes neuronales entrenadas con "
                    "enormes cantidades de texto. Utilizan la arquitectura Transformer con mecanismos "
                    "de atención (attention) para comprender y generar texto coherente. Ejemplos: "
                    "GPT-4, Claude, Gemini. Se ajustan mediante RLHF (Reinforcement Learning from "
                    "Human Feedback) para alinear su comportamiento con las preferencias humanas."
                ),
                relevancia=0.92,
            ),
            DocumentoRecuperado(
                id="doc_003",
                contenido=(
                    "RAG (Retrieval-Augmented Generation) es una técnica que combina la recuperación "
                    "de información con la generación de texto. El proceso: (1) El usuario formula "
                    "una pregunta, (2) el retriever busca documentos relevantes en una base de "
                    "conocimiento vectorial, (3) los documentos recuperados se incluyen como contexto "
                    "en el prompt del LLM, (4) el LLM genera una respuesta fundamentada y factual."
                ),
                relevancia=0.98,
            ),
            DocumentoRecuperado(
                id="doc_004",
                contenido=(
                    "LangGraph es un framework de orquestación de agentes IA que modela flujos "
                    "de trabajo como grafos dirigidos (StateGraph). Cada nodo representa una operación "
                    "y las aristas definen el flujo de control. Permite bucles, condiciones y paralelismo, "
                    "siendo ideal para sistemas multi-agente complejos. El estado fluye entre nodos "
                    "de forma tipada y predecible."
                ),
                relevancia=0.89,
            ),
            DocumentoRecuperado(
                id="doc_005",
                contenido=(
                    "FastAPI es un framework web moderno y de alto rendimiento para Python, basado "
                    "en Starlette y Pydantic. Genera documentación OpenAPI automática (Swagger UI / "
                    "ReDoc), soporta programación asíncrona (async/await) y utiliza type hints de "
                    "Python para validación automática de datos de entrada y salida."
                ),
                relevancia=0.85,
            ),
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[DocumentoRecuperado]:
        """
        Recupera los documentos más relevantes para la query dada.

        En esta implementación mock, devuelve los top_k documentos
        ordenados por relevancia. Un retriever real calcularía
        la similitud semántica (cosine similarity) con embeddings vectoriales.

        Args:
            query: La pregunta o consulta del usuario.
            top_k: Número máximo de documentos a recuperar.

        Returns:
            Lista de documentos recuperados ordenados por relevancia.
        """
        logger.debug("🔍 MockRetriever: Recuperando top-%d documentos para: '%s'", top_k, query[:80])
        sorted_docs = sorted(self._knowledge_base, key=lambda d: d.relevancia, reverse=True)
        return sorted_docs[:top_k]


# Instancia singleton del retriever (patrón Singleton ligero)
_retriever = MockRetriever()


# ---------------------------------------------------------------------------
# Prompt del Alumno
# ---------------------------------------------------------------------------

_PROMPT_RESPONDER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un alumno universitario inteligente y bien preparado. "
                "Tu objetivo es responder la pregunta del profesor de forma clara, "
                "precisa y bien argumentada, utilizando ÚNICAMENTE la información "
                "del contexto proporcionado por tu sistema de estudio.\n\n"
                "Reglas:\n"
                "- Basa tu respuesta SOLO en el contexto dado.\n"
                "- Si el contexto no cubre completamente la pregunta, indícalo honestamente.\n"
                "- Usa un lenguaje académico pero accesible.\n"
                "- Estructura tu respuesta de forma lógica y coherente.\n\n"
                "Responde ÚNICAMENTE con el JSON estructurado solicitado."
            ),
        ),
        (
            "human",
            (
                "PREGUNTA DEL PROFESOR:\n{pregunta}\n\n"
                "CONTEXTO DE ESTUDIO (documentos recuperados):\n{contexto}\n\n"
                "Responde la pregunta basándote en el contexto anterior."
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# Nodo del grafo
# ---------------------------------------------------------------------------


def responder_pregunta(state: BattleState) -> BattleState:
    """
    Nodo 2: El Alumno recupera contexto RAG y formula su respuesta.

    Flujo:
        1. Recupera documentos relevantes usando el Mock Retriever.
        2. Construye el contexto como texto concatenado.
        3. Invoca el LLM con el contexto y la pregunta.
        4. Devuelve la respuesta estructurada.

    Args:
        state: Estado actual con el campo `pregunta` disponible.

    Returns:
        Diccionario parcial del estado con la `respuesta` generada.
    """
    pregunta_obj = state["pregunta"]
    pregunta_texto = pregunta_obj.pregunta  # type: ignore[union-attr]

    logger.info("📚 Alumno: Consultando sistema RAG para la pregunta...")

    # Paso 1: Recuperar documentos relevantes
    documentos = _retriever.retrieve(query=pregunta_texto, top_k=3)
    ids_fuentes = [doc.id for doc in documentos]

    # Paso 2: Construir el contexto como string formateado
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {doc.id}]\n{doc.contenido}" for doc in documentos
    )
    logger.info("📄 Alumno: %d documentos recuperados: %s", len(documentos), ids_fuentes)

    # Paso 3: Invocar el LLM con salida estructurada
    llm = get_llm(temperature=0.5)
    structured_llm = llm.with_structured_output(RespuestaAlumno)

    chain = _PROMPT_RESPONDER | structured_llm
    respuesta: RespuestaAlumno = chain.invoke(
        {
            "pregunta": pregunta_texto,
            "contexto": contexto,
        }
    )

    # Asegurar que las fuentes recuperadas estén registradas (merge)
    if not respuesta.fuentes_usadas:
        respuesta.fuentes_usadas = ids_fuentes

    logger.info("✅ Alumno: Respuesta formulada con %d fuentes.", len(respuesta.fuentes_usadas))

    return {"respuesta": respuesta}
