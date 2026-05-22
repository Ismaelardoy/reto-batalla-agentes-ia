"""
Compilación del grafo principal de la batalla de chatbots.

Define el flujo completo usando LangGraph StateGraph:
    START -> generar_pregunta -> responder_pregunta -> evaluar_respuesta -> END

El grafo es compilado una sola vez al importar el módulo y expuesto
como `battle_graph` para su uso en la capa de API.
"""

import logging

from langgraph.graph import END, START, StateGraph

from src.alumno.agent import responder_pregunta
from src.graph.state import BattleState
from src.profesor.agent import evaluar_respuesta, generar_pregunta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nombres de los nodos (constantes para evitar magic strings)
# ---------------------------------------------------------------------------

NODE_GENERAR_PREGUNTA = "generar_pregunta"
NODE_RESPONDER_PREGUNTA = "responder_pregunta"
NODE_EVALUAR_RESPUESTA = "evaluar_respuesta"


def _build_graph() -> StateGraph:
    """
    Construye y compila el StateGraph de la batalla.

    Flujo lineal:
        START
          └─► generar_pregunta  (Profesor formula la pregunta)
                └─► responder_pregunta  (Alumno responde con RAG)
                      └─► evaluar_respuesta  (Profesor evalúa)
                            └─► END

    Returns:
        Grafo compilado listo para invocar con `graph.invoke(state)`.
    """
    logger.info("🔧 Construyendo el grafo de batalla...")

    graph_builder = StateGraph(BattleState)

    # Registro de nodos
    graph_builder.add_node(NODE_GENERAR_PREGUNTA, generar_pregunta)
    graph_builder.add_node(NODE_RESPONDER_PREGUNTA, responder_pregunta)
    graph_builder.add_node(NODE_EVALUAR_RESPUESTA, evaluar_respuesta)

    # Definición del flujo de aristas
    graph_builder.add_edge(START, NODE_GENERAR_PREGUNTA)
    graph_builder.add_edge(NODE_GENERAR_PREGUNTA, NODE_RESPONDER_PREGUNTA)
    graph_builder.add_edge(NODE_RESPONDER_PREGUNTA, NODE_EVALUAR_RESPUESTA)
    graph_builder.add_edge(NODE_EVALUAR_RESPUESTA, END)

    compiled = graph_builder.compile()
    logger.info("✅ Grafo compilado correctamente con %d nodos.", 3)

    return compiled


# Instancia del grafo compilada en el arranque del módulo (Eager initialization)
battle_graph = _build_graph()
