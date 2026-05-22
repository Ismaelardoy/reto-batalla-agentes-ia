"""
Definición del estado compartido del grafo de conversación.

El estado es la única fuente de verdad (Single Source of Truth) que
fluye a través de todos los nodos del StateGraph de LangGraph.
Cada nodo lee del estado y devuelve un diccionario parcial con las
claves que ha actualizado.
"""

from typing import Optional

from typing_extensions import TypedDict

from src.shared.schemas import EvaluacionProfesor, PreguntaProfesor, RespuestaAlumno


class BattleState(TypedDict, total=False):
    """
    Estado completo de la batalla entre el Profesor IA y el Alumno IA.

    Campos:
        temario: El texto del temario que el Profesor usa como base
                 para formular la pregunta y evaluar la respuesta.
        pregunta: La pregunta estructurada generada por el Profesor.
        respuesta: La respuesta estructurada generada por el Alumno
                   mediante el sistema RAG.
        evaluacion: La evaluación estructurada emitida por el Profesor
                    tras analizar la respuesta del Alumno.
    """

    temario: str
    pregunta: Optional[PreguntaProfesor]
    respuesta: Optional[RespuestaAlumno]
    evaluacion: Optional[EvaluacionProfesor]
