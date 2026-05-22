"""
Esquemas Pydantic compartidos entre todos los agentes.

Principio GIGO (Garbage In, Garbage Out): Estos modelos garantizan que
los LLMs siempre devuelvan datos estructurados y validados mediante
el mecanismo `with_structured_output` de LangChain.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class PreguntaProfesor(BaseModel):
    """
    Esquema para la pregunta generada por el agente Profesor.

    El LLM debe rellenar este modelo de forma estricta para garantizar
    que la pregunta sea coherente con el temario provisto.
    """

    pregunta: str = Field(
        description=(
            "Pregunta clara, concisa y pedagógicamente relevante sobre "
            "el temario proporcionado. Debe incitar al pensamiento crítico."
        )
    )
    justificacion: str = Field(
        description=(
            "Breve justificación de por qué esta pregunta es relevante "
            "para evaluar el conocimiento del alumno sobre el temario."
        )
    )


class RespuestaAlumno(BaseModel):
    """
    Esquema para la respuesta generada por el agente Alumno.

    Incluye la respuesta elaborada y las fuentes del sistema RAG
    que se usaron para construirla, garantizando trazabilidad.
    """

    respuesta: str = Field(
        description=(
            "Respuesta elaborada y bien argumentada a la pregunta del profesor, "
            "basada en el contexto recuperado por el sistema RAG."
        )
    )
    fuentes_usadas: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de identificadores o descripciones breves de los fragmentos "
            "de información (documentos/chunks) usados para formular la respuesta."
        ),
    )


class EvaluacionProfesor(BaseModel):
    """
    Esquema para la evaluación estructurada del agente Profesor.

    Proporciona una nota numérica y feedback detallado con puntos
    fuertes y débiles, siguiendo criterios pedagógicos objetivos.
    """

    nota: Annotated[float, Field(ge=0.0, le=10.0)] = Field(
        description=(
            "Nota numérica de 0.0 a 10.0 que refleja la calidad y precisión "
            "de la respuesta del alumno en relación con el temario."
        )
    )
    feedback: str = Field(
        description=(
            "Comentario general y constructivo sobre la respuesta del alumno. "
            "Debe ser objetivo, específico y orientado a la mejora."
        )
    )
    puntos_fuertes: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de aspectos positivos destacables de la respuesta del alumno."
        ),
    )
    puntos_debiles: list[str] = Field(
        default_factory=list,
        description=(
            "Lista de aspectos a mejorar o errores identificados en la respuesta."
        ),
    )
