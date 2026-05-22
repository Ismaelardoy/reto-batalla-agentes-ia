"""
Servidor FastAPI — Punto de entrada de la aplicación.

Rutas disponibles:
    GET  /               → Healthcheck para Docker y load balancers.
    POST /start_battle   → Inicia la simulación completa de la batalla.
    GET  /docs           → Swagger UI (auto-generada por FastAPI).
    GET  /redoc          → Documentación ReDoc (auto-generada por FastAPI).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import os

from src.graph.main_graph import battle_graph
from src.shared.schemas import EvaluacionProfesor, PreguntaProfesor, RespuestaAlumno

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas de Request / Response de la API
# ---------------------------------------------------------------------------


class BattleRequest(BaseModel):
    """Cuerpo de la petición para iniciar una batalla."""

    temario: str = Field(
        ...,
        min_length=50,
        description=(
            "Texto del temario sobre el cual el Profesor formulará la pregunta "
            "y evaluará al Alumno. Mínimo 50 caracteres."
        ),
        examples=[
            "Inteligencia Artificial y Machine Learning: redes neuronales, "
            "deep learning, modelos de lenguaje (LLMs), arquitectura Transformer, "
            "sistemas RAG y agentes autónomos con LangGraph."
        ],
    )


class BattleResponse(BaseModel):
    """Respuesta completa de la simulación de batalla."""

    duracion_segundos: float = Field(
        description="Tiempo total de ejecución de la batalla en segundos."
    )
    pregunta: PreguntaProfesor = Field(
        description="Pregunta formulada por el Profesor IA."
    )
    respuesta: RespuestaAlumno = Field(
        description="Respuesta elaborada por el Alumno IA mediante RAG."
    )
    evaluacion: EvaluacionProfesor = Field(
        description="Evaluación final del Profesor IA con nota y feedback."
    )


class HealthResponse(BaseModel):
    """Respuesta del healthcheck."""

    status: str = "ok"
    service: str = "batalla-chatbots-api"
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicación
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación FastAPI.
    Se ejecuta al arranque (before yield) y al apagado (after yield).
    """
    logger.info("🚀 Arrancando la API de Batalla de Chatbots...")
    logger.info("📊 Grafo de LangGraph cargado y listo.")
    yield
    logger.info("🛑 Apagando la API...")


# ---------------------------------------------------------------------------
# Instancia de FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="⚔️ Batalla de Chatbots: Profesor IA vs Alumno IA",
    description=(
        "API para simular una batalla educativa entre un Profesor IA y un Alumno IA. "
        "El Profesor formula preguntas sobre un temario, el Alumno responde usando RAG "
        "y el Profesor evalúa la respuesta con criterios pedagógicos objetivos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware CORS (permite llamadas desde cualquier origen en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rutas de la API
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck",
    tags=["Sistema"],
)
def healthcheck() -> HealthResponse:
    """
    Endpoint de salud del servicio.

    Retorna un JSON confirmando que la API está activa.
    Esencial para los health checks de Docker, Kubernetes y load balancers.
    """
    return HealthResponse()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    """
    Sirve la interfaz web de la aplicación (frontend).
    """
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend en construcción</h1><p>El archivo index.html no fue encontrado.</p>",
            status_code=404
        )


@app.post(
    "/start_battle",
    response_model=BattleResponse,
    status_code=status.HTTP_200_OK,
    summary="Iniciar Batalla de Chatbots",
    tags=["Batalla"],
)
def start_battle(request: BattleRequest) -> BattleResponse:
    """
    Inicia la simulación completa de la batalla entre el Profesor IA y el Alumno IA.

    **Flujo interno:**
    1. **Profesor** lee el temario y formula una pregunta estructurada.
    2. **Alumno** consulta su sistema RAG y elabora una respuesta argumentada.
    3. **Profesor** evalúa la respuesta con nota (0-10) y feedback detallado.

    **Retorna** el resultado completo de la batalla incluyendo pregunta,
    respuesta, evaluación con nota y tiempo de ejecución.

    **Posibles errores:**
    - `422 Unprocessable Entity`: El temario no cumple los requisitos mínimos.
    - `500 Internal Server Error`: Error en la ejecución del grafo LangGraph.
    """
    logger.info("⚔️ Iniciando batalla. Longitud del temario: %d chars.", len(request.temario))
    start_time = time.monotonic()

    try:
        # Invocar el grafo con el estado inicial
        initial_state = {"temario": request.temario}
        final_state = battle_graph.invoke(initial_state)

    except Exception as exc:
        logger.exception("❌ Error durante la ejecución del grafo: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en la ejecución de la batalla: {str(exc)}",
        ) from exc

    elapsed = round(time.monotonic() - start_time, 3)
    logger.info("🏆 Batalla finalizada en %.3f segundos. Nota del alumno: %.1f/10",
                elapsed, final_state["evaluacion"].nota)

    return BattleResponse(
        duracion_segundos=elapsed,
        pregunta=final_state["pregunta"],
        respuesta=final_state["respuesta"],
        evaluacion=final_state["evaluacion"],
    )
