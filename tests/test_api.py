from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest

from src.app import app
from src.shared.schemas import PreguntaProfesor, RespuestaAlumno, EvaluacionProfesor

client = TestClient(app)

def test_healthcheck():
    """Verifica que el endpoint /health responda correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "batalla-chatbots-api"

def test_serve_frontend():
    """Verifica que la ruta raíz sirva el frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "").lower()

@patch("src.app.battle_graph.invoke")
def test_start_battle(mock_invoke):
    """Verifica que el endpoint /start_battle procese las peticiones correctamente simulando el grafo."""
    # Mock del resultado del grafo
    mock_invoke.return_value = {
        "temario": "Inteligencia Artificial y Machine Learning con redes neuronales, deep learning y procesamiento de lenguaje natural.",
        "pregunta": PreguntaProfesor(
            pregunta="¿Qué es el Deep Learning?",
            justificacion="Concepto básico de redes neuronales profundas."
        ),
        "respuesta": RespuestaAlumno(
            respuesta="El Deep Learning es un subcampo de Machine Learning basado en redes neuronales con múltiples capas.",
            fuentes_usadas=["doc_001"]
        ),
        "evaluacion": EvaluacionProfesor(
            nota=9.5,
            feedback="Excelente y concisa explicación.",
            puntos_fuertes=["Precisión", "Claridad"],
            puntos_debiles=["Ninguno relevante"]
        )
    }

    # Petición con temario suficientemente largo
    payload = {
        "temario": "Inteligencia Artificial y Machine Learning con redes neuronales, deep learning y procesamiento de lenguaje natural."
    }
    response = client.post("/start_battle", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "duracion_segundos" in data
    assert data["pregunta"]["pregunta"] == "¿Qué es el Deep Learning?"
    assert data["respuesta"]["respuesta"] == "El Deep Learning es un subcampo de Machine Learning basado en redes neuronales con múltiples capas."
    assert data["evaluacion"]["nota"] == 9.5

def test_start_battle_validation_error():
    """Verifica que la petición falle si el temario es demasiado corto (<50 caracteres)."""
    payload = {
        "temario": "Corto"
    }
    response = client.post("/start_battle", json=payload)
    assert response.status_code == 422
