"""
Script de prueba rápida para el endpoint /start_battle.

Ejecutar con:
    python test_battle.py

Requisitos previos:
    - La API debe estar corriendo en http://localhost:8000
    - Las variables de entorno deben estar configuradas en .env
"""

import json
import sys
import time

import httpx

API_BASE_URL = "http://localhost:8000"

TEMARIO_EJEMPLO = """
Inteligencia Artificial y Sistemas Multi-Agente con LangGraph:

1. Fundamentos de IA y Machine Learning:
   - Definición y tipos de IA (estrecha, general, superinteligencia)
   - Aprendizaje supervisado, no supervisado y por refuerzo
   - Redes neuronales y Deep Learning

2. Modelos de Lenguaje Grande (LLMs):
   - Arquitectura Transformer y mecanismo de atención
   - Pre-entrenamiento y fine-tuning
   - RLHF (Reinforcement Learning from Human Feedback)
   - Modelos actuales: GPT-4, Claude, Gemini

3. Sistemas RAG (Retrieval-Augmented Generation):
   - Problema de las alucinaciones en LLMs
   - Pipeline RAG: ingesta, embedding, retrieval, generación
   - Bases de datos vectoriales (FAISS, ChromaDB, Pinecone)

4. Orquestación con LangGraph:
   - StateGraph y flujos de control
   - Sistemas multi-agente
   - Patrones: nodo, arista, estado compartido

5. APIs con FastAPI y Pydantic:
   - Validación de datos con Pydantic v2
   - Structured Output en LangChain
   - Endpoints RESTful y documentación OpenAPI
"""


def test_healthcheck() -> bool:
    """Prueba el endpoint de healthcheck."""
    print("\n🔍 Probando healthcheck (GET /health)...")
    response = httpx.get(f"{API_BASE_URL}/health")
    response.raise_for_status()
    data = response.json()
    assert data["status"] == "ok", f"Estado inesperado: {data}"
    print(f"   ✅ Healthcheck OK: {data}")
    return True


def test_start_battle() -> bool:
    """Prueba el endpoint principal de batalla."""
    print("\n⚔️  Iniciando batalla (POST /start_battle)...")
    print("   ⏳ Esto puede tardar entre 10-60 segundos dependiendo del LLM...\n")

    start = time.monotonic()
    response = httpx.post(
        f"{API_BASE_URL}/start_battle",
        json={"temario": TEMARIO_EJEMPLO},
        timeout=120.0,  # Timeout generoso para LLMs lentos
    )
    elapsed = time.monotonic() - start

    if response.status_code != 200:
        print(f"   ❌ Error HTTP {response.status_code}: {response.text}")
        return False

    data = response.json()

    # Validaciones básicas de la respuesta
    assert "pregunta" in data, "Falta campo 'pregunta'"
    assert "respuesta" in data, "Falta campo 'respuesta'"
    assert "evaluacion" in data, "Falta campo 'evaluacion'"
    assert 0.0 <= data["evaluacion"]["nota"] <= 10.0, "Nota fuera de rango"

    print("─" * 70)
    print(f"📌 PREGUNTA DEL PROFESOR:")
    print(f"   {data['pregunta']['pregunta']}")
    print(f"\n   Justificación: {data['pregunta']['justificacion']}")

    print(f"\n📝 RESPUESTA DEL ALUMNO:")
    print(f"   {data['respuesta']['respuesta']}")
    print(f"\n   Fuentes RAG usadas: {data['respuesta']['fuentes_usadas']}")

    print(f"\n🏆 EVALUACIÓN DEL PROFESOR:")
    print(f"   Nota: {data['evaluacion']['nota']:.1f}/10")
    print(f"   Feedback: {data['evaluacion']['feedback']}")
    print(f"   Puntos fuertes: {data['evaluacion']['puntos_fuertes']}")
    print(f"   Puntos débiles: {data['evaluacion']['puntos_debiles']}")
    print("─" * 70)
    print(f"\n⏱️  Duración total: {elapsed:.2f}s (API reportó: {data['duracion_segundos']}s)")

    return True


def main() -> None:
    """Ejecuta todas las pruebas en secuencia."""
    print("=" * 70)
    print("  🤖 TEST SUITE: Batalla de Chatbots API")
    print("=" * 70)

    tests = [test_healthcheck, test_start_battle]
    passed = 0

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as exc:
            print(f"   ❌ Error en {test_fn.__name__}: {exc}")

    print(f"\n{'=' * 70}")
    print(f"  Resultado: {passed}/{len(tests)} pruebas pasaron.")
    print("=" * 70)

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
