"""
Factoría centralizada de LLMs.

Principio DRY: Un único punto de creación del LLM para toda la aplicación.
Soporta múltiples proveedores configurables mediante variables de entorno.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

_SUPPORTED_PROVIDERS = ("openai", "ollama", "gemini", "groq")


def _get_provider() -> str:
    """Obtiene y valida el proveedor de LLM configurado."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. "
            f"Opciones válidas: {_SUPPORTED_PROVIDERS}"
        )
    return provider


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.7) -> BaseChatModel:
    """
    Factoría que devuelve una instancia del LLM configurada.

    Utiliza `lru_cache` para que instancias con la misma temperatura
    se reutilicen en lugar de crearse repetidas veces (patrón Flyweight).

    Args:
        temperature: Temperatura de muestreo del modelo (0.0 - 2.0).
                     Valores bajos = más deterministas, altos = más creativos.

    Returns:
        BaseChatModel: Instancia del LLM lista para usar.

    Raises:
        ValueError: Si el proveedor configurado no está soportado.
        EnvironmentError: Si faltan variables de entorno requeridas.
    """
    provider = _get_provider()

    if provider == "openai":
        return _build_openai_llm(temperature)
    elif provider == "ollama":
        return _build_ollama_llm(temperature)
    elif provider == "gemini":
        return _build_gemini_llm(temperature)
    elif provider == "groq":
        return _build_groq_llm(temperature)

    # Guarda de seguridad (nunca debería llegar aquí gracias a _get_provider)
    raise ValueError(f"Proveedor no implementado: {provider}")


def _build_openai_llm(temperature: float) -> BaseChatModel:
    """Crea una instancia de ChatOpenAI con la configuración del entorno."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "Instala 'langchain-openai' para usar el proveedor OpenAI. "
            "Ejecuta: pip install langchain-openai"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "api_key" in api_key.lower() or "tu_api_key" in api_key.lower():
        raise EnvironmentError(
            "La variable 'OPENAI_API_KEY' no está configurada o contiene un valor por defecto. "
            "Por favor, coloca tu clave real en el archivo .env."
        )

    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def _build_ollama_llm(temperature: float) -> BaseChatModel:
    """Crea una instancia de ChatOllama con la configuración del entorno."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError(
                "Instala 'langchain-ollama' o 'langchain-community' para usar Ollama. "
                "Ejecuta: pip install langchain-ollama"
            ) from e

    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        # Auto-detect base URL: use internal docker link if running in container, else localhost
        if os.path.exists("/.dockerenv"):
            base_url = "http://ollama:11434"
        else:
            base_url = "http://localhost:11434"
            
    model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")

    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
    )


def _build_gemini_llm(temperature: float) -> BaseChatModel:
    """Crea una instancia de ChatGoogleGenerativeAI con la configuración del entorno."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "Instala 'langchain-google-genai' para usar el proveedor Gemini. "
            "Ejecuta: pip install langchain-google-genai"
        ) from e

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or "api_key" in api_key.lower() or "tu_api_key" in api_key.lower():
        raise EnvironmentError(
            "La API Key de Gemini no está configurada. "
            "Por favor, obtén una clave de API válida en Google AI Studio (https://aistudio.google.com/) y colócala en tu archivo .env."
        )

    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def _build_groq_llm(temperature: float) -> BaseChatModel:
    """Crea una instancia de ChatGroq con la configuración del entorno."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "Instala 'langchain-groq' para usar el proveedor Groq. "
            "Ejecuta: pip install langchain-groq"
        ) from e

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "api_key" in api_key.lower() or "tu_api_key" in api_key.lower():
        raise EnvironmentError(
            "La variable 'GROQ_API_KEY' no está configurada o contiene un valor por defecto. "
            "Por favor, coloca tu clave real en el archivo .env."
        )

    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )
