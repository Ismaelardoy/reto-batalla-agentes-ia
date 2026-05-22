import os
import pytest
from unittest.mock import patch
from src.config.llm_factory import get_llm, _get_provider

def test_get_provider_unsupported():
    """Verifica que un proveedor no soportado lance un ValueError."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"}):
        with pytest.raises(ValueError, match="Proveedor 'invalid_provider' no soportado"):
            _get_provider()

def test_openai_missing_key():
    """Verifica que se lance un EnvironmentError si falta la API Key de OpenAI o es un placeholder."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
        get_llm.cache_clear()  # Limpiar caché de lru_cache para forzar re-evaluación
        with pytest.raises(EnvironmentError, match="La variable 'OPENAI_API_KEY' no está configurada"):
            get_llm()

    with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "tu_api_key_aqui"}):
        get_llm.cache_clear()
        with pytest.raises(EnvironmentError, match="La variable 'OPENAI_API_KEY' no está configurada"):
            get_llm()

def test_gemini_missing_key():
    """Verifica que se lance un EnvironmentError si falta la API Key de Gemini o es un placeholder."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
        get_llm.cache_clear()
        with pytest.raises(EnvironmentError, match="La API Key de Gemini no está configurada"):
            get_llm()

    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "tu_api_key_de_gemini_aqui", "GOOGLE_API_KEY": ""}):
        get_llm.cache_clear()
        with pytest.raises(EnvironmentError, match="La API Key de Gemini no está configurada"):
            get_llm()

def test_groq_missing_key():
    """Verifica que se lance un EnvironmentError si falta la API Key de Groq o es un placeholder."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}):
        get_llm.cache_clear()
        with pytest.raises(EnvironmentError, match="La variable 'GROQ_API_KEY' no está configurada"):
            get_llm()

    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "tu_api_key_aqui"}):
        get_llm.cache_clear()
        with pytest.raises(EnvironmentError, match="La variable 'GROQ_API_KEY' no está configurada"):
            get_llm()


@patch("langchain_openai.ChatOpenAI")
def test_openai_success(mock_chat_openai):
    """Verifica que se cree el modelo OpenAI correctamente con los parámetros adecuados."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-123456-valid"}):
        get_llm.cache_clear()
        get_llm(temperature=0.5)
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key="sk-123456-valid"
        )


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_gemini_success(mock_chat_gemini):
    """Verifica que se cree el modelo Gemini correctamente con los parámetros adecuados."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "real_gemini_key_1234"}):
        get_llm.cache_clear()
        get_llm(temperature=0.8)
        mock_chat_gemini.assert_called_once_with(
            model="gemini-1.5-flash",
            temperature=0.8,
            google_api_key="real_gemini_key_1234"
        )


@patch("langchain_groq.ChatGroq")
def test_groq_success(mock_chat_groq):
    """Verifica que se cree el modelo Groq correctamente con los parámetros adecuados."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "real_groq_key_1234"}):
        get_llm.cache_clear()
        get_llm(temperature=0.6)
        mock_chat_groq.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            groq_api_key="real_groq_key_1234"
        )


@patch("langchain_community.chat_models.ChatOllama")
def test_ollama_success(mock_chat_ollama):
    """Verifica que se cree el modelo Ollama correctamente con los parámetros adecuados."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL_NAME": "llama3.2:1b", "OLLAMA_BASE_URL": "http://ollama-test:11434"}):
        get_llm.cache_clear()
        get_llm(temperature=0.2)
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2:1b",
            temperature=0.2,
            base_url="http://ollama-test:11434"
        )

