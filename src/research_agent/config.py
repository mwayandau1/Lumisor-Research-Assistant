"""
Central configuration: environment variables and shared clients via OpenRouter.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLM (Liquid AI via OpenRouter)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "liquid/lfm2-2.6b:free")
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))

# Embeddings (OpenAI via OpenRouter)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


def get_llm(model: str = PLANNER_MODEL, temperature: float = PLANNER_TEMPERATURE) -> ChatOpenAI:
    """Return a ChatOpenAI client routed through OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embeddings client routed through OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
