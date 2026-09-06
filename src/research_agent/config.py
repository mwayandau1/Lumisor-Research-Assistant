"""
Central configuration: environment variables, logging, and shared clients via OpenRouter.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# General-purpose LLM - used where output is plain prose, e.g. RAG answers.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "openai/gpt-4o-mini")
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))

# Structured-output LLM - used wherever we bind a Pydantic schema (with_structured_output).
# Small free models are unreliable at strict tool-calling/JSON-schema adherence, so
# planning and extraction get a model that's proven to follow schemas consistently.
STRUCTURED_MODEL = os.getenv("STRUCTURED_MODEL", "openai/gpt-4o-mini")
STRUCTURED_TEMPERATURE = float(os.getenv("STRUCTURED_TEMPERATURE", "0"))

# Embeddings (OpenAI via OpenRouter)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Retrieval (Milestone 2/4)
# How many of the planner's expanded queries each source runs, and how many
# papers it asks for per query. Both were hardcoded in the retrieval node before
# Milestone 4; with two sources fanning out in parallel they multiply, so they
# belong somewhere you can turn them down without editing a node.
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "5"))
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "3"))

# Semantic Scholar (Milestone 4 second source).
# The API works without a key, but unauthenticated traffic shares one small
# global rate-limit pool, so 429s are routine - hence the retry/backoff in
# semantic_scholar.py. A free key raises the limit substantially.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SEMANTIC_SCHOLAR_TIMEOUT = float(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT", "20"))

# Sources the retrieval fan-out is allowed to use. Trim this to run a
# single-source comparison, or to skip a source that is rate-limiting you.
ENABLED_SOURCES = [
    s.strip()
    for s in os.getenv("ENABLED_SOURCES", "arxiv,semantic_scholar").split(",")
    if s.strip()
]


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
