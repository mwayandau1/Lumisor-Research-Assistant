"""
Central configuration: environment variables and shared LLM client.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-agent")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "claude-sonnet-4-6")
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.3"))


def get_llm(model: str = PLANNER_MODEL, temperature: float = PLANNER_TEMPERATURE) -> ChatAnthropic:
    """Return a configured ChatAnthropic client."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatAnthropic(model=model, temperature=temperature, api_key=ANTHROPIC_API_KEY)
