"""Shared state passed between LangGraph nodes."""

from typing import Optional, TypedDict

from research_agent.schemas import ResearchPlan


class GraphState(TypedDict, total=False):
    """The state object threaded through the graph."""

    topic: str
    plan: Optional[ResearchPlan]
