"""Shared state passed between LangGraph nodes."""

from typing import List, Optional, TypedDict

from research_agent.schemas import RAGAnswer, ResearchPlan, RetrievedPaper


class GraphState(TypedDict, total=False):
    """The state object threaded through the graph."""

    # Milestone 0/1
    topic: str
    plan: Optional[ResearchPlan]

    # Milestone 2
    retrieved_papers: List[RetrievedPaper]
    rag_answers: List[RAGAnswer]
