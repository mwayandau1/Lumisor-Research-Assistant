"""Shared state passed between LangGraph nodes."""

import operator
from typing import Annotated, List, Optional, TypedDict

from research_agent.schemas import (
    MergeStats,
    PaperRecord,
    RAGAnswer,
    ResearchPlan,
    RetrievedPaper,
    SourceResult,
)


class GraphState(TypedDict, total=False):
    """The state object threaded through the graph."""

    # Milestone 0/1
    topic: str
    plan: Optional[ResearchPlan]

    # Milestone 2
    retrieved_papers: List[RetrievedPaper]
    rag_answers: List[RAGAnswer]

    # Milestone 3
    extracted_records: List[PaperRecord]

    # Milestone 4
    # The source nodes run concurrently and all write this one key, so it needs
    # a reducer: without `operator.add` LangGraph treats two writes in the same
    # superstep as a conflict and raises instead of merging them.
    source_results: Annotated[List[SourceResult], operator.add]
    merge_stats: Optional[MergeStats]
