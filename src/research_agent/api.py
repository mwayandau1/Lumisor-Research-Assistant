"""HTTP API for manual/Postman testing of the research agent.

Each endpoint isolates one stage of the graph so a stage can be exercised
on its own (cheap, fast feedback) without paying for the stages before it -
the same idea as scripts/test_retrieval.py and scripts/test_extraction.py,
just reachable from Postman instead of the CLI.

Run with:
    uv run uvicorn research_agent.api:app --reload --port 8000
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from research_agent.extraction import run_extraction
from research_agent.graph import run as run_graph
from research_agent.planner import run_planner
from research_agent.rag import run_rag
from research_agent.schemas import (
    MergeStats,
    PaperRecord,
    RAGAnswer,
    ResearchPlan,
    RetrievedPaper,
)
from research_agent.search import enabled_source_nodes, merge_source_results, run_sources

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Agent API",
    description="Manual test surface for the autonomous research agent's LangGraph nodes.",
    version="0.1.0",
)


# ---- request/response models -------------------------------------------------

class TopicRequest(BaseModel):
    topic: str


class SearchRequest(BaseModel):
    topic: Optional[str] = None
    queries: Optional[List[str]] = None


class SearchResponse(BaseModel):
    merge_stats: MergeStats
    papers: List[RetrievedPaper]


class RetrieveResponse(BaseModel):
    plan: ResearchPlan
    merge_stats: Optional[MergeStats] = None
    retrieved_papers: List[RetrievedPaper]
    rag_answers: List[RAGAnswer]


class ExtractRequest(BaseModel):
    topic: Optional[str] = None
    papers: Optional[List[RetrievedPaper]] = None


class ExtractResponse(BaseModel):
    extracted_records: List[PaperRecord]


class ResearchResponse(BaseModel):
    plan: ResearchPlan
    merge_stats: Optional[MergeStats] = None
    retrieved_papers: List[RetrievedPaper]
    rag_answers: List[RAGAnswer]
    extracted_records: List[PaperRecord]


# ---- endpoints -----------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Basic connectivity check - confirms the server is up, no LLM calls."""
    return {"status": "ok"}


@app.post("/plan", response_model=ResearchPlan)
def plan(req: TopicRequest) -> ResearchPlan:
    """Milestone 1 in isolation: topic -> structured ResearchPlan."""
    try:
        state = run_planner({"topic": req.topic})
    except Exception as exc:
        logger.exception("Planner failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return state["plan"]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Milestone 4 in isolation: fan out to every enabled source, then merge.

    Deliberately skips embedding and storage, so this is the cheap way to see
    what the second source actually bought - `merge_stats` reports per-source
    counts, how many papers both sources found, and any source that failed.
    Pass `queries` to skip the planner's LLM call entirely and pay nothing.
    """
    if not req.queries and not req.topic:
        raise HTTPException(status_code=422, detail="Provide either 'queries' or 'topic'.")

    try:
        if req.queries:
            # A plan is the only thing the source nodes read, so a synthetic one
            # over the given queries is enough to exercise them for free.
            research_plan = ResearchPlan(
                topic=req.topic or "ad-hoc query",
                summary="Ad-hoc source comparison - no planner call.",
                objectives=[],
                search_queries=req.queries,
            )
        else:
            research_plan = run_planner({"topic": req.topic})["plan"]

        state = {"plan": research_plan}
        source_results = []
        for node in enabled_source_nodes().values():
            source_results.extend(node(state).get("source_results", []))
        papers, stats = merge_source_results(source_results)
    except Exception as exc:
        logger.exception("Source search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(merge_stats=stats, papers=papers)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: TopicRequest) -> RetrieveResponse:
    """Milestones 1+2+4: plan, then parallel source search + merge, then RAG answers."""
    try:
        state = run_planner({"topic": req.topic})
        state = run_sources(state)
        state = run_rag(state)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RetrieveResponse(
        plan=state["plan"],
        merge_stats=state.get("merge_stats"),
        retrieved_papers=state.get("retrieved_papers", []),
        rag_answers=state.get("rag_answers", []),
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """
    Milestone 3 in isolation.

    Pass `papers` directly to test extraction on hand-crafted abstracts
    without paying for search/embedding calls, or pass `topic` to run
    plan -> search -> merge -> extract. The RAG step is skipped either way:
    extraction reads the retrieved papers, not the answers.
    """
    if not req.papers and not req.topic:
        raise HTTPException(status_code=422, detail="Provide either 'papers' or 'topic'.")

    try:
        if req.papers:
            state = {"retrieved_papers": req.papers}
        else:
            state = run_planner({"topic": req.topic})
            state = run_sources(state)
        state = run_extraction(state)
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ExtractResponse(extracted_records=state.get("extracted_records", []))


@app.post("/research", response_model=ResearchResponse)
def research(req: TopicRequest) -> ResearchResponse:
    """Full pipeline through the compiled graph - identical to `python main.py`."""
    try:
        state = run_graph(req.topic)
    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        plan=state["plan"],
        merge_stats=state.get("merge_stats"),
        retrieved_papers=state.get("retrieved_papers", []),
        rag_answers=state.get("rag_answers", []),
        extracted_records=state.get("extracted_records", []),
    )
