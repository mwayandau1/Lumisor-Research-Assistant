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
from research_agent.planner import run_planner
from research_agent.rag import run_retrieval
from research_agent.schemas import PaperRecord, RAGAnswer, ResearchPlan, RetrievedPaper

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Agent API",
    description="Manual test surface for the autonomous research agent's LangGraph nodes.",
    version="0.1.0",
)


# ---- request/response models -------------------------------------------------

class TopicRequest(BaseModel):
    topic: str


class RetrieveResponse(BaseModel):
    plan: ResearchPlan
    retrieved_papers: List[RetrievedPaper]
    rag_answers: List[RAGAnswer]


class ExtractRequest(BaseModel):
    topic: Optional[str] = None
    papers: Optional[List[RetrievedPaper]] = None


class ExtractResponse(BaseModel):
    extracted_records: List[PaperRecord]


class ResearchResponse(BaseModel):
    plan: ResearchPlan
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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: TopicRequest) -> RetrieveResponse:
    """Milestones 1+2: plan, then arXiv search + RAG answers per sub-question."""
    try:
        state = run_planner({"topic": req.topic})
        state = run_retrieval(state)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RetrieveResponse(
        plan=state["plan"],
        retrieved_papers=state.get("retrieved_papers", []),
        rag_answers=state.get("rag_answers", []),
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """
    Milestone 3 in isolation.

    Pass `papers` directly to test extraction on hand-crafted abstracts
    without paying for arXiv/embedding calls, or pass `topic` to run the
    full plan -> retrieve -> extract chain.
    """
    if not req.papers and not req.topic:
        raise HTTPException(status_code=422, detail="Provide either 'papers' or 'topic'.")

    try:
        if req.papers:
            state = {"retrieved_papers": req.papers}
        else:
            state = run_planner({"topic": req.topic})
            state = run_retrieval(state)
        state = run_extraction(state)
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ExtractResponse(extracted_records=state.get("extracted_records", []))


@app.post("/research", response_model=ResearchResponse)
def research(req: TopicRequest) -> ResearchResponse:
    """Full pipeline: planner -> retrieval -> extraction. Same as `python main.py`."""
    try:
        state = run_planner({"topic": req.topic})
        state = run_retrieval(state)
        state = run_extraction(state)
    except Exception as exc:
        logger.exception("Research pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        plan=state["plan"],
        retrieved_papers=state.get("retrieved_papers", []),
        rag_answers=state.get("rag_answers", []),
        extracted_records=state.get("extracted_records", []),
    )
