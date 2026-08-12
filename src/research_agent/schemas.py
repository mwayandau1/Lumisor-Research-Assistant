"""Structured data models shared across the graph."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ResearchObjective(BaseModel):
    """A single objective within the overall research plan."""

    id: int = Field(..., description="1-indexed order of this objective in the plan")
    title: str = Field(..., description="Short name, e.g. 'Existing attacks'")
    description: str = Field(..., description="1-2 sentences on what this objective covers")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Concrete questions that, once answered, satisfy this objective",
    )


class ResearchPlan(BaseModel):
    """The full decomposition of a research topic, produced by the Planner Agent."""

    topic: str = Field(..., description="The original research topic as given by the user")
    summary: str = Field(..., description="A short restatement of the topic")
    objectives: List[ResearchObjective] = Field(
        ..., description="Ordered list of objectives that together cover the topic"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Expanded search queries to use for retrieval",
    )


# Milestone 2 schemas

class RetrievedPaper(BaseModel):
    """A paper retrieved from arXiv."""

    paper_id: str = Field(..., description="arXiv ID")
    title: str
    authors: List[str]
    abstract: str
    published: Optional[str] = None
    pdf_url: Optional[str] = None


class RAGAnswer(BaseModel):
    """An answer to a sub-question, grounded in retrieved evidence."""

    question: str
    answer: str
    sources: List[str] = Field(
        default_factory=list,
        description="List of paper IDs used to generate this answer",
    )
