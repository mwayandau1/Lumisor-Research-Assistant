"""Structured data models shared across the graph."""

from typing import Dict, List, Optional

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
    """A paper retrieved from one of the configured sources."""

    paper_id: str = Field(..., description="Source-local ID: arXiv ID, or Semantic Scholar paperId")
    title: str
    authors: List[str]
    abstract: str
    published: Optional[str] = None
    pdf_url: Optional[str] = None

    # Milestone 4: papers now arrive from more than one source, so each record has
    # to say where it came from and carry the identifiers used to deduplicate it.
    source: str = Field("arxiv", description="Source that returned this paper")
    url: Optional[str] = Field(None, description="Landing page for the paper")
    doi: Optional[str] = Field(None, description="DOI, when the source reports one")
    arxiv_id: Optional[str] = Field(
        None, description="Normalised arXiv ID (no version suffix), when known"
    )
    year: Optional[int] = None
    citation_count: Optional[int] = Field(
        None, description="Citations reported by the source, when available"
    )
    found_by: List[str] = Field(
        default_factory=list,
        description="Every source that returned this paper; more than one means the sources corroborate each other",
    )


class RAGAnswer(BaseModel):
    """An answer to a sub-question, grounded in retrieved evidence."""

    question: str
    answer: str
    sources: List[str] = Field(
        default_factory=list,
        description="List of paper IDs used to generate this answer",
    )


# Milestone 3 schemas

class PaperRecord(BaseModel):
    """A structured summary of a single paper, extracted from its abstract."""

    paper_id: str = Field(..., description="arXiv ID, links back to RetrievedPaper")
    title: str

    method: Optional[str] = Field(
        None, description="The core method/approach the paper proposes, in 1-2 sentences"
    )
    dataset: Optional[str] = Field(
        None, description="Dataset(s) or benchmark(s) used for evaluation, if named"
    )
    results: Optional[str] = Field(
        None, description="Key reported result(s), e.g. a metric improvement, if stated"
    )
    limitations: Optional[str] = Field(
        None, description="Limitations or weaknesses the paper itself acknowledges, if any"
    )


# Milestone 4 schemas

class SourceResult(BaseModel):
    """What a single retrieval source returned during one graph run.

    Each source node emits exactly one of these. Keeping the per-source
    envelope (rather than dumping papers straight into one list) is what lets
    the merge node report recall/overlap and lets one source fail without
    taking the run down with it.
    """

    source: str = Field(..., description="Source name, e.g. 'arxiv' or 'semantic_scholar'")
    papers: List[RetrievedPaper] = Field(default_factory=list)
    queries_run: int = Field(0, description="How many search queries this source executed")
    error: Optional[str] = Field(
        None,
        description="Set when the source failed; the run continues on the remaining sources",
    )


class MergeStats(BaseModel):
    """Recall and overlap accounting for the parallel-source merge.

    This is the evidence that adding a second source actually bought
    something: if `overlap_papers` is close to `unique_papers`, the second
    source is mostly returning what arXiv already found.
    """

    per_source: Dict[str, int] = Field(
        default_factory=dict, description="Papers returned per source, before dedup"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict, description="Per-source failure messages, if any"
    )
    total_before_dedup: int = 0
    unique_papers: int = 0
    overlap_papers: int = Field(
        0, description="Papers that more than one source returned - the recall the merge saved"
    )
    dropped_no_abstract: int = Field(
        0, description="Papers discarded because the source returned no abstract to embed"
    )
