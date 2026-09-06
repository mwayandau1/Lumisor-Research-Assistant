"""Milestone 4: parallel multi-source retrieval and merge.

Milestone 2 did search, storage and RAG in one node against one source. That
does not extend: adding a source to a sequential node makes the run slower in
proportion to how many sources there are, and one flaky API takes the whole
run down with it.

So retrieval is split into three node types:

    planner -+-> arxiv_search -------------+-> merge_sources -> rag
             +-> semantic_scholar_search --+

The source nodes have no dependency on each other, so LangGraph runs them in
one superstep - two sources cost about as much wall-clock as the slower one,
not the sum. Each returns *only* its own `source_results` entry, which the
reducer on `GraphState` concatenates; returning the whole state from parallel
branches would make them fight over every other key.

`merge_sources` is where the second source actually pays off: it deduplicates
across sources (arXiv ID, then DOI, then normalised title), folds the two
records for a paper both sources found into one richer record, and reports what
the extra source bought in `MergeStats`.
"""

import logging
import re
import time
from typing import Callable

from research_agent.config import (
    ENABLED_SOURCES,
    MAX_SEARCH_QUERIES,
    RESULTS_PER_QUERY,
)
from research_agent.retriever import embed_and_store, search_arxiv
from research_agent.schemas import MergeStats, ResearchPlan, RetrievedPaper, SourceResult
from research_agent.semantic_scholar import DELAY_BETWEEN_QUERIES, search_semantic_scholar
from research_agent.state import GraphState

logger = logging.getLogger(__name__)

# Fields worth back-filling from a duplicate record. `abstract` and `authors`
# are handled separately (longest / non-empty wins, rather than first non-null).
_MERGEABLE_FIELDS = ("published", "pdf_url", "url", "doi", "arxiv_id", "year", "citation_count")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Circuit breaker: once this many queries in a row have failed, the source is
# down rather than unlucky, and the remaining queries would only add latency to
# a run that is already going to proceed without it.
MAX_CONSECUTIVE_FAILURES = 2


# ---- one source ------------------------------------------------------------

def _run_source(
    source: str,
    search_fn: Callable[[str, int], list[RetrievedPaper]],
    plan: ResearchPlan,
    delay_between_queries: float = 0.0,
) -> SourceResult:
    """Run one source across the plan's search queries.

    A query that fails is logged and skipped rather than raised - one bad query
    (or one 429) should cost us that query's results, not the whole source. The
    source is only reported as failed if it came back with nothing at all, and
    it is abandoned early once MAX_CONSECUTIVE_FAILURES queries in a row fail.
    """
    queries = plan.search_queries[:MAX_SEARCH_QUERIES]
    papers: list[RetrievedPaper] = []
    failures: list[str] = []
    consecutive_failures = 0
    attempted = 0

    for i, query in enumerate(queries):
        if i and delay_between_queries:
            time.sleep(delay_between_queries)

        attempted += 1
        try:
            found = search_fn(query, RESULTS_PER_QUERY)
        except Exception as exc:
            failures.append(f"{query!r}: {exc}")
            consecutive_failures += 1
            logger.warning("[%s] query failed: %s", source, exc)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "[%s] %d consecutive failures - skipping the remaining %d queries.",
                    source,
                    consecutive_failures,
                    len(queries) - attempted,
                )
                break
            continue

        consecutive_failures = 0
        papers.extend(found)

    logger.info(
        "[%s] %d papers from %d/%d queries", source, len(papers), attempted - len(failures), len(queries)
    )

    # Only a source that produced nothing counts as failed; partial results are
    # still results, and the merge will happily use them.
    error = "; ".join(failures) if failures and not papers else None
    return SourceResult(source=source, papers=papers, queries_run=attempted, error=error)


def run_arxiv_search(state: GraphState) -> dict:
    """LangGraph node: the arXiv branch of the retrieval fan-out."""
    plan = state.get("plan")
    if not plan:
        return {"source_results": []}
    return {"source_results": [_run_source("arxiv", search_arxiv, plan)]}


def run_semantic_scholar_search(state: GraphState) -> dict:
    """LangGraph node: the Semantic Scholar branch of the retrieval fan-out."""
    plan = state.get("plan")
    if not plan:
        return {"source_results": []}
    return {
        "source_results": [
            _run_source(
                "semantic_scholar",
                search_semantic_scholar,
                plan,
                delay_between_queries=DELAY_BETWEEN_QUERIES,
            )
        ]
    }


# Registry the graph builds its fan-out from, so ENABLED_SOURCES actually
# changes the graph's shape instead of being re-checked inside every node.
SOURCE_NODES: dict[str, Callable[[GraphState], dict]] = {
    "arxiv": run_arxiv_search,
    "semantic_scholar": run_semantic_scholar_search,
}


def enabled_source_nodes() -> dict[str, Callable[[GraphState], dict]]:
    """The source nodes ENABLED_SOURCES asks for, ignoring unknown names."""
    nodes = {name: SOURCE_NODES[name] for name in ENABLED_SOURCES if name in SOURCE_NODES}
    if not nodes:
        logger.warning(
            "ENABLED_SOURCES matched no known source (%s) - falling back to arXiv.", ENABLED_SOURCES
        )
        return {"arxiv": run_arxiv_search}
    return nodes


# ---- merge -----------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation/whitespace, so formatting alone can't hide a duplicate."""
    return _NON_ALNUM.sub("", title.lower())


def _dedup_keys(paper: RetrievedPaper) -> list[str]:
    """Identity keys for a paper, strongest first.

    arXiv ID and DOI are exact identifiers; the normalised title is the
    fallback for records where neither source reported one.
    """
    keys = []
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        keys.append(f"doi:{paper.doi.strip().lower()}")
    title_key = _normalize_title(paper.title)
    if title_key:
        keys.append(f"title:{title_key}")
    return keys


def _merge_paper(base: RetrievedPaper, other: RetrievedPaper) -> RetrievedPaper:
    """Fold a duplicate record into the one being kept, taking the better of each field."""
    merged = base.model_copy(deep=True)

    # The longer abstract is the more complete one - and beats an empty one,
    # which is how an abstract-less Semantic Scholar hit gets rescued by arXiv.
    if len(other.abstract or "") > len(merged.abstract or ""):
        merged.abstract = other.abstract
    if not merged.authors and other.authors:
        merged.authors = other.authors

    for field in _MERGEABLE_FIELDS:
        if not getattr(merged, field) and getattr(other, field):
            setattr(merged, field, getattr(other, field))

    for source in other.found_by or [other.source]:
        if source not in merged.found_by:
            merged.found_by.append(source)

    return merged


def merge_source_results(results: list[SourceResult]) -> tuple[list[RetrievedPaper], MergeStats]:
    """Deduplicate papers across sources and report what the merge did.

    Sources are processed in ENABLED_SOURCES order, so the preferred source
    supplies the canonical record and `_merge_paper` back-fills from the rest.
    """
    order = {name: i for i, name in enumerate(ENABLED_SOURCES)}
    ordered = sorted(results, key=lambda r: order.get(r.source, len(order)))

    stats = MergeStats(
        per_source={r.source: len(r.papers) for r in ordered},
        errors={r.source: r.error for r in ordered if r.error},
        total_before_dedup=sum(len(r.papers) for r in ordered),
    )

    merged: dict[str, RetrievedPaper] = {}   # canonical id -> paper
    key_index: dict[str, str] = {}           # dedup key -> canonical id

    for result in ordered:
        for paper in result.papers:
            paper = paper.model_copy(deep=True)
            if not paper.found_by:
                paper.found_by = [paper.source]

            keys = _dedup_keys(paper)
            canonical = next((key_index[k] for k in keys if k in key_index), None)

            if canonical is None:
                canonical = paper.paper_id
                merged[canonical] = paper
            else:
                merged[canonical] = _merge_paper(merged[canonical], paper)

            # Register every key against the canonical record, including ones
            # learned from the duplicate - a DOI only the second source reported
            # still has to be able to match a third record later.
            for key in keys:
                key_index.setdefault(key, canonical)

    # A paper with no abstract from any source can be neither embedded nor
    # extracted from, so it would only pollute the index. Dropping it is counted.
    keep = [p for p in merged.values() if p.abstract and p.abstract.strip()]
    stats.dropped_no_abstract = len(merged) - len(keep)
    stats.unique_papers = len(keep)
    stats.overlap_papers = sum(1 for p in keep if len(p.found_by) > 1)

    return keep, stats


def run_merge_sources(state: GraphState) -> dict:
    """LangGraph node: the fan-in. Deduplicates across sources, then embeds and stores."""
    results = state.get("source_results") or []
    papers, stats = merge_source_results(results)

    logger.info(
        "Merged %d papers from %s -> %d unique (%d found by >1 source, %d dropped for no abstract)",
        stats.total_before_dedup,
        stats.per_source,
        stats.unique_papers,
        stats.overlap_papers,
        stats.dropped_no_abstract,
    )
    for source, error in stats.errors.items():
        logger.warning("Source %s returned nothing: %s", source, error)

    stored = embed_and_store(papers)
    logger.info("Embedded and stored %d new chunks", stored)

    return {"retrieved_papers": papers, "merge_stats": stats}


# ---- sequential convenience -------------------------------------------------

def run_sources(state: GraphState) -> GraphState:
    """Run the whole fan-out and merge sequentially, outside the graph.

    The API and the test scripts want one call they can step through; the
    parallelism is the graph's job and nothing in the merge logic depends on it,
    so running the same nodes in order produces identical output.
    """
    source_results: list[SourceResult] = []
    for node in enabled_source_nodes().values():
        source_results.extend(node(state).get("source_results", []))

    state = {**state, "source_results": source_results}
    return {**state, **run_merge_sources(state)}
