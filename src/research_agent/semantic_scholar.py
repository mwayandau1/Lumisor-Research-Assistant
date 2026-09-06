"""Milestone 4: Semantic Scholar - the second retrieval source.

Why this one: arXiv only knows about preprints, so it is blind to venue-only
publications and to most pre-2010 literature. Semantic Scholar indexes across
venues, which is exactly the recall arXiv cannot give, and it reports DOIs and
citation counts that arXiv does not - both useful later for citations
(Milestone 6) and for weighting evidence.

The public API works without a key, but unauthenticated traffic shares one
small global rate-limit pool, so HTTP 429 is a normal response here rather than
an exceptional one. Every request therefore retries with backoff, and a source
that stays down surfaces as an error on the SourceResult instead of raising -
the run continues on whatever sources did answer.
"""

import logging
import time

import httpx

from research_agent.config import (
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_TIMEOUT,
)
from research_agent.retriever import normalize_arxiv_id
from research_agent.schemas import RetrievedPaper

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Only the fields we actually map onto RetrievedPaper - the API returns nothing
# beyond what is asked for, and a shorter field list is a cheaper request.
FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "publicationDate",
    "externalIds",
    "openAccessPdf",
    "citationCount",
    "url",
])

# Kept deliberately short: without an API key most 429s are not transient
# (the shared pool is genuinely saturated), so a long retry budget just makes
# the run slow before failing anyway. The circuit breaker in search.py stops
# the source entirely once it is clearly down.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
# Unauthenticated callers share ~1 request/second globally; pausing between our
# own queries keeps us from spending every attempt on self-inflicted 429s.
DELAY_BETWEEN_QUERIES = 0.0 if SEMANTIC_SCHOLAR_API_KEY else 1.5


def _headers() -> dict:
    return {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}


def _get_with_retry(client: httpx.Client, params: dict) -> dict:
    """GET the search endpoint, retrying rate limits and transient server errors.

    Raises the last error if every attempt fails - the caller decides whether
    that kills the run (it doesn't) or just this source (it does).
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        try:
            response = client.get(SEARCH_URL, params=params, headers=_headers())
        except httpx.HTTPError as exc:  # timeout, DNS, connection reset
            last_error = exc
            logger.warning("Semantic Scholar request failed (attempt %d): %s", attempt, exc)
        else:
            if response.status_code == 200:
                return response.json()

            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                request=response.request,
                response=response,
            )
            # 429 and 5xx are worth waiting out; a 400 never fixes itself.
            if response.status_code != 429 and response.status_code < 500:
                raise last_error

            header = response.headers.get("Retry-After")
            if header and header.isdigit():
                wait = max(wait, float(header))
            logger.warning(
                "Semantic Scholar returned %d (attempt %d/%d), retrying in %.1fs",
                response.status_code,
                attempt,
                MAX_ATTEMPTS,
                wait,
            )

        if attempt < MAX_ATTEMPTS:
            time.sleep(wait)

    raise last_error or RuntimeError("Semantic Scholar request failed")


def _to_paper(raw: dict) -> RetrievedPaper | None:
    """Map one API record onto RetrievedPaper. Returns None if it has no usable ID/title."""
    paper_id = raw.get("paperId")
    title = raw.get("title")
    if not paper_id or not title:
        return None

    external = raw.get("externalIds") or {}
    open_access = raw.get("openAccessPdf") or {}

    return RetrievedPaper(
        paper_id=paper_id,
        title=title,
        authors=[a.get("name", "") for a in (raw.get("authors") or []) if a.get("name")],
        # Semantic Scholar withholds abstracts for many closed-access records.
        # Keep the paper here and let the merge node count/drop it, so the drop
        # shows up in the stats instead of vanishing silently.
        abstract=raw.get("abstract") or "",
        published=raw.get("publicationDate"),
        pdf_url=open_access.get("url"),
        source="semantic_scholar",
        url=raw.get("url"),
        doi=external.get("DOI"),
        arxiv_id=normalize_arxiv_id(external.get("ArXiv")),
        year=raw.get("year"),
        citation_count=raw.get("citationCount"),
    )


def search_semantic_scholar(query: str, max_results: int = 5) -> list[RetrievedPaper]:
    """Search Semantic Scholar and return paper metadata.

    Mirrors `retriever.search_arxiv`'s signature so the two sources are
    interchangeable from the retrieval nodes' point of view.
    """
    params = {"query": query, "limit": max_results, "fields": FIELDS}

    with httpx.Client(timeout=SEMANTIC_SCHOLAR_TIMEOUT) as client:
        payload = _get_with_retry(client, params)

    papers = []
    for raw in payload.get("data") or []:
        paper = _to_paper(raw)
        if paper:
            papers.append(paper)
    return papers
