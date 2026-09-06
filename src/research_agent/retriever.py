"""Milestone 2: arXiv retrieval, chunking, embedding, and vector storage."""

import logging
import re

import arxiv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

from research_agent.config import (
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    get_embeddings,
)
from research_agent.schemas import RetrievedPaper

logger = logging.getLogger(__name__)

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# "2401.12345v3" and "2401.12345" are the same paper. arXiv hands us the
# versioned form, Semantic Scholar the bare one - normalising here is what lets
# Milestone 4's merge recognise them as one paper.
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_arxiv_id(raw: str | None) -> str | None:
    """Strip an arXiv ID down to its version-less form. None-safe."""
    if not raw:
        return None
    return _ARXIV_VERSION_SUFFIX.sub("", raw.strip())


def get_supabase():
    """Return Supabase client."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def search_arxiv(query: str, max_results: int = 5) -> list[RetrievedPaper]:
    """Search arXiv and return paper metadata."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    papers = []
    for result in client.results(search):
        raw_id = result.entry_id.split("/")[-1]
        papers.append(
            RetrievedPaper(
                paper_id=raw_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.isoformat() if result.published else None,
                pdf_url=result.pdf_url,
                source="arxiv",
                url=result.entry_id,
                doi=result.doi,
                arxiv_id=normalize_arxiv_id(raw_id),
                year=result.published.year if result.published else None,
            )
        )
    return papers


def chunk_text(text: str) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _insert_chunks(supabase, rows: list[dict]) -> None:
    """Insert chunk rows, tolerating a schema that predates the `source` column.

    `source` arrived with Milestone 4 (sql/002_paper_source.sql). Retrying
    without it means an un-migrated database degrades to Milestone 2 behaviour
    instead of failing the whole run.
    """
    try:
        supabase.table("paper_chunks").insert(rows).execute()
    except Exception as exc:
        if "source" not in str(exc):
            raise
        logger.warning(
            "paper_chunks has no 'source' column - run sql/002_paper_source.sql. "
            "Storing chunks without source for now."
        )
        supabase.table("paper_chunks").insert(
            [{k: v for k, v in row.items() if k != "source"} for row in rows]
        ).execute()


def embed_and_store(papers: list[RetrievedPaper]) -> int:
    """Chunk paper abstracts, embed, and store in Supabase. Returns count stored."""
    supabase = get_supabase()
    embeddings = get_embeddings()
    stored = 0

    for paper in papers:
        # Nothing to embed - Semantic Scholar withholds abstracts on many
        # closed-access records, so this is a real case, not a defensive one.
        if not paper.abstract or not paper.abstract.strip():
            continue

        # Check if already stored
        existing = supabase.table("paper_chunks").select("id").eq("paper_id", paper.paper_id).limit(1).execute()
        if existing.data:
            continue

        chunks = chunk_text(paper.abstract)
        if not chunks:
            continue

        vectors = embeddings.embed_documents(chunks)

        rows = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "chunk_index": i,
                "content": chunk,
                "embedding": vec,
                "source": paper.source,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        _insert_chunks(supabase, rows)
        stored += len(rows)

    return stored


def similarity_search(query: str, top_k: int = 5) -> list[dict]:
    """Search for similar chunks using pgvector."""
    supabase = get_supabase()
    embeddings = get_embeddings()

    query_vec = embeddings.embed_query(query)

    result = supabase.rpc(
        "match_paper_chunks",
        {"query_embedding": query_vec, "match_count": top_k, "match_threshold": 0.5},
    ).execute()

    return result.data or []
