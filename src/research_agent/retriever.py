"""Milestone 2: arXiv retrieval, chunking, embedding, and vector storage."""

import arxiv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

from research_agent.config import (
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    get_embeddings,
)
from research_agent.schemas import RetrievedPaper

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


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
        papers.append(
            RetrievedPaper(
                paper_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                published=result.published.isoformat() if result.published else None,
                pdf_url=result.pdf_url,
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


def embed_and_store(papers: list[RetrievedPaper]) -> int:
    """Chunk paper abstracts, embed, and store in Supabase. Returns count stored."""
    supabase = get_supabase()
    embeddings = get_embeddings()
    stored = 0

    for paper in papers:
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
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        supabase.table("paper_chunks").insert(rows).execute()
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
