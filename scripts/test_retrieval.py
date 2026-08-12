"""Test script for Milestone 2: arXiv search, chunking, embedding, and retrieval."""

import sys

from research_agent.retriever import (
    chunk_text,
    embed_and_store,
    search_arxiv,
    similarity_search,
)


def test_arxiv_search(query: str = "LLM memory poisoning attacks", max_results: int = 3):
    """Test arXiv paper search."""
    print("=" * 60)
    print("1. ARXIV SEARCH")
    print("=" * 60)
    print(f"Query: {query}\n")

    papers = search_arxiv(query, max_results=max_results)
    print(f"Found {len(papers)} papers:\n")

    for p in papers:
        print(f"  [{p.paper_id}] {p.title}")
        print(f"    Authors: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"    Abstract: {p.abstract[:150]}...")
        print()

    return papers


def test_chunking(papers):
    """Test text chunking."""
    print("=" * 60)
    print("2. CHUNKING")
    print("=" * 60)

    if not papers:
        print("No papers to chunk.")
        return

    paper = papers[0]
    chunks = chunk_text(paper.abstract)
    print(f"Paper: {paper.title}")
    print(f"Abstract length: {len(paper.abstract)} chars")
    print(f"Chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
    print()


def test_embed_and_store(papers):
    """Test embedding and storing in Supabase."""
    print("=" * 60)
    print("3. EMBED & STORE")
    print("=" * 60)

    if not papers:
        print("No papers to embed.")
        return

    stored = embed_and_store(papers)
    print(f"Stored {stored} new chunks in Supabase.")
    print("(0 means papers were already stored)\n")


def test_similarity_search(query: str = "How do attackers poison LLM memory?", top_k: int = 5):
    """Test similarity search."""
    print("=" * 60)
    print("4. SIMILARITY SEARCH")
    print("=" * 60)
    print(f"Query: {query}\n")

    results = similarity_search(query, top_k=top_k)

    if not results:
        print("No results found. Make sure papers are stored first.")
        return

    print(f"Found {len(results)} relevant chunks:\n")
    for r in results:
        print(f"  [{r['paper_id']}] {r['title']}")
        print(f"    Similarity: {r['similarity']:.3f}")
        print(f"    Content: {r['content'][:150]}...")
        print()


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "LLM memory poisoning attacks"

    print(f"\n🧪 Testing Milestone 2 Retrieval Pipeline\n")

    # 1. Search arXiv
    papers = test_arxiv_search(query)

    # 2. Test chunking
    test_chunking(papers)

    # 3. Embed and store
    test_embed_and_store(papers)

    # 4. Similarity search
    test_similarity_search(f"What are the main techniques for {query}?")

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
