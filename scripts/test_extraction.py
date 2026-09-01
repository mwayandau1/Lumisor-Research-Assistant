"""Test script for Milestone 3: structured paper extraction.

Runs extraction in isolation - no arXiv search, no embedding, no Supabase -
so you can iterate on the extraction prompt/schema without paying for (or
waiting on) the rest of the pipeline.

Usage:
    uv run python scripts/test_extraction.py                # two built-in sample abstracts
    uv run python scripts/test_extraction.py "your topic"   # real arXiv papers on that topic
"""

import sys

from research_agent.extraction import run_extraction
from research_agent.retriever import search_arxiv
from research_agent.schemas import RetrievedPaper

# Two hand-picked cases:
# - a "rich" abstract that states method, dataset, and a result
# - a "sparse" abstract that states a method but nothing else -
#   extraction should return null for dataset/results/limitations here,
#   not invent them.
SAMPLE_PAPERS = [
    RetrievedPaper(
        paper_id="sample-rich-0001",
        title="Retrieval-Augmented Poisoning Detection for LLM Agent Memory",
        authors=["A. Researcher", "B. Researcher"],
        abstract=(
            "We propose a retrieval-augmented detector that flags poisoned entries "
            "in an LLM agent's long-term memory before they are used for reasoning. "
            "Our method embeds each memory entry and compares it against a trusted "
            "reference set using cosine similarity, flagging outliers above a "
            "learned threshold. We evaluate on a synthetic benchmark of 500 agent "
            "conversations injected with adversarial memories and show a 92% "
            "detection rate at a 3% false positive rate, outperforming a prior "
            "perplexity-based baseline by 18 points. We do not evaluate against "
            "adaptive attackers who are aware of the detector, and our threshold "
            "is tuned per-domain rather than learned automatically."
        ),
        published="2026-01-01",
        pdf_url=None,
    ),
    RetrievedPaper(
        paper_id="sample-sparse-0002",
        title="Notes on Memory Poisoning in Autonomous Agents",
        authors=["C. Researcher"],
        abstract=(
            "This position paper argues that memory poisoning is an underexplored "
            "attack surface for autonomous LLM agents with persistent memory. We "
            "outline a taxonomy of attack vectors, including direct injection, "
            "indirect injection via tool outputs, and cross-session contamination, "
            "and call for standardized benchmarks to evaluate defenses."
        ),
        published="2026-01-02",
        pdf_url=None,
    ),
]


def main() -> None:
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Fetching real arXiv papers for: {query}\n")
        papers = search_arxiv(query, max_results=3)
    else:
        print("No query given - using built-in sample abstracts (1 rich, 1 sparse).\n")
        papers = SAMPLE_PAPERS

    if not papers:
        print("No papers to extract from.")
        return

    result = run_extraction({"retrieved_papers": papers})

    print("=" * 60)
    print(f"EXTRACTED RECORDS ({len(result.get('extracted_records', []))})")
    print("=" * 60)
    for rec in result.get("extracted_records", []):
        print(f"\n[{rec.paper_id}] {rec.title}")
        print(f"  method:       {rec.method}")
        print(f"  dataset:      {rec.dataset}")
        print(f"  results:      {rec.results}")
        print(f"  limitations:  {rec.limitations}")


if __name__ == "__main__":
    main()
