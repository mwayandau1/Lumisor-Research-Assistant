"""Test script for Milestone 4: parallel multi-source retrieval and merge.

Runs the source fan-out and the merge in isolation - no planner LLM call, no
embedding, no Supabase - so you can see what the second source actually buys
without paying for the rest of the pipeline.

Two checks:

1. An offline dedup check on hand-built records. It covers the three cases the
   merge has to get right - matched on arXiv ID, matched on title alone, and a
   Semantic Scholar record whose missing abstract is rescued by the arXiv copy -
   and needs no network at all.
2. A live search across every enabled source for a real query, printing the
   per-source counts and the overlap.

Usage:
    uv run python scripts/test_sources.py                 # offline check + default query
    uv run python scripts/test_sources.py "your topic"    # offline check + your query
    uv run python scripts/test_sources.py --offline       # offline check only, no network
"""

import sys

from research_agent.config import ENABLED_SOURCES, MAX_SEARCH_QUERIES, RESULTS_PER_QUERY
from research_agent.schemas import ResearchPlan, RetrievedPaper, SourceResult
from research_agent.search import enabled_source_nodes, merge_source_results

DEFAULT_QUERY = "memory poisoning attacks on LLM agents"

# The same paper as arXiv and Semantic Scholar each report it: different IDs,
# different metadata, and S2 withholding the abstract. The merge should collapse
# these into one record that keeps arXiv's abstract and gains S2's DOI/citations.
SAME_PAPER_ARXIV = RetrievedPaper(
    paper_id="2401.11111v2",
    title="Poisoning the Well: Long-Term Memory Attacks on LLM Agents",
    authors=["A. Researcher"],
    abstract="We show that an attacker can corrupt an agent's persistent memory store.",
    source="arxiv",
    arxiv_id="2401.11111",
)
SAME_PAPER_S2 = RetrievedPaper(
    paper_id="s2-aaaa1111",
    title="Poisoning the Well: Long-term Memory Attacks on LLM Agents!",  # punctuation/case differ
    authors=["A. Researcher"],
    abstract="",  # S2 withholds abstracts on many records
    source="semantic_scholar",
    arxiv_id="2401.11111",
    doi="10.1000/poison.2024",
    citation_count=42,
    year=2024,
)
# Matched on normalised title alone - neither side reports an arXiv ID or DOI.
TITLE_ONLY_ARXIV = RetrievedPaper(
    paper_id="2402.22222v1",
    title="Detecting Adversarial Memories",
    authors=["B. Researcher"],
    abstract="A detector for adversarial entries in agent memory.",
    source="arxiv",
    arxiv_id="2402.22222",
)
TITLE_ONLY_S2 = RetrievedPaper(
    paper_id="s2-bbbb2222",
    title="detecting adversarial memories",
    authors=["B. Researcher"],
    abstract="A detector for adversarial entries in agent memory, evaluated on 500 conversations.",
    source="semantic_scholar",
    citation_count=7,
)
# Only Semantic Scholar has this one - the recall arXiv cannot give.
S2_ONLY = RetrievedPaper(
    paper_id="s2-cccc3333",
    title="A Venue-Only Study of Agent Memory Integrity",
    authors=["C. Researcher"],
    abstract="A journal-only study never posted as a preprint.",
    source="semantic_scholar",
    doi="10.1000/venue.2023",
)
# No abstract from either source - unusable downstream, so the merge drops it.
NO_ABSTRACT = RetrievedPaper(
    paper_id="s2-dddd4444",
    title="Closed Access Paper With No Abstract",
    authors=[],
    abstract="",
    source="semantic_scholar",
)


def check_offline_merge() -> bool:
    """Merge hand-built records and assert the dedup did what it claims."""
    print("=" * 60)
    print("1. OFFLINE MERGE CHECK (no network)")
    print("=" * 60)

    results = [
        SourceResult(source="arxiv", papers=[SAME_PAPER_ARXIV, TITLE_ONLY_ARXIV], queries_run=1),
        SourceResult(
            source="semantic_scholar",
            papers=[SAME_PAPER_S2, TITLE_ONLY_S2, S2_ONLY, NO_ABSTRACT],
            queries_run=1,
        ),
    ]

    papers, stats = merge_source_results(results)
    by_title = {p.title: p for p in papers}

    print(f"  per source:           {stats.per_source}")
    print(f"  total before dedup:   {stats.total_before_dedup}")
    print(f"  unique after merge:   {stats.unique_papers}")
    print(f"  found by >1 source:   {stats.overlap_papers}")
    print(f"  dropped (no abstract):{stats.dropped_no_abstract}")
    print()
    for p in papers:
        print(f"  [{p.paper_id}] ({'+'.join(p.found_by)}) {p.title}")
        print(f"      doi={p.doi} citations={p.citation_count} abstract={len(p.abstract)} chars")

    merged_same = by_title.get(SAME_PAPER_ARXIV.title)
    merged_title_only = by_title.get(TITLE_ONLY_ARXIV.title)

    checks = [
        ("6 records collapse to 3 usable", stats.unique_papers == 3),
        ("2 papers found by both sources", stats.overlap_papers == 2),
        ("abstract-less record dropped", stats.dropped_no_abstract == 1),
        ("arXiv-ID match merged", merged_same is not None),
        (
            "merged record keeps the arXiv abstract",
            merged_same is not None and merged_same.abstract == SAME_PAPER_ARXIV.abstract,
        ),
        (
            "merged record gains the S2 DOI and citations",
            merged_same is not None
            and merged_same.doi == SAME_PAPER_S2.doi
            and merged_same.citation_count == 42,
        ),
        (
            "title-only match merged, longer abstract wins",
            merged_title_only is not None
            and merged_title_only.abstract == TITLE_ONLY_S2.abstract,
        ),
        ("Semantic-Scholar-only paper survives", S2_ONLY.title in by_title),
    ]

    print()
    ok = True
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        ok = ok and passed
    print()
    return ok


def check_live_search(query: str) -> None:
    """Run every enabled source against one real query and merge the results."""
    print("=" * 60)
    print("2. LIVE SOURCE SEARCH")
    print("=" * 60)
    print(f"Query:   {query}")
    print(f"Sources: {ENABLED_SOURCES} (max {MAX_SEARCH_QUERIES} queries, {RESULTS_PER_QUERY} results each)\n")

    # The source nodes read nothing but `search_queries`, so a one-query plan is
    # all that's needed to exercise them without a planner call.
    plan = ResearchPlan(
        topic=query,
        summary="Ad-hoc single-query source comparison.",
        objectives=[],
        search_queries=[query],
    )

    state = {"plan": plan}
    source_results = []
    for name, node in enabled_source_nodes().items():
        source_results.extend(node(state).get("source_results", []))

    papers, stats = merge_source_results(source_results)

    for source, count in stats.per_source.items():
        print(f"  {source:<22} {count} papers")
    for source, error in stats.errors.items():
        print(f"  !! {source} returned nothing: {error}")

    print(f"\n  total before dedup:    {stats.total_before_dedup}")
    print(f"  unique after merge:    {stats.unique_papers}")
    print(f"  found by >1 source:    {stats.overlap_papers}")
    print(f"  dropped (no abstract): {stats.dropped_no_abstract}\n")

    for p in papers:
        print(f"  [{p.paper_id}] ({'+'.join(p.found_by)}) {p.title[:70]}")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--offline"]
    offline_only = "--offline" in sys.argv

    passed = check_offline_merge()

    if not offline_only:
        check_live_search(args[0] if args else DEFAULT_QUERY)

    print("=" * 60)
    print("Merge checks passed." if passed else "Merge checks FAILED.")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
