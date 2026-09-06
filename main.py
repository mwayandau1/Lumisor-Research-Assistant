"""CLI entry point for the Autonomous Research Scientist Agent."""

import json
import sys

from research_agent.graph import run


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "<research topic>"')
        sys.exit(1)

    topic = sys.argv[1]
    print(f"\n🔬 Running research agent on topic:\n   {topic}\n")

    result = run(topic)

    # Milestone 1: Plan
    print("=" * 60)
    print("RESEARCH PLAN")
    print("=" * 60)
    print(json.dumps(result["plan"].model_dump(), indent=2))

    # Milestone 4: what each source contributed, and what the merge saved
    stats = result.get("merge_stats")
    if stats:
        print("\n" + "=" * 60)
        print("SOURCE MERGE")
        print("=" * 60)
        for source, count in stats.per_source.items():
            print(f"  {source:<22} {count} papers")
        print(f"  {'-' * 40}")
        print(f"  {'total before dedup':<22} {stats.total_before_dedup}")
        print(f"  {'unique after merge':<22} {stats.unique_papers}")
        print(f"  {'found by >1 source':<22} {stats.overlap_papers}")
        if stats.dropped_no_abstract:
            print(f"  {'dropped (no abstract)':<22} {stats.dropped_no_abstract}")
        for source, error in stats.errors.items():
            print(f"  !! {source} returned nothing: {error}")

    # Milestone 2: Retrieved papers and RAG answers
    if result.get("retrieved_papers"):
        print("\n" + "=" * 60)
        print(f"RETRIEVED PAPERS ({len(result['retrieved_papers'])})")
        print("=" * 60)
        for p in result["retrieved_papers"]:
            found_by = "+".join(p.found_by) if p.found_by else p.source
            print(f"  [{p.paper_id}] ({found_by}) {p.title}")

    if result.get("rag_answers"):
        print("\n" + "=" * 60)
        print("RAG ANSWERS")
        print("=" * 60)
        for ans in result["rag_answers"]:
            print(f"\nQ: {ans.question}")
            print(f"A: {ans.answer}")
            if ans.sources:
                print(f"   Sources: {', '.join(ans.sources)}")

    # Milestone 3: Extracted structured records
    if result.get("extracted_records"):
        print("\n" + "=" * 60)
        print(f"EXTRACTED RECORDS ({len(result['extracted_records'])})")
        print("=" * 60)
        for rec in result["extracted_records"]:
            print(json.dumps(rec.model_dump(), indent=2))


if __name__ == "__main__":
    main()
