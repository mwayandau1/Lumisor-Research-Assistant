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

    # Milestone 2: Retrieved papers and RAG answers
    if result.get("retrieved_papers"):
        print("\n" + "=" * 60)
        print(f"RETRIEVED PAPERS ({len(result['retrieved_papers'])})")
        print("=" * 60)
        for p in result["retrieved_papers"]:
            print(f"  [{p.paper_id}] {p.title}")

    if result.get("rag_answers"):
        print("\n" + "=" * 60)
        print("RAG ANSWERS")
        print("=" * 60)
        for ans in result["rag_answers"]:
            print(f"\nQ: {ans.question}")
            print(f"A: {ans.answer}")
            if ans.sources:
                print(f"   Sources: {', '.join(ans.sources)}")


if __name__ == "__main__":
    main()
