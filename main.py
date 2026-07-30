"""CLI entry point for the Autonomous Research Scientist Agent."""

import json
import sys

from research_agent.graph import run


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "<research topic>"')
        sys.exit(1)

    topic = sys.argv[1]
    print(f"\n Running planner on topic:\n  {topic}\n")

    result = run(topic)
    print(json.dumps(result["plan"].model_dump(), indent=2))


if __name__ == "__main__":
    main()
