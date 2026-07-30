# Autonomous Research Scientist Agent

An agentic research assistant built with LangGraph + LangChain. Given a
research topic, it plans, retrieves, extracts, critiques, and reports —
gradually, milestone by milestone, rather than all at once.

This repo currently implements **Milestone 0 (skeleton)** and
**Milestone 1 (Planner Agent)**.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in ANTHROPIC_API_KEY
```

## Run

```bash
python main.py "Memory poisoning attacks against long-term memory in autonomous LLM agents"
```

This prints a structured JSON `ResearchPlan`: objectives, sub-questions,
and expanded search queries — the planner's decomposition of your topic.

## Project layout

```
research_agent/
  research_agent/
    __init__.py
    config.py     # env vars + shared LLM client
    schemas.py     # Pydantic models (ResearchPlan, etc.)
    state.py       # shared LangGraph state (GraphState)
    planner.py     # Milestone 1: Planner Agent node
    graph.py       # LangGraph wiring
  main.py          # CLI entry point
  requirements.txt
  .env.example
```

## Milestone roadmap

- [x] **0 — Skeleton**: project structure, config, one-node graph that runs end to end
- [x] **1 — Planner Agent**: topic -> structured `ResearchPlan` (objectives, sub-questions, search queries)
- [ ] **2 — Single-source retrieval + RAG**: arXiv search -> chunk -> embed -> Pinecone -> RAG answers per sub-question
- [ ] **3 — Structured paper extraction**: turn each retrieved paper into a structured record (method, dataset, results, limitations)
- [ ] **4 — Second source + parallel retrieval**: add Semantic Scholar or GitHub, run retrieval nodes in parallel
- [ ] **5 — Critic Agent**: second LLM pass flags unsupported claims / weak citations, sends corrections back
- [ ] **6 — Report Generator**: compile plan + evidence + critic-approved content into a Markdown (then DOCX) report
- [ ] **7 — Evaluation logging**: log faithfulness, groundedness, latency, tokens, cost per run
- [ ] **8 — Contradiction & gap detection**: compare structured records across papers
- [ ] **9 — Minimal frontend**: thin dashboard showing agents working live

Deliberately parked for later (only after 0-9 are solid): PPT export,
full knowledge graph, trend forecasting, multimodal figure extraction,
team collaboration, multi-format citation export.

## Extending to the next milestone

Each new milestone should:
1. Add new fields to `GraphState` in `state.py`.
2. Add a new node function (like `run_planner` in `planner.py`).
3. Wire it into `graph.py` after the existing nodes.

This keeps every milestone additive — nothing already working has to be
rewritten to add the next piece.
