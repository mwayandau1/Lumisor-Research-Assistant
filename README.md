# Autonomous Research Scientist Agent

An agentic research assistant built with LangGraph + LangChain. Given a
research topic, it plans, retrieves, extracts, critiques, and reports —
gradually, milestone by milestone, rather than all at once.

This repo currently implements **Milestone 0 (skeleton)**, **Milestone 1
(Planner Agent)**, **Milestone 2 (retrieval + RAG)**, and **Milestone 3
(structured paper extraction)**.

## Stack

- **Orchestration:** LangGraph (`planner -> retrieval -> extraction`)
- **LLM access:** OpenRouter (`langchain-openai`'s `ChatOpenAI` pointed at OpenRouter's
  base URL) — one API key covers many underlying models.
  - Plain-prose calls (RAG answer generation) use a free model by default.
  - Structured-output calls (planner JSON, Milestone 3 extraction) use a stronger
    model, since strict schema/tool-calling adherence is where small free models
    tend to fail.
- **Embeddings:** OpenAI `text-embedding-3-small`, also routed through OpenRouter.
- **Vector store:** Supabase Postgres + `pgvector` (cosine similarity via a SQL function).
- **Source:** arXiv (`arxiv` package).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in the real keys below
```

Fill in `.env`:

- `OPENROUTER_API_KEY` — used for both the LLM calls and the embeddings.
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — your Supabase project.

Then, in the Supabase SQL editor, run [`sql/001_paper_chunks.sql`](sql/001_paper_chunks.sql)
once to create the `paper_chunks` table, its pgvector index, and the
`match_paper_chunks` similarity-search function.

## Run

```bash
python main.py "Memory poisoning attacks against long-term memory in autonomous LLM agents"
```

This runs the full graph and prints, in order:

1. **Research plan** — objectives, sub-questions, expanded search queries.
2. **Retrieved papers** — arXiv results pulled from the plan's search queries.
3. **RAG answers** — each sub-question answered from embedded abstract chunks.
4. **Extracted records** — each paper distilled into `{method, dataset, results,
   limitations}`, with `null` for anything the abstract doesn't actually state.

## Testing via Postman / HTTP

Each graph node is also reachable over HTTP for manual testing, isolating
one stage at a time - the same idea as `scripts/test_retrieval.py` and
`scripts/test_extraction.py`, just reachable from Postman instead of the CLI.

Start the server:

```bash
uv run uvicorn research_agent.api:app --reload --port 8000
```

Then either:
- Import [`postman/research_agent.postman_collection.json`](postman/research_agent.postman_collection.json)
  into Postman (`base_url` variable defaults to `http://127.0.0.1:8000`), or
- Browse the auto-generated Swagger UI at `http://127.0.0.1:8000/docs` and try
  requests directly from the browser without Postman at all.

Endpoints, cheapest first:

| Endpoint | Cost | What it tests |
|---|---|---|
| `GET /health` | free | server is up |
| `POST /plan` | 1 LLM call | Milestone 1 alone: `{"topic": "..."}` -> `ResearchPlan` |
| `POST /extract` | 1 LLM call per paper | Milestone 3 alone: pass `{"papers": [...]}` directly (see the collection for a ready-made rich/sparse pair) to test the extraction prompt without arXiv/embedding cost |
| `POST /retrieve` | 1 plan + N arXiv/embedding calls | Milestones 1+2: `{"topic": "..."}` -> plan + retrieved papers + RAG answers |
| `POST /extract` (with `topic` instead of `papers`) | full M1-M3 cost | runs plan -> retrieve -> extract |
| `POST /research` | full M1-M3 cost | the whole pipeline, same as `python main.py "<topic>"` but as JSON |

## Project layout

```
research_agent/
  src/research_agent/
    __init__.py
    config.py       # env vars, logging setup, shared LLM/embedding clients
    schemas.py       # Pydantic models (ResearchPlan, RetrievedPaper, RAGAnswer, PaperRecord)
    state.py         # shared LangGraph state (GraphState)
    planner.py       # Milestone 1: Planner Agent node
    retriever.py      # Milestone 2: arXiv search, chunking, embedding, Supabase similarity search
    rag.py            # Milestone 2: retrieval node (RAG answers per sub-question)
    extraction.py      # Milestone 3: structured paper extraction node
    graph.py           # LangGraph wiring
    api.py             # FastAPI app exposing each node for Postman/HTTP testing
  scripts/
    test_retrieval.py   # isolated Milestone 2 test (arXiv/chunk/embed/search)
    test_extraction.py  # isolated Milestone 3 test (sample rich/sparse abstracts)
  postman/
    research_agent.postman_collection.json  # importable Postman collection
  sql/
    001_paper_chunks.sql  # pgvector table + similarity-search function
  main.py            # CLI entry point
  requirements.txt
  .env.example
```

## Milestone roadmap

- [x] **0 — Skeleton**: project structure, config, one-node graph that runs end to end
- [x] **1 — Planner Agent**: topic -> structured `ResearchPlan` (objectives, sub-questions, search queries)
- [x] **2 — Single-source retrieval + RAG**: arXiv search -> chunk -> embed -> Supabase/pgvector -> RAG answers per sub-question
- [x] **3 — Structured paper extraction**: each retrieved paper -> structured `PaperRecord` (method, dataset, results, limitations)
- [ ] **4 — Second source + parallel retrieval**: add Semantic Scholar or GitHub, run retrieval nodes in parallel
- [ ] **5 — Critic Agent**: second LLM pass flags unsupported claims / weak citations, sends corrections back
- [ ] **6 — Report Generator**: compile plan + evidence + critic-approved content into a Markdown (then DOCX) report
- [ ] **7 — Evaluation logging**: log faithfulness, groundedness, latency, tokens, cost per run
- [ ] **8 — Contradiction & gap detection**: compare structured records across papers
- [ ] **9 — Minimal frontend**: thin dashboard showing agents working live

Deliberately parked for later (only after 0-9 are solid): PPT export,
full knowledge graph, trend forecasting, multimodal figure extraction,
team collaboration, multi-format citation export.

See [`MILESTONES.md`](MILESTONES.md) for the full per-milestone detail.

## Extending to the next milestone

Each new milestone should:
1. Add new fields to `GraphState` in `state.py`.
2. Add a new node function (like `run_planner` in `planner.py`).
3. Wire it into `graph.py` after the existing nodes.

This keeps every milestone additive — nothing already working has to be
rewritten to add the next piece.
