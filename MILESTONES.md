# Milestones — Autonomous Research Scientist Agent

Build in order. Each milestone is a working, demoable thing on its own —
not a stub waiting on the next phase. Don't start the next milestone
until the current one runs end-to-end.

Status legend: `[x]` done · `[ ]` not started

---

## [x] Milestone 0 — Skeleton

**Goal:** prove the scaffolding works before adding any intelligence.

- Project structure, env vars, shared LLM client, logging.
- Single LangGraph graph with one node: takes a topic string, returns it.

**Implemented in:** `research_agent/config.py`, `research_agent/state.py`, `research_agent/graph.py`

---

## [x] Milestone 1 — Planner Agent

**Goal:** turn a vague research topic into a structured plan.

- One agent decomposes the topic into objectives + sub-questions.
- Output is structured JSON (Pydantic model), not prose.
- Also generates expanded search queries for later retrieval.

**Implemented in:** `research_agent/schemas.py` (`ResearchPlan`), `research_agent/planner.py`

**Demo:** `python main.py "<topic>"` → structured plan printed as JSON.

---

## [x] Milestone 2 — Single-source retrieval + RAG

**Goal:** answer the planner's sub-questions with real, grounded evidence.

- Source: arXiv.
- Fetch papers → chunk abstracts → embed (OpenAI `text-embedding-3-small` via OpenRouter) →
  store in Supabase/pgvector (switched from the original Pinecone plan — reuses the
  same embed-then-similarity-search pattern, just a different vector store).
- A retrieval node answers each sub-question from the plan using RAG.

**Implemented in:** `research_agent/retriever.py` (arXiv search, chunking, embedding,
Supabase similarity search), `research_agent/rag.py` (retrieval node), `sql/001_paper_chunks.sql`
(pgvector table + `match_paper_chunks` function).

**Note:** only abstracts are embedded (no full-text PDF parsing) — enough for RAG
answers, but a real limitation for Milestone 3's extraction (see below).

**Demo:** planner → arXiv search → RAG-grounded answers per sub-question.

---

## [x] Milestone 3 — Structured paper extraction

**Goal:** make retrieval "smart" instead of just returning raw chunks.

- For each retrieved paper, extract: title, method, dataset, results, limitations into a
  structured schema (`PaperRecord`).
- Because only abstracts are available (see Milestone 2 note), fields the abstract
  doesn't state are returned as `null` rather than guessed — extraction is grounded
  in what's actually retrievable, not hallucinated to fill the schema.

**Implemented in:** `research_agent/schemas.py` (`PaperRecord`), `research_agent/extraction.py`

**Add to state:** `extracted_records`

**Demo:** planner → retrieval → each paper's abstract distilled into a structured record,
with a per-field missing-count logged (signal for how much the abstract-only limitation bites).

---

## [x] Milestone 4 — Second source + parallel retrieval

**Goal:** improve recall without over-engineering source count.

- Second source: **Semantic Scholar** (Graph API). Chosen over GitHub because arXiv
  only indexes preprints — it is blind to venue-only publications and to most older
  literature, which is exactly the gap Semantic Scholar fills. It also reports DOIs
  and citation counts arXiv doesn't, both of which Milestone 6 needs for citations.
- Retrieval nodes run **in parallel** via LangGraph fan-out/fan-in:

  ```
  planner ─┬─> arxiv ────────────────┬─> merge_sources ─> rag ─> extraction
           └─> semantic_scholar ─────┘
  ```

  Both source nodes sit in the same superstep, so two sources cost about as much
  wall-clock as the slower one instead of the sum (measured: 3.0s vs 6.0s
  sequential on a stubbed 3-query run).
- **Merge** deduplicates across sources on arXiv ID → DOI → normalised title, and
  folds duplicate records together rather than picking one: the longer abstract
  wins, and missing fields are back-filled, so an abstract-less Semantic Scholar
  record is rescued by the arXiv copy of the same paper.

**Implemented in:** `research_agent/semantic_scholar.py` (API client with retry/backoff),
`research_agent/search.py` (parallel source nodes, circuit breaker, dedup/merge),
`research_agent/graph.py` (fan-out/fan-in wiring), `sql/002_paper_source.sql`
(`source` column so chunks stay traceable across runs).

**Add to state:** `source_results` (with an `operator.add` reducer — without it
LangGraph treats two parallel writes to one key as a conflict and raises),
`merge_stats`.

**Two things this milestone forced that are worth keeping:**

1. **Sources fail independently.** The Semantic Scholar API is usable without a key
   but shares one small global rate-limit pool, so HTTP 429 is the *normal* response
   rather than an exceptional one. A failing source therefore reports an error on its
   `SourceResult` instead of raising — the run continues on whatever sources answered.
   A circuit breaker abandons a source after 2 consecutive failed queries so a dead
   source costs seconds, not the full retry budget on every query. Set
   `SEMANTIC_SCHOLAR_API_KEY` (free) to actually get results from this source.
2. **Recall has to be measurable, not assumed.** `MergeStats` reports per-source
   counts, unique papers after dedup, and how many papers *both* sources found. If
   overlap approaches the unique count, the second source is only re-finding what
   arXiv already had — that's the number that decides whether a third source is
   worth adding.

   Measured on `"memory poisoning LLM agents"` (5 results per source): arXiv 5 +
   Semantic Scholar 5 → **8 unique, 2 collapsed cross-source**. So the second
   source was worth +3 papers (+60% recall), and the overlap is low enough to
   justify it. The three S2-only papers are exactly the kind arXiv structurally
   cannot rank for: a venue publication with an IEEE DOI, and AgentPoison
   (465 citations) which arXiv's relevance ordering missed entirely. The two
   collapsed records kept arXiv's canonical ID while gaining the DOI and citation
   count only Semantic Scholar reports.

**Demo:** `uv run python scripts/test_sources.py` — offline dedup checks (no network,
no API keys) plus a live per-source recall comparison. Or `POST /search` with
`{"queries": [...]}` for the same comparison over HTTP at zero LLM cost.

**Note:** only add a third source once these two are solid — clean extraction matters
more than raw recall.

---

## [ ] Milestone 5 — Critic Agent

**Goal:** never trust the first answer.

- Second LLM pass reviews the draft for unsupported claims, weak citations, missing evidence, logical errors.
- Sends corrections back into the loop.
- Cheap to build, high signal in interviews — build before anything flashy.

**Add to state:** `critic_notes`

---

## [ ] Milestone 6 — Report Generator

**Goal:** produce a real, shareable artifact.

- Compile plan + retrieved evidence + critic-approved content into a structured report.
- Markdown first, DOCX export second.
- This is the first "wow, it produced something real" moment.

**Add to state:** `draft_report`, `final_report`

---

## [ ] Milestone 7 — Evaluation logging

**Goal:** demonstrate production engineering maturity.

- Log faithfulness/groundedness, citation accuracy, latency, token usage, cost — per run.
- No UI needed at first: a CSV/JSON log you can chart is enough to talk through in an interview.
- This is the "boring" feature that signals seniority; don't skip it.

---

### Milestones 0–7 are the target scope.

If 0–7 are solid, you already have a strong portfolio project **and** a
tool that helps your actual thesis lit review. Everything below is
optional — only touch it if you have time left over.

---

## [ ] Milestone 8 — Contradiction & gap detection *(stretch)*

- Compare structured records (from Milestone 3) across papers.
- Detect contradictions (Paper A says X performs best, Paper B says X performs poorly) and explain possible reasons.
- Surface research gaps from limitations + missing evaluations — this is where "propose a research direction" starts to be real instead of a gimmick.

## [ ] Milestone 9 — Minimal frontend *(stretch)*

- Thin Streamlit or Next.js dashboard showing agents working live.
- Build this last, once the backend logic is proven. Don't over-invest here for a portfolio piece.

---

## Future Work Backlog (park until 0–7 are done)

Not part of the target build. Revisit only if 0–7 are solid and you
still have time:

- PPT export
- Full knowledge graph
- Trend forecasting
- Multimodal figure/table/equation extraction
- Team collaboration (shared projects, comments, version history)
- Multi-format citation generator (APA/IEEE/MLA/BibTeX/Chicago/Harvard — pick one format for now)
- Novelty estimator / similarity scoring against thousands of papers
- Paper recommendation engine
- Automated periodic literature review updates

---

## How to extend to the next milestone

Each new milestone should:

1. Add new fields to `GraphState` in `research_agent/state.py`.
2. Add a new node function (following the pattern in `research_agent/planner.py`).
3. Wire it into `research_agent/graph.py` after the existing nodes.

This keeps every milestone additive — nothing already working has to be
rewritten to add the next piece.
