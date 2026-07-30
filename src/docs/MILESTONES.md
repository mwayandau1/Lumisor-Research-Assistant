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

## [ ] Milestone 2 — Single-source retrieval + RAG

**Goal:** answer the planner's sub-questions with real, grounded evidence.

- Pick ONE source first — arXiv (matches your thesis domain).
- Fetch papers → chunk → embed → store in Pinecone (reuse the pattern from the TA project).
- Add a retrieval node that answers each sub-question from the plan using RAG.

**Add to state:** `retrieved_papers`, `rag_answers`

**Demo:** planner → arXiv search → RAG-grounded answers per sub-question.

---

## [ ] Milestone 3 — Structured paper extraction

**Goal:** make retrieval "smart" instead of just returning raw chunks.

- For each retrieved paper, extract: title, method, dataset, results, limitations into a structured schema (e.g. `PaperRecord`).
- Build this now while the dataset is small — it's the foundation for comparison and contradiction detection later.

**Add to state:** `extracted_records`

---

## [ ] Milestone 4 — Second source + parallel retrieval

**Goal:** improve recall without over-engineering source count.

- Add Semantic Scholar or GitHub as a second source.
- Run retrieval agents in parallel via LangGraph, merge results.
- Only add a third source later if the first two are solid — clean extraction matters more than raw recall.

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
