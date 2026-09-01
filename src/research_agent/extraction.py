"""Milestone 3: Structured paper extraction.

Turns each retrieved paper's abstract into a structured PaperRecord
(method, dataset, results, limitations) instead of leaving evidence as
raw chunks. This is the foundation Milestone 8 (contradiction/gap
detection) will compare across papers.
"""

import logging

from research_agent.config import STRUCTURED_MODEL, STRUCTURED_TEMPERATURE, get_llm
from research_agent.schemas import PaperRecord
from research_agent.state import GraphState

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured facts from a research paper's abstract.

Rules:
- Only use information explicitly present in the abstract. Never infer or guess.
- If the abstract does not mention a field (e.g. it never states a dataset,
  or never acknowledges a limitation), return null for that field. Do not
  invent a plausible-sounding answer.
- Keep each field to 1-2 sentences, factual, no filler.
"""


def run_extraction(state: GraphState) -> GraphState:
    """LangGraph node: reads state['retrieved_papers'], writes state['extracted_records']."""
    papers = state.get("retrieved_papers") or []
    if not papers:
        return state

    logger.info("Extracting structured records from %d papers", len(papers))

    llm = get_llm(model=STRUCTURED_MODEL, temperature=STRUCTURED_TEMPERATURE)
    structured_llm = llm.with_structured_output(PaperRecord)

    records: list[PaperRecord] = []
    missing_counts = {"method": 0, "dataset": 0, "results": 0, "limitations": 0}

    for paper in papers:
        record = structured_llm.invoke([
            ("system", EXTRACTION_SYSTEM_PROMPT),
            (
                "human",
                f"paper_id: {paper.paper_id}\n"
                f"title: {paper.title}\n"
                f"abstract: {paper.abstract}",
            ),
        ])
        # The model sometimes echoes its own paper_id/title inconsistently -
        # keep the ones we already trust from arXiv metadata.
        record.paper_id = paper.paper_id
        record.title = paper.title

        for field in missing_counts:
            if getattr(record, field) is None:
                missing_counts[field] += 1

        records.append(record)

    logger.info(
        "Extraction complete: %d records. Missing per field (of %d): %s",
        len(records),
        len(papers),
        missing_counts,
    )

    return {**state, "extracted_records": records}
