"""Planner Agent: decomposes a research topic into a structured plan."""

import logging

from research_agent.config import STRUCTURED_TEMPERATURE, STRUCTURED_MODEL, get_llm
from research_agent.schemas import ResearchPlan
from research_agent.state import GraphState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are the Research Planning Agent inside an autonomous research assistant.

Given a research topic, decompose it into a clear, structured research plan.

Guidelines:
- Produce 4-8 objectives that together give thorough coverage of the topic.
- Each objective needs 2-4 concrete sub-questions.
- Produce 8-15 expanded search queries for academic search engines.
- Be specific to the given topic. Do not pad with generic filler.
"""


def run_planner(state: GraphState) -> GraphState:
    """LangGraph node: reads state['topic'], writes state['plan']."""
    logger.info("Planning research on topic: %s", state["topic"])

    llm = get_llm(model=STRUCTURED_MODEL, temperature=STRUCTURED_TEMPERATURE)
    structured_llm = llm.with_structured_output(ResearchPlan)

    plan = structured_llm.invoke([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", f"Research topic: {state['topic']}"),
    ])

    logger.info(
        "Plan ready: %d objectives, %d search queries",
        len(plan.objectives),
        len(plan.search_queries),
    )
    return {**state, "plan": plan}
