"""Planner Agent: decomposes a research topic into a structured plan."""

from research_agent.config import get_llm
from research_agent.schemas import ResearchPlan
from research_agent.state import GraphState

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
    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchPlan)

    plan = structured_llm.invoke([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", f"Research topic: {state['topic']}"),
    ])

    return {**state, "plan": plan}
