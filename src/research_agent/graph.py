"""LangGraph wiring: builds and runs the research agent graph."""

from langgraph.graph import END, StateGraph

from research_agent.planner import run_planner
from research_agent.rag import run_retrieval
from research_agent.state import GraphState


def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    graph.add_node("planner", run_planner)
    graph.add_node("retrieval", run_retrieval)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", END)

    return graph.compile()


def run(topic: str) -> GraphState:
    """Run the compiled graph on a single topic."""
    return build_graph().invoke({"topic": topic})
