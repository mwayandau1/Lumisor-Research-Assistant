"""LangGraph wiring: builds and runs the research agent graph.

    planner -+-> arxiv_search -------------+-> merge_sources -> rag -> extraction -> END
             +-> semantic_scholar_search --+

The two source nodes sit in the same superstep, so LangGraph executes them
concurrently and `merge_sources` only fires once both have reported.
"""

from langgraph.graph import END, StateGraph

from research_agent.extraction import run_extraction
from research_agent.planner import run_planner
from research_agent.rag import run_rag
from research_agent.search import enabled_source_nodes, run_merge_sources
from research_agent.state import GraphState


def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    graph.add_node("planner", run_planner)
    graph.add_node("merge_sources", run_merge_sources)
    graph.add_node("rag", run_rag)
    graph.add_node("extraction", run_extraction)

    graph.set_entry_point("planner")

    # Fan out to every enabled source, then fan back in. Building this from the
    # registry (rather than hardcoding two edges) means ENABLED_SOURCES changes
    # the graph's shape - useful for an arXiv-only recall comparison.
    for name, node in enabled_source_nodes().items():
        graph.add_node(name, node)
        graph.add_edge("planner", name)
        graph.add_edge(name, "merge_sources")

    graph.add_edge("merge_sources", "rag")
    graph.add_edge("rag", "extraction")
    graph.add_edge("extraction", END)

    return graph.compile()


def run(topic: str) -> GraphState:
    """Run the compiled graph on a single topic."""
    return build_graph().invoke({"topic": topic})
