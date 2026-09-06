"""Milestone 2: RAG node - answers each sub-question from the embedded evidence.

Search and storage moved out to `search.py` in Milestone 4, so this node now
does one thing: answer the plan's sub-questions from whatever the merge node
put in the vector store.
"""

import logging

from research_agent.config import get_llm
from research_agent.retriever import similarity_search
from research_agent.schemas import RAGAnswer
from research_agent.state import GraphState

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """\
You are a research assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so. Be concise and cite specific papers.
"""


def run_rag(state: GraphState) -> GraphState:
    """LangGraph node: reads state['plan'], writes state['rag_answers']."""
    plan = state.get("plan")
    if not plan:
        return state

    llm = get_llm()
    rag_answers: list[RAGAnswer] = []

    for obj in plan.objectives:
        for question in obj.sub_questions:
            # Retrieve relevant chunks
            chunks = similarity_search(question, top_k=5)
            logger.info("Q: %s -> %d chunks retrieved", question, len(chunks))

            if not chunks:
                rag_answers.append(
                    RAGAnswer(question=question, answer="No relevant information found.", sources=[])
                )
                continue

            # Build context
            context = "\n\n".join(
                f"[{c['paper_id']}] {c['title']}\n{c['content']}" for c in chunks
            )
            sources = list({c["paper_id"] for c in chunks})

            # Generate answer
            response = llm.invoke([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", f"Context:\n{context}\n\nQuestion: {question}"),
            ])

            rag_answers.append(
                RAGAnswer(question=question, answer=response.content, sources=sources)
            )

    return {**state, "rag_answers": rag_answers}
