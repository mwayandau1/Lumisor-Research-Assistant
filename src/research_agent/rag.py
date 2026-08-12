"""Milestone 2: Retrieval node - fetches papers and answers sub-questions via RAG."""

from research_agent.config import get_llm
from research_agent.retriever import embed_and_store, search_arxiv, similarity_search
from research_agent.schemas import RAGAnswer, RetrievedPaper
from research_agent.state import GraphState

RAG_SYSTEM_PROMPT = """\
You are a research assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so. Be concise and cite specific papers.
"""


def run_retrieval(state: GraphState) -> GraphState:
    """
    LangGraph node: uses search queries from plan to fetch papers,
    embed them, and answer each sub-question via RAG.
    """
    plan = state["plan"]
    if not plan:
        return state

    # 1. Search arXiv using expanded queries
    all_papers: dict[str, RetrievedPaper] = {}
    for query in plan.search_queries[:5]:  # Limit queries to avoid rate limits
        papers = search_arxiv(query, max_results=3)
        for p in papers:
            all_papers[p.paper_id] = p

    retrieved = list(all_papers.values())

    # 2. Embed and store in Supabase
    embed_and_store(retrieved)

    # 3. Answer each sub-question via RAG
    llm = get_llm()
    rag_answers: list[RAGAnswer] = []

    for obj in plan.objectives:
        for question in obj.sub_questions:
            # Retrieve relevant chunks
            chunks = similarity_search(question, top_k=5)

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

    return {**state, "retrieved_papers": retrieved, "rag_answers": rag_answers}
