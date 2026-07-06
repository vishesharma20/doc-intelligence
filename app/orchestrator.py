from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.agents.summarizer import summarize
from app.agents.critic import critique
from app.agents.extractor import extract
from app.agents.qa import index_document, ask


class DocState(TypedDict):
    document_text: str
    summary: Optional[str]
    critique_result: Optional[dict]
    extracted_data: Optional[dict]
    retry_count: int
    max_retries: int


def summarizer_node(state: DocState) -> DocState:
    print(f"[Summarizer] Running (attempt {state['retry_count'] + 1})...")
    summary = summarize(state["document_text"])
    return {**state, "summary": summary}


def critic_node(state: DocState) -> DocState:
    print("[Critic] Reviewing summary...")
    result = critique(state["document_text"], state["summary"])
    return {**state, "critique_result": result, "retry_count": state["retry_count"] + 1}


def extractor_node(state: DocState) -> DocState:
    print("[Extractor] Extracting structured data...")
    data = extract(state["document_text"])
    return {**state, "extracted_data": data}


def route_after_critic(state: DocState) -> str:
    verdict = state["critique_result"].get("verdict", "approve")
    if verdict == "needs_revision" and state["retry_count"] < state["max_retries"]:
        print("[Router] Needs revision -> retrying summarizer")
        return "retry"
    print("[Router] Approved or max retries hit -> continue")
    return "continue"


# Build the graph
graph = StateGraph(DocState)

graph.add_node("summarizer", summarizer_node)
graph.add_node("critic", critic_node)
graph.add_node("extractor", extractor_node)

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "critic")
graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "retry": "summarizer",
        "continue": "extractor"
    }
)
graph.add_edge("extractor", END)

app_graph = graph.compile()


def run_pipeline(document_text: str, chunks_for_qa: list[str] = None) -> DocState:
    initial_state: DocState = {
        "document_text": document_text,
        "summary": None,
        "critique_result": None,
        "extracted_data": None,
        "retry_count": 0,
        "max_retries": 2,
    }

    if chunks_for_qa:
        index_document(chunks_for_qa)

    final_state = app_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    from app.loaders import load_document, chunk_text

    text = load_document("OfferLetter.pdf")
    chunks = chunk_text(text)

    result = run_pipeline(text, chunks_for_qa=chunks)

    print("\n=== FINAL RESULT ===")
    print("\nSUMMARY:\n", result["summary"])
    print("\nCRITIQUE:\n", result["critique_result"])
    print("\nEXTRACTED DATA:\n", result["extracted_data"])