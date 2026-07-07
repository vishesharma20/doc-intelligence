import os
from dotenv import load_dotenv
from groq import Groq
from app.agents.vectorstore import add_chunks, query_chunks, clear_collection

def index_document(chunks: list[str]):
    """Call this once per document to load it into the vector store."""
    clear_collection()
    add_chunks(chunks)

load_dotenv()
import streamlit as st

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

client = Groq(api_key=get_groq_key())

QA_PROMPT = """Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""

def index_document(chunks: list[str]):
    """Call this once per document to load it into the vector store."""
    add_chunks(chunks)

def ask(question: str) -> str:
    relevant_chunks = query_chunks(question)
    context = "\n\n".join(relevant_chunks)

    prompt = QA_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    from app.loaders import load_document, chunk_text

    text = load_document("sample.pdf")
    chunks = chunk_text(text)
    index_document(chunks)

    question = "What is this document about?"
    answer = ask(question)
    print(f"Q: {question}\nA: {answer}")

SMALL_TALK = {"hi", "hello", "hey"}

def ask(question: str) -> str:
    if question.strip().lower().rstrip("!.?") in SMALL_TALK:
        return "Hi! Ask me anything about the document you uploaded."

    relevant_chunks = query_chunks(question)
    context = "\n\n".join(relevant_chunks)

    prompt = QA_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()