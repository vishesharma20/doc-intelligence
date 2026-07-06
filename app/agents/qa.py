import os
from dotenv import load_dotenv
from groq import Groq
from app.agents.vectorstore import add_chunks, query_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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