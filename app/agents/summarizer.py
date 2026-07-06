import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SUMMARY_PROMPT = """You are a summarization agent. Read the document below and produce two summaries.

Return your response in this exact format (plain text, no JSON):

EXECUTIVE SUMMARY:
[2-3 sentence high-level summary for someone who has 10 seconds]

DETAILED SUMMARY:
[A more thorough paragraph, 5-8 sentences, covering the main points, structure, and any important details]

Document:
{document_text}
"""

def summarize(text: str) -> str:
    prompt = SUMMARY_PROMPT.format(document_text=text[:6000])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    from app.loaders import load_document
    text = load_document("sample.pdf")
    result = summarize(text)
    print(result)