import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CRITIC_PROMPT = """You are a quality-control agent reviewing another AI agent's output against the source document.

Source document (excerpt):
{document_text}

Agent output to review:
{agent_output}

Check for:
1. Factual accuracy (does the output match the document?)
2. Completeness (is anything important missing?)
3. Hallucination (did the agent invent anything not in the document?)

Return ONLY valid JSON (no markdown) with this structure:
{{
    "quality_score": <1-10>,
    "issues_found": ["list of specific issues, empty list if none"],
    "verdict": "approve" or "needs_revision",
    "notes": "brief explanation of the verdict"
}}
"""

def critique(document_text: str, agent_output: str) -> dict:
    prompt = CRITIC_PROMPT.format(
        document_text=document_text[:4000],
        agent_output=agent_output
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_output": raw_output}


if __name__ == "__main__":
    from app.loaders import load_document
    from app.agents.summarizer import summarize

    text = load_document("sample.pdf")
    summary = summarize(text)
    result = critique(text, summary)
    print(json.dumps(result, indent=2))