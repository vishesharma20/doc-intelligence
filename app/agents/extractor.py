import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """You are a document extraction agent. Extract key structured information from the document below.

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
    "title": "document title or best guess",
    "document_type": "e.g. contract, report, article, resume, invoice",
    "key_entities": ["list of important names, organizations, or places mentioned"],
    "key_dates": ["list of important dates mentioned"],
    "key_facts": ["list of 3-5 most important facts or figures"],
    "summary_one_line": "one sentence summary"
}}

Document:
{document_text}
"""

def extract(text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(document_text=text[:6000])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    # Clean up in case model wraps it in markdown fences
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
    text = load_document("sample.pdf")
    result = extract(text)
    print(json.dumps(result, indent=2))