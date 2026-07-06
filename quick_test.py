import os
from dotenv import load_dotenv
from groq import Groq
from app.loaders import load_document

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

text = load_document("sample.pdf")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{
        "role": "user",
        "content": f"Summarize this document in 3 sentences:\n\n{text[:5000]}"
    }]
)

print(response.choices[0].message.content)