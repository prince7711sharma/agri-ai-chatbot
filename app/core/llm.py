# app/core/llm.py

import os
from groq import Groq
from dotenv import load_dotenv
from app.core.config import LLM_MODEL

# LOAD .env FILE
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
