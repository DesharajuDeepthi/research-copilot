import json
import re

from langchain_openai import ChatOpenAI

from research_copilot.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    return json.loads(match.group(0) if match else text)
