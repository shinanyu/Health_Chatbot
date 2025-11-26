import os
from langchain_openai import ChatOpenAI

OPENAI_MODEL = os.getenv("OPHTHA_LLM_MODEL", "gpt-4o-mini")

def load_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

llm = load_llm()

