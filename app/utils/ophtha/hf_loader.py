from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
from typing import List, Optional, Tuple

load_dotenv()

class Settings(BaseModel):
    hf_token: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    model_repo: str = os.getenv("MODEL_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    temperature: float = float(os.getenv("HF_TEMPERATURE", "0.3"))
    max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))

settings = Settings()

def _get_client() -> InferenceClient:
    if not settings.hf_token:
        raise ValueError("❌ HUGGINGFACE_API_TOKEN is missing in .env file")
    return InferenceClient(model=settings.model_repo, token=settings.hf_token)

def ask_llm(
    prompt: str,
    history: Optional[Tuple[List[str], List[str]]] = None,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    client = _get_client()
    past_user_inputs, generated_responses = history or ([], [])
    params = {
        "temperature": temperature or settings.temperature,
        "max_new_tokens": max_new_tokens or settings.max_new_tokens,
    }
    payload = {
        "past_user_inputs": past_user_inputs,
        "generated_responses": generated_responses,
        "text": prompt,
    }
    try:
        resp = client.conversational(inputs=payload, params=params)
        return resp.get("generated_text", "").strip()
    except Exception as e:
        return f"⚠️ LLM 호출 중 오류 발생: {e}"
