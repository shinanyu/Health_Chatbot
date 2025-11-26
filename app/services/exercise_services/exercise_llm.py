# exercise_llm.py
# 포트 : 7000
print("exercise_llm.py 로딩 완료")

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import numpy as np
import asyncio
import os
import re

# ===== Embedding (384-d fixed) =====
from sentence_transformers import SentenceTransformer
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

def embed(text: str) -> List[float]:
    return embed_model.encode(text, normalize_embeddings=True).tolist()

VECTOR_DIM = len(embed_model.encode("dim"))

# ===== LLM (Qwen GGUF 모델 사용) =====
from llama_cpp import Llama
MODEL_PATH = "../../models/exercise_models/qwen2.5-3b-instruct-q4_k_m.gguf"   # ✅ 네가 다운받은 모델 이름 그대로

llm = Llama(
    model_path=MODEL_PATH,
    # 이전 : 2048
    n_ctx=2048,
    n_threads=max(1, (os.cpu_count() or 2) - 1),
    n_batch=128,
    logits_all=False,
    verbose=False,
    chat_format="chatml",
)

# ===== FastAPI =====
app = FastAPI(title="MediBear LLM Server (Local Mongo + RAG)")

# ===== MongoDB =====
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
db = client["ai_coach"]
chat_col = db["chat_history"]
profile_col = db["profile"]

# ===== Input Models =====
class ChatInput(BaseModel):
    user_id: str
    message: str

class ChatWithAnalysisInput(BaseModel):
    user_id: str
    message: str
    analysis: Dict[str, Any]

# ===== Utilities =====
def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na * nb != 0 else 0.0

def safe_get_vec(doc) -> Optional[List[float]]:
    vec = doc.get("embedding") or doc.get("vector")
    if isinstance(vec, list) and len(vec) == VECTOR_DIM:
        return vec
    return None

def clean_text_korean_only(text: str) -> str:
    return re.sub(r"[^가-힣0-9\s\.\,\?\!]", "", text)


# ===== RAG =====
def build_rag_context(user_id: str, user_msg: str, topk: int = 3, mode: str = "auto") -> str:
    qvec = embed(user_msg)

    if mode == "exercise":
        query = {"user_id": user_id, "type": "exercise"}
    elif mode == "general":
        query = {"user_id": user_id, "type": "general"}
    else:
        query = {"user_id": user_id}

    history = list(
        chat_col.find(query)
               .sort("timestamp", -1)
               .limit(60)
    )

    scored = []
    for h in history:
        vec = safe_get_vec(h)
        if vec:
            cleaned = clean_text_korean_only(h["message"])
            if cleaned.strip():
                scored.append((cosine_similarity(qvec, vec), cleaned))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join([f"User said before: {msg}" for _, msg in scored[:topk]])


# ===== Prompt 정의 =====
SYSTEM_PROMPT_EXERCISE = (
    "너는 한국어 퍼스널 트레이너이다.\n"
    "사용자가 어떤 운동을 하고 있는지 반드시 첫 줄에서 자연스럽게 언급한다.\n"
    "분석 데이터는 참고하지만 수치를 그대로 언급하지 않는다.\n"
    "숫자, 각도, cm, %, ° 등 수치 언급 금지.\n"
    "한자, 영어, 전문용어, 과학적 표현 금지.\n"
    "운동 동작에 대한 느낌, 긴장, 힘의 흐름, 체중 분배 중심으로 피드백한다.\n\n"
    "출력 형식:\n"
    "① 지금 하고 있는 운동 이름 + 자세 느낌 요약 (1~2문장)\n"
    "② 잘한 점 (1문장)\n"
    "③ 개선할 점 (• 불릿 2~3개)\n"
    "④ 코칭 큐 (• 불릿 3~5개, 명령형 4~8글자)\n"
    "⑤ 다음 세트 목표 (1문장)\n"
    "(1문장) 이런거는 언급 금지\n"
)


SYSTEM_PROMPT_GENERAL = (
    "너는 한국어 헬스케어 상담 AI다.\n"
    "공감 + 짧고 단호 + 따뜻한 톤.\n"
    "운동 템플릿(①~⑤) 절대 사용 금지.\n"
)


# ===== LLM Wrapper =====
async def llm_generate(messages):
    def _run():
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.12,
            # 이전 : 600
            max_tokens=256, 
        )
        return out["choices"][0]["message"]["content"].strip()
    return await asyncio.to_thread(_run)


# ===== Persona 요약 =====
async def update_persona_background(user_id: str):
    chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(15))
    if not chats:
        return
    text = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])
    summary = await llm_generate([
        {"role": "system", "content": "사용자의 통증 경향/운동 습관/말투를 5줄로 요약하라."},
        {"role": "user", "content": text},
    ])
    profile_col.update_one({"user_id": user_id}, {"$set": {"persona": summary}}, upsert=True)


# ===== 답변 생성 =====
async def generate_answer(user_id: str, user_msg: str, analysis: Dict[str, Any]):
    persona = profile_col.find_one({"user_id": user_id}) or {}
    persona_text = persona.get("persona", "")

    # ✅ 여기 추가: 운동 분석 JSON 키 맞춰주기
    if analysis and "exercise" in analysis and "detected_exercise" not in analysis:
        analysis["detected_exercise"] = analysis["exercise"]

    # ✅ 운동 여부 판별 (이제 정상 동작)
    is_exercise = bool(analysis and analysis.get("detected_exercise"))

    # ✅ 운동명 텍스트 앞에 붙이기
    if is_exercise:
        exercise_name = analysis["detected_exercise"]
        user_msg = f"{exercise_name} 운동 중: {user_msg}"

    if is_exercise:
        system_prompt = SYSTEM_PROMPT_EXERCISE
        rag = build_rag_context(user_id, user_msg, mode="exercise")
    else:
        system_prompt = SYSTEM_PROMPT_GENERAL
        rag = build_rag_context(user_id, user_msg, mode="general")

    user_prompt = (rag + "\n\n" + user_msg) if rag else user_msg

    messages = [
        {"role": "system", "content": system_prompt + ("\n[사용자 요약]\n" + persona_text if persona_text else "")},
        {"role": "user", "content": user_prompt},
    ]

    return await llm_generate(messages)

# ===== 저장 =====
def save_chat(user_id, message, response, embedding, analysis):
    print("save_chat함수 호출 완")
    chat_col.insert_one({
        "user_id": user_id,
        "message": message,
        "response": response,
        "embedding": embedding,
        "analysis": analysis or {},
        "timestamp": datetime.now(),
        "type": "exercise" if (analysis and analysis.get("detected_exercise")) else "general",
    })

# ===== Endpoints =====
@app.post("/chat")
async def chat_plain(data: ChatInput, background_tasks: BackgroundTasks):
    
    qvec = embed(data.message)
    answer = await generate_answer(data.user_id, data.message, analysis={})
    save_chat(data.user_id, data.message, answer, qvec, {})
    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)
    return {"answer": answer}

@app.post("/chat_with_analysis")
async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):
    
    # ✅ message 비어있으면 자동 생성
    user_msg = data.message if data.message and data.message.strip() else "운동 자세 피드백 요청"

    qvec = embed(user_msg)

    print("\n")
    print(data.analysis)
    print("\n")

    answer = await generate_answer(data.user_id, user_msg, data.analysis)

    save_chat(data.user_id, user_msg, answer, qvec, data.analysis)

    if chat_col.count_documents({"user_id": data.user_id}) >= 3:
        background_tasks.add_task(update_persona_background, data.user_id)

    return {"answer": answer}


# # exercise_llm.py
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from typing import Dict, Any, Optional, List
# from pymongo import MongoClient
# import numpy as np
# import asyncio
# import os

# # ===== Embedding (384-d fixed) =====
# from sentence_transformers import SentenceTransformer
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
# embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

# def embed(text: str) -> List[float]:
#     return embed_model.encode(text, normalize_embeddings=True).tolist()

# VECTOR_DIM = len(embed_model.encode("dim"))

# # ===== LLM (Qwen 1.5B GGUF) =====
# from llama_cpp import Llama
# MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=2048,
#     n_threads=max(1, (os.cpu_count() or 2) - 1),
#     n_batch=128,
#     logits_all=False,
#     verbose=False,
#     chat_format="chatml",
# )

# # ===== FastAPI =====
# app = FastAPI(title="MediBear LLM Server (Local Mongo + RAG)")

# # ===== MongoDB =====
# client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]


# # ===== Input Models =====
# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# class ChatWithAnalysisInput(BaseModel):
#     user_id: str
#     message: str
#     analysis: Dict[str, Any]

# # ===== Utilities =====
# def cosine_similarity(a, b) -> float:
#     a, b = np.asarray(a), np.asarray(b)
#     na, nb = np.linalg.norm(a), np.linalg.norm(b)
#     return float(np.dot(a, b) / (na * nb)) if na * nb != 0 else 0.0

# def safe_get_vec(doc) -> Optional[List[float]]:
#     vec = doc.get("embedding") or doc.get("vector")
#     if isinstance(vec, list) and len(vec) == VECTOR_DIM:
#         return vec
#     return None


# # ===== RAG (운동 대화만 검색) =====
# def build_rag_context(user_id: str, user_msg: str, topk: int = 3) -> str:
#     qvec = embed(user_msg)

#     # 🔥 변경: 운동 기록만 불러오기
#     history = list(
#         chat_col.find({"user_id": user_id, "type": "exercise"})
#                 .sort("timestamp", -1)
#                 .limit(50)
#     )

#     scored = []
#     for h in history:
#         vec = safe_get_vec(h)
#         if vec:
#             scored.append((cosine_similarity(qvec, vec), h["message"], h["response"]))

#     if not scored:
#         return ""

#     scored.sort(key=lambda x: x[0], reverse=True)
#     return "\n---\n".join([f"User: {u}\nAI: {a}" for _, u, a in scored[:topk]])


# # ===== Prompt 변환 =====
# SYSTEM_PROMPT_EXERCISE = (
#     "너는 한국어 퍼스널 트레이너이다. 분석 데이터를 바탕으로 감각 중심 피드백만 제공한다.\n"
#     "숫자, 각도, cm, %, ° 등 수치 언급 금지.\n"
#     "출력 형식은 반드시 아래 형식을 따른다:\n"
#     "① 자세 느낌 요약 (2문장)\n"
#     "② 잘한 점 (1문장)\n"
#     "③ 개선할 점 (• 불릿 2~3개)\n"
#     "④ 코칭 큐 (• 불릿 3~5개, 4~8글자 명령형)\n"
#     "⑤ 다음 세트 목표 (1문장)\n"
# )

# SYSTEM_PROMPT_GENERAL = (
#     "너는 한국어 헬스케어 상담 AI다.\n"
#     "공감 + 간단명료 + 따뜻한 톤으로 답한다.\n"
#     "운동 피드백 형식(①~⑤)은 절대 사용하지 않는다.\n"
# )


# # ===== Persona 요약 =====
# async def update_persona_background(user_id: str):
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(12))
#     if not chats:
#         return
#     text = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])
#     summary = await llm_generate([
#         {"role": "system", "content": "사용자의 운동 목표/몸 상태/말투를 5줄로 요약해라."},
#         {"role": "user", "content": text},
#     ])
#     profile_col.update_one({"user_id": user_id}, {"$set": {"persona": summary}}, upsert=True)


# # ===== LLM 호출 =====
# async def llm_generate(messages):
#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.5,
#             top_p=0.9,
#             repeat_penalty=1.12,
#             max_tokens=600,
#         )
#         return out["choices"][0]["message"]["content"].strip()
#     return await asyncio.to_thread(_run)


# # ===== 답변 생성 (핵심 로직) =====
# async def generate_answer(user_id: str, user_msg: str, analysis: Dict[str, Any]):
#     persona = profile_col.find_one({"user_id": user_id}) or {}
#     persona_text = persona.get("persona", "")

#     is_exercise = bool(analysis and analysis.get("detected_exercise"))

#     if is_exercise:
#         system_prompt = SYSTEM_PROMPT_EXERCISE
#         user_prompt = build_rag_context(user_id, user_msg) + "\n\n" + user_msg
#     else:
#         system_prompt = SYSTEM_PROMPT_GENERAL
#         user_prompt = user_msg

#     messages = [
#         {"role": "system", "content": system_prompt + ("\n[사용자 요약]\n" + persona_text if persona_text else "")},
#         {"role": "user", "content": user_prompt},
#     ]

#     return await llm_generate(messages)


# # ===== 저장 (운동/일반 자동 태그) =====
# def save_chat(user_id, message, response, embedding, analysis):
#     chat_col.insert_one({
#         "user_id": user_id,
#         "message": message,
#         "response": response,
#         "embedding": embedding,
#         "analysis": analysis or {},
#         "timestamp": datetime.now(),
#         "type": "exercise" if (analysis and analysis.get("detected_exercise")) else "general",  # 🔥 변경
#     })


# # ===== Endpoints =====
# @app.post("/chat")
# async def chat_plain(data: ChatInput, background_tasks: BackgroundTasks):
#     qvec = embed(data.message)
#     answer = await generate_answer(data.user_id, data.message, analysis={})
#     save_chat(data.user_id, data.message, answer, qvec, {})
#     if chat_col.count_documents({"user_id": data.user_id}) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)
#     return {"answer": answer}

# @app.post("/chat_with_analysis")
# async def chat_with_analysis(data: ChatWithAnalysisInput, background_tasks: BackgroundTasks):
#     qvec = embed(data.message)
#     answer = await generate_answer(data.user_id, data.message, data.analysis)
#     save_chat(data.user_id, data.message, answer, qvec, data.analysis)
#     if chat_col.count_documents({"user_id": data.user_id}) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)
#     return {"answer": answer}



# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from datetime import datetime
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import asyncio
# import os
# from llama_cpp import Llama

# app = FastAPI()

# # ---------------- MongoDB ----------------
# client = MongoClient("mongodb://localhost:27017")
# db = client["ai_coach"]
# chat_col = db["chat_history"]
# profile_col = db["profile"]

# # ---------------- Embedding Model ----------------
# embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

# # ---------------- LLM ----------------
# MODEL_PATH = "../../models/exercise_models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=1024,
#     n_threads=8,
#     n_batch=128,
#     logits_all=False,
#     verbose=False,
#     chat_format="chatml"
# )

# class ChatInput(BaseModel):
#     user_id: str
#     message: str

# def cosine_similarity(a, b):
#     a, b = np.array(a), np.array(b)
#     if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
#         return 0.0
#     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# async def generate_async(user_msg: str, persona: str, context_text: str):
#     messages = [
#         {"role": "system",
#          "content": (
#              "당신은 개인 맞춤형 건강/운동 상담 코치 AI입니다.\n"
#              "사용자의 지난 대화 내용(persona 요약 + 최근 대화 context)을 참고하여 "
#              "사용자의 상태와 감정, 습관을 기억하고 자연스럽게 이어지는 대화를 하세요."
#          )},
#         {"role": "user",
#          "content": f"[사용자 요약 정보]\n{persona}\n\n[최근 관련 대화]\n{context_text}\n\n[현재 질문]\n{user_msg}"}
#     ]

#     def _run():
#         out = llm.create_chat_completion(
#             messages=messages,
#             temperature=0.35,
#             top_p=0.9,
#             max_tokens=240,
#             stop=["</s>", "<|im_end|>"]
#         )
#         return out["choices"][0]["message"]["content"].strip()

#     return await asyncio.to_thread(_run)


# async def update_persona_background(user_id: str):
#     chats = list(chat_col.find({"user_id": user_id}).sort("timestamp", -1).limit(10))
#     text_block = "\n".join([f"User: {c['message']}\nAI: {c['response']}" for c in chats])

#     messages = [
#         {"role": "system", "content": "최근 대화를 분석하여 사용자의 건강/운동 특징을 5줄로 요약하세요."},
#         {"role": "user", "content": text_block or "(대화없음)"}
#     ]

#     def _run():
#         out = llm.create_chat_completion(messages=messages, temperature=0.2, top_p=0.9, max_tokens=120)
#         return out["choices"][0]["message"]["content"].strip()

#     summary = await asyncio.to_thread(_run)

#     profile_col.update_one(
#         {"user_id": user_id},
#         {"$set": {"persona": summary, "updated_at": datetime.now()}},
#         upsert=True
#     )


# @app.post("/chat")
# async def chat_with_ai(data: ChatInput, background_tasks: BackgroundTasks):

#     # 1) 입력 문장 임베딩
#     emb = embed_model.encode(data.message, normalize_embeddings=True)
#     user_vec = emb.tolist()

#     # 2) 최근 대화 불러오기 + RAG (유사도 상위 3개)
#     history = list(chat_col.find({"user_id": data.user_id}).sort("timestamp", -1).limit(10))

#     contexts = []
#     for h in history:
#         vec = h.get("embedding") or h.get("vector")
#         if not vec:
#             continue
#         if len(vec) != len(user_vec):
#             continue    # ✅ 차원 다르면 skip
#         sim = cosine_similarity(user_vec, vec)
#         contexts.append((sim, h["message"], h.get("response", "")))

#     if contexts:
#         contexts = sorted(contexts, key=lambda x: x[0], reverse=True)[:3]
#         context_text = "\n".join([f"User: {m}\nAI: {r}" for _, m, r in contexts])
#     else:
#         context_text = "\n".join([f"User: {h['message']}\nAI: {h['response']}" for h in history[:3]])

#     # 3) Persona 불러오기
#     profile = profile_col.find_one({"user_id": data.user_id})
#     persona = profile["persona"] if profile else "특징 미파악 사용자"

#     # 4) LLM 호출
#     answer = await generate_async(data.message, persona, context_text)

#     # 5) 저장
#     chat_col.insert_one({
#         "user_id": data.user_id,
#         "message": data.message,
#         "response": answer,
#         "embedding": user_vec,
#         "timestamp": datetime.now()
#     })

#     # 6) Persona 업데이트는 백그라운드로
#     if len(history) >= 3:
#         background_tasks.add_task(update_persona_background, data.user_id)

#     return {"answer": answer, "persona_summary": persona}

