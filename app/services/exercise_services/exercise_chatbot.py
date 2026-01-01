# exercise_chatbot.py
# 실행: uvicorn exercise_chatbot:app --host 0.0.0.0 --port 7000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from llama_cpp import Llama
import numpy as np
import re

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ===== MongoDB =====
client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=500)
db = client["ai_coach"]
chat_col = db["chat_history"]

# ===== 임베딩 모델 =====
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

# 임베딩 함수
# def embed(text: str) -> List[float]:
#     # e5 계열은 보통 "query: " / "passage: " prefix 쓰지만,
#     # 간단하게는 그냥 text만 넣어도 동작함.
#     return embed_model.encode(text, normalize_embeddings=True).tolist()

def embed(text: str, mode: str = "passage") -> List[float]:
    """
    mode = "query"  -> 검색 쿼리용 임베딩
    mode = "passage" -> 문서(저장용) 임베딩
    """
    if mode == "query":
        prefix = "query: "
    else:
        prefix = "passage: "
    return embed_model.encode(prefix + text, normalize_embeddings=True).tolist()

# 큐앤에이 저장 함수
def save_qna_to_mongo(user_id: str, question: str, answer: str):
    """
    사용자 질문 + LLM 답변을 하나의 문서로 묶어서 임베딩 후 MongoDB에 저장
    """
    doc_text = f"질문: {question}\n답변: {answer}"
    vec = embed(doc_text, mode="passage")

    qna_doc = {
        "user_id": user_id,
        "text": doc_text,
        "embedding": vec,
        "created_at": datetime.utcnow(),
    }
    chat_col.insert_one(qna_doc)

# 대화 내용 코사인 유사도 검색 함수 (mongodb벡터 검색 기능 사용x)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.dot(a_norm, b_norm.T)[0][0])

# 이전 q&a 프롬프트에 넣는 함수
def format_memory_block(docs: List[Dict[str, Any]]) -> str:
    """
    검색된 Q&A들을 LLM에게 보여줄 수 있는 텍스트 블록으로 변환.
    너무 길어지지 않게 적당히 자름.
    """
    if not docs:
        return ""

    lines = ["[이전에 이 사용자에게 제공했던 관련 조언 일부]"]
    for i, doc in enumerate(docs, start=1):
        text = doc.get("text", "")
        # 너무 길면 앞부분만 사용
        text = text[:500]
        lines.append(f"\n(Q&A #{i})\n{text}")

    return "\n".join(lines)

def retrieve_similar_qna(user_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    새 질문(query)에 대해, 같은 user_id의 과거 Q&A 중 상위 top_k개를 찾아옴
    (간단하게 전체 스캔 + 파이썬에서 코사인 유사도 정렬)
    """
    query_vec = np.array(embed(query, mode="query"), dtype=np.float32)

    # 일단 최근 N개만 제한해서 가져오기 (예: 200개)
    cursor = chat_col.find({"user_id": user_id}).sort("created_at", -1).limit(200)

    scored = []
    for doc in cursor:
        emb = doc.get("embedding")
        if not emb:
            continue
        emb_vec = np.array(emb, dtype=np.float32)
        sim = cosine_sim(query_vec, emb_vec)
        scored.append((sim, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 디버깅
    for sim, doc in scored[:5]:
        print(f"[RAG DEBUG] sim={sim:.3f}, text={doc.get('text','')[:80]}")
    
    top_docs = [d for (s, d) in scored[:top_k] if s > 0.2]  # 유사도 threshold는 대충 0.3 정도

    return top_docs

# ----------------------------------------
# FastAPI 설정
# ----------------------------------------
app = FastAPI(title="Exercise LLM Chatbot API")

# ----------------------------------------
class ChatWithAnalysisRequest(BaseModel):
    user_id: str                      # 유저 식별자 (현재는 안 써도 일단 받고 있음)
    message: str = ""                 # 사용자가 입력한 질문/메시지
    analysis: Dict[str, Any]          # /analyze 에서 생성된 운동 분석 JSON
    session_id: Optional[str] = None  # 추후 세션 관리용으로 확장 가능

class ChatWithAnalysisResponse(BaseModel):
    answer: str        # LLM이 생성한 답변
    used_summary: str  # 프롬프트에 사용된 운동 요약 텍스트 (디버깅/로그용)

# ----------------------------------------
# LLM 로딩 (Qwen 모델)
# ----------------------------------------
MODEL_PATH = "../../models/exercise_models/qwen2.5-3b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    repeat_penalty=1.5,      # repeat_penalty는 1.2~1.4 권장 : 1.25~1.3은 “방금 쓴 문장 반복 금지 효과” 
    repeat_last_n=512,       # 반복 억제
)

BAD_PHRASES = [
    "이 답변 구조를 따르세요",
    "좋은 운동이 되세요",
    "질문이 없을 경우",
    "---",
]

def clean_answer(text: str) -> str:
    # 1) 금지 문장 기준으로 뒷부분 싹 날리기
    for p in BAD_PHRASES:
        if p in text:
            text = text.split(p)[0]

    # 2) '좋았던 점:'이 여러 번 나오면 → 두 번째 이후는 전부 삭제
    parts = text.split("좋았던 점:")
    if len(parts) > 2:
        # parts[0]은 앞의 잡다한 프롬프트 잔여 텍스트일 수 있음 → 제거하고
        # 첫 번째 블록만 다시 붙임
        text = "좋았던 점:" + parts[1]

    # 3) 전체에서 첫 번째 '부상 위험 요소:' 이후 불필요한 텍스트 제거
    if "부상 위험 요소:" in text:
        # 부상 위험 요소 이후에 불필요한 반복이 붙을 수 있으므로
        sub = text.split("부상 위험 요소:")[1]
        # "부상 위험 요소:" + 첫 번째 문단만 남김
        first_line = sub.strip().splitlines()[0]
        text = (
            text.split("부상 위험 요소:")[0]
            + "부상 위험 요소: "
            + first_line
        )

    return text.strip()


# ----------------------------------------
# 작은 유틸: safe_mean
# ----------------------------------------
def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

# ----------------------------------------
# 분석,요약 JSON →  한국어 요약 텍스트 생성 함수 : llm에게 던져줄 정보 
def summarize_exercise_json(data: Dict[str, Any]) -> str:

    if data.get("exercise") is None and data.get("model") is None and data.get("reps") is None:
        msg = data.get("message", "")
        lines = ["[운동 분석 정보 없음]", "사용자가 보낸 텍스트만 전달되었습니다."]
        if msg:
            lines.append(f"사용자 메시지(원문): {msg}")
        return "\n".join(lines)

    # ---- 운동 분석 정보가 있는 경우 ----
    ex = data.get("exercise", "unknown")
    fps = data.get("fps")
    dur = data.get("duration_s")

    g = data.get("global_stats", {}) or {}
    rep_count = g.get("rep_count")
    avg_rom = g.get("avg_rom_primary_deg")
    wobble = g.get("wobble_score")
    primary_angle = g.get("primary_angle")

    reps = data.get("reps", []) or []

    # rep들에서 추가로 평균 템포/대칭 계산
    tempo_down_list = []
    tempo_up_list = []
    symmetry_list = []

    for r in reps:
        t = r.get("tempo_s") or {}
        if t.get("down") is not None:
            tempo_down_list.append(t.get("down"))
        if t.get("up") is not None:
            tempo_up_list.append(t.get("up"))
        sym = r.get("symmetry_deg")
        if sym is not None:
            symmetry_list.append(sym)

    avg_tempo_down = g.get("avg_tempo_down_s")  # 이미 global_stats에 있는 값
    if avg_tempo_down is None:
        avg_tempo_down = _safe_mean(tempo_down_list)

    avg_tempo_up = _safe_mean(tempo_up_list)
    left_right_symmetry = _safe_mean(symmetry_list)

    lines: List[str] = []

    lines.append(f"운동 종류: {ex}")
    if dur is not None:
        lines.append(f"총 운동 시간: 약 {dur:.1f}초")
    if fps is not None:
        lines.append(f"영상 FPS(추정): {fps:.1f}")

    if rep_count is not None:
        lines.append(f"총 반복 횟수(필터링 후): {rep_count}회")

    if primary_angle and avg_rom is not None:
        lines.append(
            f"주요 관절({primary_angle})의 평균 ROM: 약 {avg_rom:.1f}도"
        )

    if avg_tempo_down is not None:
        if avg_tempo_up is not None:
            lines.append(
                f"평균 템포: 내려가는 동작 {avg_tempo_down:.2f}초 / 올라오는 동작 {avg_tempo_up:.2f}초"
            )
        else:
            lines.append(
                f"평균 템포: 내려가는 동작 {avg_tempo_down:.2f}초 (올라오는 동작은 프레임 부족으로 추정 어려움)"
            )

    if wobble is not None:
        lines.append(
            f"상체 흔들림 지수(wobble_score): {wobble:.2f} (1에 가까울수록 상체가 많이 흔들리는 편)"
        )

    if left_right_symmetry is not None:
        lines.append(
            f"좌우 무릎 각도 차이 평균: {left_right_symmetry:.1f}도 (값이 클수록 좌우 비대칭이 큰 편)"
        )

    # reps 예시 2~3개 요약
    if reps:
        lines.append("")  # 빈 줄
        lines.append("반복 예시 (일부):")
        for rep in reps[:3]:
            idx = rep.get("idx")
            rom_primary = rep.get("primary_rom_deg")
            t = rep.get("tempo_s") or {}
            sym = rep.get("symmetry_deg")
            align = rep.get("alignment_knee_over_toe_ratio")
            stab = rep.get("stability") or {}
            torso_std = stab.get("torso_std_deg")

            rep_line = f"- {idx}번째 반복: "

            if rom_primary is not None:
                rep_line += f"주요 ROM {rom_primary:.1f}도, "

            if t.get("down") is not None and t.get("up") is not None:
                rep_line += f"내려가는 {t['down']:.2f}초 / 올라오는 {t['up']:.2f}초, "

            if sym is not None:
                rep_line += f"좌우 무릎 각도 차이 평균 {sym:.1f}도, "

            if align is not None:
                rep_line += f"무릎-발끝 정렬 비율 {align:.2f}, "

            if torso_std is not None:
                rep_line += f"상체 흔들림(등 각도 표준편차) {torso_std:.1f}도"

            lines.append(rep_line.rstrip(", "))

    return "\n".join(lines)

# ----------------------------------------
# LLM 프롬프트 생성
# ----------------------------------------
BASE_SYSTEM_PROMPT = """
당신은 영상 기반 운동 분석을 전문적으로 설명하는 AI 코치입니다.

반드시 지켜야 할 규칙은 다음과 같습니다:

1. 답변은 항상 제공된 "운동 분석 요약(summary)"에 기반하여 생성합니다.
2. summary에 없는 내용은 절대 추측하지 않습니다.
3. summary에 없는 정보에 대해 질문받으면 다음 문장으로 답합니다:
   - "이 영상 정보만으로는 확인이 어렵습니다."
4. 사용자의 메시지는 참고용이며, 사용자 메시지의 문장을 그대로 복사하거나 재구성하지 않습니다.
5. summary에 등장하는 문장이나 표현을 그대로 복사하거나 반복해 사용하지 않습니다.
6. 답변에는 "analysis_json", "summary", "데이터", "지시사항", "답변 구조" 같은 기술적·설명적 단어를 포함하지 않습니다.
7. 아래 세 개의 제목만 사용하고, 각 제목 아래에 한 번씩만 내용을 작성합니다.
8. 예시나 샘플 문장을 그대로 따라 하지 않습니다.
9. summary가 부족하면 해당 항목은 다음 문장으로 채웁니다:
   - "이 영상 정보만으로는 확인이 어렵습니다."
10. 답변에는 인사말, 설명, 메타 코멘트(예: "이 답변 구조를 따르세요", "질문이 없습니다")를 절대 포함하지 않습니다.
11. 각 섹션(좋았던 점 / 개선이 필요한 점 / 부상 위험 요소)의 첫 문장은 반드시 운동 종류(exercise)를 명시해야 합니다.
12. 같은 의미의 문장을 반복하지 않습니다.
13. 불필요한 문장 구조(“이는 ~ 때문이다”, “이로 인해 ~ 발생한다”)는 최소화하고 자연스럽게 표현합니다.

14. 하나의 지표(예: 평균 ROM, 상체 흔들림 지수, 좌우 무릎 각도 차이, 템포 등)는 딱 한 섹션에서만 사용합니다.
    - 이미 '좋았던 점'에서 사용한 지표는 '개선이 필요한 점'이나 '부상 위험 요소'에서 다시 언급하지 않습니다.
    - 이미 '개선이 필요한 점'에서 사용한 지표는 '부상 위험 요소'에서 다시 언급하지 않습니다.

15. 가능한 한 다음 기준을 따릅니다.
    - ROM, 템포 등 긍정적으로 해석할 수 있는 내용은 "좋았던 점"에만 사용합니다.
    - 상체 흔들림, 템포 불균형, 힘 조절 문제 등은 "개선이 필요한 점"에서 다룹니다.
    - 좌우 비대칭, 관절 정렬 문제, 부상 가능성이 있는 내용은 "부상 위험 요소"에서만 다룹니다.

16. "좋았던 점"에는 부정적인 내용(위험, 불균형, 과도한 흔들림 등)을 포함하지 않습니다.
17. "개선이 필요한 점"과 "부상 위험 요소"에 같은 내용을 두 번 쓰지 않습니다.
"""

def build_prompt(summary: str, user_msg: str, memory_block: str = "") -> str:
    print(summary)

    is_summary_available = not summary.startswith("[운동 분석 정보 없음]")

    # 🔹 1) 운동 분석 정보 없을 경우 → 일반 대화 모드
    if not is_summary_available:
        return f"""너는 친절한 한국인 운동 코치야.
[이전에 이 사용자에게 했던 조언 (없으면 비어 있을 수 있음)]
{memory_block if memory_block else "(이전에 저장된 조언이 없습니다.)"}

[지침]
- 위 블록에 사용자의 과거 운동이나 조언 내용이 있다면, 그것을 우선 참고해서 이 사용자에게 맞춰서 답변한다.
- 특히, 사용자가 "전에", "예전에", "지난번에", "그때"처럼 과거 운동이나 과거 조언을 물어보는 경우:
  - 위 블록에 관련 내용이 있으면, 그 내용을 근거로 "전에 ~ 운동을 했었다"처럼 요약해서 알려준다.
  - 위 블록에 관련 내용이 없으면, "저장된 기록만으로는 예전에 어떤 운동을 하셨는지 정확히 알기 어렵습니다."라고 솔직하게 말한다.
  - 이때 과거에 하지 않았던 운동이나 기록에 없는 내용은 절대 지어내지 않는다.
- 사용자가 앞으로 어떤 운동을 하면 좋을지, 운동 팁/계획/추천을 물어보는 경우에는,
  - 개인 기록이 있으면 그것을 참고해서 개인화된 조언을 하고,
  - 개인 기록이 부족하면 일반적인 운동 상식 수준에서 조언해도 된다.
- 답변은 1~2문장으로 짧게, 예시 대화나 새로운 질문을 만들지 말고, 지금 질문에 대한 답변만 작성한다.

[사용자 질문]
{user_msg}

[최종 답변만 작성하세요]
"""

    # 2) 운동 분석 정보 있을 경우 → 기존 운동 피드백 모드
    memory_section = ""
    if memory_block:
        memory_section = f"\n{memory_block}\n"

    return f"""{BASE_SYSTEM_PROMPT}

[운동 분석 요약]
{summary}

{memory_section}
[사용자 질문]
{user_msg}

[추가 지시사항]
- 아래 세 항목 각각을 한 번씩만 작성합니다.
- 각 항목은 최대 1~2문장만 사용합니다.
- 같은 표현을 반복해서 사용하지 않습니다.
- "분석 결과를 바탕으로"라는 문장은 0~1회만 사용합니다.
- 사용자 질문의 문장을 답변에 포함하지 않습니다.
- 아래에 제시된 형식 외의 문장을 앞이나 뒤에 덧붙이지 않습니다.
- 답변의 시작은 반드시 '좋았던 점:'으로 시작해야 하며,
  마지막 문장은 '부상 위험 요소:' 항목의 문장이어야 합니다.
- 각 항목 첫 문장은 반드시 운동 종류를 자연스럽게 포함하세요. 예: "이 스쿼트 동작에서는...", "이 해머 컬 동작에서는...".
- 동일한 문장 또는 동일한 의미의 문장을 반복하지 않습니다.
- 아래 세 개의 섹션(좋았던 점 / 개선이 필요한 점 / 부상 위험 요소) 블록은 정확히 한 번만 출력합니다.

[최종 출력 형식]
좋았던 점:
- 

개선이 필요한 점:
- 

부상 위험 요소:
- 
"""

# ----------------------------------------
# 최종 엔드포인트 : /chat_with_analysis
#   main.py 의 /analyze → LLM 서버 호출 구조와 1:1 매칭
# ----------------------------------------
@app.post("/chat_with_analysis", response_model=ChatWithAnalysisResponse)
async def chat_with_analysis(req: ChatWithAnalysisRequest):
    """
    /analyze 에서 전달한:
      - user_id
      - message (사용자 질문 텍스트)
      - analysis (운동 분석 JSON)
    을 받아서,
      1) analysis → 요약 텍스트로 변환
      2) 요약 + 질문으로 프롬프트 생성
      3) LLM(qwen)에게 전달해 답변 생성
      4) answer + used_summary 반환
    """

    # 1. JSON → 요약 텍스트
    summary_text = summarize_exercise_json(req.analysis)
    
    # 2. 과거 Q&A RAG 검색
    similar_docs = retrieve_similar_qna(user_id=req.user_id, query=req.message, top_k=3)
    memory_block = format_memory_block(similar_docs)
    
    # 3. 프롬프트 구성
    prompt = build_prompt(summary_text, req.message, memory_block=memory_block)

    # 4. LLM 호출 
    result = llm(
        prompt,
        max_tokens=512,
        temperature=0.3,
        top_p=0.9,
        stop=[
            "---",
            "이 답변 구조를 따르세요",
            "좋은 운동이 되세요",
            "질문이 없을 경우",
            "\n\n좋았던 점:",   # 두 번째 블록 시작을 강제 stop
        ],
    )
    raw_text = result["choices"][0]["text"]
    answer_text = clean_answer(raw_text)
    
    # 5. 이번 Q&A를 MongoDB + 임베딩으로 저장 (RAG 지식 추가)
    try:
        save_qna_to_mongo(
            user_id=req.user_id,
            question=req.message,
            answer=answer_text,
        )
    except Exception as e:
        # 로그만 찍고, 사용자에게는 영향 없게
        print(f"[WARN] failed to save QnA to MongoDB: {e}")

    # 4. 응답 반환
    return ChatWithAnalysisResponse(
        answer=answer_text.strip(),
        used_summary=summary_text,
    )
