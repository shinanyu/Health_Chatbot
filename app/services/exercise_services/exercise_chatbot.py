# exercise_chatbot.py
# 실행: uvicorn exercise_chatbot:app --host 0.0.0.0 --port 7000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from llama_cpp import Llama
import numpy as np

# ----------------------------------------
# FastAPI 설정
# ----------------------------------------
app = FastAPI(title="Exercise LLM Chatbot API")

# ----------------------------------------
# 요청 / 응답 모델
#   ※ main.py 에서 보내는 형식에 맞춰서 정의
#       {
#         "user_id": ...,
#         "message": ...,
#         "analysis": { ... 운동 분석 JSON ... }
#       }
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
)


# ----------------------------------------
# 작은 유틸: safe_mean
# ----------------------------------------
def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ----------------------------------------
# 분석 JSON → 요약 텍스트 생성 함수
#   main.py 의 analyze_video / analyze_frame 에서 만들어주는
#   analysis_out 구조에 맞게 작성
# ----------------------------------------
def summarize_exercise_json(data: Dict[str, Any]) -> str:
    """
    /analyze 에서 넘어온 analysis(JSON)를
    LLM이 이해하기 쉬운 한국어 텍스트 요약으로 변환.
    """
    # 텍스트만 들어온 경우 (video/image 없이)
    # main.py 에서 이 케이스는 analysis["message"] 에만 값이 있고
    # 나머지는 대부분 None 이 들어 있음.
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
너는 한국어로 답변하는 전문 퍼스널 트레이너이자 운동 분석 전문가야.
아래 '운동 분석 요약 정보'와 '사용자 질문'을 참고하여,
운동 폼과 수행 방식에 대해 전문적이고 명확하게 피드백해줘.

반드시 지켜야 할 규칙:
1. 폼의 좋은 점, 개선할 점, 잠재적인 부상 위험 요소를 구분해서 설명할 것.
2. 가능한 경우 ROM(가동 범위), 템포, 흔들림, 좌우 대칭 등의 수치를 활용해서 구체적으로 말할 것.
3. 분석 JSON에 없는 정보는 마음대로 추측하지 말고,
   "이 영상 정보만으로는 확인이 어렵습니다." 라고 말할 것.
4. 운동 초보자도 이해할 수 있도록, 너무 어려운 전문 용어는 피하고 친절한 한국어로 설명할 것.
"""

def build_prompt(summary: str, user_msg: str) -> str:
    """
    LLM에 넘길 최종 프롬프트 구성.
    - summary: 운동 분석 요약 텍스트
    - user_msg: 사용자의 질문(원문)
    """
    return f"""{BASE_SYSTEM_PROMPT}

[운동 분석 요약 정보]
{summary}

[사용자 질문]
{user_msg}

위 정보를 바탕으로, 사용자의 현재 운동 폼과 수행 방식에 대해
- 좋았던 점
- 개선이 필요한 점
- 주의해야 할 부상 위험 요소
를 중심으로 자세히 설명해줘.
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

    # 2. 프롬프트 구성
    prompt = build_prompt(summary_text, req.message)

    # 3. LLM 호출
    result = llm(
        prompt,
        max_tokens=512,
        stop=["</s>"],
    )
    answer_text = result["choices"][0]["text"].strip()

    # 4. 응답 반환
    return ChatWithAnalysisResponse(
        answer=answer_text,
        used_summary=summary_text,
    )
