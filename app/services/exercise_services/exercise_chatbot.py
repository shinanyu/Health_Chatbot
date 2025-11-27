# exercise_chatbot.py
# 실행: uvicorn exercise_chatbot:app --host 0.0.0.0 --port 7000

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from llama_cpp import Llama
import numpy as np
import re

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
    repeat_penalty=1.3,      # repeat_penalty는 1.2~1.4 권장 : 1.25~1.3은 “방금 쓴 문장 반복 금지 효과” 
    repeat_last_n=256,       # 반복 억제
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


def build_prompt(summary: str, user_msg: str) -> str:
    print(summary)
    return f"""{BASE_SYSTEM_PROMPT}

[운동 분석 요약]
{summary}

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

    # 2. 프롬프트 구성
    prompt = build_prompt(summary_text, req.message)

    # 3. LLM 호출
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

    # 4. 응답 반환
    return ChatWithAnalysisResponse(
        answer=answer_text.strip(),
        used_summary=summary_text,
    )
