"""
llm_service.py (StressCare LLM 서비스 - Groq/OpenAI 호환)
- 네가 제공한 stresscare_llm.py 내용을 FastAPI 서비스 클래스 형태로 재구성
- 보고서(코칭) + 채팅 둘 다 지원
- 환경변수:
  LLM_API_KEY, LLM_BASE_URL(기본 https://api.groq.com/openai/v1), LLM_MODEL(기본 llama-3.1-8b-instant)
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env.stress")

def _clean_repetitive_sections(text: str) -> str:
    """
    반복 문장 줄이기:
    - 같은 줄 반복 제거
    - '예를 들어/예를 들면' 연속 중복 제거
    - 앞부분(15글자) 동일 문장 연속 시 첫 문장만 유지
    """
    if not text:
        return text
    lines = text.splitlines()
    seen = set()
    cleaned: List[str] = []
    prev_prefix = None
    for line in lines:
        norm = re.sub(r"\s+", " ", line).strip()
        if norm == "":
            cleaned.append(line)
            prev_prefix = None
            continue
        if norm.startswith(("예를 들어", "예를 들면")):
            if cleaned:
                last_norm = re.sub(r"\s+", " ", cleaned[-1]).strip()
                if last_norm.startswith(("예를 들어", "예를 들면")):
                    continue
        if norm in seen:
            continue
        seen.add(norm)
        prefix = norm[:15]
        if prev_prefix is not None and prefix == prev_prefix:
            continue
        cleaned.append(line)
        prev_prefix = prefix
    return "\n".join(cleaned).strip()


# ===== 프롬프트들 (원본 규칙 반영) =====
_SYSTEM_PROMPT = """너는 개인 사용자의 스트레스와 감정을 함께 살펴보고,
부드럽고 따뜻한 말투로 현실적인 조언을 해주는 "헬스케어 감정 & 스트레스 코치" AI이다.

답변은 상담사처럼 진심 어린 어조로, 지나치게 형식적이지 않게 작성한다.
무조건 위로하거나 훈계하지 말고, 사용자의 감정에 공감하면서 작고 실천 가능한 제안을 중심으로 이야기한다.
문장은 짧고 자연스럽게, 사람의 말처럼 이어지게 써라.

입력으로 JSON 형식의 데이터를 받는다.
- ml_stress: 머신러닝 스트레스 예측 결과
- dl_emotion: 딥러닝 음성 감정 분석 결과
- user, context: 사용자의 기본 정보와 상황 설명

[반드시 지켜야 할 규칙]
1) 한국어로만 작성. 2) 공감적 톤. 3) 부정 감정/고스트레스면 3~5개 실천 팁.
4) 의료 진단/약명/처방 금지. 쉬운 한국어 사용.
5) "오늘의 한마디" 섹션: 짧은 사자성어(네 글자+괄호 표기) 또는 짧은 한국어 명언/속담 1개 + 한줄 해석.
6) 운동 권할 땐 무리 금지 문장 동반. 7) 과장된 상투문 금지. 8) 반복문장 금지.

[출력 형식]
현재 상태 요약
- 1~2문장

오늘의 한마디
"사자성어 또는 짧은 명언"
한 줄 해석

실천 팁
• 3~5개 불릿

요약 한 줄
- 핵심 메시지 한 문장
"""

_USER_PROMPT_TMPL = """다음은 한 사용자의 스트레스 예측 결과와 음성 감정 분석 결과이다.
이 정보를 바탕으로 위 System 안내에 나온 형식 그대로 코칭 멘트를 작성해줘.

```json
{json_block}
```"""

_CHAT_SYSTEM_PROMPT = """
너는 사용자의 스트레스와 감정을 이미 분석한 상태에서,
친구처럼 따뜻하고 자연스럽게 대화를 이어가는 "헬스케어 감정 & 스트레스 코치"이다.

- 보고서 형식 금지, 자연스러운 대화체
- 3~5문장 이내, 공감 + 작고 구체적 제안
- 필요하면 한 줄 명언/사자성어를 가볍게 덧붙여도 됨
- 운동 권유 시 무리 금지 문장 동반
"""


def _build_payload(
    user_info: Dict[str, Any],
    context: Dict[str, Any],
    ml_stress: Dict[str, Any],
    dl_emotion: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM 입력용 JSON payload 생성"""
    return {
        "user": user_info,
        "context": context,
        "ml_stress": ml_stress,
        "dl_emotion": dl_emotion,
        "app": {"name": "StressCare AI", "locale": "ko-KR", "version": "0.1.0"},
    }

class StressLLMService:
    """
    - 보고서(코칭) 생성: generate_coaching(...)
    - 채팅 응답 생성: chat(...)
    - 라우터에서 간단 사용을 위해 ml_score/감정만으로도 코칭 가능 (간이 payload 자동 구성)
    """

    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("❌ .env에 LLM_API_KEY가 없습니다.")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # --------- 간이 코칭(라우터 편의용) ----------
    def generate_coaching(
        self,
        ml_score: float,
        emotion: str,
        user_note: str = "",
        *,
        ml_top_features: Optional[List[Dict[str, Any]]] = None,
        user_info: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        debug: bool = False,
    ) -> str:
        """
        /stress/report 에서 바로 호출하기 좋은 간이 API.
        - ml_score(0~100), emotion(str), user_note(str)만으로 payload 자동 구성
        - 필요하면 ml_top_features / user_info / context 도 추가 가능
        """
        # 범주화(낮음/보통/높음)
        level_ko, level_en = self._level_from_score(ml_score)

        ml = {
            "stress_score_0_100": float(ml_score),
            "stress_score_raw": round(float(ml_score) / 10.0, 2),  # 대략 0~10 스케일
            "stress_level": level_en,
            "stress_level_ko": level_ko,
            "stress_comment": self._comment_from_level(level_ko),
            "top_features": ml_top_features or [],
            "model_meta": {"model_type": "random_forest_bundle"},
        }
        dl = {
            "primary_emotion": str(emotion),
            "confidence": None,
            "probabilities": None,
            "model_meta": {"model_type": "cnn_bilstm_hybrid"},
        }
        ui = user_info or {"notes": user_note}
        ctx = context or {}

        payload = _build_payload(ui, ctx, ml, dl)
        return self._generate_coaching_text(payload, temperature, max_tokens, debug)

    # --------- 풀 payload 코칭 ----------
    def generate_coaching_with_payload(
        self,
        payload: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 1200,
        debug: bool = False,
    ) -> str:
        return self._generate_coaching_text(payload, temperature, max_tokens, debug)

    # --------- 채팅 ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str:
        """
        messages 예:
        [{"role":"user","content":"실외에서 할 수 있는 운동 추천해줘"}]
        이전 대화가 있으면 user/assistant 순서로 이어 붙여 전달.
        """
        full_messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
        full_messages.extend(messages)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _clean_repetitive_sections(resp.choices[0].message.content.strip())

    # ===== 내부 구현 =====
    def _generate_coaching_text(
        self,
        payload: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        debug: bool,
    ) -> str:
        if debug:
            print("[DEBUG] Payload]", json.dumps(payload, ensure_ascii=False, indent=2))

        user_prompt = _USER_PROMPT_TMPL.format(
            json_block=json.dumps(payload, ensure_ascii=False, indent=2)
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        return _clean_repetitive_sections(raw)

    @staticmethod
    def _level_from_score(score: float) -> (str, str):
        """0~100 스코어 → ('낮음'|'보통'|'높음', 'low'|'medium'|'high')"""
        s = float(score)
        if s <= 33:
            return "낮음", "low"
        if s <= 66:
            return "보통", "medium"
        return "높음", "high"

    @staticmethod
    def _comment_from_level(level_ko: str) -> str:
        if level_ko == "낮음":
            return "일상적인 수준의 스트레스예요. 현재 페이스를 유지해봐요."
        if level_ko == "보통":
            return "스트레스가 조금 쌓여 있어요. 휴식 루틴을 가볍게 넣어볼까요?"
        return "스트레스가 높은 편이에요. 잠·식사·움직임을 작은 목표로 다시 세워봐요."