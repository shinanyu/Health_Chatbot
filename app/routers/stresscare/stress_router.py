from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field

from app.services.stress_services.dl_emotion_service import EmotionDLService
from app.services.stress_services.ml_service import StressMLService
from app.services.stress_services.llm_service import StressLLMService

router = APIRouter(prefix="/stress", tags=["stresscare"])

_dl = EmotionDLService()
_ml = StressMLService()
_llm = StressLLMService()

# 공용 스키마
class ReportIn(BaseModel):
    sleepHours: Optional[float] = Field(None, description="전날 수면 시간(시간)")
    activityLevel: Optional[float] = Field(None, description="활동 지수(0~10)")
    caffeineCups: Optional[float] = Field(None, description="카페인 섭취(잔/일)")
    primaryEmotion: Optional[str] = Field("unknown", description="음성 분석 감정(선택)")
    comment: Optional[str] = Field(None, description="사용자 메모")

class ReportOut(BaseModel):
    stressScore: float
    primaryEmotion: Optional[str]
    coachingText: str
    meta: Dict[str, Any]

class ChatTurn(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    ml: Optional[Dict[str, Any]] = None
    dl: Optional[Dict[str, Any]] = None
    coaching: Optional[str] = ""
    history: Optional[List[ChatTurn]] = []
    question: str

class ChatOut(BaseModel):
    reply: str


@router.get("/health")
def health():
    return {"ok": True, "service": "stresscare", "status": "healthy"}

# 1) 단독 음성 감정 분석
@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        # dl_emotion_service 내부 구현에 맞게 호출
        # 예) predict_emotion_from_bytes / analyze 등 사용
        label, prob = _dl.predict_emotion_from_bytes(raw)
        return {"emotion": label, "confidence": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"audio error: {e}")


# 2-A) 통합 리포트(권장): FormData 한 번에 (음성 + 입력 + 메모)
#     프론트: FormData로 'audio', 'sleepHours', 'activityLevel', 'caffeineCups', 'comment' 전송
@router.post("/report", response_model=ReportOut)
async def make_report_form(
    audio: Optional[UploadFile] = File(None),
    sleepHours: Optional[float] = Form(None),
    activityLevel: Optional[float] = Form(None),
    caffeineCups: Optional[float] = Form(None),
    comment: Optional[str] = Form(""),
):
    # 1) 유효성
    if sleepHours is None or activityLevel is None or caffeineCups is None:
        raise HTTPException(
            status_code=422,
            detail="sleepHours, activityLevel, caffeineCups는 필수입니다.(FormData)",
        )

    # 2) ML 예측 (0~100)
    try:
        ml_input = {
            "sleep_duration": float(sleepHours),
            "physical_activity_level": float(activityLevel),
            "caffeine_cups": float(caffeineCups),
        }
        # 프로젝트 구현에 맞게 predict_* 호출
        stress_score = float(_ml.predict_as_score(ml_input))
        ml_meta = {
            "stress_score_0_100": stress_score,
            "model": getattr(_ml, "model_name", "stress_rf_model.pkl"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ML predict error: {e}")

    # 3) DL 감정 (옵션: 음성 있으면 분석, 없으면 unknown)
    primary_emotion = "unknown"
    try:
        if audio is not None:
            raw = await audio.read()
            primary_emotion, _prob = _dl.predict_emotion_from_bytes(raw)
    except Exception as e:
        # DL 실패해도 리포트는 계속 진행
        primary_emotion = "unknown"

    # 4) LLM 코칭 생성
    llm_err = None
    try:
        coaching = _llm.generate_coaching(
            ml_score=stress_score,
            emotion=primary_emotion,
            user_note=(comment or ""),
            ml_top_features=None,
            user_info={},
            context={},
        )
    except Exception as e1:
        try:
            payload = {
                "user": {},
                "context": {"note": comment or ""},
                "ml_stress": {"stress_score_0_100": stress_score},
                "dl_emotion": {"primary_emotion": primary_emotion},
                "app": {"name": "StressCare AI", "locale": "ko-KR", "version": "0.1.0"},
            }
            coaching = _llm.generate_coaching_with_payload(payload)
            llm_err = f"(first_call_error){e1}"
        except Exception as e2:
            coaching = (
                f"(LLM 폴백) 현재 스트레스 점수는 {round(stress_score, 1)}점입니다. "
                "3분 복식호흡과 10분 산책으로 긴장을 풀어보세요."
            )
            llm_err = f"{e1} | {e2}"

    return ReportOut(
        stressScore=stress_score,
        primaryEmotion=primary_emotion,
        coachingText=coaching,
        meta={
            "ml_used": True,
            "llm_error": llm_err,
            "note": comment,
            "ml_meta": ml_meta,
        },
    )

# 2-B) 통합 리포트(보조): JSON 바디로만(음성 사전 분석 or 미사용)
#      프론트: application/json 로 ReportIn 바디 전송-
@router.post("/report/json", response_model=ReportOut)
def make_report_json(body: ReportIn):
    if body.sleepHours is None or body.activityLevel is None or body.caffeineCups is None:
        raise HTTPException(
            status_code=422,
            detail="sleepHours, activityLevel, caffeineCups는 필수입니다.(JSON)",
        )

    # ML
    try:
        ml_input = {
            "sleep_duration": body.sleepHours,
            "physical_activity_level": body.activityLevel,
            "caffeine_cups": body.caffeineCups,
        }
        stress_score = float(_ml.predict_as_score(ml_input))
        ml_meta = {
            "stress_score_0_100": stress_score,
            "model": getattr(_ml, "model_name", "stress_rf_model.pkl"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ML predict error: {e}")

    # DL (프론트가 전달한 값 사용)
    primary_emotion = body.primaryEmotion or "unknown"

    # LLM
    llm_err = None
    try:
        coaching = _llm.generate_coaching(
            ml_score=stress_score,
            emotion=primary_emotion,
            user_note=(body.comment or ""),
            ml_top_features=None,
            user_info={},
            context={},
        )
    except Exception as e1:
        try:
            payload = {
                "user": {},
                "context": {"note": body.comment or ""},
                "ml_stress": {"stress_score_0_100": stress_score},
                "dl_emotion": {"primary_emotion": primary_emotion},
                "app": {"name": "StressCare AI", "locale": "ko-KR", "version": "0.1.0"},
            }
            coaching = _llm.generate_coaching_with_payload(payload)
            llm_err = f"(first_call_error){e1}"
        except Exception as e2:
            coaching = (
                f"(LLM 폴백) 현재 스트레스 점수는 {round(stress_score, 1)}점입니다. "
                "3분 복식호흡과 10분 산책으로 긴장을 풀어보세요."
            )
            llm_err = f"{e1} | {e2}"

    return ReportOut(
        stressScore=stress_score,
        primaryEmotion=primary_emotion,
        coachingText=coaching,
        meta={
            "ml_used": True,
            "llm_error": llm_err,
            "note": body.comment,
            "ml_meta": ml_meta,
        },
    )

# 3) LLM 챗봇 (대화 컨텍스트 + 히스토리)
@router.post("/chat", response_model=ChatOut)
def free_chat(body: ChatIn):
    """
    프론트는 report 결과를 기반으로:
    - ml: {"stress_score": 60.5} 등
    - dl: {"primary_emotion": "..."}
    - coaching: 리포트 전체 텍스트
    - history: [{"role":"user"/"assistant","content":"..."}]
    - question: 사용자의 후속 질문
    """
    try:
        # 1) 컨텍스트를 '프리앰블' 메시지로 한 번 주입
        ctx_json = {
            "ml": body.ml or {},
            "dl": body.dl or {},
            "coaching": body.coaching or ""
        }
        preamble = {
            "role": "user",
            "content": (
                "다음 JSON은 대화에 참고할 컨텍스트입니다. "
                "분석 결과를 기억한 상태로 자연스럽게 대화해 주세요.\n"
                f"{ctx_json}"
            ),
        }

        # 2) 기존 히스토리(있다면) 이어붙인 뒤, 마지막에 사용자의 질문
        msgs: List[Dict[str, str]] = [preamble]
        msgs.extend([{"role": t.role, "content": t.content} for t in (body.history or [])])
        msgs.append({"role": "user", "content": body.question})

        reply = _llm.chat(messages=msgs)  # <- llm_service.chat(...) 사용
        return ChatOut(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"chat error: {e}")