from pydantic import BaseModel

# 수면 예측 요청 시 사용하는 입력
class UserInput(BaseModel):
    age: int
    gender: int
    caffeine_mg: float
    sleep_hours: float
    physical_activity_hours: float
    alcohol_consumption: float

# 일반 챗봇 대화 요청 → email 기반으로 변경
class ChatRequest(BaseModel):
    email: str
    message: str

# 수면 챗봇(SleepChat) 요청 → email 기반으로 변경
class SleepChatRequest(BaseModel):
    email: str
    sleep_quality: float
    fatigue_score: float
    recommended_range: str
    message: str = "오늘 내 수면 상태 어때?"
