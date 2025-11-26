from pydantic import BaseModel, Field

# ex) typescript interface?
# react에서 사용자가 입력할 값
class CalorieRequest(BaseModel):
    duration_minutes: float = Field(..., description="운동 시간 (분)")
    weight_kg: float = Field(..., description="체중 (kg)")
    activity_type: str = Field(..., description="활동 종류 (예: Running, Cycling, Swimming)")
    bmi: float = Field(..., description="체질량지수")

class CalorieResponse(BaseModel):
    predicted_calories: float = Field(..., description="예측된 칼로리 소모량 (kcal/min)")

class CalorieLog(BaseModel):
    weightKg: float
    bmi: float
    activityType: str
    durationMinutes: float
    caloriesBurned: float

class LlmResponse(BaseModel):
    prompt: str
    advice: str