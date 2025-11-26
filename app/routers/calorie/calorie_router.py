from fastapi import APIRouter, HTTPException
from app.schema.calorie.calorie_schema import CalorieRequest, CalorieResponse, CalorieLog, LlmResponse
from app.services.calorie.calorie_service import predict_calories, llm_anaylze_calorie
from typing import List

router = APIRouter()

@router.post("/predict", response_model=CalorieResponse)
async def predict(data: CalorieRequest):
    """칼로리 예측 모델"""
    result = predict_calories(
        duration_minutes=data.duration_minutes,
        weight_kg=data.weight_kg,
        activity_type=data.activity_type,
        bmi=data.bmi
    )
    
    return {"predicted_calories": round(result * data.duration_minutes, 2)}

@router.post("/llm/analyze", response_model=LlmResponse)
async def analyze(logs: List[CalorieLog]):
    """Spring boot에서 DB데이터를 받아서 LLM이 분석"""
    result = llm_anaylze_calorie(logs)
    return  result
