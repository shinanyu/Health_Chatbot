import pandas as pd
from fastapi import APIRouter
from app.models.sleep_models.sleepSchema import UserInput
from app.services.sleep_services.sleep_service import (
    predict_fatigue,
    rule_based_sleep_recommendation,
    model,
    scaler,
    columns,
)

router = APIRouter(prefix="/sleep", tags=["sleep"])

# 피로도 점수 예측
@router.post("/predict-fatigue")
async def predict_fatigue_endpoint(data: UserInput):
    """
    사용자 입력 데이터를 받아 피로도 점수를 예측합니다.
    """
    return predict_fatigue(data)


# 수면 시간 추천
@router.post("/recommend")
def recommend_rule_based(data: UserInput):
    df = pd.DataFrame([data.model_dump()])[columns]
    X_scaled = scaler.transform(df)

    # 예측 수행
    sleep_quality = float(model.predict(X_scaled)[0])
    fatigue_score = round(100 * (4 - sleep_quality) / 3, 2)

    # 추천 수면 시간 계산
    min_h, max_h = rule_based_sleep_recommendation(
        sleep_hours=data.sleep_hours,
        physical_activity_hours=data.physical_activity_hours,
        caffeine_mg=data.caffeine_mg,
        alcohol_consumption=data.alcohol_consumption,
        fatigue_score=fatigue_score,
    )

    # 결과 반환
    return {
        "predicted_sleep_quality": round(sleep_quality, 3),
        "predicted_fatigue_score": fatigue_score,
        "recommended_sleep_range": f"{min_h} ~ {max_h} 시간",
    }
