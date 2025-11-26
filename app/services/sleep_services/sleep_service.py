import os 
import joblib
import pandas as pd
from app.models.sleep_models.sleepSchema import UserInput

MODEL_PATH = "app/models/sleep_models/best_sleep_quality_rf_bundle.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]

#수면 시간 추천
def rule_based_sleep_recommendation(
    sleep_hours: float,
    physical_activity_hours: float,
    caffeine_mg: float,
    alcohol_consumption: float,
    fatigue_score: float,
) -> tuple[float, float]:
    """
    사용자의 생활 패턴을 기반으로 최적 수면시간 범위를 추천
    """
    
    # 기본 추천 범위(기본값)
    min_optimal, max_optimal = 7.0, 7.5
    
    # 수면 충분 + 활동 적음 + 카페인/알코올 거의 없음 → 6.5~7.0
    if sleep_hours >= 7.5 and physical_activity_hours < 1.0 and caffeine_mg < 100 and alcohol_consumption < 0.5:
        min_optimal, max_optimal = 6.5, 7.0

    #평균 수준 → 7.0~7.5
    elif 6.5 <= sleep_hours <= 7.5 and 1.0 <= physical_activity_hours <= 2.0 and 100 <= caffeine_mg <= 250:
        min_optimal, max_optimal = 7.5, 8.0

    # 활동 많음 / 카페인 or 알코올 높음 / 수면 부족 → 7.5~8.5
    elif sleep_hours < 6.5 or physical_activity_hours > 2.0 or caffeine_mg > 250 or alcohol_consumption > 1.0:
        min_optimal, max_optimal = 8.0, 8.5

    # 피로도 점수가 높을 때 (예: 60 이상) → 약간 더 수면 필요
    if fatigue_score >= 60:
        min_optimal += 0.5
        max_optimal += 0.5

    # 최대 9시간 제한
    max_optimal = min(max_optimal, 9.0)

    return round(min_optimal, 1), round(max_optimal, 1)

#피로토 예측
def predict_fatigue(data: UserInput):
    df = pd.DataFrame([data.model_dump()])[columns]
    X_scaled = scaler.transform(df)
    sleep_quality = float(model.predict(X_scaled)[0])

    # sleep_quality (1~4) → 피로도 (0~100) 변환
    fatigue_score = round(100 * (4 - sleep_quality) / 3, 2)

    # 컨디션 등급
    if fatigue_score < 25:
        condition = "좋음" 
    elif fatigue_score < 50:
        condition = "보통"
    elif fatigue_score < 75:
        condition = "나쁨"
    else:
        condition = "최악"

    return {
        "predicted_sleep_quality": round(sleep_quality, 3),
        "predicted_fatigue_score": fatigue_score,
        "condition_level": condition
    }
