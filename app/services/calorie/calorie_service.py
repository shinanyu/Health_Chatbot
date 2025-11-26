import os
import joblib
import pandas as pd
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(APP_DIR, "models", "calorie")

MODEL_PATH = os.path.join(MODEL_DIR, "calorie_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_dict.json")

# 4️⃣ 모델 로드
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ENCODER_PATH, 'r') as f:
    encoder_dict = json.load(f)


# 2) 환경변수 로드
load_dotenv(".env.calorie")

def predict_calories(duration_minutes: float, weight_kg: float, activity_type: str, bmi: float):
    # 모델을 사용해서 칼로리 소모량 예측
    data = pd.DataFrame({
        'duration_minutes': [duration_minutes],
        'weight_kg': [weight_kg],
        'activity_type': [activity_type],
        'bmi': [bmi]
    })
    
    # activity_type 인코딩
    if activity_type in encoder_dict['activity_type']:
        data['activity_type'] = [encoder_dict['activity_type'][activity_type]]
    else:
        data['activity_type'] = [encoder_dict['activity_type']['Unkown']]
    
    data_scaled = scaler.transform(data)
    
    # 예측 
    calorie_prediction = model.predict(data_scaled)[0]
    
    return round(float(calorie_prediction),2)

def llm_anaylze_calorie(logs):
    
    log_text = ""
    for l in logs:
        log_text += (
            f"운동종류 {l.activityType}, "
            f"운동시간 {l.caloriesBurned / 8:.0f}분, "  # 예시 변환 (칼로리 기반 임시 계산)
            f"칼로리 소모: {l.caloriesBurned:.1f} kcal, "
            f"당시 몸무게: {l.weightKg}kg\n"
        )
    
    prompt = f"""
        사용자의 최근 운동 데이터입니다.
        {log_text}
        
        위 데이터를 바탕으로,
        1. 운동 패턴 요약
        2. 칼로리 소모 추세
        4.15일 후, 30일 후 몸무게 예측
        을 한국어로 작성하고 깔끔하게 정리해주세요.
        """
    model = genai.GenerativeModel("gemini-2.5-flash")
    result = model.generate_content(prompt)
    advice = result.text.strip()
    return {"prompt": prompt, "advice": advice}



