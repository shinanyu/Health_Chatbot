import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 공통 env 파일 로드
load_dotenv(".env")   # ← 공통 설정 로드 (DB, APP_NAME, SECRET_KEY 등)

from app.routers.calorie import calorie_router
from pydantic import BaseModel
from app.routers.sleep_router import sleep_api, sleepchat_api
from app.routers.stresscare.stress_router import router as stresscare_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #모든 출처 허용
    allow_credentials=True, # 인증정보(쿠키 등) 포함 가능
    allow_methods=["*"], # 모든 http 메서드 허용
    allow_headers=["*"], #모든 헤더 허용
)

# 라우터 등록
# app.include_router(predict.router, prefix="/predict", tags=["prediction"])
app.include_router(calorie_router.router, prefix="/calorie", tags=["calorie_prediction"])
app.include_router(sleep_api.router)
app.include_router(sleepchat_api.router)
app.include_router(stresscare_router)

# 운동 라우터 
# app.include_router()

# reload=True : 코드 변경 시 서버 자동 재시작
if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)  