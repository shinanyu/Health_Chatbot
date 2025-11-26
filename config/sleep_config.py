import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(".env.sleep", override=True)

class Settings:
    """환경변수를 불러와 전역적으로 관리하는 클래스"""

    APP_NAME = os.getenv("APP_NAME", "SleepTalk AI")
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_PORT = int(os.getenv("APP_PORT", 8000))

    POSTGRES_URL = os.getenv("POSTGRES_URL")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")

settings = Settings()
