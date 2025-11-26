from sqlalchemy import create_engine
from config.sleep_config import settings

engine = create_engine(
    settings.POSTGRES_URL,
    pool_size = 10,
    max_overflow=20,
    pool_timeout=30,
    pool_pre_ping=True,
)

def get_engine():
    """다른 서비스에서 엔진을 가져갈 때 사용"""
    return engine