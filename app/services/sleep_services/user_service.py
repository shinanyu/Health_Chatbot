from sqlalchemy import text
from datetime import date
from app.services.sleep_services.db_service import get_engine

engine = get_engine()

# ---------------------------------------------------------
# 1) email → member_no 변환
# ---------------------------------------------------------
def get_member_no_by_email(email: str):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT member_no
                FROM member_tb
                WHERE email = :email
            """),
            {"email": email}
        ).mappings().first()

    if not result:
        return None

    return result["member_no"]


# ---------------------------------------------------------
# 2) 회원 기본 정보 조회
# ---------------------------------------------------------
def get_user_info(member_no: int):
    if member_no is None:
        return {
            "name": "N/A",
            "gender": "N/A",
            "age": "N/A"
        }

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT name, gender, birth_date
                FROM member_tb
                WHERE member_no = :member_no
            """),
            {"member_no": member_no}
        ).mappings().first()

    if not result:
        return {
            "name": "N/A",
            "gender": "N/A",
            "age": "N/A"
        }

    birth = result["birth_date"]
    today = date.today()
    age = today.year - birth.year - (
        (today.month, today.day) < (birth.month, birth.day)
    )

    return {
        "name": result["name"],
        "gender": result["gender"],
        "age": age
    }


# ---------------------------------------------------------
# 3) 최신 일일 활동 조회
# ---------------------------------------------------------
def get_daily_activity(member_no: int):
    if member_no is None:
        return {}

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    sleep_hours,
                    predicted_fatigue_score,
                    recommended_sleep_range,
                    predicted_sleep_quality,
                    caffeine_mg,
                    alcohol_consumption,
                    physical_activity_hours
                FROM daily_activities_tb
                WHERE member_no = :member_no
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"member_no": member_no}
        ).mappings().first()

    return result or {}


# ---------------------------------------------------------
# 4) 최근 7일 활동 조회
# ---------------------------------------------------------
def get_weekly_activity(member_no: int):
    if member_no is None:
        return []

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT *
                FROM daily_activities_tb
                WHERE member_no = :member_no
                ORDER BY date DESC
                LIMIT 7
            """),
            {"member_no": member_no}
        ).mappings().all()

    return result
