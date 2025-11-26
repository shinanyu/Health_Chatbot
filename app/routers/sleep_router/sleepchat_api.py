from fastapi import APIRouter, HTTPException
from app.graphs.sleep_graph import (
    sleep_graph_general,
    sleep_graph_daily,
    sleep_graph_weekly
)
from app.services.sleep_services.user_service import get_member_no_by_email

router = APIRouter(prefix="/sleepchat")

# 일반 대화
@router.post("/message")
def chat_message(data: dict):

    email = data.get("email")
    message = data.get("message")

    if not email or not message:
        raise HTTPException(status_code=400, detail="email and message are required")

    # email → member_no 변환
    member_no = get_member_no_by_email(email)

    # LangGraph 호출
    result = sleep_graph_general.invoke({
        "req": {"email": email, "message": message},
        "email": email,
        "member_no": member_no,
    })

    return {"response": result["response"]}

# 일일 리포트
@router.get("/report/daily/{email}")
def daily_report(email: str):

    member_no = get_member_no_by_email(email)

    result = sleep_graph_daily.invoke({
        "req": {"email": email},
        "email": email,
        "member_no": member_no,
    })

    return {"report": result["response"]}


# 주간 리포트
@router.get("/report/weekly/{email}")
def weekly_report(email: str):

    member_no = get_member_no_by_email(email)

    result = sleep_graph_weekly.invoke({
        "req": {"email": email},
        "email": email,
        "member_no": member_no,
    })

    return {"report": result["response"]}
