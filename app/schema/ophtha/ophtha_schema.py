from pydantic import BaseModel
from typing import Optional, Literal, TypedDict

class ChatIn(BaseModel):
    session_id: str
    text: str

class ChatOut(BaseModel):
    session_id: str
    category: str
    advice: str

class AnalyzeOut(BaseModel):
    session_id: str
    disease: str
    image_summary: Optional[str]
    advice: str

# LangGraph 상태 구조
class GraphState(TypedDict, total=False):
    session_id: str
    input: str
    category: Literal["general", "cataract", "glaucoma"]
    disease: Optional[str]
    image_bytes: Optional[bytes]
    image_summary: Optional[str]
    advice: Optional[str]


