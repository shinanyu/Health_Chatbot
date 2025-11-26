from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.schema.ophtha.ophtha_schema import GraphState
from app.services.ophtha.llm.chain import llm

# ------------------- Nodes -------------------
# 증상 분류 노드
def route_symptom(state: GraphState) -> GraphState:
    question = state.get("input", "")

    route_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an ophthalmology triage assistant.
Classify the user's symptom into one of the following categories:
- cataract: symptoms like blurred vision, cloudy vision, difficulty seeing at night, glare around lights, faded colors.
- glaucoma: symptoms like loss of peripheral vision (tunnel vision), severe eye pain, seeing halos, nausea or vomiting with eye pain.
- general: anything else that is mild, temporary, or unrelated to cataract or glaucoma.

Respond with only one word: "cataract", "glaucoma", or "general".
"""),
        ("human", "{question}")
    ])

    out = llm.invoke(route_prompt.format_messages(question=question))
    cat = out.content.strip().lower()

    # 안전장치
    if cat not in {"general", "cataract", "glaucoma"}:
        cat = "general"

    state["category"] = cat
    return state

def general_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an ophthalmology health assistant. Provide safe, non-diagnostic advice."),
        ("human", "Symptom description: {question}")
    ])
    question = state.get("input", "")
    out = llm.invoke(prompt.format_messages(question=question))
    state["advice"] = out.content
    return state


def cataract_gate(state: GraphState) -> GraphState:
    state["disease"] = "cataract"
    state["advice"] = (
        "백내장 증상과 비슷해보입니다. 백내장 판단을 해보시겠습니까?\n"
        "안저(눈바닥) 사진을 업로드하시면 모델로 분석해드릴게요.\n"
        "원치 않으시면 ‘안할래’라고 답해주세요."
    )
    return state


def glaucoma_gate(state: GraphState) -> GraphState:
    state["disease"] = "glaucoma"
    state["advice"] = (
        "녹내장과 증상이 비슷해보입니다. 판단을 위해 시신경 사진이 필요합니다.\n"
        "안과에서 받은 시신경 사진을 업로드해주세요.\n"
        "원치 않으시면 ‘안할래’라고 답해주세요."
    )
    return state


def declined_imaging(state: GraphState) -> GraphState:
    state["advice"] = "정확한 증상 판단을 위해 안과에서 진료를 받으세요."
    return state

# ------------------- Graph -------------------

builder = StateGraph(GraphState)

builder.add_node("route_symptom", route_symptom)
builder.add_node("general", general_node)
builder.add_node("cataract_gate", cataract_gate)
builder.add_node("glaucoma_gate", glaucoma_gate)
builder.add_node("declined", declined_imaging)

builder.set_entry_point("route_symptom")

builder.add_conditional_edges(
    "route_symptom",
    lambda s: s.get("category", "general"),
    {
        "general": "general",
        "cataract": "cataract_gate",
        "glaucoma": "glaucoma_gate",
    },
)

builder.add_edge("general", END)
builder.add_edge("cataract_gate", END)
builder.add_edge("glaucoma_gate", END)
builder.add_edge("declined", END)

memory = MemorySaver()
workflow = builder.compile(checkpointer=memory)
