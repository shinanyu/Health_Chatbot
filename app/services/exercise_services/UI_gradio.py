import json
import gradio as gr
import requests, uuid, base64, cv2, os
import numpy as np

user_id = str(uuid.uuid4())

# -------------------------------
# 이미지 / 동영상 / 텍스트 전송 함수(json으로 b64를 보내는 방식)
# -------------------------------
def send_message(message, image=None, video=None):
    payload = {"userId": user_id, "message": message}

    # 이미지가 있다면 (PIL → OpenCV → base64)
    if image is not None:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", image)
        payload["image"] = base64.b64encode(buffer).decode("utf-8")

    # 동영상이 있다면 (Gradio Video는 파일 경로를 반환함)
    elif video is not None:
        if isinstance(video, str) and os.path.exists(video):
            with open(video, "rb") as f:
                video_bytes = f.read()
            payload["video"] = base64.b64encode(video_bytes).decode("utf-8")
        else:
            return "⚠️ 동영상 파일을 읽을 수 없습니다."

    # FastAPI 서버 호출 (Spring 프록시를 거치는 경우 주소 유지)
    try:
        r = requests.post("http://localhost:8080/exercise/analyze", json=payload)
        data = r.json()
    except Exception as e:
        return f"❌ 서버 요청 실패: {e}"

    # --------------------------
    # FastAPI 응답 처리
    # --------------------------
    if isinstance(data, dict) and "detail" in data:
        return f"⚠️ 오류: {data['detail']}"

    text = []

    if "detected_exercise" in data:
        text.append(f"🏋️ 운동 종류: {data['detected_exercise']}")

    if "exercise_confidence" in data:
        text.append(f"📈 신뢰도: {data['exercise_confidence']:.2f}")

    if "stage" in data:
        text.append(f"📍 단계(Stage): {data['stage']}")

    if data.get("pose_detected"):
        joints = data["pose_data"].get("joints", {})
        back_angle = joints.get("back_angle", "?")
        text.append(f"💪 등 각도(back angle): {back_angle}°")

    # 동영상의 경우 프레임 요약 추가
    if "frames" in data and data["frames"]:
        text.append(f"🎞️ 분석된 프레임 수: {data.get('total_frames', len(data['frames']))}")
        example = data["frames"][0]
        text.append(f"🧩 예시 프레임 결과 → {example}")

    return "\n".join(text) if text else json.dumps(data, indent=2, ensure_ascii=False)

# -------------------------------
# 결과 보기 함수
# -------------------------------
def show_results():
    r = requests.get(f"http://localhost:8080/api/ai/results/{user_id}")
    data = r.json()
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, list):
        return "⚠️ 기록을 불러올 수 없습니다."
    return "\n\n".join([f"👤 {x.get('question','')}\n🤖 {x.get('answer','')}" for x in data])


# -------------------------------
# Gradio UI 구성
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI 운동 코치")
    gr.Markdown("이미지나 동영상을 업로드하면 AI가 운동 종류와 자세를 분석합니다 💪")

    with gr.Tab("💬 채팅하기"):
        msg = gr.Textbox(label="질문을 입력하세요", placeholder="예: 이 자세 어떤가요?")
        img = gr.Image(type="pil", label="운동 이미지 (선택사항)", sources=["upload", "webcam"])
        vid = gr.Video(label="운동 동영상 (선택사항)", sources=["upload"])
        btn = gr.Button("전송 🚀")
        out = gr.Textbox(lines=12, label="AI 분석 결과")
        btn.click(fn=send_message, inputs=[msg, img, vid], outputs=out)

    with gr.Tab("📜 내 기록 보기"):
        btn2 = gr.Button("기록 불러오기")
        out2 = gr.Textbox(lines=20, label="나의 대화 기록")
        btn2.click(fn=show_results, outputs=out2)

demo.launch(share=True)
