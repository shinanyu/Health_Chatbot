import torch
from PIL import Image
import io
from app.utils.ophtha.hf_vision_loader import load_glaucoma_model

_processor, _model = load_glaucoma_model()

def analyze_glaucoma(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    proc = _processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**proc).logits
        pred_idx = int(logits.argmax(-1).item())
    label = _model.config.id2label.get(pred_idx, str(pred_idx))
    score = float(torch.softmax(logits, dim=-1)[0, pred_idx].item())
    summary = f"Glaucoma model prediction: {label} ({score:.2%})"
    advice = summary + "\n정확한 판단을 위해 안과 진료를 받으세요."
    return summary, advice
