import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os

# 백내장 모델 경로
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ophtha", "cataract_resnet18_state_dict.pth")


# ResNet18 기반 모델 초기화
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # 2-class (normal / cataract)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def analyze_cataract(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
        pred = y.argmax(1).item()
        score = torch.softmax(y, dim=-1)[0, pred].item()

    label = "백내장 의심" if pred == 1 else "정상"
    summary = f"Local Cataract model (model.pth) prediction: {label} ({score:.2%})"
    advice = summary + "\n정확한 판단을 위해 안과 진료를 받으세요."
    return summary, advice
