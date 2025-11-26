import json
import matplotlib.pyplot as plt

# -----------------------------
# 1. JSON 파일 읽기
# -----------------------------
with open("history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

# history 안에 들어있을 key 예:
# ['accuracy', 'loss', 'val_accuracy', 'val_loss']

acc = history.get("accuracy", [])
val_acc = history.get("val_accuracy", [])
loss = history.get("loss", [])
val_loss = history.get("val_loss", [])

epochs = range(1, len(acc) + 1)

# -----------------------------
# 2. 정확도 그래프
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 3. 손실 그래프
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
