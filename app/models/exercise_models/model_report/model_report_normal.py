# eval_cnn_lstm.py
import os, glob, math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# 설정
# ------------------------------
DATASET_ROOT = "../dataset"
MODEL_PATH   = "../best_cnn_lstm_model.h5"  
LABEL_PATH   = "../labels.txt"

IMG_SIZE     = (160, 160)
SEQ_LEN      = 24
BATCH_SIZE   = 8

BACKBONE     = "EfficientNetB0"           # 학습 때와 동일하게!
SEED         = 42
TEST_SPLIT   = 0.15

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# 커스텀 레이어
# ------------------------------
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class TemporalAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)
    def call(self, x):
        h = self.dense(x)
        e = self.score(h)            # (B,T,1)
        a = tf.nn.softmax(e, axis=1) # (B,T,1)
        return tf.reduce_sum(a * x, axis=1)
    def get_config(self):
        cfg = super().get_config(); cfg.update({"units": self.units}); return cfg

# ------------------------------
# 유틸: 데이터 로딩
# ------------------------------
def _sample_indices(n, k):
    if n <= 0: return np.zeros((k,), dtype=int)
    if n >= k: return np.linspace(0, n - 1, num=k).astype(int)
    base = np.arange(n); pad = np.full((k - n,), n - 1)
    return np.concatenate([base, pad])

def _resize_np(img, size):
    return tf.image.resize(img, size).numpy().astype(np.float32)

def _load_frames_from_dir(dir_path):
    paths = sorted([p for p in glob.glob(os.path.join(dir_path, "*.*"))
                    if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))])
    return paths

def load_sequence_fixed(seq_path: str, seq_len: int, img_size=(160,160)) -> np.ndarray:
    import cv2
    if os.path.isdir(seq_path):
        img_paths = _load_frames_from_dir(seq_path)
        idxs = _sample_indices(len(img_paths), seq_len)
        frames = []
        for i in idxs:
            if len(img_paths) == 0:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
            else:
                from PIL import Image
                with Image.open(img_paths[i]) as im:
                    im = im.convert("RGB")
                    frames.append(_resize_np(np.array(im), img_size) / 255.0)
        return np.stack(frames, axis=0)

    video_exts = (".mp4",".avi",".mov",".mkv",".wmv")
    if os.path.isfile(seq_path) and seq_path.lower().endswith(video_exts):
        import cv2
        cap = cv2.VideoCapture(seq_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_indices(total, seq_len)
        frames, cur = [], -1
        for target in idxs:
            if cur != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)); continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _resize_np(frame, img_size) / 255.0
            frames.append(frame); cur = target
        cap.release()
        return np.stack(frames, axis=0)

    return np.zeros((seq_len, img_size[0], img_size[1], 3), dtype=np.float32)

class FrameSequenceGenerator(Sequence):
    def __init__(self, seq_paths, labels, batch_size, seq_len, img_size=(160,160),
                 preprocess=None):
        self.seq_paths = list(seq_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.preprocess = preprocess
        self.indices = np.arange(len(self.seq_paths))
    def __len__(self):
        return math.ceil(len(self.seq_paths) / self.batch_size)
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = [], []
        for i in batch_idx:
            arr = load_sequence_fixed(self.seq_paths[i], self.seq_len, self.img_size)
            if self.preprocess is not None:
                arr = self.preprocess(arr * 255.0)
            X.append(arr); y.append(self.labels[i])
        return np.stack(X, axis=0), np.array(y)

def list_sequence_dirs_and_labels(root_dir):
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    seq_paths, labels = [], []
    for cls in sorted(os.listdir(root_dir)):
        cdir = os.path.join(root_dir, cls)
        if not os.path.isdir(cdir) or cls.startswith("."): continue
        # 동영상
        for f in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, f)
            if os.path.isfile(p) and f.lower().endswith(video_exts):
                seq_paths.append(p); labels.append(cls)
        # 프레임 폴더
        for sub in sorted(os.listdir(cdir)):
            sp = os.path.join(cdir, sub)
            if os.path.isdir(sp) and not sub.startswith("."):
                if _load_frames_from_dir(sp):
                    seq_paths.append(sp); labels.append(cls)
    return seq_paths, labels

# ------------------------------
# labels.txt 순서로 인코딩
# ------------------------------
def encode_with_label_file(names, label_file_path):
    with open(label_file_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    idx_map = {name: i for i, name in enumerate(classes)}
    y = np.array([idx_map[n] for n in names])   # 존재하지 않으면 KeyError 발생
    return y, classes

def get_preprocess(backbone):
    if backbone == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    elif backbone == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError("지원 백본: EfficientNetB0 / ResNet50")

# ------------------------------
# 메인
# ------------------------------
def main():
    print("[CWD]", os.getcwd())

    # 1) 시퀀스 & 라벨명
    seq_paths, label_names = list_sequence_dirs_and_labels(DATASET_ROOT)
    print(f"[INFO] Found {len(seq_paths)} sequences.")

    # 2) labels.txt 순서로 y 인코딩
    y_all, class_names = encode_with_label_file(label_names, LABEL_PATH)
    num_classes = len(class_names)
    print(f"[INFO] Loaded labels ({num_classes}): {class_names}")

    # 3) 단순 평가용 split (랜덤)
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        seq_paths, y_all, test_size=TEST_SPLIT, random_state=SEED, stratify=y_all
    )

    # 4) 모델 로드
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    print(f"[INFO] Model loaded from {MODEL_PATH}")

    # 5) 제너레이터 (전처리 동일하게)
    preprocess = get_preprocess(BACKBONE)
    test_gen = FrameSequenceGenerator(
        X_test, y_test, BATCH_SIZE, SEQ_LEN, IMG_SIZE, preprocess=preprocess
    )

    # 6) 평가
    print("\n[TEST] Evaluating on test set ...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 7) 리포트/혼동행렬 (원하면 주석 처리 OK)
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    ys = []
    for i in range(len(test_gen)):
        _, yb = test_gen[i]
        ys.append(yb)
    y_true = np.concatenate(ys, axis=0)[:len(y_pred)]

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open("test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    np.savetxt("test_confusion_matrix.txt", cm, fmt="%d")

    print("\n[TEST] Classification Report\n", report)
    print("[TEST] Confusion Matrix\n", cm)

if __name__ == "__main__":
    main()
