import os, glob, math, random, collections
from typing import List, Tuple
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# =========================
# 설정
# =========================
DATASET_ROOT = r"./dataset"
IMG_SIZE     = (160, 160)
SEQ_LEN      = 24
BATCH_SIZE   = 8
EPOCHS       = 25
VAL_SPLIT    = 0.2
SEED         = 42
LR_BASE      = 3e-4
BACKBONE     = "EfficientNetB0"

AUG_FLIP_PROB       = 0.5
AUG_BRIGHT_DELTA    = 0.10
AUG_CONTRAST_RANGE  = (0.9, 1.1)
AUG_ROT_DEG         = 8

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 시퀀스 인덱싱
# =========================
def list_sequence_dirs_and_labels(root_dir: str):
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    classes = sorted([d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))])

    seq_paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        used = False

        subdirs = sorted([d for d in os.listdir(cls_dir)
                          if os.path.isdir(os.path.join(cls_dir, d))])
        for sd in subdirs:
            cand = os.path.join(cls_dir, sd)
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                imgs.extend(glob.glob(os.path.join(cand, ext)))
            if imgs:
                seq_paths.append(cand)
                labels.append(cls)
                used = True

        vids = [p for p in glob.glob(os.path.join(cls_dir, "*"))
                if os.path.isfile(p) and p.lower().endswith(video_exts)]
        for v in vids:
            seq_paths.append(v)
            labels.append(cls)
            used = True

        if not used:
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                imgs.extend(glob.glob(os.path.join(cls_dir, ext)))
            if imgs:
                seq_paths.append(cls_dir)
                labels.append(cls)

    return seq_paths, labels

# =========================
# 시퀀스 로딩
# =========================
def _sample_indices(n, k):
    if n <= 0:
        return np.zeros((k,), dtype=int)
    if n >= k:
        return np.linspace(0, n - 1, num=k).astype(int)
    base = np.arange(n)
    pad  = np.full((k - n,), n - 1)
    return np.concatenate([base, pad])

def _load_frames_from_dir(dir_path):
    paths = sorted([p for p in glob.glob(os.path.join(dir_path, "*.*"))
                    if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
    return paths

def _resize_np(img, size):
    return tf.image.resize(img, size).numpy().astype(np.float32)

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
                with Image.open(img_paths[i]) as im:
                    im = im.convert("RGB")
                    frames.append(_resize_np(np.array(im), img_size) / 255.0)
        return np.stack(frames, axis=0)

    video_exts = (".mp4",".avi",".mov",".mkv")
    if os.path.isfile(seq_path) and seq_path.lower().endswith(video_exts):
        import cv2
        cap = cv2.VideoCapture(seq_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = _sample_indices(total, seq_len)
        frames = []
        cur = -1
        for target in idxs:
            if cur != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if not ok:
                frames.append(np.zeros((img_size[0], img_size[1], 3), dtype=np.float32))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = _resize_np(frame, img_size) / 255.0
            frames.append(frame)
            cur = target
        cap.release()
        return np.stack(frames, axis=0)

    return np.zeros((seq_len, img_size[0], img_size[1], 3), dtype=np.float32)

# =========================
# 증강
# =========================
def augment_sequence(frames: np.ndarray) -> np.ndarray:
    T = frames.shape[0]
    x = tf.convert_to_tensor(frames)
    if random.random() < AUG_FLIP_PROB:
        x = tf.image.flip_left_right(x)
    if AUG_BRIGHT_DELTA > 0:
        delta = random.uniform(-AUG_BRIGHT_DELTA, AUG_BRIGHT_DELTA)
        x = tf.clip_by_value(x + delta, 0.0, 1.0)
    lo, hi = AUG_CONTRAST_RANGE
    if hi - lo > 1e-6:
        fac = random.uniform(lo, hi)
        mean = tf.reduce_mean(x, axis=[1,2,3], keepdims=True)
        x = tf.clip_by_value((x - mean) * fac + mean, 0.0, 1.0)
    return x.numpy()

# =========================
# 제너레이터
# =========================
class FrameSequenceGenerator(Sequence):
    def __init__(self, seq_paths: List[str], labels: np.ndarray,
                 batch_size: int, seq_len: int, img_size=(160,160),
                 shuffle=True, training=True, preprocess=None):
        self.seq_paths = seq_paths
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.img_size = img_size
        self.shuffle = shuffle
        self.training = training
        self.preprocess = preprocess
        self.indices = np.arange(len(seq_paths))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.seq_paths) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = [], []
        for i in batch_idx:
            arr = load_sequence_fixed(self.seq_paths[i], self.seq_len, self.img_size)
            if self.training:
                arr = augment_sequence(arr)
            if self.preprocess is not None:
                arr = self.preprocess(arr * 255.0)
            X.append(arr)
            y.append(self.labels[i])
        X = np.stack(X, axis=0)
        y = np.array(y)
        return X, y

# =========================
# Temporal Attention Layer
# =========================
class TemporalAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)

    def call(self, x):
        h = self.dense(x)
        e = self.score(h)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(a * x, axis=1)
        return context

# =========================
# 모델 빌드
# =========================
def build_cnn_lstm_attention(frame_shape=(160,160,3), seq_len=24, num_classes=5, backbone="EfficientNetB0"):
    if backbone == "EfficientNetB0":
        cnn = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=frame_shape, pooling="avg")
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif backbone == "ResNet50":
        cnn = tf.keras.applications.ResNet50(include_top=False, input_shape=frame_shape, pooling="avg")
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError("지원 백본: EfficientNetB0 / ResNet50")

    cnn.trainable = False

    inputs = layers.Input(shape=(seq_len,) + frame_shape)
    x = layers.TimeDistributed(cnn)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = TemporalAttention(units=128)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_BASE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, preprocess, cnn

# =========================
# Class Weight 계산
# =========================
def compute_class_weight(labels: np.ndarray):
    cnt = collections.Counter(labels.tolist())
    maxc = max(cnt.values())
    return {cls: maxc/cnt[cls] for cls in cnt}

# =========================
# 메인
# =========================
def main():
    print("[DEBUG] DATASET_ROOT:", os.path.abspath(DATASET_ROOT))

    seq_paths, label_names = list_sequence_dirs_and_labels(DATASET_ROOT)
    if len(seq_paths) == 0:
        raise RuntimeError("시퀀스를 찾지 못했습니다. 폴더 구조/확장자를 확인하세요.")

    le = LabelEncoder()
    y = le.fit_transform(label_names)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    print(f"[INFO] classes ({num_classes}): {class_names}")
    print(f"[INFO] total sequences: {len(seq_paths)}")

    X_train, X_val, y_train, y_val = train_test_split(
        seq_paths, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y
    )

    model, preprocess, backbone = build_cnn_lstm_attention(
        frame_shape=IMG_SIZE+(3,), seq_len=SEQ_LEN, num_classes=num_classes, backbone=BACKBONE
    )
    model.summary()

    train_gen = FrameSequenceGenerator(X_train, y_train, BATCH_SIZE, SEQ_LEN, IMG_SIZE,
                                       shuffle=True, training=True, preprocess=preprocess)
    val_gen   = FrameSequenceGenerator(X_val, y_val, BATCH_SIZE, SEQ_LEN, IMG_SIZE,
                                       shuffle=False, training=False, preprocess=preprocess)

    # --------------------------
    # 5) 콜백 — 현재 경로에 저장
    # --------------------------
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_cnn_lstm_model.h5",   # 최고 정확도 모델 (현재 디렉토리)
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True
    )

    cw = compute_class_weight(y_train)

    # --------------------------
    # 1단계: 백본 freeze
    # --------------------------
    warmup_epochs = 5
    print(f"\n[Stage 1] Backbone frozen | epochs={warmup_epochs}")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=warmup_epochs,
        class_weight=cw,
        callbacks=[ckpt, es],
        verbose=1
    )

    # --------------------------
    # 2단계: fine-tuning
    # --------------------------
    print("\n[Stage 2] Unfreezing backbone and fine-tuning ...")
    backbone.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_epochs = EPOCHS - warmup_epochs
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        class_weight=cw,
        callbacks=[ckpt, es],
        verbose=1
    )

    print("\n[INFO] Saving merged history (Stage1 + Stage2) ...")

    full_history = {}
    for key in history1.history.keys():
        full_history[key] = history1.history[key] + history2.history.get(key, [])

    # 파일명 원하는 걸로 바꾸면 됨.
    with open("history.json", "w", encoding="utf-8") as f:
        import json
        json.dump(full_history, f, indent=2)

    print("📈 Saved: history.json (training curves)")

    # --------------------------
    # 최종 모델 저장 (현재 경로)
    # --------------------------
    model.save("cnn_lstm_model.h5")   # ✅ 같은 위치에 저장
    with open("labels.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    print("✅ Saved: best_cnn_lstm_model.h5, cnn_lstm_model.h5, labels.txt")

if __name__ == "__main__":
    main()
