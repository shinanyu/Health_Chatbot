# -*- coding: utf-8 -*-
"""
dl_emotion_service.py (Runtime-only)
- FastAPI에서 업로드한 음성 바이트를 받아 CNN+BiLSTM으로 감정 예측
- 학습/데이터셋 스캔 없음. 모델(.h5)과 라벨(.npy)만 로드
- 환경변수(선택):
    EMOTION_MODEL_PATH, EMOTION_LABEL_PATH,
    EMOTION_UNKNOWN_THRESHOLD (기본 0.50)
"""

import io
import os
from typing import Tuple

import numpy as np
import librosa

# tf-keras / tensorflow.keras 로더 호환
try:
    from tf_keras.models import load_model as _load_model
    LOADER_NAME = "tf_keras"
except Exception:  # pragma: no cover
    from tensorflow.keras.models import load_model as _load_model
    LOADER_NAME = "tf.keras"


def _resolve_under_app(relative_path: str) -> str:
    """
    app/services/stress_services/.. 기준으로 'app' 폴더를 정확히 찾아
    그 하위의 relative_path를 절대경로로 반환.
    """
    services_dir = os.path.dirname(os.path.abspath(__file__))            # .../app/services/stress_services
    app_dir = os.path.dirname(os.path.dirname(services_dir))             # .../app
    return os.path.normpath(os.path.join(app_dir, relative_path))


class EmotionDLService:
    def __init__(self) -> None:
        """
        기본 경로:
          app/models/stresscare_models/emotion_cnn_lstm_all.h5
          app/models/stresscare_models/emotion_labels_all.npy
        .env가 있으면 EMOTION_MODEL_PATH / EMOTION_LABEL_PATH로 덮어씀.
        """
        # .env 값 우선, 없으면 app 하위 기본 경로 사용
        env_model = os.getenv("EMOTION_MODEL_PATH", "").strip()
        env_label = os.getenv("EMOTION_LABEL_PATH", "").strip()

        default_model_rel = os.path.join("models", "stresscare_models", "emotion_cnn_lstm_all.h5")
        default_label_rel = os.path.join("models", "stresscare_models", "emotion_labels_all.npy")

        # env가 절대경로가 아니면 app 기준으로 해석
        self.model_path = (
            env_model if env_model else _resolve_under_app(default_model_rel)
        )
        if env_model and not os.path.isabs(env_model):
            self.model_path = _resolve_under_app(env_model)

        self.label_path = (
            env_label if env_label else _resolve_under_app(default_label_rel)
        )
        if env_label and not os.path.isabs(env_label):
            self.label_path = _resolve_under_app(env_label)

        # unknown 임계값
        self.unknown_threshold = float(os.getenv("EMOTION_UNKNOWN_THRESHOLD", "0.50"))

        # 실제 리소스
        self.model = None
        self.labels = None
        self.model_name = f"{LOADER_NAME}:{os.path.basename(self.model_path)}"

        # 로드
        self._load_assets()

    # ---------------- internal ----------------
    def _load_assets(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"DL 모델을 찾을 수 없습니다: {self.model_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"DL 라벨 파일을 찾을 수 없습니다: {self.label_path}")

        self.model = _load_model(self.model_path)
        self.labels = np.load(self.label_path, allow_pickle=True)
        if isinstance(self.labels, np.ndarray):
            self.labels = self.labels.tolist()

        print("[DL] -------- EmotionDLService Loaded --------")
        print(f"[DL] MODEL_PATH : {self.model_path}")
        print(f"[DL] LABEL_PATH : {self.label_path}")
        print(f"[DL] LOADER     : {self.model_name}")
        print(f"[DL] THRESHOLD  : {self.unknown_threshold}")
        print("[DL] ----------------------------------------")

    @staticmethod
    def _extract_melspec_from_audio(
        y: np.ndarray,
        sr: int,
        n_mels: int = 128,
        duration: float = 3.0,
    ) -> np.ndarray:
        """
        오디오 배열 → (n_mels, time, 1) 정규화 Mel-spectrogram
        - duration(초)로 길이 고정
        - 샘플별 0~1 정규화
        """
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mn, mx = mel_db.min(), mel_db.max()
        if mx - mn > 0:
            mel_db = (mel_db - mn) / (mx - mn)

        return mel_db[..., np.newaxis].astype(np.float32)  # (n_mels, time, 1)

    # ---------------- public ----------------
    def predict_emotion_from_bytes(self, audio_bytes: bytes, sr: int = 22050) -> Tuple[str, float]:
        """
        업로드된 음성 바이트 → (예측 라벨, 신뢰도)
        - librosa가 처리 가능한 포맷(wav/mp3/m4a 등) 지원
        - 최대 확률 < threshold 이면 'unknown'
        """
        if self.model is None or self.labels is None:
            raise RuntimeError("DL 모델/라벨이 로드되지 않았습니다.")

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        X = self._extract_melspec_from_audio(y, sr)   # (n_mels, time, 1)
        X = np.expand_dims(X, axis=0)                 # (1, n_mels, time, 1)

        probs = self.model.predict(X, verbose=0)[0]   # (num_classes,)
        idx = int(np.argmax(probs))
        prob = float(probs[idx])

        if prob < self.unknown_threshold:
            return "unknown", prob

        label = self.labels[idx] if 0 <= idx < len(self.labels) else "unknown"
        return str(label), prob

    def predict_emotion_from_file(self, path: str, sr: int = 22050) -> Tuple[str, float]:
        """디버그용 파일 경로 입력"""
        with open(path, "rb") as f:
            return self.predict_emotion_from_bytes(f.read(), sr=sr)