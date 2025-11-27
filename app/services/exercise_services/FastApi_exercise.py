# main.py
# 실행 예시: uvicorn FastApi_exercise:app --host 0.0.0.0 --port 5000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple

import os
import io
import math
import base64

import numpy as np
import cv2
import av
import requests

# MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

# CORS
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 전역 설정 / 앱 기본 세팅
# ============================================================
app = FastAPI(title="AI Coach Core (FastAPI + MediaPipe + CNN-LSTM)")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 개발 중이면 ["*"]도 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델/라벨 관련 전역
cnn_lstm_model = None
LABELS_PATH = "../../models/exercise_models/labels.txt"
CLASSES: List[str] = []

# CNN-LSTM 학습 시 사용했던 설정과 동일하게 유지
SEQ_LEN = 24
IMG_SIZE = (160, 160)

# 비디오 프레임 샘플링 간격 (2 프레임마다 1개 사용)
FRAME_STRIDE = 2

# MediaPipe Pose 전역 재사용
pose_detector = None


# ============================================================
# 커스텀 레이어 (CNN-LSTM 모델 로딩용)
# ============================================================
class TemporalAttention(layers.Layer):
    """CNN-LSTM 모델에 사용된 커스텀 Attention 레이어"""

    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, activation=None)

    def call(self, x):  # x: (B,T,F)
        h = self.dense(x)           # (B,T,U)
        e = self.score(h)           # (B,T,1)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(a * x, axis=1)  # (B,F)
        return context


# ============================================================
# Pydantic 스키마
# ============================================================
class AnalyzeRequest(BaseModel):
    """프론트에서 들어오는 요청 형태"""
    userId: str
    message: Optional[str] = None     # 텍스트 (그대로 LLM 쪽에 전달)
    image: Optional[str] = None       # base64-encoded image
    video: Optional[str] = None       # base64-encoded video


# ============================================================
# 기본 기하 유틸 함수 (각도/거리)
# ============================================================
def angle_3pts(a, b, c) -> float:
    """
    세 점 a, b, c가 주어졌을 때,
    점 b 기준의 각도(도 단위)를 계산.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = (math.hypot(*ba) * math.hypot(*bc)) + 1e-6
    cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / denom
    cosv = max(-1.0, min(1.0, cosv))
    return round(math.degrees(math.acos(cosv)), 1)


def dist(a, b) -> float:
    """2D 거리"""
    return round(math.hypot(a[0] - b[0], a[1] - b[1]), 1)


# ============================================================
# MediaPipe: 포즈 특징 추출 (단일 프레임)
# ============================================================
def extract_pose_features(bgr_img, last_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    단일 BGR 이미지에서 MediaPipe Pose로
    - 관절 각도 (팔꿈치, 무릎, 등)
    - 간단한 길이/비율
    - stage (up/down/hold)
    를 추출.
    """
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    res = pose_detector.process(rgb)

    if not res.pose_landmarks:
        return {"pose_detected": False, "pose_data": {}, "stage": "unknown"}

    lm = res.pose_landmarks.landmark

    def xy(i):  # normalize -> pixel
        return (lm[i].x * w, lm[i].y * h)

    # 주요 포인트 (.value 중요) : mediapipe로 키포인트 뽑아냄
    L_SH, R_SH = xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value), xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    L_EL, R_EL = xy(mp_pose.PoseLandmark.LEFT_ELBOW.value),    xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    L_WR, R_WR = xy(mp_pose.PoseLandmark.LEFT_WRIST.value),    xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    L_HI, R_HI = xy(mp_pose.PoseLandmark.LEFT_HIP.value),      xy(mp_pose.PoseLandmark.RIGHT_HIP.value)
    L_KN, R_KN = xy(mp_pose.PoseLandmark.LEFT_KNEE.value),     xy(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    L_AN, R_AN = xy(mp_pose.PoseLandmark.LEFT_ANKLE.value),    xy(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    # 중앙점(keypoints) : post_data, 키포인트 활용해 중앙점 수치 구하고
    SHO = ((L_SH[0] + R_SH[0]) / 2, (L_SH[1] + R_SH[1]) / 2)
    HIP = ((L_HI[0] + R_HI[0]) / 2, (L_HI[1] + R_HI[1]) / 2)
    KNE = ((L_KN[0] + R_KN[0]) / 2, (L_KN[1] + R_KN[1]) / 2)
    ANK = ((L_AN[0] + R_AN[0]) / 2, (L_AN[1] + R_AN[1]) / 2)

    # 각도(joints) : pose_data
    elbow_left = angle_3pts(L_SH, L_EL, L_WR)
    elbow_right = angle_3pts(R_SH, R_EL, R_WR)
    knee_left = angle_3pts(L_HI, L_KN, L_AN)
    knee_right = angle_3pts(R_HI, R_KN, R_AN)
    back_angle = angle_3pts(SHO, HIP, KNE)  # 몸통 기울기 대략 : 위에서 구한 중앙점 수치 활용함

    # 거리/비율(distances) : pose_data
    shoulder_hip = dist(SHO, HIP)
    hip_knee = dist(HIP, KNE)
    knee_ankle = dist(KNE, ANK)
    back_to_hip_ratio = round((shoulder_hip / (hip_knee + 1e-6)), 3)

    # 간단 단계(stage)
    mean_knee = (knee_left + knee_right) / 2
    mean_elbow = (elbow_left + elbow_right) / 2
    if mean_knee < 100:
        stage = "down"
    elif mean_elbow < 100:
        stage = "down"
    elif 165 <= back_angle <= 185:
        stage = "hold"
    else:
        stage = "up"

    pose_data = {
        "joints": {
            "left_elbow_angle": elbow_left,
            "right_elbow_angle": elbow_right,
            "left_knee_angle": knee_left,
            "right_knee_angle": knee_right,
            "back_angle": back_angle,
        },
        "distances": {
            "shoulder_hip": shoulder_hip,
            "hip_knee": hip_knee,
            "knee_ankle": knee_ankle,
            "back_to_hip_ratio": back_to_hip_ratio,
        },
        "keypoints": {
            "shoulder": [round(SHO[0], 1), round(SHO[1], 1)],
            "hip": [round(HIP[0], 1), round(HIP[1], 1)],
            "knee": [round(KNE[0], 1), round(KNE[1], 1)],
            "ankle": [round(ANK[0], 1), round(ANK[1], 1)],
        },
    }
    return {"pose_detected": True, "pose_data": pose_data, "stage": stage}


# ============================================================
# CNN-LSTM 추론 유틸
# ============================================================
def predict_exercise_from_bgr_seq_model(bgr_img) -> Dict[str, Any]:
    """
    단일 이미지(BGR)를 받아서,
    - 같은 이미지를 SEQ_LEN번 복제해 시퀀스를 만든 후
    - CNN-LSTM 모델로 운동 종류를 예측.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    x = rgb.astype(np.float32)

    # EfficientNet 전처리
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # (T,H,W,3) 가짜 시퀀스 생성
    seq = np.stack([x] * SEQ_LEN, axis=0)      # (T,H,W,3)
    inp = np.expand_dims(seq, axis=0)          # (1,T,H,W,3)

    preds = cnn_lstm_model.predict(inp)[0]     # (num_classes,)
    idx = int(np.argmax(preds))
    return {
        "detected_exercise": CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx),
        "exercise_confidence": float(np.max(preds)),
        "probs": preds.tolist(),
    }


def predict_exercise_from_video_bytes(video_bytes: bytes) -> Dict[str, Any]:
    """
    비디오 전체에서 SEQ_LEN개의 프레임을 균등 샘플링하고
    CNN-LSTM 모델로 운동 종류를 예측.
    """
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]

    frames_rgb = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        frames_rgb.append(rgb)
    container.close()

    n = len(frames_rgb)
    if n == 0:
        raise HTTPException(status_code=400, detail="프레임이 없습니다.")

    # 균등 샘플링
    idxs = np.linspace(0, n - 1, num=SEQ_LEN).astype(int)
    samples = []
    for i in idxs:
        rgb = cv2.resize(frames_rgb[i], IMG_SIZE)
        x = tf.keras.applications.efficientnet.preprocess_input(rgb.astype(np.float32))
        samples.append(x)

    seq = np.stack(samples, axis=0)   # (T,H,W,3)
    inp = np.expand_dims(seq, axis=0) # (1,T,H,W,3)

    preds = cnn_lstm_model.predict(inp)[0]
    idx = int(np.argmax(preds))
    return {
        "detected_exercise": CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx),
        "exercise_confidence": float(np.max(preds)),
        "probs": preds.tolist(),
    }


# ============================================================
# 단일 이미지 분석 (폼 + CNN-LSTM 간단 분류)
# ============================================================
def analyze_frame(image_bytes: bytes) -> Dict[str, Any]:
    """
    단일 이미지 base64 → BGR → 
    - CNN-LSTM 기반 운동 종류 분류
    - MediaPipe 기반 관절 각도/키포인트 추출
    두 가지를 합쳐서 요약 JSON 반환.
    """
    npimg = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지 디코딩 실패")

    # ex : 운동 종류
    ex = predict_exercise_from_bgr_seq_model(bgr)
    # pose : 운동 관절 정보
    pose = extract_pose_features(bgr)

    return {
        "detected_exercise": ex["detected_exercise"],
        "exercise_confidence": round(ex["exercise_confidence"], 4),
        "probs": ex["probs"],
        "stage": pose["stage"],
        "pose_detected": pose["pose_detected"],
        "pose_data": pose["pose_data"],
    }


# ============================================================
# 운동 시퀀스 분석 (rep 분할/ROM/대칭/정렬/흔들림)
# ============================================================

# MediaPipe Pose 인덱스 (33개 랜드마크 기준) : 표준 index를 정의 (33개 관절좌표에는 각각 mediapipe에서 정의된 
# 고정 index번호가 있음)
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28


def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _safe_std(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return float(np.std(vals)) if vals else None


# 운동별 주관절 / 임계값 설정
EX_THRESH = {
    "squat": {
        "angle": "knee",
        "view": {
            "side": {"top": 165.0, "bottom": 105.0},
            "front": {"top": 160.0, "bottom": 110.0},
        },
    },
    "push-up": {
        "angle": "elbow",
        "view": {
            "side": {"top": 145.0, "bottom": 130.0},
            "front": {"top": 150.0, "bottom": 120.0},
        },
    },
    "barbell biceps curl": {
        "angle": "elbow",
        "view": {
            "side": {"top": 160.0, "bottom": 80.0},
            "front": {"top": 155.0, "bottom": 85.0},
        },
    },
    "hammer curl": {
        "angle": "elbow",
        "view": {
            "side": {"top": 160.0, "bottom": 85.0},
            "front": {"top": 155.0, "bottom": 90.0},
        },
    },
    "shoulder-press": {
        "angle": "elbow",
        "view": {
            "side": {"top": 160.0, "bottom": 100.0},
            "front": {"top": 155.0, "bottom": 105.0},
        },
    },
}


def get_ex_thresh(exercise: str, view: str = "side") -> dict:
    """
    운동 종류/뷰에 맞는 ROM 상하 임계값 정보 반환.
    (라벨이 'shoulder press'로 되어있는 경우 alias 처리)
    """
    if exercise == "shoulder press":
        exercise = "shoulder-press"
    conf = EX_THRESH.get(exercise)
    if not conf:
        return {"angle": "knee", "top": 165.0, "bottom": 110.0}
    view = view if view in conf["view"] else "side"
    vconf = conf["view"][view]
    return {"angle": conf["angle"], "top": vconf["top"], "bottom": vconf["bottom"]}


def get_primary_series(frames_slim: List[Dict[str, Any]], exercise: str) -> List[Tuple[int, Optional[float]]]:
    """
    각 프레임마다 주 관절(knee/elbow)의 각도 시계열을 뽑아서 반환.
    """
    kind = EX_THRESH.get(exercise, {"angle": "knee"}).get("angle", "knee")
    if exercise == "shoulder press":
        kind = EX_THRESH["shoulder-press"]["angle"]

    series: List[Tuple[int, Optional[float]]] = []
    for fr in frames_slim:
        a = fr["angles"]
        if kind == "knee":
            val = _safe_mean([a.get("knee_l"), a.get("knee_r")])
        elif kind == "elbow":
            val = _safe_mean([a.get("elbow_l"), a.get("elbow_r")])
        else:
            val = _safe_mean([a.get("knee_l"), a.get("knee_r")])
        series.append((fr["frame"], val))
    return series


def segment_reps_by_primary_angle(
    primary_series: List[Tuple[int, float]],
    fps: float,
    top_thr: float = 165.0,
    bottom_thr: float = 110.0,
    min_rep_sec: float = 0.4,
    hysteresis: float = 5.0,
) -> List[Dict[str, int]]:
    """
    주 관절 각도 시계열에서 rep(반복 구간)를 탐지.
    - top_thr: 위쪽(펴진 상태) 근처
    - bottom_thr: 아래쪽(최대 굽힘) 근처
    """
    reps: List[Dict[str, int]] = []
    min_frames = int(min_rep_sec * max(1.0, fps))

    in_down = False
    start_f: Optional[int] = None
    bottom_idx: Optional[int] = None
    last_angle: Optional[float] = None

    for idx, (f, ang) in enumerate(primary_series):
        if ang is None:
            continue

        # "down" 진입 초기 탐지
        if start_f is None and ang <= (bottom_thr + hysteresis):
            start_f = max(0, f - int(0.2 * fps))
            in_down = True
            bottom_idx = idx

        if last_angle is not None:
            # 위쪽에서 내려가기 시작
            if (not in_down) and (last_angle >= (top_thr - hysteresis)) and (ang < last_angle):
                in_down = True
                start_f = f
                bottom_idx = idx

            # down 상태 유지 + 바닥 프레임 업데이트
            if in_down:
                if (bottom_idx is None) or (ang <= primary_series[bottom_idx][1]):
                    bottom_idx = idx

                # 다시 위로 올라와 top 영역 근처 + 최소 길이 조건 만족 시 rep 하나 확정
                if (
                    ang >= (top_thr - hysteresis)
                    and (start_f is not None)
                    and ((f - start_f) >= min_frames)
                    and (bottom_idx is not None)
                ):
                    reps.append(
                        {
                            "start": start_f,
                            "bottom": primary_series[bottom_idx][0],
                            "end": f,
                        }
                    )
                    in_down = False
                    start_f = None
                    bottom_idx = None

        last_angle = ang

    # 루프가 끝난 후 아직 진행 중이던 rep 마무리
    if in_down and start_f is not None and bottom_idx is not None:
        end_f = primary_series[-1][0]
        if (end_f - start_f) >= min_frames:
            reps.append(
                {
                    "start": start_f,
                    "bottom": primary_series[bottom_idx][0],
                    "end": end_f,
                }
            )

    return reps


def compute_rep_metrics(frames_slice: List[Dict[str, Any]], fps: float) -> Dict[str, Any]:
    """
    하나의 rep 구간 안에 있는 프레임들에 대해:
    - 무릎/엉덩이 ROM
    - 좌우 대칭도
    - 무릎이 발보다 앞으로 나가는 비율
    - 상체 흔들림 정도(표준편차)
    등을 계산.
    """
    aseq = [fr["angles"] for fr in frames_slice]
    knee_l = [a.get("knee_l") for a in aseq]
    knee_r = [a.get("knee_r") for a in aseq]
    hip_l = [a.get("hip_l") for a in aseq]
    hip_r = [a.get("hip_r") for a in aseq]
    torso = [a.get("torso") for a in aseq]

    # ROM(knee)
    rom_knee = None
    both_k = [v for v in (knee_l + knee_r) if v is not None]
    if len(both_k) >= 2:
        rom_knee = float(max(both_k) - min(both_k))

    # ROM(hip)
    rom_hip = None
    both_h = [v for v in (hip_l + hip_r) if v is not None]
    if len(both_h) >= 2:
        rom_hip = float(max(both_h) - min(both_h))

    # 좌우 무릎 대칭 (각도 차이 평균)
    symmetry = None
    diffs = []
    for a, b in zip(knee_l, knee_r):
        if a is not None and b is not None:
            diffs.append(abs(a - b))
    if diffs:
        symmetry = float(np.mean(diffs))

    # 무릎이 발보다 앞으로 나간 비율
    misalign_ratio = None
    ahead_cnt = 0
    valid = 0
    for fr in frames_slice:
        pd = fr.get("pose_data") or {}
        lm = pd.get("landmarks")
        pairs = []
        # landmarks(33개 좌표)가 있으면 그걸 우선 사용
        if lm and len(lm) >= 33:
            try:
                pairs = [
                    (lm[L_KNEE]["x"], lm[L_ANKLE]["x"]),
                    (lm[R_KNEE]["x"], lm[R_ANKLE]["x"]),
                ]
            except Exception:
                pairs = []
        else:
            # 아니면 keypoints(knee/ankle) 좌표 사용
            kp = pd.get("keypoints") or {}
            knee = kp.get("knee")
            ankle = kp.get("ankle")
            if knee and ankle:
                pairs = [(knee[0], ankle[0])]

        for kx, ax in pairs:
            if kx is None or ax is None:
                continue
            valid += 1
            if (kx - ax) > 0.03:
                ahead_cnt += 1

    if valid > 0:
        misalign_ratio = ahead_cnt / valid

    # 상체 흔들림 (각도 표준편차)
    wobble = _safe_std(torso)

    return {
        "rom_deg": {"knee": rom_knee, "hip": rom_hip},
        "symmetry_deg": symmetry,
        "alignment_knee_over_toe_ratio": misalign_ratio,
        "stability": {"torso_std_deg": wobble},
        "duration_s": (len(frames_slice) / fps if fps else None),
    }


def filter_reps(reps, fps):
    """
    너무 ROM이 적거나, 템포가 너무 빠르거나/느린 rep,
    서로 너무 붙어 있는 rep들을 제거하는 후처리.
    """
    ROM_MIN_DEG = 40.0   # 가동 범위 기준
    MIN_GAP_SEC = 0.8    # rep 간 최소 간격(초)
    min_gap = int(MIN_GAP_SEC * fps)

    def ok_tempo(t):
        return (t is not None) and (0.25 <= t <= 2.0)

    filtered = []
    last_end = -10**9

    for r in reps:
        rom = r.get("primary_rom_deg") or 0.0
        tempo = r.get("tempo_s") or {}
        down = tempo.get("down")
        up = tempo.get("up")
        endf = (r.get("frames") or {}).get("end", 0)

        if rom < ROM_MIN_DEG:
            continue
        if not (ok_tempo(down) and ok_tempo(up)):
            continue
        if endf - last_end < min_gap:
            continue

        last_end = endf
        filtered.append(r)

    return filtered


def analyze_video(video_bytes: bytes) -> Dict[str, Any]:
    """
    비디오 전체를 분석하여:
    - 운동 종류(CNN-LSTM)
    - 각 프레임별 MediaPipe Pose 기반 각도/키포인트
    - rep(반복) 세그멘트 및 rep별 ROM/대칭/정렬/안정성
    를 계산하고 요약 JSON을 반환.
    """
    # 1) CNN-LSTM으로 전체 운동 종류 예측
    overall = predict_exercise_from_video_bytes(video_bytes)
    ex_name = overall["detected_exercise"]

    # 2) 프레임별 Pose/각도 추출
    frames_slim: List[Dict[str, Any]] = []
    try:
        container = av.open(io.BytesIO(video_bytes))
        stream = container.streams.video[0]

        # fps 추론
        if stream.average_rate is not None and stream.average_rate.denominator != 0:
            fps = float(stream.average_rate)
        elif hasattr(stream, "base_rate") and stream.base_rate and stream.base_rate.denominator != 0:
            fps = float(stream.base_rate)
        else:
            fps = 30.0  # fallback

        for i, frame in enumerate(container.decode(stream)):
            if i % FRAME_STRIDE != 0:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            bgr = rgb[:, :, ::-1].copy()

            pose = extract_pose_features(bgr)
            j = ((pose or {}).get("pose_data") or {}).get("joints") or {}

            angles = {
                "knee_l": j.get("left_knee_angle"),
                "knee_r": j.get("right_knee_angle"),
                "hip_l": None,
                "hip_r": None,
                "elbow_l": j.get("left_elbow_angle"),
                "elbow_r": j.get("right_elbow_angle"),
                "torso": j.get("back_angle"),
            }

            frames_slim.append(
                {
                    "frame": i,
                    "stage": pose.get("stage"),
                    "pose_detected": pose.get("pose_detected"),
                    "pose_data": pose.get("pose_data"),
                    "angles": angles,
                }
            )

        container.close()
    except av.AVError:
        raise HTTPException(status_code=400, detail="PyAV에서 동영상을 읽을 수 없습니다.")

    if not frames_slim:
        raise HTTPException(status_code=400, detail="분석 가능한 프레임이 없습니다.")

    # 3) 주 관절 각도 시계열 및 rep 분할
    thr_conf = get_ex_thresh(ex_name, view="side")
    primary_angle = thr_conf["angle"]
    primary_series = get_primary_series(frames_slim, ex_name)

    vals = [v for _, v in primary_series if v is not None]
    if vals:
        print(
            f"[debug] primary range: min={min(vals):.1f}, max={max(vals):.1f}, "
            f"top={thr_conf['top']}, bottom={thr_conf['bottom']}"
        )

    reps_idx = segment_reps_by_primary_angle(
        primary_series,
        fps=fps,
        top_thr=thr_conf["top"],
        bottom_thr=thr_conf["bottom"],
        min_rep_sec=0.4,
        hysteresis=5.0,
    )

    def primary_rom_between(start_f: int, end_f: int) -> Optional[float]:
        vals = [v for f, v in primary_series if start_f <= f <= end_f and v is not None]
        return float(max(vals) - min(vals)) if len(vals) >= 2 else None

    # 4) rep별 상세 지표 계산
    rep_summaries = []
    evidence_frames = []
    for ridx, seg in enumerate(reps_idx, start=1):
        start_f, bottom_f, end_f = seg["start"], seg["bottom"], seg["end"]
        slice_frames = [fr for fr in frames_slim if start_f <= fr["frame"] <= end_f]
        metrics = compute_rep_metrics(slice_frames, fps=fps)

        down_s = (bottom_f - start_f) / fps if fps and bottom_f > start_f else None
        up_s = (end_f - bottom_f) / fps if fps and end_f > bottom_f else None

        rep_summaries.append(
            {
                "idx": ridx,
                "frames": {"start": start_f, "bottom": bottom_f, "end": end_f},
                "rom_deg": metrics["rom_deg"],
                "primary_rom_deg": primary_rom_between(start_f, end_f),
                "tempo_s": {"down": down_s, "up": up_s, "pause": 0.0 if (down_s and up_s) else None},
                "symmetry_deg": metrics["symmetry_deg"],
                "alignment_knee_over_toe_ratio": metrics["alignment_knee_over_toe_ratio"],
                "stability": metrics["stability"],
            }
        )
        evidence_frames.append(bottom_f)

    # 5) 전체 평균/요약 지표
    avg_rom_primary = _safe_mean(
        [r.get("primary_rom_deg") for r in rep_summaries if r.get("primary_rom_deg") is not None]
    )
    avg_tempo_down = _safe_mean(
        [r["tempo_s"]["down"] for r in rep_summaries if r["tempo_s"]["down"] is not None]
    )
    wobble = _safe_mean(
        [r["stability"]["torso_std_deg"] for r in rep_summaries if r["stability"]["torso_std_deg"] is not None]
    )

    result: Dict[str, Any] = {
        "exercise": ex_name,
        "fps": fps,
        "duration_s": len(frames_slim) / fps if fps else None,
        "global_stats": {
            "rep_count": len(rep_summaries),
            "avg_rom_primary_deg": avg_rom_primary,
            "avg_tempo_down_s": avg_tempo_down,
            "wobble_score": wobble,
            "primary_angle": primary_angle,
        },
        "reps": rep_summaries,
        "evidence_frames": evidence_frames[:5],
        "model": {
            "detected_exercise": ex_name,
            "exercise_confidence": round(overall["exercise_confidence"], 4),
            "probs": overall["probs"],
        },
    }

    # rep이 하나도 잡히지 않았을 때의 fallback 처리
    if len(rep_summaries) == 0:
        prim_vals = [ang for _, ang in primary_series if ang is not None]
        global_rom = (max(prim_vals) - min(prim_vals)) if len(prim_vals) >= 2 else None

        stage_hist = {"up": 0, "down": 0, "hold": 0}
        for fr in frames_slim:
            s = fr.get("stage")
            if s in stage_hist:
                stage_hist[s] += 1

        result["global_stats"]["global_primary_rom_deg"] = float(global_rom) if global_rom is not None else None
        result["global_stats"]["stage_counts"] = stage_hist

        if prim_vals:
            min_idx = int(
                np.argmin(
                    [v if v is not None else 1e9 for v in [v for _, v in primary_series]]
                )
            )
            result["evidence_frames"] = [primary_series[min_idx][0]]

        result["flags"] = ["no_rep_detected"]

    return result


# ============================================================
# base64 디코딩 유틸
# ============================================================
def safe_b64decode(b64_string: str) -> bytes:
    """
    앞부분에 data:video/mp4;base64, ... 같은 헤더가 붙어있어도 처리 가능하도록
    패딩까지 보정해서 base64 디코딩.
    """
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",")[1]

    # 패딩 보정
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += "=" * (4 - missing_padding)

    return base64.b64decode(b64_string)


# ============================================================
# 애플리케이션 라이프사이클 (모델/포즈 로더)
# ============================================================
@app.on_event("startup")
def on_startup():
    """
    서버 시작 시:
    - 클래스 라벨 로드
    - CNN-LSTM 모델 로드
    - MediaPipe Pose 초기화
    """
    global cnn_lstm_model, CLASSES, pose_detector

    # 클래스 라벨 로드
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            CLASSES = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # labels.txt가 없을 때 기본 클래스
        CLASSES = ["barbell biceps curl", "hammer curl", "push-up", "shoulder press", "squat"]

    # 모델 로드 (커스텀 레이어 등록)
    cnn_lstm_model = load_model(
        "../../models/exercise_models/best_cnn_lstm_model_stronger.h5",
        custom_objects={"TemporalAttention": TemporalAttention},
    )

    pose_detector = mp_pose.Pose(static_image_mode=True)
    print("✅ CNN-LSTM 모델 & Pose 로드 완료. classes =", CLASSES)


@app.on_event("shutdown")
def on_shutdown():
    """서버 종료 시 Pose 리소스 정리"""
    global pose_detector
    try:
        if pose_detector is not None:
            pose_detector.close()
    except Exception:
        pass

# ============================================================
# 헬스 체크용 엔드포인트
# ============================================================
@app.get("/")
def root():
    return "OK"

# 메인 엔드포인트: /analyze
#   - 이미지 / 비디오 / 텍스트 입력을 처리 
#   - 내부에서 CNN-LSTM + MediaPipe로 분석 
#   - 분석 결과를 spirngboot프록시 서버로 반환 
# ============================================================
@app.post("/analyze")
def analyze_json(payload: AnalyzeRequest):

    try:
        msg = (payload.message or "").strip()

        # 1) 이미지(base64)
        if payload.image:
            image_bytes = safe_b64decode(payload.image)
            analysis_out = analyze_frame(image_bytes)

        # 2) 비디오(base64)
        elif payload.video:
            video_bytes = safe_b64decode(payload.video)
            analysis_out = analyze_video(video_bytes)

            # rep 후처리 필터링 + rep_count 업데이트
            if "reps" in analysis_out and "fps" in analysis_out:
                analysis_out["reps"] = filter_reps(analysis_out["reps"], analysis_out["fps"])
                if "global_stats" in analysis_out:
                    analysis_out["global_stats"]["rep_count"] = len(analysis_out["reps"])

        # 3) 텍스트만 있을 때
        elif msg:
            analysis_out = {
                "detected_exercise": None,
                "exercise_confidence": None,
                "probs": None,
                "total_frames": None,
                "frames": None,
                "stage": None,
                "pose_detected": None,
                "pose_data": None,
                "message": msg,
            }

        else:
            raise HTTPException(status_code=400, detail="image, video, message 중 하나 필요")

        return {"analysis": analysis_out}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}")
