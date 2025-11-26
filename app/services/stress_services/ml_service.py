from typing import Dict, List, Any, Optional
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


class StressMLService:
    """
    스트레스 ML 예측 서비스
    - 학습 스크립트에서 저장한 joblib 번들을 로드하여 예측
    - 번들 구조 예:
        {
          "preprocess": fitted ColumnTransformer,
          "model": fitted Regressor,
          "feature_cols": [...],
          "numeric_features": [...],
          "categorical_features": [...],
          "feature_labels_ko": {...},        # (선택)
          "stress_level_thresholds": {...}   # (선택)
        }
    - 점수 스케일: dataset의 Stress Level(대개 0~10)을 0~100으로 스케일업
    """

    DEFAULT_MODEL_PATH = os.getenv(
        "STRESS_MODEL_PATH",
        os.path.join("app", "models", "stresscare_models", "stress_rf_model.pkl")
    )

    def __init__(self) -> None:
        # 기본 속성
        self.bundle: Optional[Dict[str, Any]] = None
        self.model_name: str = "fallback-linear"
        self.expected_features: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_labels_ko: Dict[str, str] = {}
        self.thresholds: Dict[str, Any] = {}

        # ✅ 안전 초기화(속성 항상 존재하도록)
        self.preprocess = None
        self.model = None

        # ===== 모델 로드 시도 =====
        model_path = Path(self.DEFAULT_MODEL_PATH)
        if model_path.exists():
            try:
                self.bundle = joblib.load(str(model_path))
                self.preprocess = self.bundle.get("preprocess")
                self.model = self.bundle.get("model")
                self.expected_features = list(self.bundle.get("feature_cols", []))
                self.numeric_features = list(self.bundle.get("numeric_features", []))
                self.categorical_features = list(self.bundle.get("categorical_features", []))
                self.feature_labels_ko = dict(self.bundle.get("feature_labels_ko", {}))
                self.thresholds = dict(self.bundle.get("stress_level_thresholds", {}))
                self.model_name = f"joblib:{model_path.name}"
            except Exception as e:
                print("❌ ML 번들 로드 실패:", e)
                self.bundle, self.preprocess, self.model = None, None, None

        # 완전 실패 시 fallback
        if self.model is None:
            self.model_name = "fallback-linear"

    # -------------------- 내부 유틸 --------------------
    def _coerce_row(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        API로 들어온 dict → 번들이 학습한 feature_cols 순서의 DataFrame 1row로 강제 변환
        - 없는 컬럼은 NaN
        - 수치형은 float로 캐스팅(실패 시 NaN), 범주형은 str로 캐스팅
        """
        if not self.expected_features:  # 번들 없을 때 빈 DF 방지
            # 들어온 키들만이라도 DF로 만들어 둠(전처리/예측은 fallback로)
            return pd.DataFrame([features])

        row: Dict[str, Any] = {}
        for col in self.expected_features:
            val = features.get(col, None)
            if col in self.numeric_features:
                try:
                    row[col] = float(val) if val is not None else np.nan
                except Exception:
                    row[col] = np.nan
            else:
                row[col] = "" if val is None else str(val)
        return pd.DataFrame([row], columns=self.expected_features)

    @staticmethod
    def _scale_to_100(yhat: float) -> float:
        """
        데이터셋의 Stress Level이 보통 0~10 범위이므로 0~100 스케일로 확장.
        (필요 시 여기서 스케일링 로직 변경)
        """
        if yhat is None:
            return 0.0
        score = float(yhat) * 10.0
        return round(max(0.0, min(100.0, score)), 2)

    # -------------------- 공개 API --------------------
    def predict_score(self, features: Dict[str, Any]) -> float:
        """
        입력 features는 학습 번들의 feature_cols를 그대로 키로 넣는 것을 권장.
        - 예: age, gender, occupation, sleep_duration, quality_of_sleep, physical_activity_level,
              bmi_category, heart_rate, daily_steps, bp_sys, bp_dia
        """
        # 1) 번들이 있으면 그대로 예측
        if self.model is not None and self.preprocess is not None and self.expected_features:
            X_df = self._coerce_row(features)
            try:
                Xt = self.preprocess.transform(X_df)  # fitted ColumnTransformer
                yhat = float(self.model.predict(Xt)[0])
                return self._scale_to_100(yhat)
            except Exception as e:
                print("❌ 번들 예측 실패, fallback로 전환:", e)

        # 2) ===== Fallback (간단 휴리스틱) =====
        # 일부 키만으로 대략 점수 계산(모델이 없어도 최소한 동작)
        sleep = float(features.get("sleep_duration", 0) or 0)
        qsleep = float(features.get("quality_of_sleep", 0) or 0)
        phys   = float(features.get("physical_activity_level", 0) or 0)
        hr     = float(features.get("heart_rate", 0) or 0)
        steps  = float(features.get("daily_steps", 0) or 0)
        sys_   = float(features.get("bp_sys", 0) or 0)
        dia_   = float(features.get("bp_dia", 0) or 0)

        raw = (
            (10 - min(qsleep, 10)) * 4.0 +
            (10 - min(sleep, 10)) * 2.0 +
            min(hr / 10, 10) * 2.5 +
            max(0, (sys_ - 120)) * 0.1 +
            max(0, (dia_ - 80)) * 0.1 -
            min(phys / 10, 10) * 1.5 -
            min(steps / 1000, 10) * 0.8
        )
        score = max(0.0, min(100.0, raw))
        return round(score, 2)

    # ✅ 라우터가 기대하는 이름(별칭) — A방법 핵심
    def predict_as_score(self, features: Dict[str, Any]) -> float:
        return self.predict_score(features)