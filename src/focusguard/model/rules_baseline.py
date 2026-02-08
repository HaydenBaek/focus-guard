from __future__ import annotations

from enum import Enum
from typing import Dict, Any

from focusguard.features.feature_vector import FeatureVector


class FocusState(str, Enum):
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"


class RulesBaseline:
    """
    Deterministic baseline (no ML).
    Used for MVP, labeling, and fallback safety.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        baseline_cfg = cfg.get("baseline", {})
        self.nose_offset_abs_threshold: float = float(
            baseline_cfg.get("nose_offset_abs_threshold", 0.035)
        )

    def predict(self, fv: FeatureVector) -> FocusState:
        if fv.face_present == 0:
            return FocusState.AWAY

        if fv.nose_offset_abs > self.nose_offset_abs_threshold:
            return FocusState.DISTRACTED

        return FocusState.FOCUSED
