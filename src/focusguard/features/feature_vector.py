from __future__ import annotations

from dataclasses import dataclass
from typing import List

from focusguard.features.mediapipe_extractor import FaceFeatures


@dataclass(frozen=True)
class FeatureVector:
    """
    Model-ready feature vector (derived only).
    Order is stable and documented for reproducibility.
    """
    face_present: int            # 0 or 1
    nose_offset_abs: float       # |nose - eye_mid|
    nose_offset_signed: float    # nose - eye_mid

    def as_list(self) -> List[float]:
        return [
            float(self.face_present),
            float(self.nose_offset_abs),
            float(self.nose_offset_signed),
        ]


def build_feature_vector(features: FaceFeatures) -> FeatureVector:
    return FeatureVector(
        face_present=1 if features.face_present else 0,
        nose_offset_abs=features.nose_offset_abs,
        nose_offset_signed=features.nose_offset,
    )
