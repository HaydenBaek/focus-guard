from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# Landmark indices (still valid for FaceLandmarker output order)
NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263


@dataclass
class FaceFeatures:
    face_present: bool
    nose_offset: float
    nose_offset_abs: float
    landmark_count: int


class MediaPipeFaceExtractor:
    """
    Privacy-safe extractor.
    - Never stores frames
    - Never returns images
    - Only returns derived numerical features

    Uses MediaPipe Tasks FaceLandmarker (mp.tasks.vision.*).
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = "assets/models/face_landmarker.task"

        mp_model_path = Path(model_path)
        if not mp_model_path.exists():
            raise FileNotFoundError(
                "MediaPipe Tasks model not found.\n"
                f"Expected: {mp_model_path.resolve()}\n"
                "Place the file there or pass model_path=..."
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(mp_model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> FaceFeatures:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # DONE Correct Tasks API image wrapper
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return FaceFeatures(False, 0.0, 0.0, 0)

        face_landmarks = result.face_landmarks[0]  # list of normalized landmarks
        return self._compute_from_landmarks(face_landmarks)

    def _compute_from_landmarks(self, face_landmarks) -> FaceFeatures:
        try:
            nose = face_landmarks[NOSE_TIP_IDX]
            left_eye = face_landmarks[LEFT_EYE_OUTER_IDX]
            right_eye = face_landmarks[RIGHT_EYE_OUTER_IDX]
        except Exception:
            return FaceFeatures(False, 0.0, 0.0, int(len(face_landmarks)))

        eye_mid_x = (float(left_eye.x) + float(right_eye.x)) / 2.0
        nose_offset = float(nose.x) - eye_mid_x

        return FaceFeatures(
            face_present=True,
            nose_offset=nose_offset,
            nose_offset_abs=abs(nose_offset),
            landmark_count=int(len(face_landmarks)),
        )
