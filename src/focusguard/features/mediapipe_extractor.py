from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe landmark indices (Face landmarks)
# We'll use the same indices as before when available.
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

    Supports:
      - Legacy Solutions API (mp.solutions.face_mesh) if available
      - Tasks API (FaceLandmarker) otherwise
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._use_solutions = hasattr(mp, "solutions")

        self._face_mesh = None
        self._landmarker = None

        if self._use_solutions:
            # Legacy API
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            # Tasks API
            # Requires a FaceLandmarker model asset (.task file).
            # Default expected location:
            #   assets/models/face_landmarker.task
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if model_path is None:
                model_path = "assets/models/face_landmarker.task"

            mp_model_path = Path(model_path)
            if not mp_model_path.exists():
                raise FileNotFoundError(
                    "MediaPipe Tasks model not found.\n"
                    f"Expected: {mp_model_path.resolve()}\n"
                    "Provide model_path to MediaPipeFaceExtractor(...) or place the file there."
                )

            base_opts = mp_python.BaseOptions(model_asset_path=str(mp_model_path))

            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> FaceFeatures:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._use_solutions:
            # Legacy FaceMesh results
            results = self._face_mesh.process(frame_rgb)  # type: ignore[union-attr]
            if not results.multi_face_landmarks:
                return FaceFeatures(False, 0.0, 0.0, 0)

            face_landmarks = results.multi_face_landmarks[0].landmark
            return self._compute_from_landmarks(face_landmarks)

        # Tasks API results
        from mediapipe.tasks.python import vision as mp_vision

        h, w = frame_rgb.shape[:2]
        mp_image = mp_vision.MPImage(image_format=mp_vision.ImageFormat.SRGB, data=frame_rgb)

        detection = self._landmarker.detect(mp_image)  # type: ignore[union-attr]
        if not detection.face_landmarks:
            return FaceFeatures(False, 0.0, 0.0, 0)

        # detection.face_landmarks[0] is a list of normalized landmarks with x/y in [0..1]
        face_landmarks = detection.face_landmarks[0]
        return self._compute_from_landmarks(face_landmarks)

    def _compute_from_landmarks(self, face_landmarks) -> FaceFeatures:
        # face_landmarks items expose .x/.y (normalized)
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
