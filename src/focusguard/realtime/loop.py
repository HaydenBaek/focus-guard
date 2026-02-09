from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from focusguard.features.mediapipe_extractor import MediaPipeFaceExtractor
from focusguard.features.feature_vector import build_feature_vector
from focusguard.model.rules_baseline import RulesBaseline, FocusState
from focusguard.model.smoothing import SlidingWindowSmoother
from focusguard.intervention.engine import InterventionEngine
from focusguard.intervention.player import LocalInterventionPlayer, InterventionAssets
from focusguard.logging.parquet_logger import ParquetEventLogger
from focusguard.logging.schemas import make_empty_row


def _overlay_text(frame: np.ndarray, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_camera_loop(cfg: Dict[str, Any]) -> None:
    cam_cfg = cfg.get("camera", {})
    runtime_cfg = cfg.get("runtime", {})

    index = int(cam_cfg.get("index", 0))
    show_preview = bool(runtime_cfg.get("show_preview", True))
    camera_on = bool(cam_cfg.get("enabled", True))
    session_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4()) if camera_on else None

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check OS permissions.")

    # Core components
    extractor = MediaPipeFaceExtractor()
    baseline = RulesBaseline(cfg)
    smoother = SlidingWindowSmoother(cfg)

    # Interventions
    with open("configs/interventions.yaml", "r") as f:
        interventions_cfg = yaml.safe_load(f) or {}

    intervention_engine = InterventionEngine(cfg, interventions_cfg)
    player = LocalInterventionPlayer(
        InterventionAssets(base_dir=Path("assets/interventions"))
    )

    # Logging
    logger = ParquetEventLogger(cfg)

    print("FocusGuard running. Press 'q' to quit, 'c' to toggle camera.")

    fps_last = time.time()
    fps_frames = 0
    last_state_print = 0.0

    while True:
        now = time.time()

        if camera_on:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
        else:
            frame = None

        if frame is not None:
            ff = extractor.extract(frame)
            fv = build_feature_vector(ff)
            raw_state = baseline.predict(fv)
            smooth_state = smoother.update(raw_state)
        else:
            raw_state = FocusState.AWAY
            smooth_state = smoother.update(raw_state)
            fv = None
            ff = None

        # Intervention decision
        event = intervention_engine.update(smooth_state, now=now)
        if event:
            player.play(event, video_filename="sample_clip.mp4")

        # Logging (derived-only)
        if camera_on and run_id is not None:
            row = make_empty_row(logger.schema)
            row.update(
                {
                    "ts": now,
                    "date": time.strftime("%Y-%m-%d", time.localtime(now)),
                    "session_id": session_id,
                    "run_id": run_id,
                    "camera_on": int(camera_on),
                    "face_present": fv.face_present if fv else 0,
                    "nose_offset_abs": fv.nose_offset_abs if fv else 0.0,
                    "nose_offset_signed": fv.nose_offset_signed if fv else 0.0,
                    "state_raw": raw_state.value,
                    "state_smooth": smooth_state.value,
                    "intervention_kind": event.kind.value if event else None,
                    "intervention_reason": event.reason if event else None,
                }
            )
            logger.log(row)

        # UI
        if show_preview and frame is not None:
            _overlay_text(frame, smooth_state.value)
            cv2.imshow("FocusGuard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            camera_on = not camera_on
            if camera_on:
                run_id = str(uuid.uuid4())
            else:
                run_id = None
            print(f"Camera {'ON' if camera_on else 'OFF'}")

        # FPS
        fps_frames += 1
        if now - fps_last >= 1.0:
            fps = fps_frames / (now - fps_last)
            print(f"FPS: {fps:.1f} STATE={smooth_state.value}")
            fps_frames = 0
            fps_last = now

    logger.close()
    cap.release()
    cv2.destroyAllWindows()
