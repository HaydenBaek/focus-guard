from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict

import cv2
import numpy as np
import uvicorn

from focusguard.features.mediapipe_extractor import MediaPipeFaceExtractor
from focusguard.features.feature_vector import build_feature_vector
from focusguard.model.rules_baseline import RulesBaseline, FocusState
from focusguard.model.smoothing import SlidingWindowSmoother
from focusguard.logging.parquet_logger import ParquetEventLogger
from focusguard.logging.schemas import make_empty_row
from focusguard.runtime.controller import FocusGuardController
from focusguard.runtime.control_api import (
    create_control_app,
    InterventionConfig,
    set_latest_frame,
)
from focusguard.runtime.intervention import play_local_video


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


def _draw_face_box(frame: np.ndarray, bbox_norm, pad: float = 0.08) -> None:
    if not bbox_norm:
        return
    h, w = frame.shape[:2]
    x1 = int(max(0.0, bbox_norm[0] - pad) * w)
    y1 = int(max(0.0, bbox_norm[1] - pad) * h)
    x2 = int(min(1.0, bbox_norm[2] + pad) * w)
    y2 = int(min(1.0, bbox_norm[3] + pad) * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)


def run_camera_loop(cfg: Dict[str, Any]) -> None:
    cam_cfg = cfg.get("camera", {})
    runtime_cfg = cfg.get("runtime", {})

    index = int(cam_cfg.get("index", 0))
    show_preview = bool(runtime_cfg.get("show_preview", True))
    camera_on = bool(cam_cfg.get("enabled", True))
    session_id = str(uuid.uuid4())
    run_id = None

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check OS permissions.")

    # Core components
    extractor = MediaPipeFaceExtractor()
    baseline = RulesBaseline(cfg)
    smoother = SlidingWindowSmoother(cfg)

    # Logging
    logger = ParquetEventLogger(cfg)

    # Control plane
    controller = FocusGuardController()
    control_app = create_control_app(
        controller, InterventionConfig(video_path="assets/interventions/focus.mp4")
    )

    def run_api() -> None:
        uvicorn.run(control_app, host="127.0.0.1", port=8001, log_level="warning")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print("FocusGuard running. Press 'q' to quit, 'c' to toggle camera.")

    fps_last = time.time()
    fps_frames = 0
    last_mode = controller.snapshot()["mode"]

    while True:
        now = time.time()

        mode = controller.snapshot()["mode"]
        if mode != last_mode:
            if mode == "RUNNING" and camera_on:
                run_id = str(uuid.uuid4())
            else:
                run_id = None
            last_mode = mode

        effective_camera_on = camera_on and mode == "RUNNING"

        if effective_camera_on:
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
        should_fire = controller.update_observation(smooth_state.value, now=now)
        if should_fire:
            threading.Thread(
                target=lambda: play_local_video("assets/interventions/focus.mp4"),
                daemon=True,
            ).start()
            controller.mark_intervention_fired(now=now)

        # Logging (derived-only)
        if effective_camera_on and run_id is not None:
            row = make_empty_row(logger.schema)
            row.update(
                {
                    "ts": now,
                    "date": time.strftime("%Y-%m-%d", time.localtime(now)),
                    "session_id": session_id,
                    "run_id": run_id,
                    "session_mode": mode,
                    "camera_on": int(effective_camera_on),
                    "face_present": fv.face_present if fv else 0,
                    "nose_offset_abs": fv.nose_offset_abs if fv else 0.0,
                    "nose_offset_signed": fv.nose_offset_signed if fv else 0.0,
                    "state_raw": raw_state.value,
                    "state_smooth": smooth_state.value,
                    "intervention_fired": int(should_fire),
                    "intervention_type": "video" if should_fire else None,
                    "policy_reason": "distracted_streak" if should_fire else None,
                    "intervention_kind": "video" if should_fire else None,
                    "intervention_reason": "distracted_streak" if should_fire else None,
                }
            )
            logger.log(row)

        # UI
        if frame is not None:
            _overlay_text(frame, f"{smooth_state.value} | {controller.snapshot()['mode']}")
            if ff and ff.face_present:
                _draw_face_box(frame, ff.bbox_norm)
            set_latest_frame(frame)
            if show_preview:
                cv2.imshow("FocusGuard", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            camera_on = not camera_on
            if camera_on and mode == "RUNNING":
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
