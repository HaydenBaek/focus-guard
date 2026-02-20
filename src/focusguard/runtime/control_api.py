from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import threading
import time

from focusguard.runtime.controller import FocusGuardController
from focusguard.runtime.intervention import play_local_video
import cv2


frame_lock = threading.Lock()
latest_frame = None  # numpy array (BGR)


def set_latest_frame(frame) -> None:
    global latest_frame
    with frame_lock:
        latest_frame = None if frame is None else frame.copy()


def mjpeg_generator():
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )


class InterventionConfig(BaseModel):
    video_path: str = "assets/interventions/focus.mp4"


def create_control_app(
    controller: FocusGuardController, intervention_cfg: InterventionConfig
) -> FastAPI:
    app = FastAPI(title="FocusGuard Control Plane")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/status")
    def status():
        return controller.snapshot()

    @app.post("/session/start")
    def start():
        controller.start()
        return {"mode": "RUNNING"}

    @app.post("/session/pause")
    def pause():
        controller.pause()
        return {"mode": "PAUSED"}

    @app.post("/session/stop")
    def stop():
        controller.stop()
        return {"mode": "STOPPED"}

    @app.post("/intervention/test")
    def test_intervention():
        play_local_video(intervention_cfg.video_path)
        controller.mark_intervention_fired(time.time())
        return {"fired": True}

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>FocusGuard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    :root {
      --bg: #0f172a;
      --panel: #0b1220;
      --panel-2: #111827;
      --accent: #22d3ee;
      --accent-2: #a78bfa;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --border: #1f2937;
    }
    * { box-sizing: border-box; }
    body {
      font-family: "Space Grotesk", sans-serif;
      margin: 0;
      color: var(--text);
      background: radial-gradient(1200px 600px at 10% 10%, #0b1f3a 0%, var(--bg) 55%);
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 28px; }
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 18px;
    }
    .title { font-size: 24px; font-weight: 700; letter-spacing: 0.5px; }
    .pill { font-size: 12px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); }
    .btn {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      border: none; color: #07101f; font-weight: 700;
      padding: 10px 14px; margin-right: 8px; border-radius: 10px;
      cursor: pointer;
    }
    .btn.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
    .card {
      margin-top: 0; padding: 14px; border: 1px solid var(--border);
      border-radius: 14px; background: linear-gradient(180deg, var(--panel), var(--panel-2));
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .stat { padding: 10px; border: 1px solid var(--border); border-radius: 12px; background: #0a0f1d; }
    .label { font-size: 12px; color: var(--muted); }
    .value { font-size: 18px; font-weight: 600; }
    img { width: 100%; border-radius: 12px; border: 1px solid var(--border); }
    pre { background: #0b1020; border: 1px solid var(--border); padding: 10px; border-radius: 10px; overflow: auto; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="title">FocusGuard Control</div>
      <div class="pill">Local only · 127.0.0.1:8001</div>
    </div>
    <div style="margin-bottom: 14px;">
      <button class="btn" onclick="post('/session/start')">Start</button>
      <button class="btn secondary" onclick="post('/session/pause')">Pause</button>
      <button class="btn secondary" onclick="post('/session/stop')">Stop</button>
      <button class="btn secondary" onclick="post('/intervention/test')">Test Intervention</button>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Camera</h3>
        <img src="/stream" />
      </div>
      <div class="card">
        <h3>Status</h3>
        <div class="stats">
          <div class="stat"><div class="label">Mode</div><div class="value" id="mode">-</div></div>
          <div class="stat"><div class="label">State</div><div class="value" id="current_state">-</div></div>
          <div class="stat"><div class="label">Distracted (s)</div><div class="value" id="distracted_streak_s">-</div></div>
          <div class="stat"><div class="label">Focused (s)</div><div class="value" id="focused_streak_s">-</div></div>
          <div class="stat"><div class="label">Away (s)</div><div class="value" id="away_streak_s">-</div></div>
          <div class="stat"><div class="label">Last Intervention</div><div class="value" id="last_intervention_ts">-</div></div>
        </div>
        <pre id="status"><code>Loading...</code></pre>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h3>Session Summary</h3>
      <div class="stats">
        <div class="stat"><div class="label">Focus Score (%)</div><div class="value" id="focus_score">-</div></div>
        <div class="stat"><div class="label">Total Run (s)</div><div class="value" id="total_run_s">-</div></div>
        <div class="stat"><div class="label">Focused (s)</div><div class="value" id="focused_time_s">-</div></div>
        <div class="stat"><div class="label">Distracted (s)</div><div class="value" id="distracted_time_s">-</div></div>
        <div class="stat"><div class="label">Away (s)</div><div class="value" id="away_time_s">-</div></div>
        <div class="stat"><div class="label">Session Start</div><div class="value" id="session_start_ts">-</div></div>
        <div class="stat"><div class="label">Session End</div><div class="value" id="session_end_ts">-</div></div>
      </div>
    </div>
  </div>

<script>
async function post(path) {
  await fetch(path, { method: 'POST' });
  await refresh();
}
async function refresh() {
  const r = await fetch('/status');
  const j = await r.json();
  document.getElementById('mode').textContent = j.mode;
  document.getElementById('current_state').textContent = j.current_state;
  document.getElementById('distracted_streak_s').textContent = j.distracted_streak_s;
  document.getElementById('focused_streak_s').textContent = j.focused_streak_s;
  document.getElementById('away_streak_s').textContent = j.away_streak_s;
  document.getElementById('last_intervention_ts').textContent = j.last_intervention_ts;
  document.getElementById('focus_score').textContent = j.focus_score;
  document.getElementById('total_run_s').textContent = j.total_run_s;
  document.getElementById('focused_time_s').textContent = j.focused_time_s;
  document.getElementById('distracted_time_s').textContent = j.distracted_time_s;
  document.getElementById('away_time_s').textContent = j.away_time_s;
  document.getElementById('session_start_ts').textContent = j.session_start_ts;
  document.getElementById('session_end_ts').textContent = j.session_end_ts;
  document.getElementById('status').textContent = JSON.stringify(j, null, 2);
}
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""
        return HTMLResponse(html)

    @app.get("/stream")
    def stream():
        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app
