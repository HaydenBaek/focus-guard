from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time


@dataclass
class ControllerState:
    mode: str = "IDLE"  # IDLE | RUNNING | PAUSED | STOPPED
    current_state: str = "UNKNOWN"  # FOCUSED | DISTRACTED | AWAY | UNKNOWN
    distracted_streak_s: float = 0.0
    focused_streak_s: float = 0.0
    away_streak_s: float = 0.0
    last_intervention_ts: float = 0.0
    cooldown_s: float = 60.0
    distracted_trigger_s: float = 8.0
    away_ignore_s: float = 15.0
    last_update_ts: float = field(default_factory=time.time)
    session_start_ts: float = 0.0
    session_end_ts: float = 0.0
    total_run_s: float = 0.0
    focused_time_s: float = 0.0
    distracted_time_s: float = 0.0
    away_time_s: float = 0.0


class FocusGuardController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = ControllerState()

    def start(self) -> None:
        with self._lock:
            now = time.time()
            self.state.mode = "RUNNING"
            self.state.session_start_ts = now
            self.state.session_end_ts = 0.0
            self.state.total_run_s = 0.0
            self.state.focused_time_s = 0.0
            self.state.distracted_time_s = 0.0
            self.state.away_time_s = 0.0
            self.state.distracted_streak_s = 0.0
            self.state.focused_streak_s = 0.0
            self.state.away_streak_s = 0.0

    def pause(self) -> None:
        with self._lock:
            self.state.mode = "PAUSED"

    def stop(self) -> None:
        with self._lock:
            self.state.mode = "STOPPED"
            self.state.session_end_ts = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            s = self.state
            focus_score = (
                round((s.focused_time_s / s.total_run_s) * 100.0, 2)
                if s.total_run_s > 0
                else 0.0
            )
            return {
                "mode": s.mode,
                "current_state": s.current_state,
                "distracted_streak_s": round(s.distracted_streak_s, 2),
                "focused_streak_s": round(s.focused_streak_s, 2),
                "away_streak_s": round(s.away_streak_s, 2),
                "last_intervention_ts": s.last_intervention_ts,
                "cooldown_s": s.cooldown_s,
                "distracted_trigger_s": s.distracted_trigger_s,
                "away_ignore_s": s.away_ignore_s,
                "focus_score": focus_score,
                "session_start_ts": s.session_start_ts,
                "session_end_ts": s.session_end_ts,
                "total_run_s": round(s.total_run_s, 2),
                "focused_time_s": round(s.focused_time_s, 2),
                "distracted_time_s": round(s.distracted_time_s, 2),
                "away_time_s": round(s.away_time_s, 2),
            }

    def update_observation(self, predicted_state: str, now: float | None = None) -> bool:
        """Update streaks; returns True if intervention should fire."""
        now = now or time.time()
        with self._lock:
            s = self.state
            dt = max(0.0, now - s.last_update_ts)
            s.last_update_ts = now
            s.current_state = predicted_state

            if s.mode != "RUNNING":
                return False

            if predicted_state == "DISTRACTED":
                s.total_run_s += dt
                s.distracted_time_s += dt
                s.distracted_streak_s += dt
                s.focused_streak_s = 0.0
                s.away_streak_s = 0.0
            elif predicted_state == "FOCUSED":
                s.total_run_s += dt
                s.focused_time_s += dt
                s.focused_streak_s += dt
                s.distracted_streak_s = 0.0
                s.away_streak_s = 0.0
            elif predicted_state == "AWAY":
                s.total_run_s += dt
                s.away_time_s += dt
                s.away_streak_s += dt
                s.distracted_streak_s = 0.0
                s.focused_streak_s = 0.0
            else:
                s.distracted_streak_s = 0.0
                s.focused_streak_s = 0.0
                s.away_streak_s = 0.0

            if s.away_streak_s >= s.away_ignore_s:
                return False

            if s.last_intervention_ts and (now - s.last_intervention_ts) < s.cooldown_s:
                return False

            return s.distracted_streak_s >= s.distracted_trigger_s

    def mark_intervention_fired(self, now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            self.state.last_intervention_ts = now
            self.state.distracted_streak_s = 0.0
