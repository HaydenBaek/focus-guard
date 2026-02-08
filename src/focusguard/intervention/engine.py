from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, List

from focusguard.model.rules_baseline import FocusState


class InterventionType(str, Enum):
    NONE = "NONE"
    BEEP = "BEEP"
    VIDEO = "VIDEO"
    OVERLAY = "OVERLAY"


@dataclass
class InterventionEvent:
    timestamp: float
    state: FocusState
    kind: InterventionType
    reason: str


@dataclass
class InterventionConfig:
    soft_nudge_seconds: float = 4.0
    escalate_seconds: float = 12.0
    cooldown_seconds: float = 90.0


class InterventionEngine:
    """
    Local-only intervention policy engine.
    This file makes decisions only; it does NOT play audio/video.
    (Playback comes later in a separate adapter.)
    """

    def __init__(self, cfg: Dict[str, Any], interventions_cfg: Dict[str, Any]) -> None:
        icfg = cfg.get("intervention", {})
        self.cfg = InterventionConfig(
            soft_nudge_seconds=float(icfg.get("soft_nudge_seconds", 4.0)),
            escalate_seconds=float(icfg.get("escalate_seconds", 12.0)),
            cooldown_seconds=float(icfg.get("cooldown_seconds", 90.0)),
        )

        # interventions_cfg is configs/interventions.yaml loaded dict
        self._map: Dict[FocusState, List[InterventionType]] = {}
        for k, v in (interventions_cfg or {}).items():
            try:
                state = FocusState(k)
            except Exception:
                continue
            kinds: List[InterventionType] = []
            for item in (v or []):
                try:
                    kinds.append(InterventionType(item))
                except Exception:
                    pass
            self._map[state] = kinds

        self._current_bad_state: Optional[FocusState] = None
        self._bad_since: Optional[float] = None

        self._last_intervention_at: Optional[float] = None
        self._soft_fired_for_episode: bool = False
        self._escalate_fired_for_episode: bool = False

    def update(self, state: FocusState, now: Optional[float] = None) -> Optional[InterventionEvent]:
        """
        Call this every loop with the (smoothed) state.
        Returns an InterventionEvent when we should trigger something, else None.
        """
        t = now if now is not None else time.time()

        if state == FocusState.FOCUSED:
            # Immediately stop interventions when focus returns (handled by UI/player layer),
            # and reset episode tracking.
            self._reset_episode()
            return None

        # We only intervene on sustained DISTRACTED/AWAY episodes
        if self._current_bad_state != state:
            self._current_bad_state = state
            self._bad_since = t
            self._soft_fired_for_episode = False
            self._escalate_fired_for_episode = False

        if self._bad_since is None:
            self._bad_since = t

        # Cooldown check
        if self._last_intervention_at is not None:
            if (t - self._last_intervention_at) < self.cfg.cooldown_seconds:
                return None

        duration = t - self._bad_since

        # Soft nudge
        if (not self._soft_fired_for_episode) and duration >= self.cfg.soft_nudge_seconds:
            kind = self._pick_kind(state, prefer=InterventionType.BEEP, fallback=InterventionType.OVERLAY)
            self._soft_fired_for_episode = True
            self._last_intervention_at = t
            return InterventionEvent(
                timestamp=t,
                state=state,
                kind=kind,
                reason=f"sustained_{state.value.lower()}_{duration:.1f}s_soft",
            )

        # Escalation
        if (not self._escalate_fired_for_episode) and duration >= self.cfg.escalate_seconds:
            kind = self._pick_kind(state, prefer=InterventionType.VIDEO, fallback=InterventionType.BEEP)
            self._escalate_fired_for_episode = True
            self._last_intervention_at = t
            return InterventionEvent(
                timestamp=t,
                state=state,
                kind=kind,
                reason=f"sustained_{state.value.lower()}_{duration:.1f}s_escalate",
            )

        return None

    def _pick_kind(
        self,
        state: FocusState,
        prefer: InterventionType,
        fallback: InterventionType,
    ) -> InterventionType:
        kinds = self._map.get(state, [])
        if prefer in kinds:
            return prefer
        if fallback in kinds:
            return fallback
        return prefer if prefer != InterventionType.NONE else fallback

    def _reset_episode(self) -> None:
        self._current_bad_state = None
        self._bad_since = None
        self._soft_fired_for_episode = False
        self._escalate_fired_for_episode = False
