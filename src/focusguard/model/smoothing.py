from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any

from focusguard.model.rules_baseline import FocusState


@dataclass
class SmoothingConfig:
    window_size: int = 7
    min_consensus: int = 5


class SlidingWindowSmoother:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        s_cfg = cfg.get("smoothing", {})
        self.cfg = SmoothingConfig(
            window_size=int(s_cfg.get("window_size", 7)),
            min_consensus=int(s_cfg.get("min_consensus", 5)),
        )
        if self.cfg.window_size < 1:
            raise ValueError("smoothing.window_size must be >= 1")
        if self.cfg.min_consensus < 1 or self.cfg.min_consensus > self.cfg.window_size:
            raise ValueError("smoothing.min_consensus must be in [1, window_size]")

        self._buf: Deque[FocusState] = deque(maxlen=self.cfg.window_size)
        self._stable: FocusState = FocusState.AWAY

    def update(self, state: FocusState) -> FocusState:
        self._buf.append(state)

        # Vote for the most common state in the window
        counts = {}
        for s in self._buf:
            counts[s] = counts.get(s, 0) + 1

        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        winner_count = counts[winner]

        # Only switch when enough agreement is present
        if winner_count >= self.cfg.min_consensus:
            self._stable = winner

        return self._stable
