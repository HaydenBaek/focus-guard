from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from focusguard.intervention.engine import InterventionEvent, InterventionType


@dataclass
class InterventionAssets:
    base_dir: Path


class LocalInterventionPlayer:
    """
    Local-only player for interventions.
    - BEEP: system beep (no files needed)
    - VIDEO: opens a local video file using the OS default player (no uploads)
    - OVERLAY: handled by UI layer (this class does nothing for overlay)
    """

    def __init__(self, assets: InterventionAssets) -> None:
        self.assets = assets

    def play(self, event: InterventionEvent, video_filename: Optional[str] = None) -> None:
        if event.kind == InterventionType.BEEP:
            self._beep()
            return

        if event.kind == InterventionType.VIDEO:
            if not video_filename:
                return
            path = (self.assets.base_dir / video_filename).resolve()
            if not path.exists():
                return
            self._open_file(path)
            return

        # OVERLAY or NONE: no-op here
        return

    def stop(self) -> None:
        """
        Best-effort stop. We intentionally keep this conservative:
        stopping external players reliably across OSes is messy.
        UI layer should stop overlays immediately; videos are short clips.
        """
        return

    def _beep(self) -> None:
        # Terminal bell
        print("\a", end="", flush=True)

    def _open_file(self, path: Path) -> None:

        if os.name == "posix":
            if subprocess.call(["uname"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                # likely macOS/Linux
                if Path("/usr/bin/open").exists():
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        else:
            # Windows
            os.startfile(str(path))  # type: ignore[attr-defined]
