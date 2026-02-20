import subprocess
import sys
from pathlib import Path


def play_local_video(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Intervention video not found: {path}")

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["cmd", "/c", "start", "", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])
