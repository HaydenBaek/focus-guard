from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen, Request

# Official MediaPipe FaceLandmarker model bundle (.task)
# Source: Google AI Edge docs (storage.googleapis.com)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

OUT_PATH = Path("assets/models/face_landmarker.task")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "FocusGuard/1.0"})
    with urlopen(req) as resp:
        data = resp.read()

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out_path)


def main() -> None:
    print(f"Downloading FaceLandmarker model to: {OUT_PATH}")
    download(MODEL_URL, OUT_PATH)
    print("Done.")
    print(f"Size: {OUT_PATH.stat().st_size / (1024*1024):.2f} MB")
    print(f"SHA256: {_sha256(OUT_PATH)}")


if __name__ == "__main__":
    main()
