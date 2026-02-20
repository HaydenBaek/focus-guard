# FocusGuard

On-device focus monitoring and intervention prototype.

FocusGuard captures webcam frames locally, extracts privacy-safe numeric features, predicts focus state in real time, and logs derived events to Parquet. It also exposes a local control UI (`/ui`) with MJPEG streaming (`/stream`) and session controls (`start/pause/stop`).

## What Is Implemented

- Real-time camera loop with OpenCV capture and overlay
- Face feature extraction with MediaPipe Face Landmarker
- Deterministic baseline state model (`FOCUSED`, `DISTRACTED`, `AWAY`)
- Sliding-window smoothing for stable state transitions
- Local control plane (FastAPI in background thread) with:
- `/ui` dashboard
- `/stream` MJPEG feed
- `/status` session/state metrics
- `/session/start`, `/session/pause`, `/session/stop`
- `/intervention/test` demo trigger
- Session/run tracking in logs (`session_id`, `run_id`)
- Derived-only event logging to partitioned Parquet (`data/events/date=YYYY-MM-DD/*.parquet`)
- Basic training pipeline (scikit-learn logistic regression + MLflow tracking)
- Basic model serving API (`/predict`, `/latency`, `/health`)

## Tech Stack (from codebase)

- Language: Python
- CV/vision: OpenCV (`cv2`), MediaPipe
- Data: pandas, NumPy, PyArrow/Parquet
- API/backend: FastAPI, Uvicorn, Pydantic
- ML: scikit-learn (logistic regression), joblib, MLflow
- Config: YAML (`PyYAML`)
- Local playback: OS-native `open`/`xdg-open`/`cmd start`

## Repository Structure

```text
focus-guard/
  configs/
    app.yaml
    interventions.yaml
    logging.yaml
    train.yaml
  assets/
    interventions/
      focus.mp4
  data/
    events/date=YYYY-MM-DD/*.parquet
  models/
  reports/
  scripts/
    download_mediapipe_models.py
  src/focusguard/
    cli.py
    realtime/loop.py
    runtime/{controller.py,control_api.py,intervention.py}
    features/{mediapipe_extractor.py,feature_vector.py}
    model/{rules_baseline.py,smoothing.py}
    logging/{schemas.py,parquet_logger.py}
    data/{dataset.py,labeling.py,quality.py,splits.py}
    train/train_sklearn.py
    serve/{app.py,model_loader.py,latency.py}
```

## Privacy Model

- No camera frames are written to disk by the pipeline.
- Event logs store only derived numeric/state fields.
- MJPEG frames are streamed over `127.0.0.1` for local UI preview only.

## Prerequisites

- Python 3.11+ recommended
- Webcam access enabled for terminal/python process
- A local model file for MediaPipe face landmarker:
- `assets/models/face_landmarker.task`

Download the MediaPipe model:

```bash
python3 scripts/download_mediapipe_models.py
```

## Setup

`pyproject.toml` is currently empty, so install dependencies manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe numpy pandas pyarrow pyyaml fastapi uvicorn pydantic scikit-learn joblib mlflow
```

Run commands with:

```bash
PYTHONPATH=src ...
```

## Run Real-Time App + Control UI

```bash
PYTHONPATH=src python3 -m focusguard.cli camera --config configs/app.yaml
```

Open:

- UI: `http://127.0.0.1:8001/ui`
- Stream: `http://127.0.0.1:8001/stream`
- Status JSON: `http://127.0.0.1:8001/status`

Keyboard controls in camera loop:

- `q`: quit process
- `c`: toggle camera capture on/off

Session controls in UI:

- `Start`: mode `RUNNING`
- `Pause`: mode `PAUSED`
- `Stop`: mode `STOPPED`

## Event Logging Schema

Required columns enforced by loader (`src/focusguard/data/dataset.py`):

- `ts`
- `date`
- `session_id`
- `run_id`
- `session_mode`
- `camera_on`
- `face_present`
- `nose_offset_abs`
- `nose_offset_signed`
- `state_raw`
- `state_smooth`
- `intervention_fired`
- `intervention_type`
- `policy_reason`
- `intervention_kind`
- `intervention_reason`

## Train a Model

```bash
PYTHONPATH=src python3 -m focusguard.train.train_sklearn --config configs/train.yaml
```

Outputs:

- Model artifact: `models/focusguard_logreg.joblib`
- Report: `reports/latest_eval.json`
- MLflow tracking: `mlflow.db` and `mlruns/`

Notes:

- Training expects Parquet event files under `data/events/date=*/`.
- Loader fails fast if required schema columns are missing.

## Serve Trained Model

```bash
PYTHONPATH=src python3 -m focusguard.serve.app
```

Server defaults to `127.0.0.1:8000`.

Endpoints:

- `GET /health`
- `GET /config`
- `GET /latency`
- `POST /predict`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"face_present":1,"nose_offset_abs":0.02,"nose_offset_signed":0.01}'
```

## Current Gaps / Scaffolded Files

These files exist but are currently empty placeholders:

- `pyproject.toml`
- `Makefile`
- `scripts/bootstrap_venv.sh`
- `tests/test_config.py`
- `tests/test_rules_baseline.py`
- `tests/test_smoothing.py`
- `src/focusguard/train/train_torch.py`
- `src/focusguard/train/registry.py`
- `src/focusguard/eval/metrics.py`
- `src/focusguard/eval/report.py`
- `src/focusguard/eval/slices.py`
- `src/focusguard/eval/calibration.py`
- `src/focusguard/privacy.py`
- `src/focusguard/realtime/camera.py`
- `src/focusguard/realtime/overlay.py`

## Configuration Files

- `configs/app.yaml`: runtime camera/smoothing/intervention/logging defaults
- `configs/interventions.yaml`: mapping from state to intervention type
- `configs/logging.yaml`: Parquet logger flush/partition settings
- `configs/train.yaml`: data path, feature columns, split strategy, model + MLflow settings

## Development Notes

- `.gitignore` excludes `data/events/`, `*.parquet`, `models/`, and `reports/`.
- Existing local artifacts in this repo indicate prior runs:
- `data/events/...`
- `models/focusguard_logreg.joblib`
- `reports/latest_eval.json`
- `mlflow.db`, `mlruns/`
