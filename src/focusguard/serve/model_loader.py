from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Dict

import joblib
import numpy as np


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    feature_cols: List[str]


def load_model(model_path: str | Path) -> LoadedModel:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p.resolve()}")

    obj = joblib.load(p)

    # Support both keys (we used "features" in the latest file, older used "feature_cols")
    cols = obj.get("features") or obj.get("feature_cols")
    if not cols:
        raise ValueError("Model artifact missing 'features' or 'feature_cols'")

    return LoadedModel(model=obj["model"], feature_cols=list(cols))


def predict_one(loaded: LoadedModel, features: Dict[str, float]) -> Dict[str, Any]:
    x = np.array([[float(features.get(c, 0.0)) for c in loaded.feature_cols]], dtype=np.float32)
    pred = loaded.model.predict(x)[0]

    out: Dict[str, Any] = {"prediction": str(pred)}

    # Optional probability output if classifier supports it
    if hasattr(loaded.model, "predict_proba"):
        proba = loaded.model.predict_proba(x)[0].tolist()
        classes = [str(c) for c in loaded.model.classes_]
        out["proba"] = dict(zip(classes, proba))

    return out
