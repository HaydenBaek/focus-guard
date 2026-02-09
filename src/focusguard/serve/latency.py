from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from focusguard.serve.model_loader import LoadedModel, predict_one


@dataclass(frozen=True)
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def _percentile_ms(samples_s: List[float], q: float) -> float:
    arr = np.array(samples_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, q))


def measure_latency(
    loaded: LoadedModel,
    *,
    n: int = 200,
    warmup: int = 20,
    sample_features: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Measures end-to-end Python inference latency for /predict path
    (model.predict + feature ordering).
    """
    if sample_features is None:
        sample_features = {
            "face_present": 1.0,
            "nose_offset_abs": 0.01,
            "nose_offset_signed": 0.01,
        }

    # Warmup
    for _ in range(warmup):
        predict_one(loaded, sample_features)

    samples: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = predict_one(loaded, sample_features)
        t1 = time.perf_counter()
        samples.append(t1 - t0)

    stats = LatencyStats(
        n=n,
        p50_ms=_percentile_ms(samples, 50),
        p95_ms=_percentile_ms(samples, 95),
        p99_ms=_percentile_ms(samples, 99),
        mean_ms=float(np.mean(np.array(samples)) * 1000.0),
    )
    return {"latency": stats.__dict__}
