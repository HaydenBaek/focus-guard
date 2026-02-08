from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    rows: int
    missing_rates: Dict[str, float]
    label_distribution: Dict[str, int]
    warnings: List[str]


def check_quality(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    feature_cols: List[str] | None = None,
    max_missing_rate: float = 0.2,
) -> QualityReport:
    """
    Basic data quality checks for training.
    - missing rates per column
    - label distribution
    - simple warnings for skew / missingness
    """
    warnings: List[str] = []

    if feature_cols is None:
        feature_cols = []

    cols_to_check = list(set(feature_cols + [label_col]))

    missing_rates: Dict[str, float] = {}
    for c in cols_to_check:
        if c not in df.columns:
            missing_rates[c] = 1.0
            warnings.append(f"missing_column:{c}")
            continue
        rate = float(df[c].isna().mean())
        missing_rates[c] = rate
        if rate > max_missing_rate:
            warnings.append(f"high_missing_rate:{c}:{rate:.2f}")

    label_distribution: Dict[str, int] = {}
    if label_col in df.columns:
        vc = df[label_col].value_counts(dropna=False)
        label_distribution = {str(k): int(v) for k, v in vc.items()}

        total = sum(label_distribution.values())
        if total > 0:
            for k, v in label_distribution.items():
                if v / total > 0.95:
                    warnings.append(f"label_skew:{k}:{v/total:.2f}")

    return QualityReport(
        rows=int(len(df)),
        missing_rates=missing_rates,
        label_distribution=label_distribution,
        warnings=warnings,
    )
