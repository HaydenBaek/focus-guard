from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd


@dataclass(frozen=True)
class LabelingConfig:
    strategy: str = "bootstrap_rules"
    away_label_if_face_missing: bool = True


def apply_bootstrap_labels(
    df: pd.DataFrame,
    *,
    away_label_if_face_missing: bool = True,
) -> pd.Series:
    """
    Weak labeling strategy for MVP training.
    Uses system outputs + simple guards to create labels.

    Returns a pandas Series of labels (str).
    """
    if "state_smooth" not in df.columns:
        raise ValueError("DataFrame must contain 'state_smooth' for bootstrap labeling")

    labels = df["state_smooth"].astype("string").copy()

    if away_label_if_face_missing and "face_present" in df.columns:
        labels.loc[df["face_present"] == 0] = "AWAY"

    return labels


def make_labels(
    df: pd.DataFrame,
    train_cfg: Dict[str, Any],
) -> pd.Series:
    """
    Entry point for labeling.
    """
    labels_cfg = train_cfg.get("labels", {})
    cfg = LabelingConfig(
        strategy=str(labels_cfg.get("strategy", "bootstrap_rules")),
        away_label_if_face_missing=bool(
            labels_cfg.get("away_label_if_face_missing", True)
        ),
    )

    if cfg.strategy != "bootstrap_rules":
        raise ValueError(f"Unsupported labeling strategy: {cfg.strategy}")

    return apply_bootstrap_labels(
        df,
        away_label_if_face_missing=cfg.away_label_if_face_missing,
    )
