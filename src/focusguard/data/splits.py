from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    method: str = "by_date"
    test_days: int = 2
    val_days: int = 2


def split_by_date(
    df: pd.DataFrame,
    *,
    test_days: int,
    val_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leak-safe split by date partitions.
    Assumes df["date"] exists and is YYYY-MM-DD.

    Oldest dates → train
    Newer dates   → val
    Newest dates  → test
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column for by_date split")

    dates = sorted(d for d in df["date"].dropna().unique())
    if len(dates) < (test_days + val_days + 1):
        raise ValueError(
            f"Not enough unique dates for split: "
            f"have={len(dates)} need>={test_days + val_days + 1}"
        )

    test_dates = set(dates[-test_days:])
    val_dates = set(dates[-(test_days + val_days):-test_days])
    train_dates = set(dates[: -(test_days + val_days)])

    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    val_df = df[df["date"].isin(val_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)

    return train_df, val_df, test_df


def make_splits(
    df: pd.DataFrame,
    train_cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Entry point used by training scripts.
    """
    split_cfg = train_cfg.get("split", {})
    cfg = SplitConfig(
        method=str(split_cfg.get("method", "by_date")),
        test_days=int(split_cfg.get("test_days", 2)),
        val_days=int(split_cfg.get("val_days", 2)),
    )

    if cfg.method != "by_date":
        raise ValueError(f"Unsupported split method: {cfg.method}")

    return split_by_date(
        df,
        test_days=cfg.test_days,
        val_days=cfg.val_days,
    )
