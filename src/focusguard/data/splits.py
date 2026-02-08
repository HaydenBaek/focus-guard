from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    method: str = "by_date"
    test_days: int = 0
    val_days: int = 0


def split_by_date(
    df: pd.DataFrame,
    *,
    test_days: int,
    val_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leak-safe split by date partitions.
    Supports bootstrapping with test_days=0 / val_days=0.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column for by_date split")

    dates = sorted(d for d in df["date"].dropna().unique())
    need = test_days + val_days + 1
    if len(dates) < need:
        raise ValueError(
            f"Not enough unique dates for split: have={len(dates)} need>={need}"
        )

    test_dates = set(dates[-test_days:]) if test_days > 0 else set()
    val_dates = (
        set(dates[-(test_days + val_days):-test_days]) if val_days > 0 else set()
    )

    train_mask = ~df["date"].isin(test_dates | val_dates)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = (
        df[df["date"].isin(val_dates)].reset_index(drop=True)
        if val_days > 0
        else df.iloc[0:0].copy()
    )
    test_df = (
        df[df["date"].isin(test_dates)].reset_index(drop=True)
        if test_days > 0
        else df.iloc[0:0].copy()
    )

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
        test_days=int(split_cfg.get("test_days", 0)),
        val_days=int(split_cfg.get("val_days", 0)),
    )

    if cfg.method != "by_date":
        raise ValueError(f"Unsupported split method: {cfg.method}")

    return split_by_date(
        df,
        test_days=cfg.test_days,
        val_days=cfg.val_days,
    )