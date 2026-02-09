from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "ts",
    "date",
    "session_id",
    "run_id",
    "camera_on",
    "face_present",
    "nose_offset_abs",
    "nose_offset_signed",
    "state_raw",
    "state_smooth",
    "intervention_kind",
    "intervention_reason",
]


@dataclass(frozen=True)
class DatasetConfig:
    events_dir: Path
    min_rows: int = 0
    required_cols: Tuple[str, ...] = tuple(REQUIRED_COLUMNS)


class DatasetError(RuntimeError):
    pass


def _find_parquet_files(events_dir: Path) -> List[Path]:
    # Supports partitioned layout: data/events/date=YYYY-MM-DD/*.parquet
    pattern = str(events_dir / "date=*" / "*.parquet")
    files = [Path(p) for p in glob.glob(pattern)]
    files.sort()
    return files


def _enforce_schema(df: pd.DataFrame, required_cols: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required columns: {missing}")

    # Keep only required + any extra columns (we’ll allow extras but keep them)
    # (Downstream code should select feature cols explicitly.)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Make types predictable across Parquet writers/versions
    df = df.copy()

    # numeric
    for c in ["ts", "nose_offset_abs", "nose_offset_signed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["camera_on", "face_present"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # strings (nullable)
    for c in [
        "date",
        "session_id",
        "run_id",
        "state_raw",
        "state_smooth",
        "intervention_kind",
        "intervention_reason",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def load_events_df(
    train_cfg: Dict[str, Any],
    *,
    limit_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads FocusGuard event logs from partitioned Parquet into a single DataFrame.
    Enforces required columns and applies basic type coercions.

    train_cfg expects:
      train_cfg["data"]["events_dir"] : str
      train_cfg["data"]["min_rows"]   : int (optional)
    """
    data_cfg = train_cfg.get("data", {})
    events_dir = Path(data_cfg.get("events_dir", "data/events"))
    min_rows = int(data_cfg.get("min_rows", 0))

    cfg = DatasetConfig(events_dir=events_dir, min_rows=min_rows)

    if not cfg.events_dir.exists():
        raise DatasetError(f"events_dir not found: {cfg.events_dir.resolve()}")

    files = _find_parquet_files(cfg.events_dir)
    if not files:
        raise DatasetError(f"No parquet files found under: {cfg.events_dir.resolve()}")

    if limit_files is not None:
        files = files[: int(limit_files)]

    # Read all parquet files (simple pandas approach for MVP)
    dfs: List[pd.DataFrame] = []
    for fp in files:
        try:
            dfs.append(pd.read_parquet(fp))
        except Exception as e:
            raise DatasetError(f"Failed reading parquet: {fp} ({e})") from e

    df = pd.concat(dfs, ignore_index=True)

    df = _enforce_schema(df, cfg.required_cols)
    df = _coerce_types(df)

    # Basic cleanup: drop rows without timestamps
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    if cfg.min_rows and len(df) < cfg.min_rows:
        raise DatasetError(f"Not enough rows: {len(df)} < min_rows={cfg.min_rows}")

    return df


def summarize_events_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Small helper for debugging and reporting.
    """
    out: Dict[str, Any] = {
        "rows": int(len(df)),
        "dates": int(df["date"].nunique()) if "date" in df.columns else 0,
        "camera_on_rate": float(df["camera_on"].mean()) if "camera_on" in df.columns else None,
        "face_present_rate": float(df["face_present"].mean()) if "face_present" in df.columns else None,
        "state_smooth_counts": (
            df["state_smooth"].value_counts(dropna=False).to_dict()
            if "state_smooth" in df.columns
            else {}
        ),
    }
    return out
