from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from focusguard.logging.schemas import EventSchema


@dataclass
class ParquetLoggerConfig:
    output_dir: Path = Path("data/events")
    flush_every_n: int = 50


class ParquetEventLogger:
    """
    Privacy-safe Parquet logger.
    Stores ONLY derived features + timestamps + predictions + interventions.
    Never writes frames/video.
    """

    def __init__(self, cfg: Dict[str, Any], logging_cfg: Optional[Dict[str, Any]] = None) -> None:
        app_logging = cfg.get("logging", {})
        parquet_cfg = (logging_cfg or {}).get("parquet", {})

        self.cfg = ParquetLoggerConfig(
            output_dir=Path(app_logging.get("output_dir", "data/events")),
            flush_every_n=int(parquet_cfg.get("flush_every_n", 50)),
        )
        self.schema = EventSchema.default()

        self._buffer: List[Dict[str, Any]] = []
        self._enabled: bool = bool(app_logging.get("enabled", True))

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, row: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        # Ensure all expected columns exist (don’t silently drift)
        for col in self.schema.columns:
            if col not in row:
                row[col] = None

        self._buffer.append(row)

        if len(self._buffer) >= self.cfg.flush_every_n:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._enabled:
            return None
        if not self._buffer:
            return None

        # Partition by date if provided
        date_val = self._buffer[-1].get("date")
        if not date_val:
            # fallback to today's date (local)
            date_val = time.strftime("%Y-%m-%d", time.localtime())

        out_dir = self.cfg.output_dir / f"date={date_val}"
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"events_{int(time.time())}_{uuid.uuid4().hex[:8]}.parquet"
        out_path = out_dir / filename

        table = pa.Table.from_pylist(self._buffer)
        pq.write_table(table, out_path)

        self._buffer.clear()
        return out_path

    def close(self) -> None:
        self.flush()
