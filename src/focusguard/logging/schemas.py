from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass(frozen=True)
class EventSchema:
    """
    Stable logging contract (derived-only, privacy-safe).
    Keep this consistent across pipeline + training.
    """
    columns: List[str]

    @staticmethod
    def default() -> "EventSchema":
        return EventSchema(
            columns=[
                # Time
                "ts",                  # float epoch seconds
                "date",                # YYYY-MM-DD (partition key)

                # Runtime
                "fps",                 # float
                "camera_on",           # int 0/1

                # Derived features (NO frames)
                "face_present",        # int 0/1
                "nose_offset_abs",     # float
                "nose_offset_signed",  # float

                # Predictions
                "state_raw",           # str
                "state_smooth",        # str

                # Interventions (nullable)
                "intervention_kind",   # str or None
                "intervention_reason", # str or None
            ]
        )


def make_empty_row(schema: EventSchema) -> Dict[str, Any]:
    """
    Convenience helper for building rows with all columns present.
    """
    return {c: None for c in schema.columns}
