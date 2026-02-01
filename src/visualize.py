from __future__ import annotations

from pathlib import Path
import json


def load_summary(path: Path | str) -> dict:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)