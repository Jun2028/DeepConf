from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json


@dataclass(slots=True)
class ProblemRecord:
    problem_id: str
    problem: str
    answer: int
    raw: dict[str, Any]


def _coerce_answer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid AIME answer.")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    raise ValueError(f"Cannot coerce answer value: {value!r}")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {path}.") from exc


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON list entries must be objects.")
            yield item
        return

    if isinstance(payload, dict) and "rows" in payload:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Dataset server JSON must contain a list under 'rows'.")
        for row_entry in rows:
            if not isinstance(row_entry, dict):
                raise ValueError("Each rows entry must be an object.")
            row = row_entry.get("row", row_entry)
            if not isinstance(row, dict):
                raise ValueError("Row entry is not an object.")
            yield row
        return

    if isinstance(payload, dict):
        yield payload
        return

    raise ValueError("Unsupported JSON dataset format.")


def load_problems(
    dataset_path: Path | str,
    *,
    problem_key: str = "problem",
    answer_key: str = "answer",
    id_key: str = "problem_idx",
    limit: int | None = None,
) -> list[ProblemRecord]:
    """Load AIME-style problems from JSON or JSONL files.

    Supports simple JSONL/JSON lists as well as the Hugging Face datasets-server
    response shape with a top-level 'rows' field.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if path.suffix.lower() == ".jsonl":
        iterator = _iter_jsonl(path)
    else:
        iterator = _iter_json(path)

    problems: list[ProblemRecord] = []
    for idx, record in enumerate(iterator):
        if problem_key not in record:
            raise KeyError(f"Missing problem key '{problem_key}' in dataset record {idx}.")
        if answer_key not in record:
            raise KeyError(f"Missing answer key '{answer_key}' in dataset record {idx}.")

        problem_id_value = record.get(id_key, idx)
        problem_id = str(problem_id_value)
        problem_text = str(record[problem_key])
        answer = _coerce_answer(record[answer_key])

        problems.append(
            ProblemRecord(
                problem_id=problem_id,
                problem=problem_text,
                answer=answer,
                raw=record,
            )
        )

        if limit is not None and len(problems) >= limit:
            break

    if not problems:
        raise ValueError(f"No problems found in dataset: {path}")

    return problems