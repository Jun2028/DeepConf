from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping, Sequence, Any


@dataclass(slots=True)
class VoteResult:
    answer: int | None
    votes: dict[int, float]
    tie: bool
    valid_count: int


def _sample_answer(sample: Mapping[str, Any]) -> int | None:
    answer = sample.get("answer")
    if answer is None:
        return None
    return int(answer)


def _sample_confidence(sample: Mapping[str, Any]) -> float:
    confidence = sample.get("confidence")
    if confidence is None:
        return 0.0
    return float(confidence)


def _sample_id(sample: Mapping[str, Any], fallback: int) -> str:
    sample_id = sample.get("sample_id")
    if sample_id is None:
        return str(fallback)
    return str(sample_id)


def _valid_samples(samples: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [sample for sample in samples if _sample_answer(sample) is not None]


def majority_vote(samples: Sequence[Mapping[str, Any]]) -> VoteResult:
    valid = _valid_samples(samples)
    if not valid:
        return VoteResult(answer=None, votes={}, tie=False, valid_count=0)

    counts = Counter(_sample_answer(sample) for sample in valid)
    max_count = max(counts.values())
    candidates = [answer for answer, count in counts.items() if count == max_count]
    chosen = min(candidates)
    tie = len(candidates) > 1

    votes = {int(answer): float(count) for answer, count in counts.items()}
    return VoteResult(answer=chosen, votes=votes, tie=tie, valid_count=len(valid))


def weighted_vote(samples: Sequence[Mapping[str, Any]]) -> VoteResult:
    valid = _valid_samples(samples)
    if not valid:
        return VoteResult(answer=None, votes={}, tie=False, valid_count=0)

    totals: dict[int, float] = defaultdict(float)
    for sample in valid:
        answer = _sample_answer(sample)
        assert answer is not None
        totals[answer] += _sample_confidence(sample)

    max_total = max(totals.values())
    candidates = [answer for answer, total in totals.items() if total == max_total]
    chosen = min(candidates)
    tie = len(candidates) > 1

    return VoteResult(answer=chosen, votes=dict(totals), tie=tie, valid_count=len(valid))


def filter_top_eta(samples: Sequence[Mapping[str, Any]], eta: float) -> list[Mapping[str, Any]]:
    if not (0 < eta <= 1):
        raise ValueError("eta must be in (0, 1].")

    valid = _valid_samples(samples)
    if not valid:
        return []

    keep_count = max(1, ceil(eta * len(valid)))

    def sort_key(item: Mapping[str, Any], idx: int) -> tuple[float, str]:
        confidence = _sample_confidence(item)
        sample_id = _sample_id(item, fallback=idx)
        # Negative confidence for descending order, then stable tie-break by id.
        return (-confidence, sample_id)

    ranked = sorted(
        enumerate(valid),
        key=lambda pair: sort_key(pair[1], pair[0]),
    )

    kept_pairs = ranked[:keep_count]
    kept = [pair[1] for pair in kept_pairs]
    return kept


def cons_at_k(samples: Sequence[Mapping[str, Any]]) -> VoteResult:
    return majority_vote(samples)


def measure_at_k(samples: Sequence[Mapping[str, Any]]) -> VoteResult:
    return weighted_vote(samples)


def measure_top_eta_at_k(samples: Sequence[Mapping[str, Any]], eta: float) -> VoteResult:
    kept = filter_top_eta(samples, eta=eta)
    return weighted_vote(kept)