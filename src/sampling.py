from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import threading
import hashlib
import json
import re
import time

from tqdm import tqdm

from .aime_loader import ProblemRecord
from .confidence import compute_trace_confidence
from .config import RunConfig, ensure_directories
from .deepseek_client import ChatCompletionResult, DeepSeekClient, RetryableStatusError
from .extract_answer import extract_answer
from .prompting import BOXED_INSTRUCTION, build_prompt


@dataclass(slots=True)
class PoolMetadata:
    pool_id: str
    dataset_path: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    top_logprobs: int
    n_pool: int
    prompt_instruction: str
    built_at: str
    problems: list[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "pool"


def compute_pool_id(config: RunConfig) -> str:
    """Compute a stable, human-friendly pool id from key configuration values."""

    payload = {
        "dataset_path": str(config.dataset.dataset_path),
        "problem_key": config.dataset.problem_key,
        "answer_key": config.dataset.answer_key,
        "id_key": config.dataset.id_key,
        "model": config.api.model,
        "base_url": config.api.base_url,
        "temperature": config.decoding.temperature,
        "top_p": config.decoding.top_p,
        "max_tokens": config.decoding.max_tokens,
        "top_logprobs": config.decoding.top_logprobs,
        "n_pool": config.pooling.n_pool,
        "instruction": BOXED_INSTRUCTION,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:8]

    dataset_stem = Path(config.dataset.dataset_path).stem
    name_parts = [
        _slugify(dataset_stem),
        _slugify(config.api.model),
        f"n{config.pooling.n_pool}",
        f"mt{config.decoding.max_tokens}",
        f"t{_format_float(config.decoding.temperature)}",
        f"p{_format_float(config.decoding.top_p)}",
        f"lp{config.decoding.top_logprobs}",
        digest,
    ]
    return "__".join(name_parts)


def pool_dir(config: RunConfig, pool_id: str | None = None) -> Path:
    resolved_id = pool_id or compute_pool_id(config)
    return config.pools_dir / resolved_id


def _metadata_matches_config(metadata: PoolMetadata, config: RunConfig) -> bool:
    dataset_match = (
        metadata.dataset_path == str(config.dataset.dataset_path)
        or Path(metadata.dataset_path).name == Path(config.dataset.dataset_path).name
    )
    return (
        dataset_match
        and metadata.model == config.api.model
        and metadata.temperature == config.decoding.temperature
        and metadata.top_p == config.decoding.top_p
        and metadata.max_tokens == config.decoding.max_tokens
        and metadata.top_logprobs == config.decoding.top_logprobs
        and metadata.n_pool == config.pooling.n_pool
        and metadata.prompt_instruction == BOXED_INSTRUCTION
    )


def find_matching_pool_dir(config: RunConfig) -> Path | None:
    if not config.pools_dir.exists():
        return None
    for candidate in config.pools_dir.iterdir():
        if not candidate.is_dir():
            continue
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = load_pool_metadata(candidate)
        except Exception:
            continue
        if _metadata_matches_config(metadata, config):
            return candidate
    return None


def pool_problem_path(pool_path: Path, problem_id: str) -> Path:
    return pool_path / "problems" / f"{problem_id}.json"


def _token_to_dict(token: Any) -> dict[str, Any]:
    token_value = getattr(token, "token", None)
    logprob_value = getattr(token, "logprob", None)
    top_value = getattr(token, "top_logprobs", None)

    top_logprobs: dict[str, float] = {}
    if isinstance(top_value, dict):
        top_logprobs = {str(k): float(v) for k, v in top_value.items()}

    result = {
        "token": str(token_value) if token_value is not None else "",
        "logprob": float(logprob_value) if logprob_value is not None else None,
        "top_logprobs": top_logprobs,
    }
    return result


def _sample_from_result(
    result: ChatCompletionResult,
    *,
    sample_id: int,
    confidence_metric: str,
    group_size: int,
    tail_tokens: int,
    bottom_fraction: float,
) -> dict[str, Any]:
    tokens = [_token_to_dict(token) for token in result.tokens]
    answer = extract_answer(result.text)
    confidence = compute_trace_confidence(
        tokens,
        metric=confidence_metric,
        group_size=group_size,
        tail_tokens=tail_tokens,
        bottom_fraction=bottom_fraction,
    )
    return {
        "sample_id": sample_id,
        "text": result.text,
        "tokens": tokens,
        "finish_reason": result.finish_reason,
        "answer": answer,
        "confidence": confidence,
    }


def _make_client_from_config(config: RunConfig) -> DeepSeekClient:
    return DeepSeekClient(
        base_url=config.api.base_url,
        model=config.api.model,
        timeout_seconds=config.api.timeout_seconds,
        api_key_env=config.api.api_key_env,
    )


def build_pool_for_problem(
    *,
    client: DeepSeekClient,
    problem: ProblemRecord,
    config: RunConfig,
) -> dict[str, Any]:
    messages = build_prompt(problem.problem)

    total = config.pooling.n_pool
    concurrency = max(1, int(config.pooling.concurrency))

    samples: list[dict[str, Any]] = [None] * total  # type: ignore[list-item]
    lock = threading.Lock()
    next_id = 0

    def next_sample_id() -> int | None:
        nonlocal next_id
        with lock:
            if next_id >= total:
                return None
            sample_id = next_id
            next_id += 1
            return sample_id

    def run_one(sample_id: int) -> dict[str, Any]:
        # Use per-thread client to avoid shared session issues.
        local_client = client if concurrency == 1 else _make_client_from_config(config)
        result = local_client.chat_completion(
            messages=messages,
            temperature=config.decoding.temperature,
            top_p=config.decoding.top_p,
            max_tokens=config.decoding.max_tokens,
            logprobs=config.decoding.logprobs,
            top_logprobs=config.decoding.top_logprobs,
            require_logprobs=True,
        )
        return _sample_from_result(
            result,
            sample_id=sample_id,
            confidence_metric=config.confidence.metric,
            group_size=config.confidence.group_size,
            tail_tokens=config.confidence.tail_tokens,
            bottom_fraction=config.confidence.bottom_fraction,
        )

    progress = tqdm(total=total, desc=f"pool {problem.problem_id}", leave=False)
    if concurrency == 1:
        for sample_id in range(total):
            samples[sample_id] = run_one(sample_id)
            progress.update(1)
        progress.close()
    else:
        class AdaptiveConcurrency:
            def __init__(self, max_concurrency: int, *, min_concurrency: int = 1) -> None:
                resolved_max = max(1, int(max_concurrency))
                self.max = resolved_max
                self.min = max(1, int(min_concurrency))
                self.current = resolved_max
                self.cooldown_until = 0.0
                self.throttle_streak = 0
                self.success_streak = 0

            def note_throttle(self) -> None:
                self.throttle_streak += 1
                self.success_streak = 0
                if self.current > self.min:
                    reduced = max(self.min, int(self.current * 0.7))
                    if reduced == self.current:
                        reduced = max(self.min, self.current - 1)
                    self.current = reduced
                backoff = min(60.0, 2 ** min(self.throttle_streak, 5))
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + backoff)

            def note_success(self) -> None:
                self.throttle_streak = 0
                self.success_streak += 1
                if (
                    self.current < self.max
                    and time.monotonic() >= self.cooldown_until
                    and self.success_streak >= 5
                ):
                    self.current += 1
                    self.success_streak = 0

            def cooldown_remaining(self) -> float:
                remaining = self.cooldown_until - time.monotonic()
                return max(0.0, remaining)

        def is_throttle_error(exc: Exception) -> bool:
            if isinstance(exc, RetryableStatusError):
                return exc.status_code == 429
            return False

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            in_flight: dict[object, int] = {}
            pending: deque[int] = deque()
            limiter = AdaptiveConcurrency(concurrency)

            def submit_sample(sample_id: int) -> None:
                future = executor.submit(run_one, sample_id)
                in_flight[future] = sample_id

            def next_scheduled_id() -> int | None:
                if pending:
                    return pending.popleft()
                return next_sample_id()

            # Prime the queue.
            while len(in_flight) < limiter.current:
                sample_id = next_scheduled_id()
                if sample_id is None:
                    break
                submit_sample(sample_id)

            while in_flight or pending or next_id < total:
                delay = limiter.cooldown_remaining()
                if delay > 0 and not in_flight:
                    time.sleep(delay)

                if delay == 0:
                    while len(in_flight) < limiter.current:
                        next_id_value = next_scheduled_id()
                        if next_id_value is None:
                            break
                        submit_sample(next_id_value)

                if not in_flight:
                    continue

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    sample_id = in_flight.pop(future)
                    try:
                        samples[sample_id] = future.result()
                        limiter.note_success()
                        progress.update(1)
                    except Exception as exc:
                        if is_throttle_error(exc):
                            limiter.note_throttle()
                        # Retry the same sample_id on failure.
                        pending.append(sample_id)
                        continue
        progress.close()

    return {
        "problem_id": problem.problem_id,
        "problem": problem.problem,
        "answer": problem.answer,
        "samples": samples,
    }


def write_pool_problem(pool_path: Path, problem_data: dict[str, Any]) -> Path:
    ensure_directories([pool_path, pool_path / "problems"])
    path = pool_problem_path(pool_path, problem_data["problem_id"])
    with path.open("w", encoding="utf-8") as handle:
        json.dump(problem_data, handle, ensure_ascii=False)
    return path


def write_pool_metadata(pool_path: Path, metadata: PoolMetadata) -> Path:
    ensure_directories([pool_path])
    metadata_path = pool_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, ensure_ascii=False, indent=2)
    return metadata_path


def load_pool_problem(pool_path: Path, problem_id: str) -> dict[str, Any]:
    path = pool_problem_path(pool_path, problem_id)
    if not path.exists():
        raise FileNotFoundError(f"Pool problem file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pool_metadata(pool_path: Path) -> PoolMetadata:
    path = pool_path / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Pool metadata missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PoolMetadata(**payload)


def load_pool(pool_path: Path, problem_ids: Iterable[str]) -> list[dict[str, Any]]:
    return [load_pool_problem(pool_path, problem_id) for problem_id in problem_ids]


def pool_is_complete(pool_path: Path, problem_ids: Iterable[str]) -> bool:
    metadata_path = pool_path / "metadata.json"
    if not metadata_path.exists():
        return False
    problems_dir = pool_path / "problems"
    if not problems_dir.exists():
        return False
    return all(pool_problem_path(pool_path, problem_id).exists() for problem_id in problem_ids)


def build_pool(
    *,
    client: DeepSeekClient,
    problems: list[ProblemRecord],
    config: RunConfig,
    pool_id: str | None = None,
) -> Path:
    """Build or reuse a pool of traces for all problems."""

    resolved_pool_id = pool_id or compute_pool_id(config)
    pool_path = pool_dir(config, pool_id=resolved_pool_id)

    if not pool_path.exists():
        existing = find_matching_pool_dir(config)
        if existing is not None:
            pool_path = existing
            resolved_pool_id = pool_path.name

    problem_ids = [problem.problem_id for problem in problems]
    if pool_is_complete(pool_path, problem_ids) and not config.rebuild_pool:
        return pool_path

    ensure_directories([pool_path, pool_path / "problems"])

    for problem in tqdm(problems, desc="build pool", leave=True):
        problem_path = pool_problem_path(pool_path, problem.problem_id)
        if problem_path.exists() and not config.rebuild_pool:
            continue
        problem_data = build_pool_for_problem(client=client, problem=problem, config=config)
        write_pool_problem(pool_path, problem_data)

    metadata = PoolMetadata(
        pool_id=resolved_pool_id,
        dataset_path=str(config.dataset.dataset_path),
        model=config.api.model,
        temperature=config.decoding.temperature,
        top_p=config.decoding.top_p,
        max_tokens=config.decoding.max_tokens,
        top_logprobs=config.decoding.top_logprobs,
        n_pool=config.pooling.n_pool,
        prompt_instruction=BOXED_INSTRUCTION,
        built_at=_now_iso(),
        problems=problem_ids,
    )
    write_pool_metadata(pool_path, metadata)
    return pool_path
