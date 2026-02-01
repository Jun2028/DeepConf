from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import re

import numpy as np

from .aggregate import cons_at_k, measure_at_k, measure_top_eta_at_k
from .aime_loader import ProblemRecord, load_problems
from .confidence import (
    bottom_fraction_mean,
    compute_confidence_inputs,
    compute_trace_confidence,
    tail_mean,
)
from .config import (
    ApiConfig,
    ConfidenceConfig,
    DatasetConfig,
    DecodingConfig,
    PoolingConfig,
    RunConfig,
    ensure_directories,
)
from .deepseek_client import DeepSeekClient
from .sampling import (
    build_pool,
    compute_pool_id,
    find_matching_pool_dir,
    load_pool,
    pool_dir,
    pool_is_complete,
    pool_problem_path,
)


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resample_indices(n_pool: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k > n_pool:
        raise ValueError(f"k={k} cannot exceed n_pool={n_pool}.")
    return rng.choice(n_pool, size=k, replace=False)


def _prepare_sample(sample: dict[str, Any], confidence: float) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "answer": sample.get("answer"),
        "confidence": confidence,
    }


_SAMPLE_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*(null|"(?:\\.|[^"\\])*"|[-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
)
_SAMPLE_CONF_RE = re.compile(
    r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|null)'
)


def _iter_sample_strings(path: Path) -> Iterable[str]:
    """Stream sample objects from a pool problem JSON without loading everything into memory."""

    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_samples = False

        while not in_samples:
            chunk = handle.read(65536)
            if not chunk:
                return
            buffer += chunk
            idx = buffer.find('"samples"')
            if idx == -1:
                buffer = buffer[-200:]
                continue
            bracket = buffer.find("[", idx)
            if bracket == -1:
                buffer = buffer[idx:]
                continue
            buffer = buffer[bracket + 1 :]
            in_samples = True

        in_string = False
        escape = False
        depth = 0
        collecting = False
        sample_buf: list[str] = []

        while True:
            if not buffer:
                chunk = handle.read(65536)
                if not chunk:
                    return
                buffer = chunk

            for ch in buffer:
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    if collecting:
                        sample_buf.append(ch)
                    continue

                if ch == '"':
                    in_string = True
                    if collecting:
                        sample_buf.append(ch)
                    continue

                if ch == "{":
                    if depth == 0:
                        collecting = True
                        sample_buf = ["{"]
                    else:
                        if collecting:
                            sample_buf.append(ch)
                    depth += 1
                    continue

                if ch == "}":
                    if collecting:
                        sample_buf.append(ch)
                    depth -= 1
                    if depth == 0 and collecting:
                        yield "".join(sample_buf)
                        collecting = False
                        sample_buf = []
                    continue

                if collecting:
                    sample_buf.append(ch)

                if ch == "]" and depth == 0:
                    return

            buffer = ""


def _extract_sample_fields(sample_text: str) -> tuple[str | None, float | None]:
    answer_match = _SAMPLE_ANSWER_RE.search(sample_text)
    confidence_match = _SAMPLE_CONF_RE.search(sample_text)

    answer: str | None | int = None
    if answer_match:
        raw = answer_match.group(1)
        if raw != "null":
            if raw.startswith('"'):
                answer = json.loads(raw)
            else:
                try:
                    answer = int(float(raw))
                except ValueError:
                    answer = None

    confidence = None
    if confidence_match:
        raw = confidence_match.group(1)
        if raw != "null":
            confidence = float(raw)

    return answer, confidence


def _load_problem_light(path: Path, n_pool: int) -> tuple[list[str | None], list[float | None]]:
    answers: list[str | None] = []
    confidences: list[float | None] = []

    for sample_text in _iter_sample_strings(path):
        answer, confidence = _extract_sample_fields(sample_text)
        answers.append(answer)
        confidences.append(confidence)
        if len(answers) >= n_pool:
            break

    if len(answers) < n_pool:
        raise ValueError(f"Pool file {path} has {len(answers)} samples; expected {n_pool}.")

    return answers, confidences


def _compute_confidence_metrics(
    tokens: list[dict[str, Any]],
    *,
    metrics: Iterable[str],
    group_size: int,
    tail_tokens: int,
    bottom_fraction: float,
) -> dict[str, float]:
    inputs = compute_confidence_inputs(tokens, group_size=group_size)
    token_confs = inputs.token_confidences
    group_confs = inputs.group_confidences

    results: dict[str, float] = {}
    for metric in metrics:
        if metric == "avg":
            results[metric] = float(np.mean(token_confs)) if token_confs else 0.0
        elif metric == "bottom10_group":
            results[metric] = bottom_fraction_mean(group_confs, fraction=bottom_fraction)
        elif metric == "lowest_group":
            results[metric] = min(group_confs) if group_confs else 0.0
        elif metric == "tail":
            results[metric] = tail_mean(token_confs, tail_tokens=tail_tokens)
        else:
            raise ValueError(f"Unknown confidence metric: {metric}")
    return results


def _load_problem_metrics_streaming(
    path: Path,
    *,
    n_pool: int,
    metrics: Iterable[str],
    group_size: int,
    tail_tokens: int,
    bottom_fraction: float,
) -> tuple[list[str | None], dict[str, list[float]]]:
    answers: list[str | None] = []
    confidences_by_metric: dict[str, list[float]] = {metric: [] for metric in metrics}

    for sample_text in _iter_sample_strings(path):
        sample = json.loads(sample_text)
        answer = sample.get("answer")
        tokens = sample.get("tokens") or []
        metrics_result = _compute_confidence_metrics(
            tokens,
            metrics=metrics,
            group_size=group_size,
            tail_tokens=tail_tokens,
            bottom_fraction=bottom_fraction,
        )

        answers.append(answer)
        for metric, value in metrics_result.items():
            confidences_by_metric[metric].append(value)

        if len(answers) >= n_pool:
            break

    if len(answers) < n_pool:
        raise ValueError(f"Pool file {path} has {len(answers)} samples; expected {n_pool}.")

    return answers, confidences_by_metric


def _compute_sample_confidence(sample: dict[str, Any], config: RunConfig) -> float:
    tokens = sample.get("tokens") or []
    return compute_trace_confidence(
        tokens,
        metric=config.confidence.metric,
        group_size=config.confidence.group_size,
        tail_tokens=config.confidence.tail_tokens,
        bottom_fraction=config.confidence.bottom_fraction,
    )


def _confidence_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _jsonify(value: Any) -> Any:
    """Convert non-JSON-native values (e.g., Paths) recursively."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def evaluate_from_pool(
    *,
    pool_path: Path,
    problems: list[ProblemRecord],
    config: RunConfig,
    use_stored_confidence: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate offline by resampling working sets from a saved pool."""

    problem_ids = [problem.problem_id for problem in problems]
    pool_records = None
    light_records: dict[str, tuple[list[str | None], list[float | None]]] = {}
    if use_stored_confidence:
        for problem_id in problem_ids:
            problem_path = pool_problem_path(pool_path, problem_id)
            light_records[problem_id] = _load_problem_light(problem_path, config.pooling.n_pool)
    else:
        pool_records = load_pool(pool_path, problem_ids)
        if not pool_records:
            raise ValueError("Pool is empty.")

    n_pool = config.pooling.n_pool
    k = config.pooling.k
    etas = config.pooling.etas

    repeat_summaries: list[dict[str, Any]] = []
    per_problem_results: list[dict[str, Any]] = []

    for repeat_idx in range(config.pooling.repeat):
        rng = np.random.default_rng(config.pooling.seed + repeat_idx)

        pass1_correct = 0
        cons_correct = 0
        measure_correct = 0
        eta_correct = {eta: 0 for eta in etas}

        cons_ties = 0
        valid_counts: list[int] = []
        all_confidences: list[float] = []

        for problem_idx, problem in enumerate(problems):
            if use_stored_confidence:
                answers, confidences = light_records[problem.problem_id]
                indices = _resample_indices(n_pool, k, rng)
                working_with_conf = []
                for i in indices:
                    answer = answers[int(i)]
                    confidence = confidences[int(i)]
                    if confidence is None:
                        raise ValueError(
                            f"Missing confidence for problem {problem.problem_id} sample {int(i)}."
                        )
                    working_with_conf.append({"answer": answer, "confidence": confidence})
                all_confidences.extend([item["confidence"] for item in working_with_conf])
            else:
                record = pool_records[problem_idx]
                samples: list[dict[str, Any]] = record.get("samples", [])
                if len(samples) < n_pool:
                    raise ValueError(
                        f"Pool for problem {problem.problem_id} has {len(samples)} samples; expected {n_pool}."
                    )

                indices = _resample_indices(n_pool, k, rng)
                working_samples = [samples[int(i)] for i in indices]

                confidences = [_compute_sample_confidence(sample, config) for sample in working_samples]
                working_with_conf = [
                    _prepare_sample(sample, confidence)
                    for sample, confidence in zip(working_samples, confidences)
                ]
                all_confidences.extend(confidences)

            first_answer = working_with_conf[0].get("answer")
            pass1_is_correct = first_answer == problem.answer
            pass1_correct += int(pass1_is_correct)

            cons_result = cons_at_k(working_with_conf)
            cons_is_correct = cons_result.answer == problem.answer
            cons_correct += int(cons_is_correct)
            cons_ties += int(cons_result.tie)
            valid_counts.append(cons_result.valid_count)

            measure_result = measure_at_k(working_with_conf)
            measure_correct += int(measure_result.answer == problem.answer)

            eta_results: dict[str, dict[str, Any]] = {}
            for eta in etas:
                eta_result = measure_top_eta_at_k(working_with_conf, eta=eta)
                eta_correct[eta] += int(eta_result.answer == problem.answer)
                eta_results[str(eta)] = asdict(eta_result)

            per_problem_results.append(
                {
                    "repeat_idx": repeat_idx,
                    "problem_id": problem.problem_id,
                    "ground_truth": problem.answer,
                    "indices": indices.tolist(),
                    "pass_at_1": {
                        "answer": first_answer,
                        "correct": pass1_is_correct,
                    },
                    "cons_at_k": asdict(cons_result),
                    "measure_at_k": asdict(measure_result),
                    "measure_top_eta_at_k": eta_results,
                }
            )

        problem_count = len(problems)
        repeat_summary = {
            "repeat_idx": repeat_idx,
            "problem_count": problem_count,
            "pass_at_1": pass1_correct / problem_count,
            "cons_at_k": cons_correct / problem_count,
            "measure_at_k": measure_correct / problem_count,
            "measure_top_eta_at_k": {
                str(eta): eta_correct[eta] / problem_count for eta in etas
            },
            "cons_tie_rate": cons_ties / problem_count,
            "valid_samples": {
                "mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
                "min": int(min(valid_counts)) if valid_counts else 0,
            },
            "confidence_summary": _confidence_summary(all_confidences),
        }
        repeat_summaries.append(repeat_summary)

    def _avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "pool_id": pool_path.name,
        "n_pool": n_pool,
        "k": k,
        "repeat": config.pooling.repeat,
        "confidence_metric": config.confidence.metric,
        "etas": [float(eta) for eta in etas],
        "pass_at_1": _avg([item["pass_at_1"] for item in repeat_summaries]),
        "cons_at_k": _avg([item["cons_at_k"] for item in repeat_summaries]),
        "measure_at_k": _avg([item["measure_at_k"] for item in repeat_summaries]),
        "measure_top_eta_at_k": {
            str(eta): _avg([item["measure_top_eta_at_k"][str(eta)] for item in repeat_summaries])
            for eta in etas
        },
        "cons_tie_rate": _avg([item["cons_tie_rate"] for item in repeat_summaries]),
        "valid_samples": {
            "mean": _avg([item["valid_samples"]["mean"] for item in repeat_summaries]),
            "min": int(min(item["valid_samples"]["min"] for item in repeat_summaries)),
        },
        "confidence_summary": {
            key: _avg([item["confidence_summary"][key] for item in repeat_summaries])
            for key in ["mean", "p10", "p50", "p90"]
        },
        "repeats": repeat_summaries,
    }

    details = {
        "pool_id": pool_path.name,
        "config": _jsonify(asdict(config)),
        "results": per_problem_results,
    }

    return summary, details


def _evaluate_with_precomputed_confidences(
    *,
    problems: list[ProblemRecord],
    answers_by_problem: list[list[str | None]],
    confidences_by_problem: list[list[float]],
    config: RunConfig,
    metric_name: str,
    pool_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_pool = config.pooling.n_pool
    k = config.pooling.k
    etas = config.pooling.etas

    repeat_summaries: list[dict[str, Any]] = []
    per_problem_results: list[dict[str, Any]] = []

    for repeat_idx in range(config.pooling.repeat):
        rng = np.random.default_rng(config.pooling.seed + repeat_idx)

        pass1_correct = 0
        cons_correct = 0
        measure_correct = 0
        eta_correct = {eta: 0 for eta in etas}

        cons_ties = 0
        valid_counts: list[int] = []
        all_confidences: list[float] = []

        for problem_idx, problem in enumerate(problems):
            answers = answers_by_problem[problem_idx]
            confidences = confidences_by_problem[problem_idx]

            if len(answers) < n_pool or len(confidences) < n_pool:
                raise ValueError(
                    f"Pool for problem {problem.problem_id} has {len(answers)} samples; expected {n_pool}."
                )

            indices = _resample_indices(n_pool, k, rng)
            working_with_conf = []
            for i in indices:
                idx = int(i)
                working_with_conf.append(
                    {
                        "answer": answers[idx],
                        "confidence": confidences[idx],
                    }
                )
                all_confidences.append(confidences[idx])

            first_answer = working_with_conf[0].get("answer")
            pass1_is_correct = first_answer == problem.answer
            pass1_correct += int(pass1_is_correct)

            cons_result = cons_at_k(working_with_conf)
            cons_is_correct = cons_result.answer == problem.answer
            cons_correct += int(cons_is_correct)
            cons_ties += int(cons_result.tie)
            valid_counts.append(cons_result.valid_count)

            measure_result = measure_at_k(working_with_conf)
            measure_correct += int(measure_result.answer == problem.answer)

            eta_results: dict[str, dict[str, Any]] = {}
            for eta in etas:
                eta_result = measure_top_eta_at_k(working_with_conf, eta=eta)
                eta_correct[eta] += int(eta_result.answer == problem.answer)
                eta_results[str(eta)] = asdict(eta_result)

            per_problem_results.append(
                {
                    "repeat_idx": repeat_idx,
                    "problem_id": problem.problem_id,
                    "ground_truth": problem.answer,
                    "indices": indices.tolist(),
                    "pass_at_1": {
                        "answer": first_answer,
                        "correct": pass1_is_correct,
                    },
                    "cons_at_k": asdict(cons_result),
                    "measure_at_k": asdict(measure_result),
                    "measure_top_eta_at_k": eta_results,
                }
            )

        problem_count = len(problems)
        repeat_summary = {
            "repeat_idx": repeat_idx,
            "problem_count": problem_count,
            "pass_at_1": pass1_correct / problem_count,
            "cons_at_k": cons_correct / problem_count,
            "measure_at_k": measure_correct / problem_count,
            "measure_top_eta_at_k": {
                str(eta): eta_correct[eta] / problem_count for eta in etas
            },
            "cons_tie_rate": cons_ties / problem_count,
            "valid_samples": {
                "mean": float(np.mean(valid_counts)) if valid_counts else 0.0,
                "min": int(min(valid_counts)) if valid_counts else 0,
            },
            "confidence_summary": _confidence_summary(all_confidences),
        }
        repeat_summaries.append(repeat_summary)

    def _avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "pool_id": pool_id,
        "n_pool": n_pool,
        "k": k,
        "repeat": config.pooling.repeat,
        "confidence_metric": metric_name,
        "etas": [float(eta) for eta in etas],
        "pass_at_1": _avg([item["pass_at_1"] for item in repeat_summaries]),
        "cons_at_k": _avg([item["cons_at_k"] for item in repeat_summaries]),
        "measure_at_k": _avg([item["measure_at_k"] for item in repeat_summaries]),
        "measure_top_eta_at_k": {
            str(eta): _avg([item["measure_top_eta_at_k"][str(eta)] for item in repeat_summaries])
            for eta in etas
        },
        "cons_tie_rate": _avg([item["cons_tie_rate"] for item in repeat_summaries]),
        "valid_samples": {
            "mean": _avg([item["valid_samples"]["mean"] for item in repeat_summaries]),
            "min": int(min(item["valid_samples"]["min"] for item in repeat_summaries)),
        },
        "confidence_summary": {
            key: _avg([item["confidence_summary"][key] for item in repeat_summaries])
            for key in ["mean", "p10", "p50", "p90"]
        },
        "repeats": repeat_summaries,
    }

    details = {
        "pool_id": pool_id,
        "config": _jsonify(asdict(config)),
        "results": per_problem_results,
    }

    return summary, details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepConf offline evaluation with pooling.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--id_key", default="problem_idx")

    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--timeout_seconds", type=float, default=60.0)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_logprobs", type=int, default=20)

    parser.add_argument("--n_pool", type=int, default=4096)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--etas", type=float, nargs="+", default=[0.10, 0.90])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--use_stored_confidence", action="store_true")
    parser.add_argument("--all_metrics", action="store_true")

    parser.add_argument(
        "--confidence_metric",
        choices=["avg", "bottom10_group", "lowest_group", "tail"],
        default="lowest_group",
    )
    parser.add_argument("--group_size", type=int, default=2048)
    parser.add_argument("--tail_tokens", type=int, default=2048)
    parser.add_argument("--bottom_fraction", type=float, default=0.10)

    parser.add_argument("--limit_problems", type=int, default=None)
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    parser.add_argument("--pools_dir", type=Path, default=Path("runs") / "pools")

    parser.add_argument("--pool_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--rebuild_pool", action="store_true")

    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> RunConfig:
    dataset_config = DatasetConfig(
        dataset_path=args.dataset,
        problem_key=args.problem_key,
        answer_key=args.answer_key,
        id_key=args.id_key,
    )
    api_config = ApiConfig(
        base_url=str(args.api_base_url),
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    decoding_config = DecodingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
    )
    confidence_config = ConfidenceConfig(
        metric=args.confidence_metric,
        group_size=args.group_size,
        tail_tokens=args.tail_tokens,
        bottom_fraction=args.bottom_fraction,
    )
    pooling_config = PoolingConfig(
        n_pool=args.n_pool,
        k=args.k,
        etas=tuple(float(eta) for eta in args.etas),
        repeat=args.repeat,
        seed=args.seed,
        concurrency=args.concurrency,
    )

    config = RunConfig(
        dataset=dataset_config,
        api=api_config,
        decoding=decoding_config,
        confidence=confidence_config,
        pooling=pooling_config,
        runs_dir=args.runs_dir,
        pools_dir=args.pools_dir,
        limit_problems=args.limit_problems,
        rebuild_pool=args.rebuild_pool,
        pool_only=args.pool_only,
        eval_only=args.eval_only,
    )
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directories([path.parent])
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    ensure_directories([config.runs_dir, config.pools_dir])

    problems = load_problems(
        config.dataset.dataset_path,
        problem_key=config.dataset.problem_key,
        answer_key=config.dataset.answer_key,
        id_key=config.dataset.id_key,
        limit=config.limit_problems,
    )

    pool_id = compute_pool_id(config)
    pool_path = pool_dir(config, pool_id=pool_id)
    if not pool_path.exists():
        existing = find_matching_pool_dir(config)
        if existing is not None:
            pool_path = existing
            pool_id = pool_path.name

    if not config.eval_only:
        client = DeepSeekClient(
            base_url=config.api.base_url,
            model=config.api.model,
            timeout_seconds=config.api.timeout_seconds,
            api_key_env=config.api.api_key_env,
        )
        pool_path = build_pool(client=client, problems=problems, config=config, pool_id=pool_id)
        print(f"Pool ready at: {pool_path}")

    if config.pool_only:
        return

    problem_ids = [problem.problem_id for problem in problems]
    if not pool_is_complete(pool_path, problem_ids):
        existing = find_matching_pool_dir(config)
        if existing is not None and pool_is_complete(existing, problem_ids):
            pool_path = existing
        else:
            raise RuntimeError(
                "Pool is incomplete. Build the pool first (run without --eval_only, or pass --rebuild_pool)."
            )

    if args.all_metrics:
        if args.use_stored_confidence:
            raise ValueError("Cannot use --all_metrics with --use_stored_confidence.")

        metrics = ["avg", "bottom10_group", "lowest_group", "tail"]
        answers_by_problem: list[list[str | None]] = []
        confidences_by_metric: dict[str, list[list[float]]] = {metric: [] for metric in metrics}

        for problem in problems:
            problem_path = pool_problem_path(pool_path, problem.problem_id)
            answers, confs_by_metric = _load_problem_metrics_streaming(
                problem_path,
                n_pool=config.pooling.n_pool,
                metrics=metrics,
                group_size=config.confidence.group_size,
                tail_tokens=config.confidence.tail_tokens,
                bottom_fraction=config.confidence.bottom_fraction,
            )
            answers_by_problem.append(answers)
            for metric in metrics:
                confidences_by_metric[metric].append(confs_by_metric[metric])

        run_dir = config.runs_dir / _now_timestamp()

        for metric in metrics:
            summary, details = _evaluate_with_precomputed_confidences(
                problems=problems,
                answers_by_problem=answers_by_problem,
                confidences_by_problem=confidences_by_metric[metric],
                config=config,
                metric_name=metric,
                pool_id=pool_path.name,
            )
            metric_dir = run_dir / metric
            _write_json(metric_dir / "summary.json", summary)
            _write_json(metric_dir / "results.json", details)

            print("Summary metrics:")
            print(
                json.dumps(
                    {
                        "pass_at_1": summary["pass_at_1"],
                        "cons_at_k": summary["cons_at_k"],
                        "measure_at_k": summary["measure_at_k"],
                        "measure_top_eta_at_k": summary["measure_top_eta_at_k"],
                        "pool_id": summary["pool_id"],
                        "run_dir": str(metric_dir),
                        "confidence_metric": metric,
                    },
                    indent=2,
                )
            )
    else:
        summary, details = evaluate_from_pool(
            pool_path=pool_path,
            problems=problems,
            config=config,
            use_stored_confidence=args.use_stored_confidence,
        )

        run_dir = config.runs_dir / _now_timestamp()
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "results.json", details)

        print("Summary metrics:")
        print(
            json.dumps(
                {
                    "pass_at_1": summary["pass_at_1"],
                    "cons_at_k": summary["cons_at_k"],
                    "measure_at_k": summary["measure_at_k"],
                    "measure_top_eta_at_k": summary["measure_top_eta_at_k"],
                    "pool_id": summary["pool_id"],
                    "run_dir": str(run_dir),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
