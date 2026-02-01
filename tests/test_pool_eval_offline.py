from __future__ import annotations

from pathlib import Path
import json

from src.aime_loader import load_problems
from src.config import DatasetConfig, PoolingConfig, RunConfig
from src.evaluate import evaluate_from_pool
from src.sampling import PoolMetadata, write_pool_metadata, write_pool_problem


def _make_token() -> dict:
    return {"token": "x", "logprob": -1.0, "top_logprobs": {"x": -1.0, "y": -2.0}}


def test_evaluate_from_saved_pool_offline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_rows = [
        {"problem_idx": 1, "problem": "p1", "answer": 10},
        {"problem_idx": 2, "problem": "p2", "answer": 20},
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row) + "\n")

    problems = load_problems(dataset_path)

    config = RunConfig(
        dataset=DatasetConfig(dataset_path=dataset_path),
        pooling=PoolingConfig(n_pool=4, k=2, etas=(0.5,), repeat=2, seed=123),
    )

    pool_path = tmp_path / "pools" / "testpool"
    (pool_path / "problems").mkdir(parents=True, exist_ok=True)

    for problem in problems:
        samples = []
        for i in range(config.pooling.n_pool):
            answer = problem.answer if i % 2 == 0 else (problem.answer + 1)
            samples.append(
                {
                    "sample_id": i,
                    "text": "...",
                    "tokens": [_make_token(), _make_token()],
                    "answer": answer,
                    "confidence": 1.0,
                }
            )
        write_pool_problem(
            pool_path,
            {
                "problem_id": problem.problem_id,
                "problem": problem.problem,
                "answer": problem.answer,
                "samples": samples,
            },
        )

    metadata = PoolMetadata(
        pool_id=pool_path.name,
        dataset_path=str(dataset_path),
        model="deepseek-chat",
        temperature=0.6,
        top_p=0.95,
        max_tokens=256,
        top_logprobs=20,
        n_pool=config.pooling.n_pool,
        prompt_instruction="boxed",
        built_at="now",
        problems=[problem.problem_id for problem in problems],
    )
    write_pool_metadata(pool_path, metadata)

    summary, details = evaluate_from_pool(pool_path=pool_path, problems=problems, config=config)

    assert 0.0 <= summary["pass_at_1"] <= 1.0
    assert 0.0 <= summary["cons_at_k"] <= 1.0
    assert summary["pool_id"] == "testpool"
    assert len(details["results"]) == len(problems) * config.pooling.repeat
    # Details should be JSON-serializable even though config contains Paths.
    json.dumps(details)
