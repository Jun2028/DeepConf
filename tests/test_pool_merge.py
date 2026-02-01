from __future__ import annotations

import json
from pathlib import Path

from src.merge_pools import merge_pools
from src.sampling import PoolMetadata, write_pool_metadata, write_pool_problem


def _write_pool(base: Path, pool_id: str, n_pool: int, problem_id: str) -> Path:
    pool_path = base / pool_id
    (pool_path / "problems").mkdir(parents=True, exist_ok=True)

    samples = []
    for idx in range(n_pool):
        samples.append(
            {
                "sample_id": idx,
                "text": f"sample {idx}",
                "tokens": [{"token": "x", "logprob": -1.0, "top_logprobs": {"x": -1.0}}],
                "answer": 42,
                "confidence": 1.0,
            }
        )

    write_pool_problem(
        pool_path,
        {
            "problem_id": problem_id,
            "problem": "dummy",
            "answer": 42,
            "samples": samples,
        },
    )

    meta = PoolMetadata(
        pool_id=pool_id,
        dataset_path="data/aime.jsonl",
        model="deepseek-chat",
        temperature=0.6,
        top_p=0.95,
        max_tokens=128,
        top_logprobs=20,
        n_pool=n_pool,
        prompt_instruction="boxed",
        built_at="now",
        problems=[problem_id],
    )
    write_pool_metadata(pool_path, meta)
    return pool_path


def test_merge_pools(tmp_path: Path) -> None:
    pool_a = _write_pool(tmp_path, "pool_a", n_pool=2, problem_id="1")
    pool_b = _write_pool(tmp_path, "pool_b", n_pool=3, problem_id="1")

    output_dir = tmp_path / "merged"
    merged = merge_pools([pool_a, pool_b], output_dir)

    merged_meta = json.loads((merged / "metadata.json").read_text(encoding="utf-8"))
    assert merged_meta["n_pool"] == 5
    assert merged_meta["problems"] == ["1"]

    merged_problem = json.loads((merged / "problems" / "1.json").read_text(encoding="utf-8"))
    samples = merged_problem["samples"]
    assert len(samples) == 5
    assert [sample["sample_id"] for sample in samples] == list(range(5))
    assert all("source_pool" in sample for sample in samples)
    assert all("source_sample_id" in sample for sample in samples)

