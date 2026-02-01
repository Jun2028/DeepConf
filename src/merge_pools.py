from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import re

from .config import ensure_directories
from .sampling import (
    PoolMetadata,
    load_pool_metadata,
    load_pool_problem,
    write_pool_metadata,
    write_pool_problem,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "pool"


def _metadata_core(metadata: PoolMetadata) -> dict[str, Any]:
    return {
        "dataset_path": metadata.dataset_path,
        "model": metadata.model,
        "temperature": metadata.temperature,
        "top_p": metadata.top_p,
        "max_tokens": metadata.max_tokens,
        "top_logprobs": metadata.top_logprobs,
        "prompt_instruction": metadata.prompt_instruction,
        "problems": metadata.problems,
    }


def _validate_metadata(metadata_list: list[PoolMetadata]) -> None:
    if not metadata_list:
        raise ValueError("No pools provided.")
    core = _metadata_core(metadata_list[0])
    for meta in metadata_list[1:]:
        if _metadata_core(meta) != core:
            raise ValueError(
                "Pools are incompatible. Ensure dataset, model, decoding, prompt, and problem list match."
            )


def _default_output_name(pool_paths: list[Path], total_n_pool: int) -> str:
    names = [path.name for path in pool_paths]
    joined = "__".join(names[:3])
    if len(names) > 3:
        joined += "__more"
    raw = f"merged__{joined}__n{total_n_pool}"
    return _slugify(raw)


def merge_pools(pool_paths: list[Path], output_dir: Path) -> Path:
    metadata_list = [load_pool_metadata(path) for path in pool_paths]
    _validate_metadata(metadata_list)

    total_n_pool = sum(meta.n_pool for meta in metadata_list)
    problems = metadata_list[0].problems

    ensure_directories([output_dir, output_dir / "problems"])

    for problem_id in problems:
        base_data = load_pool_problem(pool_paths[0], problem_id)
        combined_samples: list[dict[str, Any]] = []
        sample_id = 0

        for pool_path, meta in zip(pool_paths, metadata_list):
            data = load_pool_problem(pool_path, problem_id)
            if data.get("answer") != base_data.get("answer") or data.get("problem") != base_data.get("problem"):
                raise ValueError(f"Problem data mismatch for problem_id={problem_id}.")

            for sample in data.get("samples", []):
                new_sample = dict(sample)
                new_sample["source_pool"] = meta.pool_id
                new_sample["source_sample_id"] = sample.get("sample_id")
                new_sample["sample_id"] = sample_id
                sample_id += 1
                combined_samples.append(new_sample)

        write_pool_problem(
            output_dir,
            {
                "problem_id": problem_id,
                "problem": base_data.get("problem"),
                "answer": base_data.get("answer"),
                "samples": combined_samples,
            },
        )

    merged_meta = PoolMetadata(
        pool_id=output_dir.name,
        dataset_path=metadata_list[0].dataset_path,
        model=metadata_list[0].model,
        temperature=metadata_list[0].temperature,
        top_p=metadata_list[0].top_p,
        max_tokens=metadata_list[0].max_tokens,
        top_logprobs=metadata_list[0].top_logprobs,
        n_pool=total_n_pool,
        prompt_instruction=metadata_list[0].prompt_instruction,
        built_at=_now_iso(),
        problems=problems,
    )
    write_pool_metadata(output_dir, merged_meta)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge compatible DeepConf pools.")
    parser.add_argument("--pools", nargs="+", type=Path, required=True)
    parser.add_argument("--pools_dir", type=Path, default=Path("runs") / "pools")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pools = [path if path.is_absolute() else args.pools_dir / path for path in args.pools]

    total_n_pool = sum(load_pool_metadata(path).n_pool for path in pools)
    output_dir = args.output_dir
    if output_dir is None:
        output_name = _default_output_name(pools, total_n_pool)
        output_dir = args.pools_dir / output_name

    merged_path = merge_pools(pools, output_dir)
    print(f"Merged pool saved to: {merged_path}")


if __name__ == "__main__":
    main()

