from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class DecodingConfig:
    """Decoding parameters aligned with the paper's DeepSeek-8B settings."""

    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: int = 4096
    top_logprobs: int = 20
    logprobs: bool = True


@dataclass(slots=True)
class ConfidenceConfig:
    """Configuration for trace-level confidence measurements."""

    metric: str = "lowest_group"
    group_size: int = 2048
    tail_tokens: int = 2048
    bottom_fraction: float = 0.10


@dataclass(slots=True)
class PoolingConfig:
    """Pool-first evaluation settings."""

    n_pool: int = 4096
    k: int = 512
    etas: tuple[float, ...] = (0.10, 0.90)
    repeat: int = 1
    seed: int = 0
    concurrency: int = 8


@dataclass(slots=True)
class DatasetConfig:
    """Dataset file and schema configuration."""

    dataset_path: Path
    problem_key: str = "problem"
    answer_key: str = "answer"
    id_key: str = "problem_idx"


@dataclass(slots=True)
class ApiConfig:
    """DeepSeek API configuration."""

    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout_seconds: float = 60.0


@dataclass(slots=True)
class RunConfig:
    """Top-level configuration used across pool building and evaluation."""

    dataset: DatasetConfig
    api: ApiConfig = field(default_factory=ApiConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    runs_dir: Path = Path("runs")
    pools_dir: Path = Path("runs") / "pools"
    limit_problems: int | None = None
    rebuild_pool: bool = False
    pool_only: bool = False
    eval_only: bool = False


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
