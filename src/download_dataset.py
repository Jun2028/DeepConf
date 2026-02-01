from __future__ import annotations

from pathlib import Path
import argparse
import json

import requests


def download_rows(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AIME 2025 rows from the Hugging Face dataset server.")
    parser.add_argument(
        "--url",
        default=(
            "https://datasets-server.huggingface.co/rows?dataset=MathArena%2Faime_2025"
            "&config=default&split=train&offset=0&length=100"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("data") / "aime_2025_rows.json")
    args = parser.parse_args()

    payload = download_rows(args.url)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved dataset rows to: {args.output}")


if __name__ == "__main__":
    main()