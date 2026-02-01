from __future__ import annotations

BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_prompt(problem_text: str) -> list[dict[str, str]]:
    """Build a paper-aligned chat prompt without a system message."""

    user_content = f"{problem_text}\n\n{BOXED_INSTRUCTION}"
    return [{"role": "user", "content": user_content}]