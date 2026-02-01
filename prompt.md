# Prompt Template

The prompt is built **without a system message**. It is a single user message that contains:

1. The raw problem text.
2. A blank line.
3. The boxed-answer instruction.

**Instruction string (exact):**

```
Please reason step by step, and put your final answer within \boxed{}.
```

**Template (exact):**

```
{problem_text}

Please reason step by step, and put your final answer within \boxed{}.
```

**Example message payload:**

```json
[
  {
    "role": "user",
    "content": "<problem text here>\n\nPlease reason step by step, and put your final answer within \\boxed{}."
  }
]
```
