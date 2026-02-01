# instructions.md - Reproducing DeepConf (offline) on AIME 2025 with DeepSeek API (top-20 logprobs)

## Goal
Demonstrate that a DeepConf-style confidence method improves over a non-DeepConf baseline on Hugging Face's `MathArena/aime_2025` by using:
- multiple sampled solutions per problem (K samples),
- token-level logprobs to score each sample,
- confidence-weighted and confidence-filtered voting.

This is method reproduction, not matching the paper's exact reported numbers.

Reference paper: `docs/2508.15260v1.pdf` ("Deep Think with Confidence").

## Hard constraints
- Use a DeepSeek-v3.2 chat model (named `deepseek-chat`) because it supports `logprobs=true` and `top_logprobs<=20`.
- Do not use the reasoning model (commonly `deepseek-reasoner`) because logprobs/top_logprobs are not supported there.
- Keep the prompt identical across baseline and DeepConf runs.

## Paper-aligned decisions to implement

1) Prompt template and answer format
- Append this instruction to every problem prompt: "Please reason step by step, and put your final answer within \\boxed{}."
- Expect the final answer to appear inside `\\boxed{...}` and extract from there during post-processing.
- For DeepSeek, keep the official system prompt (if you have one configured) and place the problem in the user message.

2) Decoding defaults (from the paper's DeepSeek-8B settings)
- `temperature = 0.6`
- `top_logprobs = 20`
- `max_tokens`: set large enough to allow long reasoning. Practical default: `4096`.
Important note (from DeepSeek docs): use either `temperature` or `top_p`, not both. This repo uses temperature only and leaves `top_p` at its default (no truncation).

3) Confidence windows and retention ratios
- Use 2,048-token windows for group and tail confidence.
- Implement filtering with `eta` in `{0.10, 0.90}` (top 10% and top 90% retention).
- Recompute the top-eta cutoff within each problem's working set of K samples.

## What to build (minimal)
1. Dataset loader for AIME 2025 problems.
2. Sampler that queries the DeepSeek API K times per question, returning generated text, per-token chosen-token logprob, and per-token top-k logprobs (k up to 20).
3. Answer extractor that maps each generated text to a single integer answer (AIME: 0-999).
4. Baseline aggregator (self-consistency / majority vote).
5. DeepConf-style aggregators (confidence score per sample plus weighted or filtered vote).
6. Evaluator that computes accuracy across problems and prints a small results table.

Keep every module small and readable.

---

## Dataset loader
Target dataset: Hugging Face `MathArena/aime_2025`(`curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=MathArena%2Faime_2025&config=default&split=train&offset=0&length=100"`).

Because field names can vary, make the loader configurable:
- Accept a local dataset file (recommended) such as `data/aime_2025.jsonl`.
- Default expected keys per record: `problem` for the question text and `answer` for the ground-truth integer answer.
- Allow overrides via CLI/config, e.g. `--problem_key` and `--answer_key`.

Suggested workflow:
1. Fetch the dataset once (outside this repo if needed).
2. Convert it into a simple JSONL with `{ "problem": ..., "answer": ... }`.
3. Point the loader at that file.

---

## Repository structure (suggested)
- `instructions.md` (this file)
- `requirements.txt`
- `src/` with: `config.py`, `deepseek_client.py`, `aime_loader.py`, `prompting.py`, `sampling.py`, `extract_answer.py`, `confidence.py`, `aggregate.py`, `evaluate.py`, `visualize.py`
- `data/` (AIME dataset, once provided)
- `runs/` (saved raw responses plus parsed outputs)

---

## Dependencies (.venv requirements)
Add the following to `requirements.txt`:
- `requests`
- `python-dotenv`
- `tqdm`
- `numpy`
- `pandas`
- `tenacity`

Notes:
- `tenacity` is for simple, robust retry logic around API calls.
- `numpy` makes the percentile and window math straightforward.

---

## Python environment (.venv)
Do not use the base environment. Use the repo-local `.venv`.

Setup (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run tests with the `.venv` interpreter:

```powershell
.\.venv\Scripts\python -m pytest -q
```

---

## API client requirements
Implement a DeepSeek API wrapper that can:
- call the chat completion endpoint,
- request `logprobs=true`,
- request `top_logprobs=20`,
- set sampling params: `temperature`, `max_tokens`, and `top_p`,
- return full generated text and a per-token list. Each token entry should include the chosen token string, the chosen token logprob, and a mapping of top tokens to logprobs (up to 20).

Configuration:
- Read API key from environment variable, e.g. `DEEPSEEK_API_KEY`.
- Model name set by config, default `deepseek-chat`.
- API url is https://api.deepseek.com
- refer to docs\chat-completion-deepsek-api.pdf for API help.
- no system prompt.
Important note (from docs): only the top-`top_logprobs` token logprobs are provided (at most 20, configurable). Other tokens may appear with sentinel logprobs (e.g., `-9999`). Confidence computation must ignore these sentinel values.
---

## Prompting
Provide a single function:
- `build_prompt(problem_text: str) -> list[dict]`

Requirements:
- Keep the system prompt stable.
- Put the problem text in the user message.
- Append the paper-aligned instruction to the user message: "Please reason step by step, and put your final answer within \\boxed{}."

Example shape:

```python
# src/prompting.py

def build_prompt(problem_text: str) -> list[dict]:
    instruction = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    user_content = f"{problem_text}\n\n{instruction}"
    return [
        # Optionally include an official DeepSeek system prompt here.
        {"role": "user", "content": user_content},
    ]
```

---

## Sampling procedure
Pool-first workflow (required for this repo):
- Step 1 (pool build, uses the LLM/API): for each problem, generate a fixed pool of `N_pool` complete traces once and save it to disk.
- Step 2 (evaluation, no LLM/API calls): resample working sets of size `K` from the saved pool and run all aggregators offline.

Per-sample fields to save in the pool:
- `sample_id`, raw `text`, `tokens` (if available), `token_logprobs`, per-token `top_logprobs`, extracted `answer` (int or `None`), and confidence `score` (float).

Practical defaults (configurable):
- `N_pool`: 4096 (paper-style; especially important if you will use `K=512`).
- `K`: 512 for paper-style results; 128 or 256 for quicker iteration.
- `temperature`: 0.6
- `max_tokens`: 4096
- Concurrency: moderate; avoid rate-limit failures. Configure with `--concurrency` (default 8).

Operational note:
- Tests and evaluation should depend on the saved pool artifacts. The LLM/API should not be called during tests or evaluation runs unless the pool is missing, the prompt/decoding settings changed, or a `--rebuild_pool` style flag is passed.

---

## Answer extraction
Implement `extract_answer(text: str) -> int | None`.

Requirements:
1. First, try to extract from `\\boxed{...}`.
2. If no boxed answer is found, fall back to the last integer in the text.
3. Validate the AIME range: keep only integers in `[0, 999]`. Otherwise return `None`.

Implementation tips:
- Accept forms like `\\boxed{123}` and `\\boxed{ 123 }`.
- If the boxed content includes extra text, extract the last integer inside the box.

---

## Confidence scoring (paper-aligned, offline)
The paper defines token confidence using the model's top-k token distribution at each position.

### Token confidence (paper Eq. 2)
Let the API return up to `k=20` top tokens with logprobs at each position `i`.
Define token confidence as `C_i = - mean(logprob(top_k_tokens_at_i))`.

Notes:
- Use the top-k logprobs, not only the chosen token logprob.
- If top-k data is missing for a token, fall back to `C_i = - chosen_logprob`.

### Trace-level confidence measurements to implement
Implement all of the following and keep them configurable:

1) Average trace confidence (paper Eq. 3)
- `C_avg = mean(C_i)` over all tokens in the trace.

2) Group confidence (paper Eq. 4)
- Use a sliding window over token confidences.
- Paper-aligned default window size: `group_size = 2048` tokens.
- Define each group as the last `group_size` tokens ending at position `i`.
- Group confidence is the mean token confidence within the window.

3) Bottom-10% group confidence (paper Eq. 5)
- Compute all group confidences in the trace.
- Take the bottom 10% of those group scores.
- Define `C_bottom10` as the mean of that bottom set.

4) Lowest group confidence (paper Eq. 6)
- `C_lowest = min(group_confidences)`.

5) Tail confidence (paper Eq. 7)
- Paper-aligned default tail length: `tail_tokens = 2048`.
- `C_tail = mean(C_i)` over the last `tail_tokens` tokens (or all tokens if shorter).

Suggested defaults for offline AIME:
- Use `C_lowest` or `C_bottom10` as the main score.
- Also report `C_tail` and `C_avg` for comparison.

---

## Aggregators to implement
Match the paper's offline evaluation definitions.

### Baseline: Cons@K (majority vote)
- Collect all valid extracted answers among the `K` samples.
- Compute the most frequent answer. Break ties deterministically (for example, smallest value).
- This predicted answer is used for accuracy.

Also compute Pass@1:
- Accuracy of the first sample only (or a single-sample run).

### DeepConf-style offline methods
Let `C_t` be the chosen trace-level confidence metric.

1) Measure@K (confidence-weighted majority voting)
- For each answer `a`, compute `V(a) = sum(C_t for traces with answer a)`.
- Predict `argmax_a V(a)`.

2) Measure+top-eta%@K (filter then weighted vote)
- Filter to the top `eta` fraction by confidence within the working set.
- Use `eta in {0.10, 0.90}` by default.
- After filtering, apply the same weighted vote as above.

Important details:
- Ignore invalid samples (no extracted answer).
- Recompute the top-eta cutoff within each problem's working set.
- If there are zero valid answers, mark the problem unanswered (count as incorrect unless specified otherwise).

---

## Evaluation output
For the chosen dataset, print at least:
- `Pass@1` accuracy
- `Cons@K` accuracy
- `Measure@K` accuracy (for each confidence metric you enable)
- `Measure+top-eta%@K` accuracy for `eta=0.10` and `eta=0.90`

Also print:
- valid samples per problem (mean and min)
- tie rate for Cons@K
- basic confidence summaries (mean and percentile stats)

Save artifacts:
- `runs/<timestamp>/results.json` (per-problem details)
- `runs/<timestamp>/summary.json` (aggregate metrics)
- Optionally raw API responses. If you save them, compress large payloads.

---

## CLI (example)
Implement a CLI roughly like this:

```bash
.\.venv\Scripts\python -m src.evaluate \
  --dataset data/aime_2025.jsonl \
  --problem_key problem \
  --answer_key answer \
  --k 64 \
  --temperature 0.6 \
  --max_tokens 4096 \
  --confidence_metric lowest_group \
  --group_size 2048 \
  --tail_tokens 2048 \
  --etas 0.1 0.9
```

Optional but useful:
- `--repeat 8` to resample and average metrics across multiple runs.
- `--seed` for deterministic resampling where applicable.

---

## Success criteria (what "reproduced" means here)
Using the same model, prompt, and sampling budget `K`:
- `Measure+top-eta%@K` and or `Measure@K` should beat `Cons@K` by a noticeable margin on the selected AIME set.
- The report should be reproducible from saved `runs/` artifacts without re-calling the API.

## Non-goals (do not prioritize)
- Matching the paper's exact reported percentages.
- Reproducing online early-stopping or token-savings precisely via API billing/accounting.
