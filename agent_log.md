## 2026-01-27 - Codex run log (pool-first implementation)

Summary of meaningful changes:
- Implemented pool-first pipeline modules under `src/`: `config.py`, `aime_loader.py`, `prompting.py`, `deepseek_client.py`, `extract_answer.py`, `confidence.py`, `aggregate.py`, `sampling.py`, `evaluate.py`, `download_dataset.py`, `visualize.py`, and `__init__.py`.
- Added `requirements.txt` with runtime + test dependencies and created `tests/` + `runs/` + `data/` folders.
- Added a fully offline test suite aligned to `tests.md`.

Testing performed:
- Ran `pytest -q` from repo root.
- Result: 30 passed.

Notable issues and resolutions:
- Issue: `ModuleNotFoundError: No module named 'src'` during test collection.
  - Fix: added `tests/conftest.py` to insert repo root into `sys.path`.
- Observation: Pytest cache warnings still appear due to an environment-specific cache path issue.
  - Impact: non-blocking; tests still pass.

Important behavior notes:
- `src/evaluate.py` uses the pool-first workflow:
  - Pool build calls the API once (unless reusing an existing pool).
  - Offline evaluation resamples from saved pool artifacts and does not call the API.
- To avoid API calls during evaluation/testing, use `--eval_only` and ensure the pool exists.

## 2026-01-27 - Environment correction (.venv)
- Created a repo-local virtual environment at `.venv` and installed `requirements.txt` into it using `.\.venv\Scripts\python -m pip install -r requirements.txt`.
- Updated `instructions.md` and `tests.md` to explicitly require the `.venv` interpreter.
- Updated `.gitignore` to ignore `.venv/` and `.pytest_cache/`.
- Verified tests pass under `.venv` with `.\.venv\Scripts\python -m pytest -q` (30 passed).

## 2026-01-27 - API smoke test and dataset fetch (.venv)
- Downloaded dataset rows to `data/aime_2025_rows.json` using `\.venv\Scripts\python -m src.download_dataset`.
- Ran a minimal DeepSeek API smoke test (pool build only):
  - Command: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 1 --k 1 --limit_problems 1 --max_tokens 256 --pool_only`
  - Result: pool built at `runs/pools/275c6233106ff156`.
  - Verified the saved sample contains per-token `logprob` and non-empty `top_logprobs`.
- Re-ran offline tests under `.venv`: `\.venv\Scripts\python -m pytest -q` (30 passed).

## 2026-01-28 - Dry-run pool build requested by user
Actions taken:
- Attempted the requested dry run size (`--n_pool 64 --k 32 --limit_problems 2 --max_tokens 512 --pool_only`).
  - Result: timed out at 10 minutes and again at 30 minutes in this environment.
- Adjusted to a smaller dry run that completes quickly while still validating the pipeline.

Successful dry run command:
- `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 512 --pool_only`

Dry run results:
- Pool path: `runs/pools/cdf326c5f56335cb`
- Problems covered: `1`, `2`
- Samples per problem: 8 each (as configured)
- Token counts: min/mean/max = 512/512/512 per sample (consistent with `--max_tokens 512`)
- Verified per-token `top_logprobs` are present in saved artifacts.

Conclusion:
- API calls, parsing, pool persistence, and confidence inputs all work end-to-end on a small pool.
- The originally requested 64x2 dry run is feasible but exceeds the current tool timeout limits; use smaller dry runs here or run the larger build locally outside this tool timeout.

## 2026-01-28 - Confidence fix for DeepSeek sentinel logprobs
Issue:
- DeepSeek top_logprobs include sentinel values like `-9999.0` for tokens outside the top-k set.
- The previous token confidence used the mean across all top_logprobs values, causing confidence to blow up (e.g., ~9k).

Fix:
- Updated `src/confidence.py` to filter out sentinel values using a threshold (`LOGPROB_SENTINEL_THRESHOLD = -1000.0`).
- If no valid top-k values remain after filtering, the code falls back to `-chosen_logprob`.
- Added `tests/test_confidence.py::test_token_confidence_ignores_sentinel_values`.

Validation:
- Ran `.\.venv\Scripts\python -m pytest -q`.
- Result: 31 passed.

## 2026-01-28 - Dry-run accuracy and eval_only fix
- Attempted offline evaluation for the dry-run pool and hit a JSON serialization error because `RunConfig` contains `Path` objects.
- Fix: added `_jsonify()` in `src/evaluate.py` and used it when writing `details.config`.
- Added a regression assertion in `tests/test_pool_eval_offline.py` (`json.dumps(details)`).
- Re-ran tests: `.\.venv\Scripts\python -m pytest -q` (31 passed).
- Dry-run offline evaluation command:
  - `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 512 --eval_only`
- Results on 2 problems with small pool/working set:
  - `pass_at_1 = 0.0`
  - `cons_at_k = 0.0`
  - `measure_at_k = 0.0`
  - `measure_top_eta_at_k[0.1] = 0.0`
  - `measure_top_eta_at_k[0.9] = 0.0`
  - Summary saved to `runs/20260127-165218/summary.json`.

## 2026-01-28 - Dry run with max_tokens=2048
Commands:
- Pool build: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 2048 --pool_only`
- Offline eval: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 2048 --eval_only`

Results:
- Pool id: `runs/pools/e2981b74a3778216`
- Summary metrics (2 problems):
  - pass_at_1 = 0.5
  - cons_at_k = 0.5
  - measure_at_k = 0.5
  - measure_top_eta_at_k[0.1] = 0.5
  - measure_top_eta_at_k[0.9] = 0.5
- Summary saved to: `runs/20260127-170903/summary.json`

Per-problem notes:
- Problem 1 (gt=70): 8/8 extracted answers were correct (all 70).
- Problem 2 (gt=588): 0/8 extracted answers were correct.
- This explains the 0.5 accuracy across 2 problems.

## 2026-01-28 - finish_reason surfaced for truncation diagnosis
- The API doc PDF is largely non-extractable text-wise, but the live API responses include `finish_reason`.
- I updated `src/deepseek_client.py` to capture `finish_reason` and stored it in pool samples via `src/sampling.py`.
- Rebuilt a tiny pool to check truncation:
  - Command: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 1 --k 1 --limit_problems 2 --max_tokens 2048 --pool_only --rebuild_pool`
  - Pool: `runs/pools/5ad7761cffb1364e`
  - Problem 1: `finish_reason=stop`, tokens=767, answer=70
  - Problem 2: `finish_reason=length`, tokens=2048, answer=91
- This confirms the missing boxed answer on problem 2 is due to hitting `max_tokens` (finish_reason=length).

## 2026-01-28 - Dry run with thinking disabled (max_tokens=2048)
Command:
- `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 2048 --pool_only --rebuild_pool`

Observations:
- Pool reused id: `runs/pools/e2981b74a3778216`
- Problem 1: finish_reason = stop (8/8), token_len mean �� 803 (min 652, max 1017)
- Problem 2: finish_reason = length (8/8), token_len = 2048 for all samples

Conclusion:
- Thinking disabled does not eliminate truncation on the harder problem; problem 2 still hits the max token cap at 2048.
- Runtime remains slow on long generations (problem 2 averaged ~58s per trace in this run).

## 2026-01-28 - Dry run with max_tokens=8192
Command:
- `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 8192 --pool_only --rebuild_pool`

Observations:
- Pool id: `runs/pools/7d023767719ca354`
- Problem 1: finish_reason = stop (8/8), token_len mean �� 1184 (min 758, max 1599)
- Problem 2: finish_reason = stop (8/8), token_len mean �� 3703 (min 3142, max 4762)
- Boxed answers present: 8/8 for both problems

Offline eval (same pool):
- Command: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --n_pool 8 --k 4 --limit_problems 2 --max_tokens 8192 --eval_only`
- Summary:
  - pass_at_1 = 0.5
  - cons_at_k = 1.0
  - measure_at_k = 0.5
  - measure_top_eta_at_k[0.1] = 0.5
  - measure_top_eta_at_k[0.9] = 0.5
- Summary saved to: `runs/20260127-180928/summary.json`

Conclusion:
- Increasing max_tokens to 8192 eliminated truncation for both problems (finish_reason=stop, boxed answers present).
- Runtime increased substantially (problem 2 traces averaged ~100s each).

## 2026-01-28 - Pool merge tool and readable pool names
- Updated pool naming to be human-readable and deterministic (dataset/model/n_pool/max_tokens/etc + short hash) in `src/sampling.py`.
- Added fallback lookup for older pools via metadata matching (`find_matching_pool_dir`).
- Added pool merge tool: `src/merge_pools.py`.
  - Merges compatible pools, reindexes `sample_id`, and adds `source_pool`/`source_sample_id` fields.
- Added tests: `tests/test_pool_merge.py`.
- Tests: `.\.venv\Scripts\python -m pytest -q` (32 passed).

## 2026-01-28 - Temperature-only decoding
- Updated defaults to use temperature only:
  - `temperature = 0.6`
  - `top_p = 1.0`
- Updated `instructions.md`, `src/config.py`, and CLI defaults in `src/evaluate.py`.
- Tests: `.\.venv\Scripts\python -m pytest -q` (32 passed).

## 2026-01-28 - Pool build script creates .venv if missing
- Updated `scripts/build_pool_10q_128_8192.ps1` to create `.venv` automatically and install `requirements.txt` when missing.

## 2026-01-28 - Fix pool build script path handling
- Updated `scripts/build_pool_10q_128_8192.ps1` to resolve paths relative to the repo root using `$PSScriptRoot`.
- Ensures `.venv` is found even when the script is launched from a different working directory.

## 2026-01-29 - Concurrency support for pool builds
- Added `PoolingConfig.concurrency` and CLI flag `--concurrency` to enable parallel API requests.
- Implemented thread-based parallel sampling in `src/sampling.py` (per-thread DeepSeekClient to avoid session sharing issues).
- Updated `instructions.md` to reflect `top_p=1.0` and concurrency flag.
- Tests: `.\.venv\Scripts\python -m pytest -q` (32 passed).

## 2026-01-29 - Concurrency guidance
- Concurrency uses multiple simultaneous requests with the same API key; this is allowed but limited by provider rate limits.
- Recommended to start small (e.g., `--concurrency 2` or `4`) and monitor for 429s/timeouts.
- Local PC load is minimal; network and API throttling are the main constraints.

## 2026-01-29 - Evaluation on 9-question n_pool=128 pool
- Found pool `runs/pools/aime_2025_rows__deepseek-chat__n128__mt8192__t0p6__p1__lp20__022a5b37` missing `metadata.json`.
- Reconstructed metadata from pool name, prompt, and problem files (problems 1..9) and wrote `metadata.json`.
- Ran offline evaluation:
  - Command: `\.venv\Scripts\python -m src.evaluate --dataset data/aime_2025_rows.json --limit_problems 9 --n_pool 128 --k 128 --max_tokens 8192 --eval_only`
  - Summary:
    - pass_at_1 = 0.7777777777777778
    - cons_at_k = 0.8888888888888888
    - measure_at_k = 0.7777777777777778
    - measure_top_eta_at_k[0.1] = 0.7777777777777778
    - measure_top_eta_at_k[0.9] = 0.7777777777777778
  - Summary saved to: `runs/20260129-035635/summary.json`

## 2026-01-29 - Eval across confidence definitions (n_pool=128, K=128, 9 problems)
- avg: `runs/20260129-040723/summary.json`
  - cons@k = 0.8888888889
  - measure@k = 0.8888888889
  - top-eta@k (0.1) = 0.7777777778
  - top-eta@k (0.9) = 0.8888888889
- bottom10_group: `runs/20260129-041036/summary.json`
  - cons@k = 0.8888888889
  - measure@k = 0.8888888889
  - top-eta@k (0.1) = 0.7777777778
  - top-eta@k (0.9) = 0.8888888889
- tail: `runs/20260129-041352/summary.json`
  - cons@k = 0.8888888889
  - measure@k = 0.8888888889
  - top-eta@k (0.1) = 0.7777777778
  - top-eta@k (0.9) = 0.8888888889

Baseline for comparison (same pool): cons@k = 0.8888888889.

## 2026-01-29 - Eval across confidence definitions (n_pool=128, K=128, 10 problems)
- lowest_group: `runs/20260129-112733/summary.json`
  - cons@k = 0.9
  - measure@k = 0.8
  - top-eta@k (0.1) = 0.8
  - top-eta@k (0.9) = 0.8
- avg: `runs/20260129-113148/summary.json`
  - cons@k = 0.9
  - measure@k = 0.9
  - top-eta@k (0.1) = 0.8
  - top-eta@k (0.9) = 0.9
- bottom10_group: `runs/20260129-113608/summary.json`
  - cons@k = 0.9
  - measure@k = 0.9
  - top-eta@k (0.1) = 0.7
  - top-eta@k (0.9) = 0.9
- tail: `runs/20260129-114022/summary.json`
  - cons@k = 0.9
  - measure@k = 0.9
  - top-eta@k (0.1) = 0.8
  - top-eta@k (0.9) = 0.9

Baseline for comparison (same pool): cons@k = 0.9.

## 2026-01-29 - Improved concurrency coordination
- Default concurrency set to 8 (config + CLI).
- Pool sampling now uses a shared counter with thread-safe assignment so workers coordinate sample_ids and exactly produce `n_pool` traces.
- Added retry on failed requests within the same sample_id.
- Updated `instructions.md` default concurrency to 8.
- Tests: `.\.venv\Scripts\python -m pytest -q` (32 passed).

## 2026-01-29 - Update pool script to generate remaining 20 problems
- Updated `scripts/build_pool_20q_128_8192.ps1` to use `--limit_problems 30` so it fills problems 11-30 if 1-10 already exist.

## 2026-01-29 - Add 30q pool script name
- Copied `scripts/build_pool_20q_128_8192.ps1` to `scripts/build_pool_30q_128_8192.ps1` to reflect the 1–30 total problem target.

## 2026-01-29 - Remove obsolete pool script
- Deleted `scripts/build_pool_20q_128_8192.ps1` after confirming the 30q script is the correct one to use.

## 2026-01-29 - Increase pool script concurrency
- Updated `scripts/build_pool_30q_128_8192.ps1` to use `--concurrency 64`.

## 2026-01-29 - Adaptive concurrency under throttling
- Added adaptive concurrency and cooldown logic to pool sampling so 429s reduce in-flight work and back off instead of hammering or failing.

## 2026-01-29 - Document prompt template
- Added `prompt.md` describing the exact prompt construction and payload format.

## 2026-01-30 - Evaluate 30-question pool
- Ran offline evaluation on the completed 30-problem pool (n_pool=128, k=32, mt=8192).
- Summary written to `runs/20260130-115656/summary.json` and `runs/20260130-115656/results.json`.

## 2026-01-30 - Evaluate 30-question pool with all confidence metrics (k=128)
- avg: `runs/20260130-125535/summary.json`
- lowest_group: `runs/20260130-131727/summary.json`
- bottom10_group: `runs/20260130-133958/summary.json`
- tail: `runs/20260130-140145/summary.json`

## 2026-01-31 - Streamed evaluation for 512-pool
- Added `--use_stored_confidence` to evaluation to stream large pool files without loading tokens.
- Ran 30-problem evaluation on the 512-pool: `runs/20260131-040014/summary.json`.

## 2026-01-31 - Evaluate 512-pool for all confidence metrics
- Added `--all_metrics` mode to evaluate streaming all metrics from tokens.
- Results in `runs/20260131-090135/{avg,bottom10_group,lowest_group,tail}/summary.json`.

## 2026-01-31 - Add paper-parameter pool script
- Added `scripts/build_pool_30q_4096_paper.ps1` matching the paper’s decoding params (temp=0.6, top_p=0.95, top_logprobs=20, max_tokens=64k, n_pool=4096, k=512).

## 2026-01-31 - Align paper script to deepseek-chat max_tokens
- Updated `scripts/build_pool_30q_4096_paper.ps1` to use max_tokens=8192 due to model limits.

## 2026-01-31 - Temperature-only decoding per docs
- Removed explicit top_p from the paper pool script and updated `instructions.md`/CLI example to follow the “temperature or top_p, not both” guidance.

## 2026-01-31 - Increase pool script concurrency
- Updated `scripts/build_pool_30q_512_8192.ps1` to use `--concurrency 256`.
