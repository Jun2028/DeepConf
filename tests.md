# tests.md - How the agent should test while implementing DeepConf (offline) on AIME

## Testing strategy
Use a pool-first workflow:
- Tests should not call the LLM/API.
- Build the pool once, then run evaluation offline from saved artifacts.
- Only run minimal real API smoke tests for pool building.
- Use the repo-local `.venv` interpreter, not the base environment.
- Run tests with: `.\.venv\Scripts\python -m pytest -q`.

Test in layers: (1) pure functions, (2) mocked parsing and wiring, (3) pool-build smoke, (4) offline evaluation from pool, (5) reproducibility.

---

## 1) Unit tests (no API)
Focus on pure, deterministic components.

### `extract_answer.py`
Create test cases for boxed extraction first, then fallback:
- `\\boxed{123}` -> 123
- `\\boxed{ 007 }` -> 7
- boxed with extra text: `\\boxed{answer is 42}` -> 42 (last integer inside the box)
- no box but trailing number: `... therefore 456` -> 456 (fallback rule)
- multiple numbers present -> extracts from `\\boxed{}` if present, else last integer
- no integers -> `None`
- out-of-range (e.g., `1000`, `-1`) -> `None`
- whitespace/newlines around the boxed answer

### `aggregate.py`
Majority vote / tie-breaking:
- majority clear winner
- tie case (verify deterministic tie-break rule, e.g., smallest answer)
- behavior with invalid samples (ignored)
- behavior with zero valid answers (unanswered)

Confidence-weighted vote (Measure@K):
- verify weights are applied to the correct answers
- verify `V(a) = sum(confidence)` per answer
- verify argmax selection matches expected totals
- verify deterministic behavior under equal totals

Top-eta filtering (Measure+top-eta%@K):
- verify kept sample count is `ceil(eta * K_valid)`
- verify cutoff is recomputed within each working set
- verify it handles small `K_valid` (e.g., 1)
- verify it handles `K_valid = 0`

### `confidence.py`
Use small synthetic sequences so results are easy to check.

Token confidence (paper Eq. 2):
- compute `C_i` from top-k logprobs (mean of top-k logprobs, negated)
- fallback when top-k is missing: `C_i = -chosen_logprob`

Trace-level metrics:
- `avg` confidence on a short list
- group confidence with a small `group_size` in tests (e.g., 3 or 4)
- bottom-10% group confidence on a known set of group scores
- lowest group confidence equals the min group score
- tail confidence with a small `tail_tokens` in tests

---

## 2) Mocked integration tests (no real calls)
Goal: ensure API response parsing and pipeline wiring works without spending tokens.

### Fixtures
Create small fake API responses that include:
- generated `text`
- per-token chosen-token logprobs
- per-token top-k logprobs (top_logprobs)
- corner cases: missing `logprobs`, missing top-k, empty token lists, missing `text`, truncated outputs

### `deepseek_client.py` parsing tests
Assertions:
- correctly extracts generated text
- correctly extracts chosen-token logprob sequence
- correctly extracts per-token top-k logprobs
- falls back cleanly when top-k logprobs are missing
- produces a clear, actionable error if required schema fields are missing

### Pool and evaluation wiring tests
Given a tiny mocked pool:
- evaluation should run without any API calls
- resampling from the pool should produce valid working sets
- all metrics should compute from the pool artifacts

---

## 3) Real API smoke tests (pool build only)
Run only after unit + mock tests pass.

Use minimal spend and write the pool to disk:
- `--limit_problems 1`
- small `--n_pool` (e.g., 4 or 8)
- small `--max_tokens`
- paper-aligned decoding defaults are fine (`temperature=0.6`, `top_p=0.95`)

Assertions:
- API call succeeds
- logprobs exist and per-token top-k logprobs are present when requested
- pool artifacts are written and can be loaded

Important:
- Do not call the API during offline evaluation tests.

---

## 4) Offline end-to-end from saved pool
Use a tiny saved pool to validate full orchestration offline:
- 3 problems
- `n_pool` can be small for tests
- evaluate with `K=8` (or smaller if needed)
- test both `eta=0.1` and `eta=0.9`

Confirm:
- run folder is created and populated
- `Pass@1`, `Cons@K`, `Measure@K`, and `Measure+top-eta%@K` compute without API calls
- robust to malformed samples (no crash)

---

## 5) Regression and reproducibility checks
Determinism from artifacts:
- tie-breaking is deterministic
- sorting by confidence is deterministic
- top-eta selection is deterministic

Resampling determinism:
- given a fixed seed and a saved pool, resampling should be reproducible
- metrics computed from the same saved working sets should be identical

Artifact-based reproducibility:
- store pool artifacts with enough detail to recompute confidence and metrics offline
- re-run evaluation from saved artifacts and assert identical metrics

---

## Development conveniences (recommended)
- Add `--limit_problems N` to keep runs tiny during development.
- Add `--n_pool` and allow very small values in smoke tests.
- Add explicit modes such as `--pool_only` and `--eval_only`.
- Add a `--rebuild_pool` flag; otherwise reuse the saved pool.
- Fail fast if required logprobs are missing during pool build.
