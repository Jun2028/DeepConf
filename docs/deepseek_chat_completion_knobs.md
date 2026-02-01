# DeepSeek Chat Completion (API “knobs”) — single-page reference

This note consolidates the **/chat/completions** request fields you can tune for **speed / cost / output length**, plus the few mode-specific gotchas (thinking mode, JSON mode, tools, caching).

## Canonical reference pages (copy/paste)
```text
https://api-docs.deepseek.com/api/create-chat-completion
https://api-docs.deepseek.com/guides/thinking_mode
https://api-docs.deepseek.com/quick_start/pricing
https://api-docs.deepseek.com/guides/kv_cache
https://api-docs.deepseek.com/guides/json_mode
https://api-docs.deepseek.com/guides/tool_calls
```

---

## Endpoint

- Method: `POST`
- Path: `/chat/completions`
- Base URL (docs): `https://api.deepseek.com` (OpenAI-compatible; also `.../v1` works but is unrelated to model version)

---

## Models + hard limits (important for budgeting)

From the pricing page:

- `deepseek-chat`
  - Context length: **128K**
  - Max output: **default 4K**, **max 8K**
- `deepseek-reasoner`
  - Context length: **128K**
  - Max output: **default 32K**, **max 64K**

Billing is based on **input + output tokens**.

---

## Fast/short defaults (recommended recipe)

If you want “fast and short” for `deepseek-chat`, do this:

- Disable thinking: `thinking: {"type":"disabled"}`
- Set a strict output cap: `max_tokens: 64..512` (pick what you need)
- Add a `stop` sequence to cut rambling (often `["\n\n"]`)
- Use deterministic sampling: `temperature: 0.0`
- Avoid extras: omit `tools`, omit `logprobs`
- Enable streaming if you care about time-to-first-token: `stream: true`

Example:
```json
{
  "model": "deepseek-chat",
  "thinking": { "type": "disabled" },
  "messages": [{ "role": "user", "content": "Answer in 1 sentence: ..." }],
  "max_tokens": 128,
  "temperature": 0.0,
  "stop": ["\n\n"],
  "stream": true
}
```

---

## Request schema (fields you can tune)

### 1) Core fields

#### `messages` (required)
Array of chat messages. Each message has:
- `role`: `"system" | "user" | "assistant" | "tool"`
- `content`: string (assistant content can be `null` if the model tool-calls)
- optional `name`: differentiate participants of same role

Special cases inside `messages`:
- Assistant message can set **`prefix: true`** (Beta) to force the model to begin the next answer with that exact content. Requires `base_url="https://api.deepseek.com/beta"`.
- For tool results, message role is `"tool"` and you must include `tool_call_id`.

#### `model` (required)
`"deepseek-chat"` or `"deepseek-reasoner"`.

---

### 2) “Thinking mode” toggle

#### `thinking` (optional object)
```json
"thinking": { "type": "enabled" | "disabled" }
```
- `enabled` switches to thinking mode
- `disabled` is non-thinking mode

**Thinking mode gotchas (critical):**
- Output includes `reasoning_content` (CoT) plus final `content`.
- `max_tokens` counts **CoT + final answer**.
- Not supported / no-op in thinking mode:
  - `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` (no effect)
  - `logprobs`, `top_logprobs` (will error if set)

If you want short/fast: keep thinking **disabled**.

---

### 3) Output length controls

#### `max_tokens` (optional int)
Hard cap on generated tokens (for thinking mode: includes CoT + final answer).

#### `stop` (optional string or string[])
Up to **16** stop sequences where generation halts.

---

### 4) Speed/latency knobs

#### `stream` (optional bool)
If true, returns Server-Sent Events (SSE) token deltas, ending with `data: [DONE]`.

#### `stream_options` (optional object; only with `stream: true`)
- `include_usage` (bool): stream one final chunk with `usage` populated (and `choices=[]`).

---

### 5) Sampling controls (verbosity/creativity)

#### `temperature` (optional number, default 1, <=2)
Lower → more deterministic, usually shorter/more direct.

DeepSeek’s own “temperature” quick-start suggests:
- Coding/Math: `0.0`
- Data cleaning/analysis: `1.0`
- General conversation: `1.3`
- Translation: `1.3`
- Creative writing: `1.5`

#### `top_p` (optional number, default 1, <=1)
Nucleus sampling alternative to temperature.
Use **either** `temperature` **or** `top_p`, not both.

---

### 6) Repetition/topic drift controls

#### `presence_penalty` (optional number, default 0, range [-2, 2])
Higher → encourages new topics.

#### `frequency_penalty` (optional number, default 0, range [-2, 2])
Higher → discourages repeating tokens/phrases.

(These are **ignored** in thinking mode.)

---

### 7) Tool calling (functions)

#### `tools` (optional array)
A list of tool definitions. Currently only `"function"` tools are supported.
- Max **128** functions.
- Each tool includes:
  - `type: "function"`
  - `function: { name, description?, parameters?, strict? }`
- `strict` (bool, default false): Beta strict schema adherence for tool outputs.

Strict-mode requirements (per Tool Calls guide):
- Use `base_url="https://api.deepseek.com/beta"`
- Set `"strict": true` on each function in `tools`
- Server validates JSON Schema; invalid schema returns error.

#### `tool_choice` (optional)
Controls tool usage:
- `"none"`: never call tools
- `"auto"`: model may call tools or answer normally
- `"required"`: must call at least one tool
- Or force a specific tool:
```json
{"type":"function","function":{"name":"my_function"}}
```
Default: `"none"` if no tools; `"auto"` if tools are present.

**For maximum speed**: omit tools or set `tool_choice: "none"` unless you truly need tool calls.

---

### 8) Log probabilities (debugging; slower/bigger payload)

#### `logprobs` (optional bool)
If true, returns logprobs per output token.

#### `top_logprobs` (optional int, <=20)
How many alternative tokens to include per position.
Requires `logprobs: true`.

**Thinking mode:** `logprobs` / `top_logprobs` are **not supported** (errors).

---

### 9) Structured output (JSON mode)

#### `response_format` (optional object)
```json
"response_format": { "type": "text" | "json_object" }
```

If you use `json_object`:
- You must also instruct the model to output JSON (include the word “json” and show an example format).
- Set `max_tokens` high enough so the JSON doesn’t get truncated.
- Docs warn that the model can otherwise “run whitespace” to the limit if not instructed properly.

---

## Response fields you should read (for performance + caching)

The response includes a `usage` object with:
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `prompt_cache_hit_tokens`
- `prompt_cache_miss_tokens`
- `completion_tokens_details.reasoning_tokens` (reasoning tokens)

**Context caching is enabled by default**:
- Cache hits happen only for **identical prefixes from the 0th token**.
- Track cache behavior via `prompt_cache_hit_tokens` vs `prompt_cache_miss_tokens`.

---

## Practical speed/cost checklist

1) Cap output: `max_tokens` + a `stop` sequence  
2) Keep thinking off unless you need it  
3) Keep prompts short; reuse a stable system prompt to increase cache hits  
4) Avoid `logprobs`, tools, and strict JSON mode unless required  
5) Use `stream: true` for better perceived latency  

