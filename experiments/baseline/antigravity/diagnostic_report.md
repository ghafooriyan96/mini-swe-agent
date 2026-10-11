# AntiGravity Manager Integration Diagnostic Report

## 1. Local Manager Overview
- **Base URL:** `http://127.0.0.1:8045/v1`
- **Protocol:** OpenAI-compatible REST API
- **Endpoint tested:** `/v1/models`, `/v1/chat/completions`
- **Authentication:** Accepts bearer tokens (e.g. `Bearer local`) without requiring real upstream external API keys.

### Discovered Model IDs
The endpoint exposes 42 models, including:
- `claude-sonnet-4-6` (Preferred model)
- `claude-sonnet-4-6-thinking`
- `claude-sonnet-5-5`
- `claude-opus-4-6`, `claude-opus-5-5`
- `gemini-3.1-pro-high` (Alternative model)
- `gemini-3.1-pro`, `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
- `gpt-oss-120b-medium`

---

## 2. Integration Architecture & Configuration
Mini-SWE-agent v2 integrates with language models via LiteLLM using `minisweagent.models.litellm_model.LitellmModel`.

To route requests through the local AntiGravity Manager without modifying core agent or model classes:
1. **Model name prefix:** `openai/<model-id>` (e.g. `openai/claude-sonnet-4-6` or `openai/gemini-3.1-pro-high`) instructs LiteLLM to use its OpenAI-compatible provider.
2. **Endpoint kwargs:** `api_base: http://127.0.0.1:8045/v1` and `api_key: local` are passed via `model_kwargs`.
3. **Cost tracking:** Local proxy models are not registered in LiteLLM's public pricing table. `cost_tracking: "ignore_errors"` prevents LiteLLM from raising `RuntimeError`. `cost_limit: 0.0` disables agent shutdown due to unmetered cost.

The reusable configuration is saved at:
`config/research/antigravity.yaml`

---

## 3. Tool Calling & Action Parsing Validation
- **Native Tool Calling (`LitellmModel`):**
  - Sends OpenAI function specification `{"name": "bash", "parameters": {"properties": {"command": ...}}}`.
  - The local AntiGravity Manager returns native OpenAI `tool_calls` (`toolu_vrtx_...` for Claude, `call_...` for Gemini).
  - The model parses tool calls via `parse_toolcall_actions()` into `[{"command": ...}]`.
  - Multi-turn execution with `role: "tool"` observation returns valid subsequent turns and completion actions (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`).
- **Text-Based Parsing (`LitellmTextbasedModel`):**
  - Tested with triple backtick `mswea_bash_command` extraction; works cleanly as an alternative.

---

## 4. End-to-End Calculator Benchmark Results

Directory: `/Users/ehsang/projects/mini-agent-sandbox`
Initial defect: `calculator.py` had `return a - b` instead of `return a + b`, causing `test_calculator.py` to fail (`assert -1 == 5`).

### Experiment 1: Preferred Model (`claude-sonnet-4-6`)
- **Trajectory:** `experiments/baseline/antigravity/trajectory_calculator_claude.json`
- **Total Steps:** 6 steps
- **Steps summary:**
  1. Inspect directory and files (`ls -la && cat calculator.py && echo "---" && cat test_calculator.py`)
  2. Apply fix (`sed -i '' 's/return a - b/return a + b/' calculator.py && cat calculator.py`)
  3. Attempt pytest (`pytest test_calculator.py -v`) -> not found in base shell
  4. Attempt python module pytest (`python -m pytest test_calculator.py -v`)
  5. Install and run pytest (`pip install pytest && python -m pytest test_calculator.py -v`) -> **1 passed in 0.00s**
  6. Submit work (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- **Exit Status:** `Submitted` (Exit code 0)
- **Result:** Successfully fixed and verified.

### Experiment 2: Alternative Model (`gemini-3.1-pro-high`)
- **Trajectory:** `experiments/baseline/antigravity/trajectory_calculator_gemini.json`
- **Total Steps:** 7 steps
- **Steps summary:**
  1. Inspect directory (`ls -la`)
  2. Inspect files (`cat calculator.py && echo "---" && cat test_calculator.py`)
  3. Reproduce failure (`pytest test_calculator.py`) -> 1 failed
  4. Fix file (`cat <<'EOF' > calculator.py\ndef add(a, b):\n    return a + b\nEOF`)
  5. Verify fix (`cat calculator.py && pytest test_calculator.py`) -> **1 passed in 0.00s**
  6. Edge-case test assertions (`python3 -c "assert add(0, 0) == 0; ...'`) -> passed
  7. Submit work (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- **Exit Status:** `Submitted` (Exit code 0)
- **Result:** Successfully fixed and verified.

---

## 5. Compatibility Observations & Recommendations
1. **`claude-sonnet-4-6` Intermittent Deprecation Message:**
   - During stress testing, `claude-sonnet-4-6` intermittently returned text: `"Claude Sonnet 4.6 is no longer available. Please switch to Claude Sonnet 5.5."` with 0 tokens.
   - When this happens without a tool call, mini-SWE-agent raises a `FormatError` and prompts the model to correct its output.
   - `claude-sonnet-5-5` and `gemini-3.1-pro-high` had 100% reliability and never emitted this message.
   - **Recommendation:** `claude-sonnet-4-6` works and solved the benchmark, but `claude-sonnet-5-5` or `gemini-3.1-pro-high` are optimal for production runs if strict zero-interruption behavior is needed.
2. **Cost Tracking Configuration:**
   - Must set `cost_tracking: "ignore_errors"` and `agent.cost_limit: 0.0` in the YAML config.
3. **Environment CWD in Sandbox Experiments:**
   - Use `-c environment.cwd=/path/to/sandbox` when running tasks outside the mini-swe-agent repository root.

---

## 6. Exact Reproduction Commands
```bash
# 1. Run interactive agent with preferred model (claude-sonnet-4-6):
MSWEA_CONFIGURED=true ./.venv/bin/mini \
  -c config/research/antigravity.yaml \
  -c environment.cwd=/Users/ehsang/projects/mini-agent-sandbox \
  -c agent.step_limit=10 \
  -o experiments/baseline/antigravity/trajectory_calculator_claude.json \
  -t "There is a bug in calculator.py causing test_calculator.py to fail. Please inspect the files, fix the bug in calculator.py, run pytest test_calculator.py to verify, and submit your work."

# 2. Run with alternative model (gemini-3.1-pro-high):
MSWEA_CONFIGURED=true ./.venv/bin/mini \
  -c config/research/antigravity.yaml \
  -m openai/gemini-3.1-pro-high \
  -c environment.cwd=/Users/ehsang/projects/mini-agent-sandbox \
  -c agent.step_limit=10 \
  -o experiments/baseline/antigravity/trajectory_calculator_gemini.json \
  -t "There is a bug in calculator.py causing test_calculator.py to fail. Please inspect the files, fix the bug in calculator.py, run pytest test_calculator.py to verify, and submit your work."

# 3. Run regression test suite:
PATH="/Users/ehsang/projects/mini-swe-agent/.venv/bin:$PATH" ./.venv/bin/pytest \
  --ignore=tests/environments/extra/test_contree.py \
  --ignore=tests/environments/extra/test_swerex_modal.py
```
