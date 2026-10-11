import json
import math
import os
import subprocess
import time
from pathlib import Path

SANDBOX_DIR = Path("/Users/ehsang/projects/mini-agent-sandbox")
REPO_DIR = Path("/Users/ehsang/projects/mini-swe-agent")
EXP_DIR = REPO_DIR / "experiments" / "phase2_test_feedback"
TRAJ_DIR = EXP_DIR / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# --- Task Definitions ---
def setup_task1():
    """Task 1: Simple assertion failure."""
    cleanup_sandbox()
    (SANDBOX_DIR / "math_utils.py").write_text(
        "def power(base: int, exp: int) -> int:\n"
        "    # Bug: multiplication instead of exponentiation\n"
        "    return base * exp\n"
    )
    (SANDBOX_DIR / "test_math_utils.py").write_text(
        "from math_utils import power\n\n"
        "def test_power():\n"
        "    assert power(2, 3) == 8\n"
    )


def setup_task2():
    """Task 2: Multiple test failures."""
    cleanup_sandbox()
    (SANDBOX_DIR / "string_utils.py").write_text(
        "def reverse_words(s: str) -> str:\n"
        "    # Bug: returns original string\n"
        "    return s\n\n"
        "def capitalize_words(s: str) -> str:\n"
        "    # Bug: returns original string\n"
        "    return s\n"
    )
    (SANDBOX_DIR / "test_string_utils.py").write_text(
        "from string_utils import reverse_words, capitalize_words\n\n"
        "def test_reverse_words():\n"
        "    assert reverse_words('hello world') == 'world hello'\n\n"
        "def test_capitalize_words():\n"
        "    assert capitalize_words('hello world') == 'Hello World'\n"
    )


def setup_task3():
    """Task 3: Realistic exception / input-validation bug."""
    cleanup_sandbox()
    (SANDBOX_DIR / "user_validator.py").write_text(
        "def parse_user_age(age_input):\n"
        "    # Bug: does not validate positive integer or raise expected ValueError message\n"
        "    return int(age_input)\n"
    )
    (SANDBOX_DIR / "test_user_validator.py").write_text(
        "import pytest\n"
        "from user_validator import parse_user_age\n\n"
        "def test_valid_age():\n"
        "    assert parse_user_age('25') == 25\n\n"
        "def test_negative_age():\n"
        "    with pytest.raises(ValueError, match='Age must be a positive integer'):\n"
        "        parse_user_age('-5')\n\n"
        "def test_invalid_type():\n"
        "    with pytest.raises(ValueError, match='Age must be a positive integer'):\n"
        "        parse_user_age('not_a_number')\n"
    )


def cleanup_sandbox():
    for f in SANDBOX_DIR.glob("*.py"):
        f.unlink()


def restore_sandbox_clean():
    cleanup_sandbox()
    (SANDBOX_DIR / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (SANDBOX_DIR / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )


TASKS = {
    "task1_simple_assertion": {
        "setup_fn": setup_task1,
        "test_file": "test_math_utils.py",
        "prompt": (
            "There is a bug in math_utils.py causing test_math_utils.py to fail. "
            "Please inspect the code, run pytest test_math_utils.py to see the test failure, "
            "fix the bug in math_utils.py, re-verify with pytest test_math_utils.py, and submit your work."
        ),
    },
    "task2_multi_failure": {
        "setup_fn": setup_task2,
        "test_file": "test_string_utils.py",
        "prompt": (
            "Multiple tests are failing in test_string_utils.py. "
            "Please inspect the codebase, run pytest test_string_utils.py to analyze the failures, "
            "fix the functions in string_utils.py, verify by running pytest test_string_utils.py, and submit your work."
        ),
    },
    "task3_input_validation": {
        "setup_fn": setup_task3,
        "test_file": "test_user_validator.py",
        "prompt": (
            "There is an input-validation bug in user_validator.py causing test_user_validator.py to fail. "
            "Please run pytest test_user_validator.py to inspect the errors, "
            "implement the required input validation in user_validator.py, "
            "re-verify with pytest test_user_validator.py, and submit your work."
        ),
    },
}

CONFIGS = {
    "A_baseline": "config/research/antigravity.yaml",
    "B_loop_detection_only": "config/research/antigravity_loop_detection.yaml",
    "C_test_feedback_only": "config/research/antigravity_test_feedback.yaml",
    "D_combined": "config/research/antigravity_combined.yaml",
}


def run_agent(config_file: str, output_path: Path, prompt: str) -> dict:
    cmd = [
        str(REPO_DIR / ".venv" / "bin" / "mini"),
        "-c",
        config_file,
        "-m",
        "openai/gemini-3.1-pro-high",
        "-c",
        f"environment.cwd={SANDBOX_DIR}",
        "-c",
        "agent.step_limit=10",
        "-y",
        "--exit-immediately",
        "-o",
        str(output_path),
        "-t",
        prompt,
    ]
    env = os.environ.copy()
    env["MSWEA_CONFIGURED"] = "true"
    env["PATH"] = f"{REPO_DIR / '.venv' / 'bin'}:{env.get('PATH', '')}"

    start_time = time.time()
    proc = subprocess.run(cmd, cwd=REPO_DIR, env=env, capture_output=True, text=True)
    runtime = time.time() - start_time

    return {
        "returncode": proc.returncode,
        "runtime": runtime,
    }


def parse_run_trajectory(traj_path: Path, task_key: str) -> dict:
    data = json.loads(traj_path.read_text())
    messages = data.get("messages", [])
    info = data.get("info", {})

    total_steps = 0
    repeated_commands = 0
    failed_test_runs = 0
    commands_seen = []
    feedbacks_received_in_history = 0
    feedback_chars_total = 0

    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0

    for m in messages:
        if m.get("role") == "assistant":
            total_steps += 1
            usage = m.get("extra", {}).get("response", {}).get("usage", {})
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
            total_tokens += usage.get("total_tokens") or 0
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens += details.get("reasoning_tokens") or 0

            for a in m.get("extra", {}).get("actions", []):
                cmd = a.get("command", "").strip()
                if cmd:
                    if cmd in commands_seen:
                        repeated_commands += 1
                    commands_seen.append(cmd)
        elif m.get("role") in ("user", "tool"):
            content = m.get("content", "")
            if isinstance(content, str) and "[STRUCTURED TEST FEEDBACK]" in content:
                feedbacks_received_in_history += 1
                feedback_chars_total += len(content)
            # Count failed test executions
            if "returncode" in m and m.get("returncode") != 0:
                if any(t in str(m) for t in ["pytest", "test_"]):
                    failed_test_runs += 1

    # Run independent external test to verify actual resolution
    test_file = TASKS[task_key]["test_file"]
    test_proc = subprocess.run(
        [str(REPO_DIR / ".venv" / "bin" / "pytest"), test_file],
        cwd=SANDBOX_DIR,
        capture_output=True,
        text=True,
    )
    resolved = test_proc.returncode == 0

    ld_stats = info.get("loop_detection_stats", {})
    tf_stats = info.get("test_feedback_stats", {})
    model_stats = info.get("model_stats", {})

    return {
        "trajectory": traj_path.name,
        "submitted": info.get("exit_status") == "Submitted",
        "resolved": resolved,
        "total_steps": total_steps,
        "failed_test_runs": failed_test_runs,
        "repeated_commands": repeated_commands,
        "loop_detections": ld_stats.get("loop_detections", 0),
        "interventions": ld_stats.get("interventions", 0),
        "feedbacks_generated": tf_stats.get("feedbacks_generated", feedbacks_received_in_history),
        "feedbacks_received_in_history": feedbacks_received_in_history,
        "feedback_chars_total": feedback_chars_total,
        "api_calls": model_stats.get("api_calls", total_steps),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def compute_stats(runs: list[dict]) -> dict:
    n = len(runs)
    if n == 0:
        return {}

    def mean(vals):
        return sum(vals) / n

    def std(vals, m):
        if n <= 1:
            return 0.0
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))

    steps = [r["total_steps"] for r in runs]
    runtimes = [r["runtime_seconds"] for r in runs]
    t_toks = [r["total_tokens"] for r in runs]
    p_toks = [r["prompt_tokens"] for r in runs]
    c_toks = [r["completion_tokens"] for r in runs]
    r_toks = [r["reasoning_tokens"] for r in runs]
    failed_tests = [r["failed_test_runs"] for r in runs]
    rep_cmds = [r["repeated_commands"] for r in runs]

    m_steps = mean(steps)
    m_run = mean(runtimes)
    m_ttok = mean(t_toks)
    m_ptok = mean(p_toks)
    m_ctok = mean(c_toks)
    m_rtok = mean(r_toks)
    m_fail = mean(failed_tests)
    m_rep = mean(rep_cmds)

    return {
        "n_runs": n,
        "resolution_rate": sum(1 for r in runs if r["resolved"]) / n,
        "submission_rate": sum(1 for r in runs if r["submitted"]) / n,
        "steps_mean": round(m_steps, 2),
        "steps_std": round(std(steps, m_steps), 2),
        "failed_test_runs_mean": round(m_fail, 2),
        "repeated_commands_mean": round(m_rep, 2),
        "runtime_seconds_mean": round(m_run, 2),
        "runtime_seconds_std": round(std(runtimes, m_run), 2),
        "prompt_tokens_mean": round(m_ptok, 2),
        "completion_tokens_mean": round(m_ctok, 2),
        "reasoning_tokens_mean": round(m_rtok, 2),
        "total_tokens_mean": round(m_ttok, 2),
        "total_tokens_std": round(std(t_toks, m_ttok), 2),
        "loop_detections_total": sum(r["loop_detections"] for r in runs),
        "feedbacks_generated_total": sum(r["feedbacks_generated"] for r in runs),
    }


def main():
    try:
        from experiments.phase2_test_feedback.ablation_runner import main as runner_main
    except ImportError:
        from ablation_runner import main as runner_main
    runner_main()


if __name__ == "__main__":
    main()
