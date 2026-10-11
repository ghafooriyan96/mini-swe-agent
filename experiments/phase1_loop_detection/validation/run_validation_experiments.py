import json
import math
import os
import subprocess
import time
from pathlib import Path

SANDBOX_DIR = Path("/Users/ehsang/projects/mini-agent-sandbox")
REPO_DIR = Path("/Users/ehsang/projects/mini-swe-agent")
VAL_DIR = REPO_DIR / "experiments" / "phase1_loop_detection" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

TASK_PROMPT = (
    "We are stress testing service validation for service.py.\n"
    "1. First, verify the failure by running 'python test_service.py' 3 times consecutively without changing any files.\n"
    "2. If you see a repetitive action warning or after confirming the failure 3 times, fix service_config.py by setting ENABLED = True.\n"
    "3. Verify by running 'python test_service.py' once more.\n"
    "4. Finish by running 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
)


def reset_sandbox_service():
    """Reset sandbox to initial service stress test state."""
    service_config = SANDBOX_DIR / "service_config.py"
    test_service = SANDBOX_DIR / "test_service.py"

    service_config.write_text("ENABLED = False\n")
    test_service.write_text(
        "import sys\n"
        "from service_config import ENABLED\n\n"
        "if not ENABLED:\n"
        "    print('Service validation failed: ENABLED is False', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('Service validation successful: ENABLED is True')\n"
    )


def reset_sandbox_calculator():
    """Reset sandbox to initial calculator defect state."""
    (SANDBOX_DIR / "calculator.py").write_text("def add(a, b):\n    return a - b\n")
    (SANDBOX_DIR / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )


def restore_sandbox_clean():
    """Clean sandbox and leave calculator in clean state."""
    for fname in ["service_config.py", "test_service.py"]:
        p = SANDBOX_DIR / fname
        if p.exists():
            p.unlink()
    (SANDBOX_DIR / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (SANDBOX_DIR / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )


def run_agent_task(config_file: str, output_traj: Path, prompt: str, step_limit: int = 10) -> dict:
    cmd = [
        str(REPO_DIR / ".venv" / "bin" / "mini"),
        "-c",
        config_file,
        "-m",
        "openai/gemini-3.1-pro-high",
        "-c",
        f"environment.cwd={SANDBOX_DIR}",
        "-c",
        f"agent.step_limit={step_limit}",
        "-y",
        "--exit-immediately",
        "-o",
        str(output_traj),
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
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_trajectory(traj_path: Path) -> dict:
    data = json.loads(traj_path.read_text())
    messages = data.get("messages", [])
    info = data.get("info", {})

    total_steps = 0
    repeated_commands = 0
    loop_detections_in_traj = 0
    interventions_in_traj = 0
    false_positives = 0
    commands_seen = []

    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0

    assistant_step_idx = 0
    for m in messages:
        if m.get("role") == "assistant":
            assistant_step_idx += 1
            total_steps += 1
            usage = m.get("extra", {}).get("response", {}).get("usage", {})
            p_tok = usage.get("prompt_tokens") or 0
            c_tok = usage.get("completion_tokens") or 0
            t_tok = usage.get("total_tokens") or 0
            details = usage.get("completion_tokens_details") or {}
            r_tok = details.get("reasoning_tokens") or 0

            prompt_tokens += p_tok
            completion_tokens += c_tok
            reasoning_tokens += r_tok
            total_tokens += t_tok

            for action in m.get("extra", {}).get("actions", []):
                cmd = action.get("command", "").strip()
                if cmd:
                    if cmd in commands_seen:
                        repeated_commands += 1
                    commands_seen.append(cmd)
        elif m.get("role") in ("user", "tool"):
            extra = m.get("extra", {})
            content = m.get("content", "")
            is_warning = (
                extra.get("loop_detected") is True
                or (isinstance(content, str) and "[WARNING] Repeated action detected" in content)
            )
            if is_warning:
                interventions_in_traj += 1
                loop_detections_in_traj += 1
                # Check for false positive: did this fire prematurely before 3 repetitions?
                # In our task, the failure is repeated on steps 1, 2, 3. Detection should only happen at step 3.
                if assistant_step_idx < 3:
                    false_positives += 1

    stats = info.get("loop_detection_stats", {})
    model_stats = info.get("model_stats", {})

    return {
        "trajectory_file": traj_path.name,
        "exit_status": info.get("exit_status", "Unknown"),
        "success": info.get("exit_status") == "Submitted",
        "total_steps": total_steps,
        "repeated_commands": repeated_commands,
        "loop_detections": stats.get("loop_detections", loop_detections_in_traj),
        "interventions": stats.get("interventions", interventions_in_traj),
        "false_positives": false_positives,
        "api_calls": model_stats.get("api_calls", total_steps),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "token_formula_verified": total_tokens == (prompt_tokens + completion_tokens + reasoning_tokens),
    }


def compute_group_stats(runs: list[dict]) -> dict:
    n = len(runs)
    if n == 0:
        return {}

    def mean(values):
        return sum(values) / n

    def std(values, m):
        if n <= 1:
            return 0.0
        return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))

    steps = [r["total_steps"] for r in runs]
    runtimes = [r["runtime_seconds"] for r in runs]
    p_toks = [r["prompt_tokens"] for r in runs]
    c_toks = [r["completion_tokens"] for r in runs]
    r_toks = [r["reasoning_tokens"] for r in runs]
    t_toks = [r["total_tokens"] for r in runs]
    rep_cmds = [r["repeated_commands"] for r in runs]

    m_steps = mean(steps)
    m_run = mean(runtimes)
    m_ptok = mean(p_toks)
    m_ctok = mean(c_toks)
    m_rtok = mean(r_toks)
    m_ttok = mean(t_toks)
    m_rep = mean(rep_cmds)

    return {
        "n_runs": n,
        "success_rate": sum(1 for r in runs if r["success"]) / n,
        "steps_mean": round(m_steps, 2),
        "steps_std": round(std(steps, m_steps), 2),
        "repeated_commands_mean": round(m_rep, 2),
        "repeated_commands_std": round(std(rep_cmds, m_rep), 2),
        "loop_detections_total": sum(r["loop_detections"] for r in runs),
        "interventions_total": sum(r["interventions"] for r in runs),
        "false_positives_total": sum(r["false_positives"] for r in runs),
        "runtime_seconds_mean": round(m_run, 2),
        "runtime_seconds_std": round(std(runtimes, m_run), 2),
        "prompt_tokens_mean": round(m_ptok, 2),
        "prompt_tokens_std": round(std(p_toks, m_ptok), 2),
        "completion_tokens_mean": round(m_ctok, 2),
        "completion_tokens_std": round(std(c_toks, m_ctok), 2),
        "reasoning_tokens_mean": round(m_rtok, 2),
        "reasoning_tokens_std": round(std(r_toks, m_rtok), 2),
        "total_tokens_mean": round(m_ttok, 2),
        "total_tokens_std": round(std(t_toks, m_ttok), 2),
    }


def main():
    print("=== Starting Phase 1.1 Validation: 5 Paired Experiments ===")

    baseline_runs = []
    loop_detection_runs = []

    for run_i in range(1, 6):
        print(f"\n--- Pair {run_i} / 5 ---")

        # 1. Baseline
        reset_sandbox_service()
        base_traj = VAL_DIR / f"trajectory_baseline_run{run_i}.json"
        print(f"Running Baseline {run_i}...")
        res_b = run_agent_task("config/research/antigravity.yaml", base_traj, TASK_PROMPT)
        data_b = parse_trajectory(base_traj)
        data_b["runtime_seconds"] = round(res_b["runtime"], 2)
        baseline_runs.append(data_b)
        print(f"Baseline {run_i}: steps={data_b['total_steps']}, success={data_b['success']}, time={data_b['runtime_seconds']}s")

        # 2. Loop Detection
        reset_sandbox_service()
        mod_traj = VAL_DIR / f"trajectory_loop_detection_run{run_i}.json"
        print(f"Running Loop Detection {run_i}...")
        res_m = run_agent_task("config/research/antigravity_loop_detection.yaml", mod_traj, TASK_PROMPT)
        data_m = parse_trajectory(mod_traj)
        data_m["runtime_seconds"] = round(res_m["runtime"], 2)
        loop_detection_runs.append(data_m)
        print(f"Loop Detection {run_i}: steps={data_m['total_steps']}, detections={data_m['loop_detections']}, success={data_m['success']}, time={data_m['runtime_seconds']}s")

    print("\n--- Running Non-Loop Benchmark (Calculator Bugfix) with Loop Detection Enabled ---")
    reset_sandbox_calculator()
    calc_traj = VAL_DIR / "trajectory_calculator_with_loop_detection.json"
    calc_prompt = (
        "There is a bug in calculator.py causing test_calculator.py to fail. "
        "Please inspect the files, fix the bug in calculator.py, run pytest test_calculator.py to verify, and submit your work."
    )
    res_c = run_agent_task("config/research/antigravity_loop_detection.yaml", calc_traj, calc_prompt, step_limit=10)
    data_c = parse_trajectory(calc_traj)
    data_c["runtime_seconds"] = round(res_c["runtime"], 2)
    print(f"Calculator Benchmark: steps={data_c['total_steps']}, detections={data_c['loop_detections']}, interventions={data_c['interventions']}, success={data_c['success']}")

    restore_sandbox_clean()
    print("Sandbox restored to clean state.")

    # Compute Group Summary
    base_stats = compute_group_stats(baseline_runs)
    mod_stats = compute_group_stats(loop_detection_runs)

    full_results = {
        "description": "Phase 1.1 5-Pair Replication and Validation Results",
        "model": "openai/gemini-3.1-pro-high (local AntiGravity Manager)",
        "token_accounting_rule": "total_tokens = prompt_tokens + completion_tokens + reasoning_tokens",
        "summary": {
            "baseline": base_stats,
            "modified": mod_stats,
        },
        "runs": {
            "baseline": baseline_runs,
            "modified": loop_detection_runs,
            "non_loop_calculator_benchmark": data_c,
        },
    }

    result_json_path = VAL_DIR / "validation_results.json"
    result_json_path.write_text(json.dumps(full_results, indent=2))
    print(f"\nSaved structured validation results to: {result_json_path}")


if __name__ == "__main__":
    main()
