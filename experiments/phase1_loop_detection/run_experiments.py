import json
import os
import subprocess
import time
from pathlib import Path

SANDBOX_DIR = Path("/Users/ehsang/projects/mini-agent-sandbox")
REPO_DIR = Path("/Users/ehsang/projects/mini-swe-agent")
EXP_DIR = REPO_DIR / "experiments" / "phase1_loop_detection"
EXP_DIR.mkdir(parents=True, exist_ok=True)

TASK_PROMPT = (
    "We are stress testing service validation for service.py.\n"
    "1. First, verify the failure by running 'python test_service.py' 3 times consecutively without changing any files.\n"
    "2. If you see a repetitive action warning or after confirming the failure 3 times, fix service_config.py by setting ENABLED = True.\n"
    "3. Verify by running 'python test_service.py' once more.\n"
    "4. Finish by running 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
)


def reset_sandbox():
    """Reset sandbox to initial state."""
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
    print("Sandbox reset successfully.")


def run_experiment(config_file: str, output_traj: Path) -> dict:
    reset_sandbox()
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
        str(output_traj),
        "-t",
        TASK_PROMPT,
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


def analyze_trajectory(traj_path: Path) -> dict:
    if not traj_path.exists():
        return {"error": "Trajectory file not found"}

    data = json.loads(traj_path.read_text())
    messages = data.get("messages", [])
    info = data.get("info", {})

    total_steps = 0
    repeated_commands = 0
    loop_detections_in_traj = 0
    interventions_in_traj = 0
    commands_seen = []

    for msg in messages:
        if msg.get("role") == "assistant":
            total_steps += 1
            for action in msg.get("extra", {}).get("actions", []):
                cmd = action.get("command", "").strip()
                if cmd:
                    if cmd in commands_seen:
                        repeated_commands += 1
                    commands_seen.append(cmd)
        elif msg.get("role") in ("user", "tool"):
            content = msg.get("content", "")
            if isinstance(content, str) and "[WARNING] Repeated action detected" in content:
                interventions_in_traj += 1
                loop_detections_in_traj += 1

    stats = info.get("loop_detection_stats", {})
    model_stats = info.get("model_stats", {})

    # Check token usage
    tokens = model_stats.get("tokens")
    if tokens is None:
        tokens_display = "Not reported / unmetered local proxy"
    else:
        tokens_display = str(tokens)

    return {
        "total_agent_steps": total_steps,
        "repeated_commands": repeated_commands,
        "loop_detections": stats.get("loop_detections", loop_detections_in_traj),
        "interventions": stats.get("interventions", interventions_in_traj),
        "task_completion_status": info.get("exit_status", "Unknown"),
        "submission": info.get("submission", ""),
        "api_calls": model_stats.get("api_calls", total_steps),
        "token_usage": tokens_display,
    }


def main():
    print("=== Running Baseline Experiment (Loop Detection Disabled) ===")
    base_traj = EXP_DIR / "trajectory_baseline.json"
    base_run = run_experiment("config/research/antigravity.yaml", base_traj)
    base_analysis = analyze_trajectory(base_traj)
    base_analysis["runtime_seconds"] = round(base_run["runtime"], 2)
    print("Baseline Analysis:", json.dumps(base_analysis, indent=2))

    print("\n=== Running Modified Experiment (Loop Detection Enabled) ===")
    mod_traj = EXP_DIR / "trajectory_loop_detection.json"
    mod_run = run_experiment("config/research/antigravity_loop_detection.yaml", mod_traj)
    mod_analysis = analyze_trajectory(mod_traj)
    mod_analysis["runtime_seconds"] = round(mod_run["runtime"], 2)
    print("Modified Analysis:", json.dumps(mod_analysis, indent=2))

    comparison = {
        "model": "gemini-3.1-pro-high (local AntiGravity Manager)",
        "task": "Synthetic repetitive failure & resolution in sandbox",
        "baseline": base_analysis,
        "modified": mod_analysis,
    }

    result_path = EXP_DIR / "experiment_results.json"
    result_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved comparison to {result_path}")

    # Cleanup synthetic files in sandbox, keeping calculator.py
    for fname in ["service_config.py", "test_service.py"]:
        p = SANDBOX_DIR / fname
        if p.exists():
            p.unlink()
    print("Cleaned up sandbox synthetic files.")


if __name__ == "__main__":
    main()
