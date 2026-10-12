import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest

from experiments.phase2_test_feedback.ablation_runner import (
    CONFIGS,
    TASKS,
    AblationExperimentRunner,
    SandboxLock,
    atomic_write_json,
    check_test_failed,
    check_test_passed,
    compute_group_stats,
    extract_run_runtime_and_status,
    is_test_command,
    parse_tool_observation,
    reprocess_trajectories,
    run_command_with_timeout,
    validate_trajectory,
    verify_report_consistency,
)


def _make_sample_trajectory(
    exit_status: str = "Submitted",
    test_passed: bool = True,
    ld_enabled: bool = False,
    tf_enabled: bool = False,
    step_count: int = 2,
    raw_tool_json: bool = True,
) -> dict:
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Please fix the issue"},
    ]
    for i in range(step_count):
        call_id = f"call_{i + 1}"
        messages.append({
            "role": "assistant",
            "content": f"Step {i + 1}",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "pytest test_math_utils.py"}),
                    },
                }
            ],
            "extra": {
                "actions": [{"command": "pytest test_math_utils.py", "tool_call_id": call_id}],
                "response": {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "completion_tokens_details": {"reasoning_tokens": 5},
                    }
                },
            },
        })
        rc = 0 if test_passed else 1
        output_text = "test_math_utils.py . [100%]\n=== 1 passed in 0.05s ===" if test_passed else "FAILED test_math_utils.py\n=== 1 failed in 0.05s ==="
        if raw_tool_json:
            content_val = json.dumps({"returncode": rc, "output": output_text})
        else:
            content_val = {"returncode": rc, "output": output_text}

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content_val,
            "extra": {"returncode": rc, "raw_output": output_text},
        })

    return {
        "info": {
            "exit_status": exit_status,
            "config": {
                "agent": {
                    "loop_detection": {"enabled": ld_enabled},
                    "test_feedback": {"enabled": tf_enabled},
                }
            },
            "loop_detection_stats": {"loop_detections": 0, "interventions": 0},
            "test_feedback_stats": {"feedbacks_generated": 0},
            "model_stats": {"api_calls": step_count},
        },
        "messages": messages,
    }


def test_successful_child_process(tmp_path: Path):
    stdout_file = tmp_path / "stdout.log"
    stderr_file = tmp_path / "stderr.log"
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('out_ok'); sys.stderr.write('err_ok')",
    ]

    res = run_command_with_timeout(
        cmd=cmd,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5.0,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )

    assert res["status"] == "SUCCESS"
    assert res["returncode"] == 0
    assert res["timed_out"] is False
    assert res["stdout"] == "out_ok"
    assert res["stderr"] == "err_ok"
    assert stdout_file.read_text() == "out_ok"
    assert stderr_file.read_text() == "err_ok"
    assert res["duration_seconds"] >= 0.0


def test_timeout_child_process(tmp_path: Path):
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]

    res = run_command_with_timeout(
        cmd=cmd,
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=0.2,
    )

    assert res["status"] == "TIMEOUT"
    assert res["timed_out"] is True
    assert res["returncode"] != 0
    assert res["duration_seconds"] >= 0.15


def test_process_group_termination_and_sigkill(tmp_path: Path):
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(10)\n"
    )
    script_file = tmp_path / "stubborn.py"
    script_file.write_text(script)

    res = run_command_with_timeout(
        cmd=[sys.executable, str(script_file)],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=0.2,
    )

    assert res["status"] == "TIMEOUT"
    assert res["timed_out"] is True
    pid = res["pid"]
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_validate_trajectory_complete(tmp_path: Path):
    traj_path = tmp_path / "complete.json"
    traj_path.write_text(json.dumps(_make_sample_trajectory(exit_status="Submitted", test_passed=True)))

    # Without live independent verification: independent_resolved is UNKNOWN
    valid, parsed, reason = validate_trajectory(
        traj_path,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
        expected_rep=1,
    )

    assert valid is True
    assert reason == "Verified complete trajectory"
    assert parsed is not None
    assert parsed["exit_status"] == "Submitted"
    assert parsed["submitted"] is True
    assert parsed["observed_test_success"] is True
    assert parsed["independent_resolved"] == "UNKNOWN"
    assert parsed["total_steps"] == 2
    assert parsed["passed_test_runs"] == 2
    assert parsed["failed_test_runs"] == 0
    assert parsed["prompt_tokens"] == 200
    assert parsed["completion_tokens"] == 40
    assert parsed["reasoning_tokens"] == 10
    assert parsed["total_tokens"] == 240

    # With live independent verification = True
    _, parsed_verified, _ = validate_trajectory(
        traj_path,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
        expected_rep=1,
        independent_verified=True,
    )
    assert parsed_verified["independent_resolved"] is True
    assert parsed_verified["resolved"] is True


def test_validate_trajectory_incomplete(tmp_path: Path):
    traj_path = tmp_path / "incomplete.json"
    traj_path.write_text(json.dumps(_make_sample_trajectory(exit_status="")))

    valid, parsed, reason = validate_trajectory(
        traj_path,
        expected_task="task2_multi_failure",
        expected_config="A_baseline",
        expected_rep=1,
    )

    assert valid is False
    assert parsed is None
    assert reason == "Incomplete trajectory: empty exit_status"


def test_validate_trajectory_non_terminal_status(tmp_path: Path):
    traj_path = tmp_path / "running.json"
    traj_path.write_text(json.dumps(_make_sample_trajectory(exit_status="InProgress")))

    valid, parsed, reason = validate_trajectory(
        traj_path,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
        expected_rep=1,
    )

    assert valid is False
    assert parsed is None
    assert "Non-terminal exit status" in reason


def test_validate_trajectory_invalid_json(tmp_path: Path):
    broken_file = tmp_path / "broken.json"
    broken_file.write_text("{broken json")

    valid, parsed, reason = validate_trajectory(
        broken_file,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
    )

    assert valid is False
    assert parsed is None
    assert "Invalid JSON" in reason

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")

    valid_empty, parsed_empty, reason_empty = validate_trajectory(
        empty_file,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
    )
    assert valid_empty is False
    assert reason_empty == "File is empty"


def test_validate_trajectory_missing_file(tmp_path: Path):
    missing_file = tmp_path / "nonexistent.json"

    valid, parsed, reason = validate_trajectory(
        missing_file,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
    )

    assert valid is False
    assert parsed is None
    assert reason == "File does not exist"


def test_validate_trajectory_config_mismatch(tmp_path: Path):
    traj_path = tmp_path / "mismatch.json"
    traj_path.write_text(json.dumps(_make_sample_trajectory(ld_enabled=True, tf_enabled=False)))

    valid, parsed, reason = validate_trajectory(
        traj_path,
        expected_task="task1_simple_assertion",
        expected_config="A_baseline",
    )

    assert valid is False
    assert parsed is None
    assert "Config mismatch" in reason


def test_atomic_write_preserves_on_interrupt(tmp_path: Path):
    target = tmp_path / "nested" / "summary.json"
    atomic_write_json(target, {"initial": 1})
    assert json.loads(target.read_text()) == {"initial": 1}

    atomic_write_json(target, {"updated": 2})
    assert json.loads(target.read_text()) == {"updated": 2}
    assert list(target.parent.glob("*.tmp.*")) == []


def test_correct_run_counting():
    assert len(TASKS) * len(CONFIGS) * 5 == 60


def test_resume_complete_and_rerun_incomplete(tmp_path: Path):
    resume_source = tmp_path / "resume_source"
    resume_source.mkdir()
    out_dir = tmp_path / "run_out"
    sandbox = tmp_path / "mock_sandbox"
    sandbox.mkdir()

    rep1_file = resume_source / "traj_task1_simple_assertion_A_baseline_rep1.json"
    rep1_file.write_text(json.dumps(_make_sample_trajectory(exit_status="Submitted", test_passed=True)))

    rep2_file = resume_source / "traj_task1_simple_assertion_A_baseline_rep2.json"
    rep2_file.write_text(json.dumps(_make_sample_trajectory(exit_status="")))

    runner = AblationExperimentRunner(
        output_dir=out_dir,
        sandbox_dir=sandbox,
        resume=True,
        resume_source=resume_source,
        repetitions=2,
        dry_run=True,
    )

    summary = runner.execute_suite()

    assert len(runner.manifest) == 24
    manifest_by_key = {
        (m["task"], m["config"], m["rep"]): m
        for m in runner.manifest
    }

    rep1_manifest = manifest_by_key[("task1_simple_assertion", "A_baseline", 1)]
    assert rep1_manifest["status"] == "RESUMED_COMPLETE"
    assert rep1_manifest["resumed"] is True
    assert rep1_manifest["observed_test_success"] is True
    assert rep1_manifest["independent_test_resolved"] == "UNKNOWN"
    assert (out_dir / "trajectories" / "traj_task1_simple_assertion_A_baseline_rep1.json").exists()

    rep2_manifest = manifest_by_key[("task1_simple_assertion", "A_baseline", 2)]
    assert rep2_manifest["status"] == "DRY_RUN"
    assert rep2_manifest["resumed"] is False

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "ablation_results.json").exists()
    assert summary["resumed"] is True


def test_sandbox_lock(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with SandboxLock(sandbox) as lock:
        assert lock.lock_file.exists()
        assert int(lock.lock_file.read_text().strip()) == os.getpid()

        with pytest.raises(RuntimeError, match="Sandbox is locked"):
            SandboxLock(sandbox).acquire()

    assert not lock.lock_file.exists()


def test_compute_group_stats_empty_and_populated():
    assert compute_group_stats([]) == {}

    stats = compute_group_stats([
        {"resolved": True, "submitted": True, "observed_test_success": True, "independent_resolved": True, "failed_test_runs": 1, "passed_test_runs": 2, "total_steps": 4, "runtime_seconds": 10.0, "total_tokens": 100},
        {"resolved": False, "submitted": True, "observed_test_success": False, "independent_resolved": False, "failed_test_runs": 2, "passed_test_runs": 0, "total_steps": 6, "runtime_seconds": 20.0, "total_tokens": 200},
    ])
    assert stats["n_runs"] == 2
    assert stats["resolution_rate"] == 0.5
    assert stats["submission_rate"] == 1.0
    assert stats["observed_test_success_rate"] == 0.5
    assert stats["independent_resolution_rate"] == 0.5
    assert stats["failed_test_runs_mean"] == 1.5
    assert stats["passed_test_runs_mean"] == 1.0
    assert stats["steps_mean"] == 5.0
    assert stats["runtime_seconds_mean"] == 15.0
    assert stats["total_tokens_mean"] == 150.0


# --- Phase 2 Specific Trajectory Parsing Regression Tests ---

def test_json_encoded_successful_pytest_output():
    # Real observation format from Phase 2 trajectories
    content = json.dumps({
        "returncode": 0,
        "output": "============================= test session starts ==============================\ntest_math_utils.py . [100%]\n============================== 1 passed in 0.02s ===============================\n"
    })
    msg = {"role": "tool", "tool_call_id": "call_1", "content": content}
    rc, out = parse_tool_observation(msg)

    assert rc == 0
    assert "1 passed in 0.02s" in out
    assert is_test_command("pytest test_math_utils.py") is True
    assert check_test_passed(rc, out) is True
    assert check_test_failed(rc, out) is False


def test_json_encoded_failing_pytest_output():
    content = json.dumps({
        "returncode": 1,
        "output": "FAILED test_math_utils.py::test_power - assert 6 == 8\n============================== 1 failed in 0.02s ===============================\n"
    })
    msg = {"role": "tool", "tool_call_id": "call_1", "content": content}
    rc, out = parse_tool_observation(msg)

    assert rc == 1
    assert "1 failed in 0.02s" in out
    assert is_test_command("pytest test_math_utils.py") is True
    assert check_test_passed(rc, out) is False
    assert check_test_failed(rc, out) is True


def test_non_test_command_returning_zero():
    # Non-test command returning 0 must not count as passed test
    content = json.dumps({"returncode": 0, "output": "math_utils.py  test_math_utils.py\n"})
    msg = {"role": "tool", "tool_call_id": "call_1", "content": content}
    rc, out = parse_tool_observation(msg)

    assert rc == 0
    assert is_test_command("ls -la") is False
    assert is_test_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT") is False
    assert is_test_command("cat test_user_validator.py") is False
    assert is_test_command("grep -n 'pytest' file.py") is False


def test_malformed_observation_content():
    # Graceful handling of invalid JSON with fallback to extra
    msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "{\"returncode\": 0, unquoted_broken_json",
        "extra": {"returncode": 0, "raw_output": "=== 1 passed in 0.01s ==="},
    }
    rc, out = parse_tool_observation(msg)

    assert rc == 0
    assert "1 passed in 0.01s" in out
    assert check_test_passed(rc, out) is True


def test_submitted_trajectory_without_passing_tests(tmp_path: Path):
    # Agent failed test, edited file, ran cat, then submitted without re-running test
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task prompt"},
        {
            "role": "assistant",
            "content": "Run test",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}],
            "extra": {"actions": [{"command": "pytest", "tool_call_id": "c1"}]},
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": json.dumps({"returncode": 1, "output": "=== 1 failed in 0.01s ==="}),
        },
        {
            "role": "assistant",
            "content": "Check file",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "cat math_utils.py"}'}}],
            "extra": {"actions": [{"command": "cat math_utils.py", "tool_call_id": "c2"}]},
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": json.dumps({"returncode": 0, "output": "def power(a, b): return a**b"}),
        },
    ]
    traj = {
        "info": {"exit_status": "Submitted", "config": {}},
        "messages": messages,
    }
    traj_file = tmp_path / "submitted_no_pass.json"
    traj_file.write_text(json.dumps(traj))

    valid, parsed, _ = validate_trajectory(traj_file, "task1_simple_assertion", "A_baseline")
    assert valid is True
    assert parsed["submitted"] is True
    assert parsed["observed_test_success"] is False
    assert parsed["independent_resolved"] == "UNKNOWN"
    assert parsed["failed_test_runs"] == 1
    assert parsed["passed_test_runs"] == 0


def test_multiple_test_executions(tmp_path: Path):
    # 2 failures followed by 1 pass
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task prompt"},
        {
            "role": "assistant",
            "content": "Test 1",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}],
            "extra": {"actions": [{"command": "pytest", "tool_call_id": "c1"}]},
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": json.dumps({"returncode": 1, "output": "=== 1 failed in 0.01s ==="}),
        },
        {
            "role": "assistant",
            "content": "Test 2",
            "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}],
            "extra": {"actions": [{"command": "pytest", "tool_call_id": "c2"}]},
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": json.dumps({"returncode": 1, "output": "=== 1 failed in 0.01s ==="}),
        },
        {
            "role": "assistant",
            "content": "Test 3",
            "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}],
            "extra": {"actions": [{"command": "pytest", "tool_call_id": "c3"}]},
        },
        {
            "role": "tool",
            "tool_call_id": "c3",
            "content": json.dumps({"returncode": 0, "output": "=== 1 passed in 0.01s ==="}),
        },
    ]
    traj = {
        "info": {"exit_status": "Submitted", "config": {}},
        "messages": messages,
    }
    traj_file = tmp_path / "multi_tests.json"
    traj_file.write_text(json.dumps(traj))

    valid, parsed, _ = validate_trajectory(traj_file, "task1_simple_assertion", "A_baseline")
    assert valid is True
    assert parsed["failed_test_runs"] == 2
    assert parsed["passed_test_runs"] == 1
    assert parsed["observed_test_success"] is True


def test_missing_tool_return_codes():
    # Tool output has no returncode, but failure is detected via output pattern
    msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"output": "FAILED test_math_utils.py\n=== 1 failed in 0.01s ==="}),
    }
    rc, out = parse_tool_observation(msg)

    assert rc is None
    assert check_test_failed(rc, out) is True
    assert check_test_passed(rc, out) is False


def test_distinction_between_observed_and_independent_resolution(tmp_path: Path):
    traj_file = tmp_path / "distinction.json"
    traj_file.write_text(json.dumps(_make_sample_trajectory(exit_status="Submitted", test_passed=True)))

    # Resumed / Historical run: workspace state unknown -> independent_resolved is UNKNOWN
    _, hist_parsed, _ = validate_trajectory(traj_file, "task1_simple_assertion", "A_baseline", independent_verified=None)
    assert hist_parsed["observed_test_success"] is True
    assert hist_parsed["independent_resolved"] == "UNKNOWN"

    # Live run: independently failed on disk
    _, live_fail, _ = validate_trajectory(traj_file, "task1_simple_assertion", "A_baseline", independent_verified=False)
    assert live_fail["observed_test_success"] is True
    assert live_fail["independent_resolved"] is False

    # Live run: independently passed on disk
    _, live_pass, _ = validate_trajectory(traj_file, "task1_simple_assertion", "A_baseline", independent_verified=True)
    assert live_pass["observed_test_success"] is True
    assert live_pass["independent_resolved"] is True


def test_runtime_recovery_from_manifest_and_timestamps(tmp_path: Path):
    # Case 1: Manifest with wall_clock_duration_seconds
    traj_file = tmp_path / "timed.json"
    sample = _make_sample_trajectory()
    # Inject timestamps into messages
    sample["messages"][0].setdefault("extra", {})["timestamp"] = 1000.0
    sample["messages"][-1].setdefault("extra", {})["timestamp"] = 1025.5
    traj_file.write_text(json.dumps(sample))

    manifest_item = {
        "wall_clock_duration_seconds": 32.14,
        "status": "SUCCESS",
        "process_returncode": 0,
        "timed_out": False,
        "start_time": "2026-10-11T12:00:00Z",
        "end_time": "2026-10-11T12:00:32Z",
    }
    dur, st, et, proc_comp, to_val = extract_run_runtime_and_status(traj_file, None, manifest_item)
    assert dur == 32.14
    assert proc_comp is True
    assert to_val is False
    assert st == "2026-10-11T12:00:00Z"

    # Case 2: No manifest entry, recovers from message timestamps
    dur_fb, st_fb, et_fb, proc_fb, to_fb = extract_run_runtime_and_status(traj_file, None, None)
    assert dur_fb == 25.5
    assert to_fb is False


def test_unknown_excluded_from_denominator():
    runs = [
        {"submitted": True, "observed_test_success": True, "independent_resolved": True},
        {"submitted": True, "observed_test_success": True, "independent_resolved": True},
        {"submitted": False, "observed_test_success": False, "independent_resolved": False},
        {"submitted": True, "observed_test_success": True, "independent_resolved": "UNKNOWN"},
        {"submitted": True, "observed_test_success": True, "independent_resolved": "UNKNOWN"},
    ]
    stats = compute_group_stats(runs)
    assert stats["n_runs"] == 5
    assert stats["indep_resolved_count"] == 2
    assert stats["indep_failed_count"] == 1
    assert stats["indep_unknown_count"] == 2
    assert stats["verified_known_count"] == 3
    # Denominator must be verified_known_count (3), not total runs (5)
    assert stats["independent_resolution_rate"] == round(2 / 3, 3)

    # All unknown
    all_unk = [{"independent_resolved": "UNKNOWN"} for _ in range(3)]
    unk_stats = compute_group_stats(all_unk)
    assert unk_stats["verified_known_count"] == 0
    assert unk_stats["independent_resolution_rate"] == "UNKNOWN"


def test_explicit_status_distinctions(tmp_path: Path):
    traj_file = tmp_path / "distinctions.json"
    traj_file.write_text(json.dumps(_make_sample_trajectory(exit_status="LimitsExceeded", test_passed=True)))

    _, parsed, _ = validate_trajectory(
        traj_path=traj_file,
        expected_task="task3_input_validation",
        expected_config="A_baseline",
        independent_verified=True,
        runtime_seconds=45.0,
        timed_out=False,
        process_completed=True,
    )

    # All five dimensions are distinct
    assert parsed["process_completed"] is True
    assert parsed["submitted"] is False
    assert parsed["observed_test_success"] is True
    assert parsed["independent_resolved"] is True
    assert parsed["timed_out"] is False
    assert parsed["runtime_seconds"] == 45.0


def test_consistency_assertions_valid_and_invalid():
    # Valid report
    runs = [
        {
            "task": "task1_simple_assertion",
            "config": "A_baseline",
            "rep": 1,
            "process_completed": True,
            "submitted": True,
            "observed_test_success": True,
            "independent_resolved": "UNKNOWN",
            "timed_out": False,
            "failed_test_runs": 1,
            "passed_test_runs": 1,
            "runtime_seconds": 10.0,
            "trajectory": "t1.json",
        },
        {
            "task": "task2_multi_failure",
            "config": "A_baseline",
            "rep": 1,
            "process_completed": True,
            "submitted": False,
            "observed_test_success": False,
            "independent_resolved": False,
            "timed_out": False,
            "failed_test_runs": 1,
            "passed_test_runs": 0,
            "runtime_seconds": 20.0,
            "trajectory": "t2.json",
        },
    ]

    summary = {
        "total_runs": 2,
        "total_process_completed": 2,
        "total_submitted": 1,
        "total_observed_test_success": 1,
        "total_independent_resolved_true": 0,
        "total_independent_resolved_false": 1,
        "total_independent_resolved_unknown": 1,
        "total_independent_verified_known": 1,
        "total_timed_out": 0,
        "total_failed_test_runs": 2,
        "total_passed_test_runs": 1,
        "total_runtime_seconds": 30.0,
    }

    t1_stats = compute_group_stats([runs[0]])
    t2_stats = compute_group_stats([runs[1]])
    cfg_stats = compute_group_stats(runs)

    valid_report = {
        "summary": summary,
        "detailed_runs": runs,
        "summary_by_task": {"task1_simple_assertion": t1_stats, "task2_multi_failure": t2_stats},
        "summary_by_config": {"A_baseline": {"overall": cfg_stats}},
        "data_quality_limitations": ["overall rate of verified runs is 0/1 = 0.0%"],
    }

    errors = verify_report_consistency(valid_report)
    assert errors == []

    # Invalid report: broken sum
    broken_report = dict(valid_report)
    broken_summary = dict(summary)
    broken_summary["total_independent_resolved_true"] = 5  # sum becomes 7 != 2
    broken_report["summary"] = broken_summary
    broken_errors = verify_report_consistency(broken_report)
    assert any("Independent resolved sum" in e for e in broken_errors)


def test_reprocess_existing_phase2_report_consistency():
    run_dir = Path("experiments/phase2_test_feedback/runs/run_20261011_183454")
    report = reprocess_trajectories(run_dir)

    assert report["consistency_check"]["status"] == "PASSED"
    sum_data = report["summary"]

    assert sum_data["total_runs"] == 60
    assert sum_data["total_process_completed"] == 58
    assert sum_data["total_timed_out"] == 2
    assert sum_data["total_submitted"] == 28
    assert sum_data["total_observed_test_success"] == 42
    assert sum_data["total_independent_resolved_true"] == 23
    assert sum_data["total_independent_resolved_false"] == 17
    assert sum_data["total_independent_resolved_unknown"] == 20
    assert sum_data["total_independent_verified_known"] == 40
    # Verified denominator: 23 / 40 = 0.575
    assert sum_data["independent_resolution_rate_verified"] == 0.575
    assert sum_data["total_failed_test_runs"] == 60
    assert sum_data["total_passed_test_runs"] == 77
    assert sum_data["total_runtime_seconds"] > 0.0


def test_default_full_matrix_behavior(tmp_path: Path):
    runner = AblationExperimentRunner(output_dir=tmp_path, dry_run=True, repetitions=1)
    assert len(runner.active_tasks) == len(TASKS)
    assert len(runner.active_configs) == len(CONFIGS)
    summary = runner.execute_suite()
    assert len(runner.manifest) == len(TASKS) * len(CONFIGS)
    assert summary["total_runs"] == len(TASKS) * len(CONFIGS)
    assert set(summary["summary_by_config"].keys()) == set(CONFIGS.keys())


def test_task_only_filtering(tmp_path: Path):
    runner = AblationExperimentRunner(
        task_filter="task2_multi_failure",
        output_dir=tmp_path,
        dry_run=True,
        repetitions=1,
    )
    assert len(runner.active_tasks) == 1
    assert "task2_multi_failure" in runner.active_tasks
    assert len(runner.active_configs) == len(CONFIGS)
    summary = runner.execute_suite()
    assert len(runner.manifest) == 4
    assert all(m["task"] == "task2_multi_failure" for m in runner.manifest)
    assert summary["total_runs"] == 4


def test_config_only_filtering(tmp_path: Path):
    runner = AblationExperimentRunner(
        config_filter="C_test_feedback_only",
        output_dir=tmp_path,
        dry_run=True,
        repetitions=1,
    )
    assert len(runner.active_tasks) == len(TASKS)
    assert len(runner.active_configs) == 1
    assert "C_test_feedback_only" in runner.active_configs
    summary = runner.execute_suite()
    assert len(runner.manifest) == 3
    assert all(m["config"] == "C_test_feedback_only" for m in runner.manifest)
    assert summary["total_runs"] == 3


def test_combined_filtering_and_single_run_counting(tmp_path: Path, capsys):
    runner = AblationExperimentRunner(
        task_filter="task1_simple_assertion",
        config_filter="D_combined",
        repetitions=1,
        output_dir=tmp_path,
        dry_run=True,
    )
    assert len(runner.active_tasks) == 1
    assert len(runner.active_configs) == 1
    summary = runner.execute_suite()
    assert len(runner.manifest) == 1
    assert runner.manifest[0]["task"] == "task1_simple_assertion"
    assert runner.manifest[0]["config"] == "D_combined"
    assert summary["total_runs"] == 1
    assert summary["task_filter"] == "task1_simple_assertion"
    assert summary["config_filter"] == "D_combined"

    captured = capsys.readouterr().out
    assert "SINGLE-RUN EXECUTION SUMMARY" in captured
    assert "Agent Exit Status:" in captured
    assert "Independent Test Result:" in captured
    assert "Snapshot Location:" in captured
    assert "Patch Location:" in captured
    assert "SHA-256 Integrity:" in captured


def test_filtered_resume_behavior(tmp_path: Path):
    resume_dir = tmp_path / "prior_traj"
    resume_dir.mkdir()
    sample_traj = _make_sample_trajectory(
        exit_status="Submitted",
        test_passed=True,
        ld_enabled=True,
        tf_enabled=True,
    )
    traj_file = resume_dir / "traj_task1_simple_assertion_D_combined_rep1.json"
    traj_file.write_text(json.dumps(sample_traj))

    runner = AblationExperimentRunner(
        task_filter="task1_simple_assertion",
        config_filter="D_combined",
        repetitions=1,
        resume=True,
        resume_source=resume_dir,
        output_dir=tmp_path / "new_run",
        dry_run=True,
    )
    runner.execute_suite()
    assert len(runner.manifest) == 1
    assert runner.manifest[0]["status"] == "RESUMED_COMPLETE"
    assert runner.manifest[0]["task"] == "task1_simple_assertion"
    assert runner.manifest[0]["config"] == "D_combined"


def test_invalid_task_and_config_values(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid task filter"):
        AblationExperimentRunner(output_dir=tmp_path, task_filter="invalid_task")

    with pytest.raises(ValueError, match="Invalid config filter"):
        AblationExperimentRunner(output_dir=tmp_path, config_filter="invalid_config")


def test_cli_replay_invalid_path_exits_status_1_without_traceback():
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "experiments/phase2_test_feedback/ablation_runner.py",
            "--replay",
            "nonexistent.snapshot.json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "NameError" not in proc.stderr
    assert "Snapshot file not found" in proc.stdout


def test_cli_replay_valid_snapshot_exits_status_0(tmp_path: Path):
    from experiments.phase2_test_feedback.workspace_preserver import WorkspacePreserver

    ws = tmp_path / "sandbox"
    ws.mkdir()
    (ws / "math_mod.py").write_text("def add(a, b):\n    return a - b\n")
    (ws / "test_math.py").write_text("from math_mod import add\ndef test_add():\n    assert add(2, 3) == 5\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    (ws / "math_mod.py").write_text("def add(a, b):\n    return a + b\n")
    snapshot = preserver.capture_snapshot(ws, baseline)
    _, snapshot_file = preserver.save_snapshot_and_patch(snapshot, tmp_path / "exp_out", "run_valid")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "experiments/phase2_test_feedback/ablation_runner.py",
            "--replay",
            str(snapshot_file),
            "--replay-test-file",
            "test_math.py",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
    assert "PASS (0)" in proc.stdout


