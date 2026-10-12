"""Unit and regression tests for Phase 3 Experimental Evaluation Infrastructure.

Covers all 18 mandatory requirements:
1. Pristine task fails correctly
2. Reference fix passes
3. Already-passing task rejection
4. Missing dependency classification
5. Nondeterministic task rejection
6. Preflight blocks LLM invocation
7. Exact task/config filtering
8. Paired run counting
9. Randomized or alternating execution order
10. Resume behavior
11. Timeout handling
12. Independent patch verification
13. Snapshot and replay compatibility
14. Provider token extraction
15. Missing cache data stays UNKNOWN
16. ACM metrics correctness
17. Paired summary aggregation
18. Backward compatibility with existing Phase 2 and 3.1 experiments
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from experiments.phase2_test_feedback.ablation_runner import (
    ALL_CONFIGS,
    ALL_TASKS,
    CONFIGS,
    TASKS,
    WorkspacePreserver,
    extract_acm_metrics,
    extract_trajectory_test_metrics,
    run_command_with_timeout,
)
from experiments.phase3_adaptive_context.benchmark_tasks import (
    PHASE3_BENCHMARK_TASKS,
    reference_fix_task_e1,
    reference_fix_task_e2,
    reference_fix_task_e3,
    setup_task_e1,
    setup_task_e2,
    setup_task_e3,
)
from experiments.phase3_adaptive_context.evaluation_runner import (
    Phase3EvaluationRunner,
    build_paired_schedule,
    compute_median,
    compute_std,
    generate_markdown_phase3_report,
    generate_pilot_disclaimer,
    summarize_paired_evaluation,
)
from experiments.phase3_adaptive_context.preflight import (
    TaskPreflightStatus,
    run_all_preflights,
    validate_single_task,
)

PYTEST_BIN = Path(sys.executable).parent / "pytest"


# 1. Pristine task fails correctly
@pytest.mark.parametrize(
    ("task_id", "setup_fn", "test_file"),
    [
        ("task_e1_multifile", setup_task_e1, "test_order_pipeline.py"),
        ("task_e2_multifailure", setup_task_e2, "test_report_pipeline.py"),
        ("task_e3_verbose_investigation", setup_task_e3, "test_rolling_window.py"),
    ],
)
def test_pristine_tasks_fail_correctly(tmp_path: Path, task_id: str, setup_fn, test_file: str):
    setup_fn(tmp_path)
    res = subprocess.run([str(PYTEST_BIN), test_file], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode != 0
    assert "FAILED" in res.stdout or "FAILURES" in res.stdout or "Error" in res.stdout


# 2. Reference fix passes
@pytest.mark.parametrize(
    ("task_id", "setup_fn", "ref_fix_fn", "test_file"),
    [
        ("task_e1_multifile", setup_task_e1, reference_fix_task_e1, "test_order_pipeline.py"),
        ("task_e2_multifailure", setup_task_e2, reference_fix_task_e2, "test_report_pipeline.py"),
        ("task_e3_verbose_investigation", setup_task_e3, reference_fix_task_e3, "test_rolling_window.py"),
    ],
)
def test_reference_fixes_pass_correctly(tmp_path: Path, task_id: str, setup_fn, ref_fix_fn, test_file: str):
    setup_fn(tmp_path)
    ref_fix_fn(tmp_path)
    res = subprocess.run([str(PYTEST_BIN), test_file], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0
    assert "passed" in res.stdout


# 3. Already-passing task rejection
def test_already_passing_task_rejection(tmp_path: Path):
    dummy_task = {
        "setup_fn": lambda p: (p / "test_dummy.py").write_text("def test_ok(): assert True\n"),
        "test_file": "test_dummy.py",
        "reference_fix_fn": lambda p: None,
    }
    res = validate_single_task("dummy_passing", dummy_task, pytest_bin=PYTEST_BIN, temp_base=tmp_path)
    assert res["status"] == TaskPreflightStatus.INVALID_TASK_ALREADY_PASSING.value


# 4. Missing dependency or test execution error classification
def test_missing_dependency_classification(tmp_path: Path):
    dummy_task = {
        "setup_fn": lambda p: (p / "test_nonexistent.py").write_text("import non_existent_pkg_xyz_123\n"),
        "test_file": "test_nonexistent.py",
        "reference_fix_fn": lambda p: None,
    }
    # Pass an invalid pytest executable or non-zero usage error
    dummy_pytest = tmp_path / "fake_pytest"
    dummy_pytest.write_text("#!/bin/sh\nexit 4\n")
    dummy_pytest.chmod(0o755)

    res = validate_single_task("dummy_error", dummy_task, pytest_bin=dummy_pytest, temp_base=tmp_path)
    assert res["status"] == TaskPreflightStatus.INVALID_TASK_TEST_ERROR.value


# 5. Nondeterministic task rejection
def test_nondeterministic_task_rejection(tmp_path: Path):
    state_file = tmp_path / "counter.txt"
    state_file.write_text("0")

    def nondet_setup(p: Path):
        code = (
            f"from pathlib import Path\n"
            f"def test_flaky():\n"
            f"    p = Path('{state_file}')\n"
            f"    count = int(p.read_text())\n"
            f"    p.write_text(str(count + 1))\n"
            f"    if count == 0: assert False\n"
            f"    else: assert True\n"
        )
        (p / "test_flaky.py").write_text(code)

    dummy_task = {
        "setup_fn": nondet_setup,
        "test_file": "test_flaky.py",
        "reference_fix_fn": lambda p: None,
    }

    res = validate_single_task("dummy_nondet", dummy_task, pytest_bin=PYTEST_BIN, temp_base=tmp_path)
    assert res["status"] == TaskPreflightStatus.INVALID_TASK_NONDETERMINISTIC.value


# 6. Preflight blocks LLM invocation
def test_preflight_blocks_llm_invocation(tmp_path: Path):
    runner = Phase3EvaluationRunner(
        output_dir=tmp_path / "run_blocked",
        task_filter="task_e1_multifile",
    )
    # Sabotage preflight result
    runner.active_tasks["task_e1_multifile"] = {
        "setup_fn": lambda p: (p / "test_order_pipeline.py").write_text("def test_p(): assert True\n"),
        "test_file": "test_order_pipeline.py",
        "reference_fix_fn": lambda p: None,
    }

    with pytest.raises(RuntimeError, match="Task preflight validation failed"):
        runner.execute_suite()


# 7. Exact task/config filtering
def test_exact_task_config_filtering():
    runner = Phase3EvaluationRunner(
        task_filter="task_e1_multifile",
        config_filter="G_context_baseline",
    )
    assert list(runner.active_tasks.keys()) == ["task_e1_multifile"]
    assert list(runner.active_configs.keys()) == ["G_context_baseline"]

    with pytest.raises(ValueError, match="Invalid task filter"):
        Phase3EvaluationRunner(task_filter="invalid_task_name")

    with pytest.raises(ValueError, match="Invalid config filter"):
        Phase3EvaluationRunner(config_filter="invalid_config_name")


# 8. Paired run counting
def test_paired_run_counting():
    # 3 tasks x 2 configs x 2 reps = 12 runs
    sched12 = build_paired_schedule(
        tasks=["task_e1_multifile", "task_e2_multifailure", "task_e3_verbose_investigation"],
        configs=["G_context_baseline", "H_context_compaction"],
        repetitions=2,
    )
    assert len(sched12) == 12

    # 1 task x 2 configs x 1 rep = 2 runs (1 pair)
    sched2 = build_paired_schedule(
        tasks=["task_e1_multifile"],
        configs=["G_context_baseline", "H_context_compaction"],
        repetitions=1,
    )
    assert len(sched2) == 2


# 9. Randomized or alternating execution order
def test_randomized_or_alternating_execution_order():
    sched = build_paired_schedule(
        tasks=["task_e1_multifile"],
        configs=["G_context_baseline", "H_context_compaction"],
        repetitions=2,
        order_strategy="alternating",
    )
    # Rep 1: G then H
    assert sched[0]["config"] == "G_context_baseline"
    assert sched[1]["config"] == "H_context_compaction"
    # Rep 2: H then G (alternated)
    assert sched[2]["config"] == "H_context_compaction"
    assert sched[3]["config"] == "G_context_baseline"

    # Test randomized
    sched_rand = build_paired_schedule(
        tasks=["task_e1_multifile"],
        configs=["G_context_baseline", "H_context_compaction"],
        repetitions=10,
        order_strategy="randomized",
        seed=123,
    )
    assert len(sched_rand) == 20
    first_configs = [sched_rand[i]["config"] for i in range(0, 20, 2)]
    assert "G_context_baseline" in first_configs
    assert "H_context_compaction" in first_configs


# 10. Resume behavior
def test_resume_behavior(tmp_path: Path):
    source_dir = tmp_path / "prior_run"
    traj_dir = source_dir / "trajectories"
    traj_dir.mkdir(parents=True)

    fake_traj = {
        "messages": [{"role": "system", "content": "sys"}],
        "info": {
            "exit_status": "Submitted",
            "model_stats": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "config": {
                "agent": {
                    "loop_detection": {"enabled": True},
                    "test_feedback": {"enabled": True},
                    "adaptive_context": {"enabled": False},
                }
            },
        },
    }
    traj_file = traj_dir / "traj_task_e1_multifile_G_context_baseline_rep1.json"
    traj_file.write_text(json.dumps(fake_traj))

    runner = Phase3EvaluationRunner(
        output_dir=tmp_path / "new_run",
        task_filter="task_e1_multifile",
        config_filter="G_context_baseline",
        repetitions=1,
        resume=True,
        resume_source=source_dir,
        dry_run=True,
    )
    report = runner.execute_suite()
    assert any(m.get("config") == "G_context_baseline" for m in runner.manifest)


# 11. Timeout handling
def test_timeout_handling():
    res = run_command_with_timeout(
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=Path.cwd(),
        env={},
        timeout_seconds=0.5,
    )
    assert res["timed_out"] is True
    assert res["status"] == "TIMEOUT"
    assert res["returncode"] in (-1, -15)


# 12. Independent patch verification
def test_independent_patch_verification(tmp_path: Path):
    setup_task_e1(tmp_path)
    res_before = subprocess.run([str(PYTEST_BIN), "test_order_pipeline.py"], cwd=tmp_path, capture_output=True)
    assert res_before.returncode != 0

    reference_fix_task_e1(tmp_path)
    res_after = subprocess.run([str(PYTEST_BIN), "test_order_pipeline.py"], cwd=tmp_path, capture_output=True)
    assert res_after.returncode == 0


# 13. Snapshot and replay compatibility
def test_snapshot_and_replay_compatibility(tmp_path: Path):
    preserver = WorkspacePreserver()
    setup_task_e1(tmp_path)
    baseline = preserver.record_baseline(tmp_path)

    # Apply reference fix
    reference_fix_task_e1(tmp_path)
    snapshot = preserver.capture_snapshot(tmp_path, baseline, metadata={"task": "task_e1_multifile"})

    snap_file = tmp_path / "snap.json"
    snap_file.write_text(json.dumps(snapshot))

    res = preserver.replay_and_evaluate(
        snapshot_path=snap_file,
        setup_fn=setup_task_e1,
        test_file="test_order_pipeline.py",
    )
    assert res["status"] == "PASS"
    assert res["returncode"] == 0


# 14. Provider token extraction
def test_provider_token_extraction():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "step 1",
            "extra": {
                "actions": [{"command": "pytest test_order_pipeline.py"}],
                "response": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 40,
                        "total_tokens": 540,
                        "completion_tokens_details": {"reasoning_tokens": 30},
                    }
                },
            },
        },
    ]
    metrics = extract_trajectory_test_metrics(messages)
    assert metrics["prompt_tokens"] == 500
    assert metrics["completion_tokens"] == 40
    assert metrics["reasoning_tokens"] == 30
    assert metrics["total_tokens"] == 540


# 15. Missing cache data stays UNKNOWN
def test_missing_cache_data_stays_unknown(tmp_path: Path):
    # Simulate a run where provider does NOT provide cached_tokens
    results = [
        {
            "config": "G_context_baseline",
            "task": "task_e1_multifile",
            "rep": 1,
            "prompt_tokens": 1000,
            "total_tokens": 1100,
            "cached_tokens": None,  # Missing from API
            "cached_tokens_status": "UNKNOWN",
        },
        {
            "config": "G_context_baseline",
            "task": "task_e1_multifile",
            "rep": 2,
            "prompt_tokens": 1200,
            "total_tokens": 1300,
            "cached_tokens": 0,  # Explicitly reported 0
            "cached_tokens_status": "MEASURED",
        },
    ]
    summary = summarize_paired_evaluation(results, tmp_path)
    g_sum = summary["summary_by_config"]["G_context_baseline"]
    assert g_sum["cached_tokens_unknown_count"] == 1
    assert g_sum["cached_tokens_reported_count"] == 1
    # Average of measured values reflects measured data, not assuming unknown is zero
    assert g_sum["cached_tokens_mean"] == 0.0


# 16. ACM metrics correctness
def test_acm_metrics_correctness():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "a1",
            "extra": {
                "context_compaction": {
                    "enabled": True,
                    "compaction_performed": True,
                    "observations_compacted": 2,
                    "estimated_reduction_tokens": 300,
                    "duration_ms": 1.5,
                }
            },
        },
    ]
    info = {
        "context_manager_stats": {
            "enabled": True,
            "total_compactions": 1,
            "total_observations_compacted": 2,
            "total_estimated_tokens_saved": 300,
            "total_compaction_time_ms": 1.5,
        }
    }
    acm_metrics, issues = extract_acm_metrics(messages, info, expected_config="H_context_compaction")
    assert acm_metrics["acm_enabled"] is True
    assert acm_metrics["acm_compactions"] == 1
    assert acm_metrics["acm_observations_compacted"] == 2
    assert acm_metrics["acm_estimated_reduction_tokens"] == 300
    assert len(issues) == 0


# 17. Paired summary aggregation
def test_paired_summary_aggregation(tmp_path: Path):
    results = [
        {
            "pair_id": "pair_e1_rep1",
            "task": "task_e1_multifile",
            "config": "G_context_baseline",
            "rep": 1,
            "prompt_tokens": 5000,
            "total_tokens": 5200,
            "independent_resolved": True,
            "acm_compactions": 0,
        },
        {
            "pair_id": "pair_e1_rep1",
            "task": "task_e1_multifile",
            "config": "H_context_compaction",
            "rep": 1,
            "prompt_tokens": 4200,
            "total_tokens": 4400,
            "independent_resolved": True,
            "acm_compactions": 2,
            "acm_estimated_reduction_tokens": 800,
        },
    ]
    summary = summarize_paired_evaluation(results, tmp_path)
    pairs = summary["matched_pair_comparisons"]
    assert len(pairs) == 1
    assert pairs[0]["diff_prompt_tokens"] == -800  # H used 800 fewer tokens
    assert pairs[0]["g_prompt_tokens"] == 5000
    assert pairs[0]["h_prompt_tokens"] == 4200
    assert pairs[0]["h_compactions"] == 2


# 18. Backward compatibility with existing Phase 2 and 3.1 experiments
def test_backward_compatibility_with_existing_experiments():
    # Phase 2 tasks present
    assert "task1_simple_assertion" in TASKS
    assert "task2_multi_failure" in TASKS
    assert "task3_input_validation" in TASKS

    # Phase 2 configs present
    assert "A_baseline" in CONFIGS
    assert "B_loop_detection_only" in CONFIGS
    assert "C_test_feedback_only" in CONFIGS
    assert "D_combined" in CONFIGS

    # Phase 3 tasks present in ALL_TASKS
    assert "task4_verbose_investigation" in ALL_TASKS
    assert "task_e1_multifile" in ALL_TASKS
    assert "task_e2_multifailure" in ALL_TASKS
    assert "task_e3_verbose_investigation" in ALL_TASKS

    # Phase 3 configs present in ALL_CONFIGS
    assert "G_context_baseline" in ALL_CONFIGS
    assert "H_context_compaction" in ALL_CONFIGS


# 19. Sample-size accuracy regression tests
def test_report_one_run_per_configuration(tmp_path: Path):
    results = [
        {
            "pair_id": "pair_task_e1_multifile_rep1",
            "task": "task_e1_multifile",
            "config": "G_context_baseline",
            "rep": 1,
            "prompt_tokens": 36497,
            "total_tokens": 38624,
            "runtime_seconds": 34.28,
            "total_steps": 9,
            "independent_resolved": True,
            "submitted": True,
            "acm_compactions": 0,
            "acm_estimated_reduction_tokens": 0,
        },
        {
            "pair_id": "pair_task_e1_multifile_rep1",
            "task": "task_e1_multifile",
            "config": "H_context_compaction",
            "rep": 1,
            "prompt_tokens": 33050,
            "total_tokens": 35873,
            "runtime_seconds": 35.45,
            "total_steps": 8,
            "independent_resolved": True,
            "submitted": True,
            "acm_compactions": 3,
            "acm_estimated_reduction_tokens": 1846,
        },
    ]
    summary = summarize_paired_evaluation(results, tmp_path)
    g_sum = summary["summary_by_config"]["G_context_baseline"]
    h_sum = summary["summary_by_config"]["H_context_compaction"]

    assert summary["total_runs"] == 2
    assert summary["total_complete_matched_pairs"] == 1
    assert summary["total_partial_matched_pairs"] == 0
    assert g_sum["total_runs"] == 1
    assert g_sum["prompt_tokens_mean"] == 36497.0
    assert g_sum["prompt_tokens_std"] is None
    assert g_sum["total_tokens_std"] is None
    assert h_sum["prompt_tokens_std"] is None

    raw_json = json.dumps(summary)
    assert '"prompt_tokens_std": null' in raw_json
    assert '"prompt_tokens_std": 0.0' not in raw_json

    md = generate_markdown_phase3_report(summary)
    assert "36,497.0 ± N/A" in md
    assert "33,050.0 ± N/A" in md
    assert "± 0.0" not in md
    assert "- **Complete Matched Pairs**: 1" in md
    assert "- **Partial Matched Pairs**" not in md


def test_report_two_runs_per_configuration(tmp_path: Path):
    results = [
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 30000, "total_tokens": 32000, "runtime_seconds": 30.0, "total_steps": 8},
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "H_context_compaction", "rep": 1, "prompt_tokens": 25000, "total_tokens": 27000, "runtime_seconds": 31.0, "total_steps": 7},
        {"pair_id": "pair_e1_rep2", "task": "task_e1", "config": "G_context_baseline", "rep": 2, "prompt_tokens": 40000, "total_tokens": 42000, "runtime_seconds": 40.0, "total_steps": 10},
        {"pair_id": "pair_e1_rep2", "task": "task_e1", "config": "H_context_compaction", "rep": 2, "prompt_tokens": 25000, "total_tokens": 27000, "runtime_seconds": 31.0, "total_steps": 7},
    ]
    summary = summarize_paired_evaluation(results, tmp_path)
    g_sum = summary["summary_by_config"]["G_context_baseline"]
    h_sum = summary["summary_by_config"]["H_context_compaction"]

    assert summary["total_runs"] == 4
    assert summary["total_complete_matched_pairs"] == 2
    assert summary["total_partial_matched_pairs"] == 0
    assert g_sum["total_runs"] == 2
    assert g_sum["prompt_tokens_mean"] == 35000.0
    assert g_sum["prompt_tokens_std"] == 7071.1
    assert h_sum["prompt_tokens_mean"] == 25000.0
    assert h_sum["prompt_tokens_std"] == 0.0

    md = generate_markdown_phase3_report(summary)
    assert "35,000.0 ± 7,071.1" in md
    assert "25,000.0 ± 0.0" in md
    assert "± N/A" not in md
    assert "- **Complete Matched Pairs**: 2" in md


def test_report_partial_matched_pairs(tmp_path: Path):
    results = [
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 30000, "total_tokens": 32000, "runtime_seconds": 30.0, "total_steps": 8},
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "H_context_compaction", "rep": 1, "prompt_tokens": 28000, "total_tokens": 30000, "runtime_seconds": 31.0, "total_steps": 8},
        {"pair_id": "pair_e2_rep1", "task": "task_e2", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 35000, "total_tokens": 37000, "runtime_seconds": 35.0, "total_steps": 9},
    ]
    summary = summarize_paired_evaluation(results, tmp_path)

    assert summary["total_runs"] == 3
    assert summary["total_complete_matched_pairs"] == 1
    assert summary["total_partial_matched_pairs"] == 1
    assert len(summary["matched_pair_comparisons"]) == 1
    assert summary["summary_by_config"]["G_context_baseline"]["prompt_tokens_std"] == 3535.5
    assert summary["summary_by_config"]["H_context_compaction"]["prompt_tokens_std"] is None

    md = generate_markdown_phase3_report(summary)
    assert "- **Total Evaluated Runs**: 3" in md
    assert "- **Complete Matched Pairs**: 1" in md
    assert "- **Partial Matched Pairs**: 1" in md
    assert "32,500.0 ± 3,535.5" in md
    assert "28,000.0 ± N/A" in md
    assert "1 complete matched pair, 1 partial pair across 2 tasks and 2 conditions" in md


def test_report_incomplete_or_timed_out_run(tmp_path: Path):
    # Run 1 completed; Run 2 timed out before starting with no steps and missing prompt tokens
    results_unexecuted = [
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 36497, "total_tokens": 38624, "runtime_seconds": 34.0, "total_steps": 9, "exit_status": "Submitted"},
        {"pair_id": "pair_e1_rep2", "task": "task_e1", "config": "G_context_baseline", "rep": 2, "prompt_tokens": None, "total_tokens": None, "runtime_seconds": 180.0, "total_steps": 0, "exit_status": "TIMEOUT"},
    ]
    summary_unexec = summarize_paired_evaluation(results_unexecuted, tmp_path)
    g_sum_unexec = summary_unexec["summary_by_config"]["G_context_baseline"]
    assert g_sum_unexec["total_runs"] == 2
    assert g_sum_unexec["prompt_tokens_valid_count"] == 1
    assert g_sum_unexec["prompt_tokens_mean"] == 36497.0
    assert g_sum_unexec["prompt_tokens_std"] is None
    assert "36,497.0 ± N/A" in generate_markdown_phase3_report(summary_unexec)

    # Incomplete run that executed partially (took steps and consumed tokens before timing out)
    results_partial_exec = [
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 30000, "total_tokens": 32000, "runtime_seconds": 34.0, "total_steps": 8, "exit_status": "Submitted"},
        {"pair_id": "pair_e1_rep2", "task": "task_e1", "config": "G_context_baseline", "rep": 2, "prompt_tokens": 20000, "total_tokens": 22000, "runtime_seconds": 180.0, "total_steps": 5, "exit_status": "TIMEOUT"},
    ]
    summary_partial = summarize_paired_evaluation(results_partial_exec, tmp_path)
    g_sum_partial = summary_partial["summary_by_config"]["G_context_baseline"]
    assert g_sum_partial["total_runs"] == 2
    assert g_sum_partial["prompt_tokens_valid_count"] == 2
    assert g_sum_partial["prompt_tokens_mean"] == 25000.0
    assert g_sum_partial["prompt_tokens_std"] == 7071.1
    assert "25,000.0 ± 7,071.1" in generate_markdown_phase3_report(summary_partial)


def test_report_dynamic_disclaimer_accuracy(tmp_path: Path):
    # 2 runs (the live E1 pilot): 1 pair, 1 task, 2 conditions
    pilot_runs = [
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 36497},
        {"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "H_context_compaction", "rep": 1, "prompt_tokens": 33050},
    ]
    summary_pilot = summarize_paired_evaluation(pilot_runs, tmp_path)
    md_pilot = generate_markdown_phase3_report(summary_pilot)
    pilot_note = summary_pilot["data_quality_notes"][-1]
    assert pilot_note == "Sample size disclaimer: 2 runs (1 complete matched pair across 1 task and 2 conditions) provide preliminary directional comparison."
    assert "12 runs" not in pilot_note
    assert pilot_note in md_pilot

    # Single run
    single_run = [{"pair_id": "pair_e1_rep1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 36497}]
    summary_single = summarize_paired_evaluation(single_run, tmp_path)
    single_note = summary_single["data_quality_notes"][-1]
    assert single_note == "Sample size disclaimer: 1 run (0 complete matched pairs, 1 partial pair across 1 task and 1 condition) provides preliminary directional comparison."
    assert "12 runs" not in single_note

    # Multi-task 12 runs
    sched12 = [
        {"pair_id": f"p_{t}_{r}", "task": t, "config": c, "rep": r, "prompt_tokens": 1000}
        for t in ["t1", "t2", "t3"]
        for r in [1, 2]
        for c in ["G_context_baseline", "H_context_compaction"]
    ]
    summary12 = summarize_paired_evaluation(sched12, tmp_path)
    note12 = summary12["data_quality_notes"][-1]
    assert note12 == "Sample size disclaimer: 12 runs (6 complete matched pairs across 3 tasks and 2 conditions) provide preliminary directional comparison."


def test_report_json_and_markdown_consistency(tmp_path: Path):
    results = [
        {"pair_id": "p1", "task": "task_e1", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 36497, "total_tokens": 38624, "runtime_seconds": 34.28, "total_steps": 9},
        {"pair_id": "p1", "task": "task_e1", "config": "H_context_compaction", "rep": 1, "prompt_tokens": 33050, "total_tokens": 35873, "runtime_seconds": 35.45, "total_steps": 8},
        {"pair_id": "p2", "task": "task_e2", "config": "G_context_baseline", "rep": 1, "prompt_tokens": 40000, "total_tokens": 42000, "runtime_seconds": 40.0, "total_steps": 10},
    ]
    summary = summarize_paired_evaluation(results, tmp_path)
    md = generate_markdown_phase3_report(summary)

    # Total runs and pair counts consistent
    assert f"- **Total Evaluated Runs**: {summary['total_runs']}" in md
    assert f"- **Complete Matched Pairs**: {summary['total_complete_matched_pairs']}" in md
    assert f"- **Partial Matched Pairs**: {summary['total_partial_matched_pairs']}" in md

    # Configs consistency
    for cfg, s in summary["summary_by_config"].items():
        assert f"`{cfg}`" in md
        mean_p = s["prompt_tokens_mean"]
        assert f"{mean_p:,.1f}" in md
        if s["prompt_tokens_std"] is None:
            assert f"{mean_p:,.1f} ± N/A" in md
        else:
            assert f"{mean_p:,.1f} ± {s['prompt_tokens_std']:,.1f}" in md

    # Data quality notes consistency
    for note in summary["data_quality_notes"]:
        assert note in md

    # Roundtrip JSON consistency
    loaded = json.loads(json.dumps(summary))
    assert loaded["total_runs"] == summary["total_runs"]
    assert loaded["total_complete_matched_pairs"] == summary["total_complete_matched_pairs"]
    assert loaded["summary_by_config"]["H_context_compaction"]["prompt_tokens_std"] is None

