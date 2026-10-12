"""Regression tests for Phase 3: Reproducible Offline Trajectory Analysis Tool."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from experiments.phase3_adaptive_context.trajectory_analysis import (
    DEFAULT_PILOT_RUN_DIRS,
    analyze_run_directory,
    analyze_single_trajectory,
    app,
    classify_inspection,
    parse_file_actions,
    run_trajectory_analysis_suite,
    verify_trajectory_steps,
)

runner = CliRunner()

E1_DIR = DEFAULT_PILOT_RUN_DIRS[0]
E2_DIR = DEFAULT_PILOT_RUN_DIRS[1]
E3_DIR = DEFAULT_PILOT_RUN_DIRS[2]


def test_correctness_step_identification():
    e1_g_traj = E1_DIR / "trajectories" / "traj_task_e1_multifile_G_context_baseline_rep1.json"
    e1_h_traj = E1_DIR / "trajectories" / "traj_task_e1_multifile_H_context_compaction_rep1.json"

    res_g = analyze_single_trajectory(e1_g_traj, skip_replay=False)
    assert res_g["first_step_modified_production_code"] == 5
    assert res_g["first_step_independently_correct"] == 5
    assert res_g["correctness_verification_method"] == "REPLAY_VERIFIED"
    assert res_g["independent_resolved"] is True

    res_h = analyze_single_trajectory(e1_h_traj, skip_replay=False)
    assert res_h["first_step_modified_production_code"] == 4
    assert res_h["first_step_independently_correct"] == 4
    assert res_h["correctness_verification_method"] == "REPLAY_VERIFIED"
    assert res_h["independent_resolved"] is True


def test_e3_limits_exceeded_behavior():
    e3_h_traj = E3_DIR / "trajectories" / "traj_task_e3_verbose_investigation_H_context_compaction_rep1.json"
    report_data = json.loads((E3_DIR / "phase3_evaluation_report.json").read_text())
    h_entry = next(r for r in report_data["detailed_runs"] if r["config"] == "H_context_compaction")

    res = analyze_single_trajectory(e3_h_traj, report_detailed_entry=h_entry, skip_replay=False)

    assert res["exit_status"] == "LimitsExceeded"
    assert res["api_calls"] == 15
    assert res["independent_resolved"] is True
    assert res["first_step_modified_production_code"] == 11
    assert res["first_step_independently_correct"] == 11
    assert res["calls_before_first_repair"] == 10
    assert res["calls_after_first_repair"] == 4

    # Step-by-step correctness: 10 FAIL followed by 5 PASS
    assert res["step_correctness"] == ["FAIL"] * 10 + ["PASS"] * 5
    assert [s["independent_patch_correctness"] for s in res["steps"]] == ["FAIL"] * 10 + ["PASS"] * 5

    # Shell return codes vs patch correctness
    # Step 1: pytest failed (RC=1), patch FAIL
    assert res["steps"][0]["command_exit_code"] == 1
    assert res["steps"][0]["independent_patch_correctness"] == "FAIL"

    # Step 2: ls -la succeeded (RC=0), yet patch FAIL
    assert res["steps"][1]["command_exit_code"] == 0
    assert res["steps"][1]["independent_patch_correctness"] == "FAIL"

    # Step 6: git status failed (RC=128), patch FAIL
    assert res["steps"][5]["command_exit_code"] == 128
    assert res["steps"][5]["independent_patch_correctness"] == "FAIL"

    # Step 11: edit succeeded (RC=0), patch PASS (1st correct step)
    assert res["steps"][10]["command_exit_code"] == 0
    assert res["steps"][10]["independent_patch_correctness"] == "PASS"

    # Step 13: pytest succeeded (RC=0), patch PASS
    assert res["steps"][12]["command_exit_code"] == 0
    assert res["steps"][12]["independent_patch_correctness"] == "PASS"

    # Steps 12-15 were post-repair validation
    post_repair_cmds = [s["command"] for s in res["steps"][11:]]
    assert any("pytest" in c for c in post_repair_cmds)
    assert any("DataPoint" in c for c in post_repair_cmds)

    # Within-H same history reduction
    assert res["acm_enabled"] is True
    assert res["same_history_context_reduction"] is not None
    assert res["same_history_context_reduction"]["total_estimated_reduction_tokens"] == 18677


def test_shell_exit_codes_independent_of_patch_correctness():
    """Verify shell command exit codes never dictate independent benchmark patch correctness."""
    e3_h_traj = E3_DIR / "trajectories" / "traj_task_e3_verbose_investigation_H_context_compaction_rep1.json"
    res = analyze_single_trajectory(e3_h_traj, skip_replay=False)

    # RC=0 commands prior to step 11 must be FAIL
    rc0_pre_fix = [s for s in res["steps"][:10] if s["command_exit_code"] == 0]
    assert len(rc0_pre_fix) > 0
    assert all(s["independent_patch_correctness"] == "FAIL" for s in rc0_pre_fix)

    # RC=128 command at step 6 must have exit code 128 preserved and independent status FAIL
    step6 = res["steps"][5]
    assert step6["step"] == 6
    assert step6["command_exit_code"] == 128
    assert step6["observation_returncode"] == 128
    assert step6["independent_patch_correctness"] == "FAIL"

    # Steps after step 11 must be PASS
    assert all(s["independent_patch_correctness"] == "PASS" for s in res["steps"][10:])


def test_missing_data_and_unknown_handling(tmp_path: Path):
    # Unknown task returns UNKNOWN without crashing
    mod_step, correct_step, method, step_corr = verify_trajectory_steps("unknown_task_xyz", ["echo 1"])
    assert mod_step is None
    assert correct_step == "UNKNOWN"
    assert method == "NO_TASK_ARTIFACTS"
    assert step_corr == ["UNKNOWN"]

    # Empty commands return UNKNOWN
    mod_step, correct_step, method, step_corr = verify_trajectory_steps("task_e1_multifile", [])
    assert mod_step is None
    assert correct_step == "UNKNOWN"
    assert method == "NO_TASK_ARTIFACTS"
    assert step_corr == []

    # Skip replay flag returns UNKNOWN rather than guessing from final patch
    e1_g_traj = E1_DIR / "trajectories" / "traj_task_e1_multifile_G_context_baseline_rep1.json"
    res_skipped = analyze_single_trajectory(e1_g_traj, skip_replay=True)
    assert res_skipped["first_step_independently_correct"] == "UNKNOWN"
    assert res_skipped["correctness_verification_method"] == "REPLAY_SKIPPED"
    assert all(s["independent_patch_correctness"] == "UNKNOWN" for s in res_skipped["steps"])

    # Nonexistent run dir raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        analyze_run_directory(tmp_path / "nonexistent_dir")


def test_replay_isolation(tmp_path: Path):
    # Verify workspace isolation: running replay in temp dir never writes to repo root
    files_before = set(Path.cwd().glob("*"))
    commands = [
        "echo 'print(1)' > isolated_test_scratch.py",
        "pytest test_order_pipeline.py",
    ]
    verify_trajectory_steps("task_e1_multifile", commands)
    files_after = set(Path.cwd().glob("*"))

    assert files_before == files_after
    assert not Path("isolated_test_scratch.py").exists()


def test_repeated_inspection_classification():
    dummy_steps = [
        {"step": 1, "command": "cat window_aggregator.py"},
        {"step": 2, "command": "sed -n '10,25p' window_aggregator.py"},
        {"step": 3, "command": "cat <<'EOF' > window_aggregator.py\ncode\nEOF"},
        {"step": 4, "command": "nl -ba window_aggregator.py | sed -n '10,25p'"},
        {"step": 5, "command": "pytest test_rolling_window.py"},
        {"step": 6, "command": "cat window_aggregator.py"},
    ]
    prod_edits = [3]

    # Pre-edit line check at step 2
    c2 = classify_inspection("window_aggregator.py", 2, dummy_steps, prod_edits)
    assert c2["category"] == "PRE_EDIT_LINE_CHECK"
    assert c2["is_normal_verification"] is True

    # Post-edit verification at step 4
    c4 = classify_inspection("window_aggregator.py", 4, dummy_steps, prod_edits)
    assert c4["category"] == "POST_EDIT_VERIFICATION"
    assert c4["is_normal_verification"] is True

    # Post-repair review at step 6
    c6 = classify_inspection("window_aggregator.py", 6, dummy_steps, prod_edits)
    assert c6["category"] == "POST_REPAIR_REVIEW"
    assert c6["is_normal_verification"] is True


def test_report_reproducibility_and_cli(tmp_path: Path):
    # Test programmatic execution
    suite = run_trajectory_analysis_suite(
        run_dirs=[E1_DIR],
        output_dir=tmp_path / "out",
        skip_replay=True,
    )
    assert suite["total_evaluated_runs"] == 2
    json_path = tmp_path / "out" / "phase3_trajectory_analysis.json"
    md_path = tmp_path / "out" / "phase3_trajectory_analysis.md"
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text())
    assert data["total_evaluated_runs"] == 2
    assert "task_e1_multifile" in data["paired_comparisons"]
    assert "Six-Run Master Comparison Table" in md_path.read_text()

    # Test CLI execution with Typer runner
    cli_out = tmp_path / "cli_out"
    res = runner.invoke(app, [
        str(E1_DIR),
        "--output-dir", str(cli_out),
        "--skip-replay",
    ])
    assert res.exit_code == 0
    assert (cli_out / "phase3_trajectory_analysis.json").exists()
    assert (cli_out / "phase3_trajectory_analysis.md").exists()


def test_same_history_acm_reduction_separation():
    e1_h_traj = E1_DIR / "trajectories" / "traj_task_e1_multifile_H_context_compaction_rep1.json"
    e1_g_traj = E1_DIR / "trajectories" / "traj_task_e1_multifile_G_context_baseline_rep1.json"

    res_h = analyze_single_trajectory(e1_h_traj, skip_replay=True)
    res_g = analyze_single_trajectory(e1_g_traj, skip_replay=True)

    # Provider prompt tokens and same-history reduction are distinct
    assert res_h["provider_prompt_tokens"] == 33050
    assert res_h["same_history_context_reduction"] is not None
    assert res_h["same_history_context_reduction"]["total_estimated_reduction_tokens"] == 1846

    # G run has no same-history ACM reduction
    assert res_g["acm_enabled"] is False
    assert res_g["same_history_context_reduction"] is None
