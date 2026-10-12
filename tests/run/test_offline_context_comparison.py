"""Regression tests for Phase 3.1: Controlled Offline Context Comparison."""

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from experiments.phase3_adaptive_context.offline_comparison import (
    DEFAULT_G_PATH,
    DEFAULT_H_PATH,
    analyze_between_runs,
    analyze_within_h,
    app,
    compare_prefix_views,
    extract_usage_metrics,
    get_command_from_message,
    load_trajectory,
    run_full_offline_comparison,
)
from minisweagent.agents.utils.context_manager import AdaptiveContextConfig, AdaptiveContextManager

runner = CliRunner()


@pytest.fixture
def h_trajectory_data() -> dict:
    return load_trajectory(DEFAULT_H_PATH)


@pytest.fixture
def g_trajectory_data() -> dict:
    return load_trajectory(DEFAULT_G_PATH)


def test_identical_history_offline_comparison(h_trajectory_data: dict):
    acm_cfg = h_trajectory_data["info"]["config"]["agent"]["adaptive_context"]
    acm = AdaptiveContextManager(AdaptiveContextConfig(**acm_cfg))
    within_h = analyze_within_h(h_trajectory_data, acm)

    assert len(within_h["steps"]) == 15
    assert within_h["summary"]["total_compactions_performed"] == 5
    assert within_h["summary"]["total_observations_compacted"] == 7
    assert within_h["summary"]["total_estimated_reduction_tokens"] == 5412

    # Verify per-step savings match recorded trajectory telemetry
    expected_step_savings = {
        11: 163,
        12: 164,
        13: 164,
        14: 2461,
        15: 2460,
    }
    for step_rec in within_h["steps"]:
        s = step_rec["step"]
        expected_saved = expected_step_savings.get(s, 0)
        assert step_rec["view_comparison"]["estimated_reduction_tokens"] == expected_saved
        assert step_rec["recorded_telemetry"].get("estimated_reduction_tokens", 0) == expected_saved


def test_canonical_history_nonmutation(h_trajectory_data: dict):
    acm_cfg = h_trajectory_data["info"]["config"]["agent"]["adaptive_context"]
    acm = AdaptiveContextManager(AdaptiveContextConfig(**acm_cfg))
    messages = h_trajectory_data["messages"]
    snapshot_before = copy.deepcopy(messages)

    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            prefix = messages[:i]
            snapshot_prefix = copy.deepcopy(prefix)
            step_view, compacted = compare_prefix_views(prefix, acm)
            assert prefix == snapshot_prefix
            if step_view["compaction_performed"]:
                assert compacted != prefix

    assert messages == snapshot_before


def test_metric_aggregation(g_trajectory_data: dict, h_trajectory_data: dict):
    acm_cfg = h_trajectory_data["info"]["config"]["agent"]["adaptive_context"]
    acm = AdaptiveContextManager(AdaptiveContextConfig(**acm_cfg))
    within_h = analyze_within_h(h_trajectory_data, acm)
    between = analyze_between_runs(g_trajectory_data, h_trajectory_data)

    assert within_h["summary"]["total_provider_prompt_tokens"] == 81668
    assert between["totals"]["g_prompt_tokens"] == 75077
    assert between["totals"]["h_prompt_tokens"] == 81668
    assert between["totals"]["diff_prompt_tokens"] == 6591
    assert between["totals"]["g_observation_chars"] == 7451
    assert between["totals"]["h_observation_chars"] == 23019
    assert between["totals"]["g_cached_tokens"] == 4012
    assert between["totals"]["h_cached_tokens"] == 0


def test_observation_structure_preservation(h_trajectory_data: dict):
    acm_cfg = h_trajectory_data["info"]["config"]["agent"]["adaptive_context"]
    acm = AdaptiveContextManager(AdaptiveContextConfig(**acm_cfg))

    # Test Step 14 prefix where Step 10 observation is compacted
    step_14_idx = [i for i, m in enumerate(h_trajectory_data["messages"]) if m.get("role") == "assistant"][13]
    prefix = h_trajectory_data["messages"][:step_14_idx]
    _, compacted = compare_prefix_views(prefix, acm)

    assert len(compacted) == len(prefix)
    for orig, comp in zip(prefix, compacted):
        assert comp.get("role") == orig.get("role")
        if orig.get("role") == "tool":
            assert comp.get("tool_call_id") == orig.get("tool_call_id")
            if "{" in orig.get("content", ""):
                try:
                    orig_dict = json.loads(orig["content"])
                    comp_dict = json.loads(comp["content"])
                    assert comp_dict.get("returncode") == orig_dict.get("returncode")
                except json.JSONDecodeError:
                    pass


def test_cli_offline_comparison_execution(tmp_path: Path):
    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"

    res = runner.invoke(app, [
        "--g-traj", str(DEFAULT_G_PATH),
        "--h-traj", str(DEFAULT_H_PATH),
        "--output-json", str(json_out),
        "--output-md", str(md_out),
    ])

    assert res.exit_code == 0
    assert json_out.exists()
    assert md_out.exists()

    report_data = json.loads(json_out.read_text())
    assert report_data["comparison_a_within_h"]["summary"]["total_estimated_reduction_tokens"] == 5412
    assert report_data["comparison_b_between_runs"]["totals"]["diff_prompt_tokens"] == 6591
    assert "Executive Summary" in md_out.read_text()
