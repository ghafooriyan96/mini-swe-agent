"""Unit and regression tests for Phase 3.1: Adaptive Context Management (ACM)."""

import copy
import json
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.context_manager import (
    AdaptiveContextConfig,
    AdaptiveContextManager,
    compact_observation_content,
    estimate_tokens,
    fold_text,
    is_code_modification_output,
    is_critical_test_failure,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def test_default_off_behavior():
    config = AgentConfig(system_template="system", instance_template="task")
    assert config.adaptive_context.enabled is False

    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task",
    )
    assert agent.context_manager is None


def test_initial_prompt_preservation():
    acm = AdaptiveContextManager(AdaptiveContextConfig(enabled=True, recent_turns_protected=1))
    messages = [
        {"role": "system", "content": "IMPORTANT SYSTEM INSTRUCTIONS " * 50},
        {"role": "user", "content": "INITIAL TASK PROMPT " * 50},
        {"role": "assistant", "content": "turn 1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join([f"file_{i}.txt" for i in range(100)])},
        {"role": "assistant", "content": "turn 2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "/workspace\n"},
    ]
    compacted, _ = acm.compact(messages)
    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]


def test_latest_three_turns_preserved():
    acm = AdaptiveContextManager(
        AdaptiveContextConfig(enabled=True, recent_turns_protected=3, max_observation_lines=20)
    )
    long_content = "\n".join([f"line {i}" for i in range(100)])
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        # Turn 1 (historical, should be folded)
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": long_content},
        # Turn 2 (historical, should be folded)
        {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": long_content},
        # Turn 3 (protected turn 3)
        {"role": "assistant", "content": "a3", "tool_calls": [{"id": "c3", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "content": long_content},
        # Turn 4 (protected turn 2)
        {"role": "assistant", "content": "a4", "tool_calls": [{"id": "c4", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c4", "content": long_content},
        # Turn 5 (protected turn 1 - latest)
        {"role": "assistant", "content": "a5", "tool_calls": [{"id": "c5", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c5", "content": long_content},
    ]

    compacted, telemetry = acm.compact(messages)
    assert telemetry["compaction_performed"] is True
    assert telemetry["observations_compacted"] == 2
    assert telemetry["observations_protected"] == 3

    # Turns 1 and 2 folded
    assert "AdaptiveContext: Folded" in compacted[3]["content"]
    assert "AdaptiveContext: Folded" in compacted[5]["content"]

    # Turns 3, 4, 5 completely untouched
    assert compacted[7]["content"] == long_content
    assert compacted[9]["content"] == long_content
    assert compacted[11]["content"] == long_content


def test_multi_tool_call_message_pairing():
    acm = AdaptiveContextManager(
        AdaptiveContextConfig(enabled=True, recent_turns_protected=1, max_observation_lines=10)
    )
    long_content1 = "\n".join([f"cat file1 line {i}" for i in range(50)])
    long_content2 = "\n".join([f"cat file2 line {i}" for i in range(50)])

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        # Turn 1 with 2 tool calls
        {
            "role": "assistant",
            "content": "multi tool call",
            "tool_calls": [
                {"id": "call_a", "function": {"name": "bash", "arguments": '{"command":"cmd1"}'}},
                {"id": "call_b", "function": {"name": "bash", "arguments": '{"command":"cmd2"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": long_content1},
        {"role": "tool", "tool_call_id": "call_b", "content": long_content2},
        # Turn 2 (latest protected)
        {
            "role": "assistant",
            "content": "recent",
            "tool_calls": [{"id": "call_c", "function": {"name": "bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_c", "content": "ok"},
    ]

    compacted, telemetry = acm.compact(messages)
    assert len(compacted) == len(messages)
    assert compacted[2]["tool_calls"][0]["id"] == "call_a"
    assert compacted[2]["tool_calls"][1]["id"] == "call_b"
    assert compacted[3]["tool_call_id"] == "call_a"
    assert compacted[4]["tool_call_id"] == "call_b"
    assert "AdaptiveContext: Folded" in compacted[3]["content"]
    assert "AdaptiveContext: Folded" in compacted[4]["content"]
    assert compacted[6]["tool_call_id"] == "call_c"


def test_json_encoded_tool_observation_validity():
    acm = AdaptiveContextManager(
        AdaptiveContextConfig(enabled=True, recent_turns_protected=1, max_observation_lines=15)
    )
    raw_output = "\n".join([f"build log item {i}" for i in range(60)])
    json_obs = json.dumps({"returncode": 0, "output": raw_output, "custom_field": "preserved"})

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json_obs},
        {"role": "assistant", "content": "new", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "done"},
    ]

    compacted, _ = acm.compact(messages)
    compacted_content = compacted[3]["content"]
    parsed = json.loads(compacted_content)

    assert parsed["returncode"] == 0
    assert parsed["custom_field"] == "preserved"
    assert parsed.get("acm_folded") is True
    assert "AdaptiveContext: Folded" in parsed["output"]
    assert "sha256:" in parsed["output"]


def test_return_code_preservation():
    config = AdaptiveContextConfig(max_observation_lines=10)
    # 1. JSON-encoded error
    json_err = json.dumps({"returncode": 127, "output": "\n".join([f"err {i}" for i in range(50)])})
    folded_json, was_folded, _ = compact_observation_content(json_err, config)
    assert was_folded is True
    assert json.loads(folded_json)["returncode"] == 127

    # 2. Template XML error
    xml_err = (
        "<returncode>2</returncode>\n"
        "<output>\n"
        + "\n".join([f"compiler out {i}" for i in range(50)])
        + "\n</output>"
    )
    folded_xml, was_folded_xml, _ = compact_observation_content(xml_err, config)
    assert was_folded_xml is True
    assert "<returncode>2</returncode>" in folded_xml


def test_large_single_line_output():
    config = AdaptiveContextConfig(max_observation_chars=1000, head_chars=200, tail_chars=200)
    huge_single_line = "A" * 3000
    folded, was_folded, meta = fold_text(
        huge_single_line,
        max_chars=config.max_observation_chars,
        head_chars=config.head_chars,
        tail_chars=config.tail_chars,
    )
    assert was_folded is True
    assert meta["strategy"] == "single_line"
    assert folded.startswith("A" * 200)
    assert folded.endswith("A" * 200)
    assert "AdaptiveContext: Folded 2600 chars" in folded
    assert "sha256:" in folded


def test_large_multiline_output():
    config = AdaptiveContextConfig(max_observation_lines=20, head_lines=5, tail_lines=5)
    multiline_text = "\n".join([f"line_{i:03d}" for i in range(100)])
    folded, was_folded, meta = fold_text(
        multiline_text,
        max_lines=config.max_observation_lines,
        head_lines=config.head_lines,
        tail_lines=config.tail_lines,
    )
    assert was_folded is True
    assert meta["strategy"] == "multiline"
    assert "line_000" in folded
    assert "line_004" in folded
    assert "line_095" in folded
    assert "line_099" in folded
    assert "AdaptiveContext: Folded 90 lines" in folded


def test_active_test_failure_preservation():
    config = AdaptiveContextConfig(
        max_observation_lines=10,
        max_observation_chars=500,
        preserve_test_failures=True,
    )
    # Pytest failure trace
    failure_trace = (
        "=== FAILURES ===\n"
        "_____ test_divide_zero _____\n"
        "    with pytest.raises(ValueError):\n"
        ">       divide(10, 0)\n"
        "E       ZeroDivisionError: division by zero\n"
        + "\n".join([f"trace line {i}" for i in range(40)])
        + "\nFAILED test_math.py::test_divide_zero - ZeroDivisionError"
    )
    assert is_critical_test_failure(failure_trace) is True

    # Ensure compact_observation_content preserves it completely
    folded, was_folded, meta = compact_observation_content(failure_trace, config)
    assert was_folded is False
    assert meta.get("protected") == "test_failure"
    assert folded == failure_trace

    # Also test within JSON
    json_fail = json.dumps({"returncode": 1, "output": failure_trace})
    folded_json, was_folded_json, meta_json = compact_observation_content(json_fail, config)
    assert was_folded_json is False
    assert meta_json.get("protected") == "test_failure"
    assert folded_json == json_fail


def test_relevant_patch_diff_preservation():
    config = AdaptiveContextConfig(
        max_observation_lines=10,
        max_observation_chars=500,
        preserve_file_modifications=True,
    )
    diff_text = (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -10,3 +10,4 @@\n"
        "- return a / b\n"
        "+ if b == 0: raise ValueError('Zero')\n"
        "+ return a / b\n"
        + "\n".join([f"context line {i}" for i in range(30)])
    )
    assert is_code_modification_output(diff_text) is True
    folded, was_folded, meta = compact_observation_content(diff_text, config)
    assert was_folded is False
    assert meta.get("protected") == "code_modification"
    assert folded == diff_text


def test_deepcopy_nonmutation_guarantees():
    acm = AdaptiveContextManager(AdaptiveContextConfig(enabled=True, recent_turns_protected=1))
    original_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join([f"verbose {i}" for i in range(100)])},
        {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "latest"},
    ]
    snapshot_before = copy.deepcopy(original_messages)

    compacted, _ = acm.compact(original_messages)

    # Input messages must remain identical
    assert original_messages == snapshot_before
    # Compacted must be different
    assert compacted != original_messages
    assert "AdaptiveContext: Folded" in compacted[3]["content"]
    assert "AdaptiveContext: Folded" not in original_messages[3]["content"]


def test_deterministic_repeated_compaction():
    acm = AdaptiveContextManager(AdaptiveContextConfig(enabled=True, recent_turns_protected=1))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join([f"verbose line {i}" for i in range(80)])},
        {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "done"},
    ]
    res1, meta1 = acm.compact(messages)
    res2, meta2 = acm.compact(messages)

    assert res1 == res2
    assert meta1["compacted_estimated_tokens"] == meta2["compacted_estimated_tokens"]
    assert meta1["estimated_reduction_tokens"] == meta2["estimated_reduction_tokens"]


def test_full_canonical_trajectory_serialization(tmp_path: Path):
    out_file = tmp_path / "trajectory.json"
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[{"role": "assistant", "content": "run echo", "extra": {"actions": [{"command": "echo test"}]}}]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task",
        output_path=out_file,
        step_limit=1,
        adaptive_context={"enabled": True, "recent_turns_protected": 0, "max_observation_lines": 5},
    )
    agent.run("test task")

    assert out_file.exists()
    saved = json.loads(out_file.read_text())
    assert "messages" in saved
    assert "context_manager_stats" in saved["info"]
    assert saved["info"]["context_manager_stats"]["enabled"] is True


def test_compatibility_with_loop_detection_and_test_feedback(tmp_path: Path):
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[{"role": "assistant", "content": "run pytest", "extra": {"actions": [{"command": "pytest"}]}}]
        ),
        env=LocalEnvironment(),
        system_template="sys",
        instance_template="task",
        output_path=tmp_path / "traj.json",
        step_limit=1,
        loop_detection={"enabled": True, "repetition_threshold": 3},
        test_feedback={"enabled": True},
        adaptive_context={"enabled": True},
    )
    assert agent.loop_detector is not None
    assert agent.test_feedback_analyzer is not None
    assert agent.context_manager is not None

    agent.run("run pytest")
    serialized = agent.serialize()
    assert "loop_detection_stats" in serialized["info"]
    assert "test_feedback_stats" in serialized["info"]
    assert "context_manager_stats" in serialized["info"]


def test_malformed_and_unsupported_observations():
    acm = AdaptiveContextManager(AdaptiveContextConfig(enabled=True, recent_turns_protected=1))
    malformed_json_content = "{\n" + '  "unclosed": "missing_bracket\n' * 30 + "}"
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": malformed_json_content},
        {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
    ]
    compacted, telemetry = acm.compact(messages)
    assert telemetry["unsupported_skipped"] == 1
    assert compacted[3]["content"] == messages[3]["content"]


def test_no_unexpected_increase_in_estimated_prompt_size():
    acm = AdaptiveContextManager(AdaptiveContextConfig(enabled=True, recent_turns_protected=1))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "short output"},
        {"role": "assistant", "content": "a2", "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "short output 2"},
    ]
    compacted, telemetry = acm.compact(messages)
    assert estimate_tokens(compacted) <= estimate_tokens(messages)
    assert telemetry["estimated_reduction_tokens"] == 0


def test_preservation_of_existing_snapshot_replay_behavior():
    real_snapshot = Path(
        "experiments/phase2_test_feedback/runs/run_20261012_000341/snapshots/task1_simple_assertion_D_combined_rep1.snapshot.json"
    )
    if not real_snapshot.exists():
        return

    from experiments.phase2_test_feedback.ablation_runner import TASKS
    from experiments.phase2_test_feedback.workspace_preserver import WorkspacePreserver

    preserver = WorkspacePreserver()
    patch_path = real_snapshot.parent.parent / "patches" / real_snapshot.name.replace(".snapshot.json", ".patch")
    task_info = TASKS["task1_simple_assertion"]

    res = preserver.replay_and_evaluate(
        snapshot_path=real_snapshot,
        setup_fn=task_info["setup_fn"],
        test_file=task_info["test_file"],
        patch_path=patch_path,
    )
    assert res["status"] == "PASS"
    assert res["returncode"] == 0
