from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.loop_detection import LoopDetectionAgent, LoopDetectionAgentConfig
from minisweagent.agents.utils.loop_detection import (
    LoopDetectionConfig,
    LoopDetector,
    fingerprint_output,
    get_repo_fingerprint,
    normalize_command,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


@pytest.fixture
def agent_config():
    config_path = Path("src/minisweagent/config/default.yaml")
    return yaml.safe_load(config_path.read_text())["agent"]


# 1. Three consecutive identical unsuccessful commands
def test_three_consecutive_identical_unsuccessful_commands():
    detector = LoopDetector(LoopDetectionConfig(enabled=True, repetition_threshold=3))
    action = {"command": "pytest tests/test_calc.py"}
    fail_output = {"returncode": 1, "output": "FAILED test_calc.py - AssertionError"}

    out1 = detector.process(action, dict(fail_output))
    assert "[WARNING]" not in out1["output"]
    assert detector.consecutive_count == 1
    assert detector.total_loop_detections == 0

    out2 = detector.process(action, dict(fail_output))
    assert "[WARNING]" not in out2["output"]
    assert detector.consecutive_count == 2
    assert detector.total_loop_detections == 0

    out3 = detector.process(action, dict(fail_output))
    assert "[WARNING]" in out3["output"]
    assert out3.get("extra", {}).get("loop_detected") is True
    assert detector.total_loop_detections == 1
    assert detector.total_interventions == 1
    assert detector.consecutive_count == 0


# 2. Identical command executed after a repository change
def test_identical_command_after_repo_change(tmp_path: Path):
    test_file = tmp_path / "code.py"
    test_file.write_text("x = 1\n")
    detector = LoopDetector(LoopDetectionConfig(enabled=True, repetition_threshold=3))
    action = {"command": "pytest"}
    fail_output = {"returncode": 1, "output": "FAILED"}

    class DummyEnv:
        def __init__(self, root: Path):
            self.root = root

        def get_repo_fingerprint(self):
            return get_repo_fingerprint(self.root)

    env = DummyEnv(tmp_path)

    # 1st execution
    detector.process(action, dict(fail_output), env=env)
    assert detector.consecutive_count == 1

    # 2nd execution - no change
    detector.process(action, dict(fail_output), env=env)
    assert detector.consecutive_count == 2

    # Modify repository file
    test_file.write_text("x = 2\n")

    # 3rd execution - repo changed, streak resets
    out3 = detector.process(action, dict(fail_output), env=env)
    assert "[WARNING]" not in out3["output"]
    assert detector.consecutive_count == 1
    assert detector.total_loop_detections == 0


# 3. Repeated pytest executions with different outputs
def test_repeated_command_different_outputs():
    detector = LoopDetector(LoopDetectionConfig(enabled=True, repetition_threshold=3))
    action = {"command": "pytest"}

    detector.process(action, {"returncode": 1, "output": "FAILED test_a.py"})
    assert detector.consecutive_count == 1

    detector.process(action, {"returncode": 1, "output": "FAILED test_b.py"})
    assert detector.consecutive_count == 1
    assert detector.total_loop_detections == 0

    detector.process(action, {"returncode": 0, "output": "PASSED 2 tests"})
    assert detector.consecutive_count == 1
    assert detector.total_loop_detections == 0


# 4. Commands that look similar but have different arguments
def test_similar_commands_different_arguments():
    detector = LoopDetector(LoopDetectionConfig(enabled=True, repetition_threshold=3))
    fail_output = {"returncode": 1, "output": "FAILED"}

    detector.process({"command": "pytest tests/test_one.py"}, dict(fail_output))
    assert detector.consecutive_count == 1

    detector.process({"command": "pytest tests/test_two.py"}, dict(fail_output))
    assert detector.consecutive_count == 1

    detector.process({"command": "pytest tests/test_one.py --verbose"}, dict(fail_output))
    assert detector.consecutive_count == 1
    assert detector.total_loop_detections == 0


# 5. Feature flag disabled
def test_feature_flag_disabled():
    detector = LoopDetector(LoopDetectionConfig(enabled=False, repetition_threshold=3))
    action = {"command": "pytest"}
    fail_output = {"returncode": 1, "output": "FAILED"}

    for _ in range(5):
        out = detector.process(action, dict(fail_output))
        assert "[WARNING]" not in out.get("output", "")
        assert "loop_detected" not in out.get("extra", {})

    assert detector.total_loop_detections == 0
    assert detector.total_interventions == 0


# 6. Corrective feedback generated correctly
def test_corrective_feedback_generated_correctly():
    custom_msg = "Custom corrective message: Change your strategy."
    detector = LoopDetector(
        LoopDetectionConfig(enabled=True, repetition_threshold=2, warning_template=custom_msg)
    )
    action = {"command": "python script.py"}
    fail_output = {"returncode": 1, "output": "Traceback: Error"}

    detector.process(action, dict(fail_output))
    out2 = detector.process(action, dict(fail_output))

    assert f"[WARNING] {custom_msg}" in out2["output"]
    assert out2["extra"]["loop_detected"] is True
    assert out2["extra"]["loop_warning"] == custom_msg
    assert out2["extra"]["loop_intervention"] == "warn"


# 7. Agent continuation after intervention
def test_agent_continuation_after_intervention(tmp_path: Path, agent_config):
    # Agent runs 3 identical failing commands, gets warning on 3rd, then submits
    failing_cmd = "false"
    model = DeterministicModel(
        outputs=[
            make_output("Try 1", [{"command": failing_cmd}]),
            make_output("Try 2", [{"command": failing_cmd}]),
            make_output("Try 3", [{"command": failing_cmd}]),
            make_output("Finished", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Done'"}]),
        ]
    )
    cfg = AgentConfig(
        **agent_config,
        loop_detection=LoopDetectionConfig(enabled=True, repetition_threshold=3),
    )
    agent = DefaultAgent(model=model, env=LocalEnvironment(cwd=str(tmp_path)), **cfg.model_dump())
    info = agent.run("Perform task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "Done\n"
    assert agent.n_calls == 4
    assert agent.loop_detector.total_loop_detections == 1
    assert agent.loop_detector.total_interventions == 1

    # Check that warning was delivered in observation message #3
    obs_messages = [m for m in agent.messages if m.get("role") == "user" and "[WARNING]" in m.get("content", "")]
    assert len(obs_messages) == 1
    assert "Repeated action detected without observable progress" in obs_messages[0]["content"]

    # Trajectory serialization includes stats
    serialized = agent.serialize()
    assert serialized["info"]["loop_detection_stats"]["loop_detections"] == 1
    assert serialized["info"]["loop_detection_stats"]["interventions"] == 1


# 8. Detection state reset between independent runs
def test_detection_state_reset_between_independent_runs(tmp_path: Path, agent_config):
    cfg = LoopDetectionAgentConfig(
        **agent_config,
        loop_detection=LoopDetectionConfig(enabled=True, repetition_threshold=3),
    )
    failing_cmd = "false"
    model1 = DeterministicModel(
        outputs=[
            make_output("Try 1", [{"command": failing_cmd}]),
            make_output("Try 2", [{"command": failing_cmd}]),
            make_output("Finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Done 1'"}]),
        ]
    )
    agent = LoopDetectionAgent(model=model1, env=LocalEnvironment(cwd=str(tmp_path)), **cfg.model_dump())
    agent.run("First run")
    assert agent.loop_detector.total_loop_detections == 0
    assert agent.loop_detector.consecutive_count == 2

    # Second independent run: loop detector consecutive count must reset to 0
    model2 = DeterministicModel(
        outputs=[
            make_output("Try 1", [{"command": failing_cmd}]),
            make_output("Finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Done 2'"}]),
        ]
    )
    agent.model = model2
    agent.run("Second run")
    # If not reset, count would have become 3 and triggered a loop; but with reset it is 1
    assert agent.loop_detector.total_loop_detections == 0
    assert agent.loop_detector.consecutive_count == 1


# Helper normalization tests
def test_normalization_and_duration_fingerprint():
    assert normalize_command("  pytest   -v    tests/test_foo.py  ") == "pytest -v tests/test_foo.py"
    assert normalize_command("git commit -m 'hello   world'") == "git commit -m 'hello   world'"
    assert fingerprint_output("1 failed in 0.05s") == "1 failed in Xs"
    assert fingerprint_output("1 failed in 1.20s") == "1 failed in Xs"
    assert fingerprint_output("Memory at 0x7ffee1234560") == "Memory at 0xHEX"
    assert fingerprint_output("2026-10-11 12:30:45 error") == "TIMESTAMP error"


# Tests for Task 3: Content-based fingerprinting & mtime resilience
def test_unchanged_content_with_changed_mtime(tmp_path: Path):
    """Test that modifying mtime without changing content does NOT cause false streak reset."""
    import os
    import time

    test_file = tmp_path / "app.py"
    test_file.write_text("print('hello world')\n")

    class DummyEnv:
        def __init__(self, root: Path):
            self.root = root

        def get_repo_fingerprint(self):
            return get_repo_fingerprint(self.root)

    env = DummyEnv(tmp_path)
    fp1 = get_repo_fingerprint(tmp_path)

    # Change mtime into the future without modifying content
    future_time = time.time() + 1000
    os.utime(test_file, (future_time, future_time))

    fp2 = get_repo_fingerprint(tmp_path)
    # Content is identical, so fingerprint must match exactly
    assert fp1 == fp2

    # Verify detector does not reset streak when mtime changes on identical content
    detector = LoopDetector(LoopDetectionConfig(enabled=True, repetition_threshold=3))
    action = {"command": "pytest"}
    fail_output = {"returncode": 1, "output": "FAILED"}

    detector.process(action, dict(fail_output), env=env)
    assert detector.consecutive_count == 1

    # Touch file mtime
    os.utime(test_file, (future_time + 100, future_time + 100))

    # Second execution should increment to 2 (NOT reset to 1)
    detector.process(action, dict(fail_output), env=env)
    assert detector.consecutive_count == 2


def test_ephemeral_and_generated_files_ignored(tmp_path: Path):
    """Test that ephemeral files (.pyc, .log, .coverage, .tmp) do not trigger false repo change."""
    test_file = tmp_path / "main.py"
    test_file.write_text("a = 1\n")

    fp1 = get_repo_fingerprint(tmp_path)

    # Create various ephemeral files
    (tmp_path / "test.log").write_text("log line\n")
    (tmp_path / "cache.pyc").write_text("bytecode\n")
    (tmp_path / ".coverage").write_text("coverage data\n")
    (tmp_path / "scratch.tmp").write_text("temp data\n")

    fp2 = get_repo_fingerprint(tmp_path)
    assert fp1 == fp2


def test_feature_flag_does_not_affect_unrelated_agent_behavior(tmp_path: Path, agent_config):
    """Verify that disabled loop detection behaves identically to baseline DefaultAgent."""
    model_output = [
        make_output("Step 1", [{"command": "echo 'step 1'"}]),
        make_output("Finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Done'"}]),
    ]
    # Agent with loop detection disabled
    cfg_disabled = AgentConfig(**agent_config, loop_detection=LoopDetectionConfig(enabled=False))
    agent_disabled = DefaultAgent(
        model=DeterministicModel(outputs=list(model_output)),
        env=LocalEnvironment(cwd=str(tmp_path)),
        **cfg_disabled.model_dump(),
    )
    info_disabled = agent_disabled.run("Run task")
    assert agent_disabled.loop_detector is None
    assert "loop_detection_stats" not in agent_disabled.serialize()["info"]
    assert info_disabled["exit_status"] == "Submitted"
