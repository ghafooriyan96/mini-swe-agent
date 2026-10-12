"""Unit tests for Phase 2.3: Reproducible Patch and Workspace Preservation."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from experiments.phase2_test_feedback.workspace_preserver import (
    SecurityError,
    WorkspacePreserver,
    compute_sha256,
    is_binary_file,
)
from experiments.phase2_test_feedback.ablation_runner import (
    TASKS,
    AblationExperimentRunner,
    atomic_write_json,
    run_command_with_timeout,
)


def test_binary_file_detection():
    assert is_binary_file(b"hello\x00world")
    assert not is_binary_file(b"plain utf-8 text with symbols !@#$%^&*()")


def test_modified_files(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    f1 = ws / "module.py"
    f1.write_text("def hello():\n    return 'v1'\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    assert "module.py" in baseline

    f1.write_text("def hello():\n    return 'v2'\n")
    snapshot = preserver.capture_snapshot(ws, baseline)

    assert len(snapshot["modified_files"]) == 1
    assert snapshot["modified_files"][0]["path"] == "module.py"
    assert snapshot["modified_files"][0]["initial_sha256"] != snapshot["modified_files"][0]["final_sha256"]
    assert "--- a/module.py" in snapshot["patch_content"]
    assert "+++ b/module.py" in snapshot["patch_content"]

    # Replay on pristine clone
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "module.py").write_text("def hello():\n    return 'v1'\n")
    preserver.apply_snapshot(clone, snapshot)
    assert (clone / "module.py").read_text() == "def hello():\n    return 'v2'\n"


def test_new_files(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "existing.py").write_text("a = 1\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    new_file = ws / "pkg" / "created.py"
    new_file.parent.mkdir()
    new_file.write_text("b = 2\n")

    snapshot = preserver.capture_snapshot(ws, baseline)
    assert len(snapshot["created_files"]) == 1
    assert snapshot["created_files"][0]["path"] == "pkg/created.py"
    assert "+++ b/pkg/created.py" in snapshot["patch_content"]

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "existing.py").write_text("a = 1\n")
    preserver.apply_snapshot(clone, snapshot)
    assert (clone / "pkg" / "created.py").read_text() == "b = 2\n"


def test_deleted_files(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    f = ws / "obsolete.py"
    f.write_text("to delete\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    f.unlink()
    snapshot = preserver.capture_snapshot(ws, baseline)
    assert len(snapshot["deleted_files"]) == 1
    assert snapshot["deleted_files"][0]["path"] == "obsolete.py"
    assert "--- a/obsolete.py" in snapshot["patch_content"]

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "obsolete.py").write_text("to delete\n")
    preserver.apply_snapshot(clone, snapshot)
    assert not (clone / "obsolete.py").exists()


def test_multiple_changed_files(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "mod.py").write_text("mod v1\n")
    (ws / "del.py").write_text("del v1\n")
    (ws / "keep.py").write_text("keep v1\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    (ws / "mod.py").write_text("mod v2\n")
    (ws / "del.py").unlink()
    (ws / "new.py").write_text("new v1\n")

    snapshot = preserver.capture_snapshot(ws, baseline)
    assert len(snapshot["modified_files"]) == 1
    assert len(snapshot["deleted_files"]) == 1
    assert len(snapshot["created_files"]) == 1

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "mod.py").write_text("mod v1\n")
    (clone / "del.py").write_text("del v1\n")
    (clone / "keep.py").write_text("keep v1\n")

    preserver.apply_snapshot(clone, snapshot)
    assert (clone / "mod.py").read_text() == "mod v2\n"
    assert not (clone / "del.py").exists()
    assert (clone / "new.py").read_text() == "new v1\n"
    assert (clone / "keep.py").read_text() == "keep v1\n"


def test_unchanged_workspace(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "app.py").write_text("same\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    snapshot = preserver.capture_snapshot(ws, baseline)

    assert snapshot["modified_files"] == []
    assert snapshot["created_files"] == []
    assert snapshot["deleted_files"] == []
    assert snapshot["patch_content"] == ""


def test_exclusion_rules(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Excluded dirs
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("git config\n")
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "file.cpython-313.pyc").write_bytes(b"\x00\x01\x02")
    (ws / ".venv").mkdir()
    (ws / ".venv" / "pyvenv.cfg").write_text("venv\n")
    
    # Excluded secrets & extensions
    (ws / ".env").write_text("API_KEY=secret\n")
    (ws / "secret_key.pem").write_text("PRIVATE KEY\n")
    (ws / "token.txt").write_text("123456\n")
    (ws / "binary.so").write_bytes(b"\x00ELF\x00")

    # Regular file
    (ws / "app.py").write_text("print('ok')\n")

    preserver = WorkspacePreserver()
    scanned = preserver.scan_workspace(ws)
    assert list(scanned.keys()) == ["app.py"]


def test_replay_integrity_pass(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    math_file = ws / "math_mod.py"
    math_file.write_text("def add(a, b):\n    return a - b  # bug\n")
    test_file = ws / "test_math.py"
    test_file.write_text("from math_mod import add\ndef test_add():\n    assert add(2, 3) == 5\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    # Apply fix
    math_file.write_text("def add(a, b):\n    return a + b\n")

    snapshot = preserver.capture_snapshot(ws, baseline)
    out_dir = tmp_path / "exp_out"
    patch_path, snapshot_path = preserver.save_snapshot_and_patch(snapshot, out_dir, "run1")

    def setup_pristine(target_dir: Path):
        (target_dir / "math_mod.py").write_text("def add(a, b):\n    return a - b  # bug\n")
        (target_dir / "test_math.py").write_text("from math_mod import add\ndef test_add():\n    assert add(2, 3) == 5\n")

    res = preserver.replay_and_evaluate(
        snapshot_path=snapshot_path,
        setup_fn=setup_pristine,
        test_file="test_math.py",
        patch_path=patch_path,
    )
    assert res["status"] == "PASS"
    assert res["returncode"] == 0
    assert "External verification succeeded" in res["explanation"]


def test_corrupted_artifact_fail(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "file.py").write_text("x = 1\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    (ws / "file.py").write_text("x = 2\n")
    snapshot = preserver.capture_snapshot(ws, baseline)

    out_dir = tmp_path / "exp_out"
    patch_path, snapshot_path = preserver.save_snapshot_and_patch(snapshot, out_dir, "run_corrupt")

    # 1. Corrupt patch content
    patch_path.write_text("tampered patch content\n")
    res_tamper_patch = preserver.replay_and_evaluate(snapshot_path, patch_path=patch_path)
    assert res_tamper_patch["status"] == "FAIL"
    assert "Corrupted patch artifact" in res_tamper_patch["explanation"]

    # 2. Corrupt snapshot JSON
    corrupted_snap = out_dir / "bad.snapshot.json"
    corrupted_snap.write_text("{invalid json")
    res_bad_json = preserver.replay_and_evaluate(corrupted_snap)
    assert res_bad_json["status"] == "FAIL"
    assert "Corrupted snapshot JSON" in res_bad_json["explanation"]

    # 3. Missing snapshot
    res_missing = preserver.replay_and_evaluate(out_dir / "nonexistent.snapshot.json")
    assert res_missing["status"] == "UNKNOWN"


def test_invalid_paths_and_path_traversal(tmp_path: Path):
    preserver = WorkspacePreserver()
    ws = tmp_path / "workspace"
    ws.mkdir()

    assert not preserver.is_safe_rel_path("../outside.py")
    assert not preserver.is_safe_rel_path("/etc/passwd")
    assert not preserver.is_safe_rel_path("foo/../../bar.py")
    assert not preserver.is_safe_rel_path("\x00bad")
    assert not preserver.is_safe_rel_path("")
    assert not preserver.is_safe_rel_path(".")
    assert preserver.is_safe_rel_path("pkg/sub/mod.py")

    with pytest.raises(SecurityError):
        preserver.validate_workspace_path(ws, "../outside.py")

    with pytest.raises(SecurityError):
        preserver.validate_workspace_path(ws, "/etc/passwd")


def test_symlink_safety(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret\n")

    # Symlink directory pointing outside
    sym_dir = ws / "symlink_dir"
    sym_dir.symlink_to(outside_dir)

    preserver = WorkspacePreserver()

    with pytest.raises(SecurityError):
        preserver.validate_workspace_path(ws, "symlink_dir/secret.txt")

    # Symlink file pointing outside
    sym_file = ws / "sym_secret.txt"
    sym_file.symlink_to(outside_dir / "secret.txt")

    with pytest.raises(SecurityError):
        preserver.validate_workspace_path(ws, "sym_secret.txt")


def test_timeout_with_partial_modifications(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    code_file = ws / "code.py"
    code_file.write_text("initial code\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    # Simulate child process timing out after partial edits
    code_file.write_text("partial modified code\n")
    (ws / "temp_trace.txt").write_text("debug trace\n")

    # Run command that times out
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    res = run_command_with_timeout(cmd=cmd, cwd=ws, env={}, timeout_seconds=0.2)
    assert res["timed_out"] is True
    assert res["status"] == "TIMEOUT"

    # Capture snapshot immediately
    snapshot = preserver.capture_snapshot(
        ws,
        baseline,
        metadata={"run_status": res["status"], "timed_out": res["timed_out"]},
    )
    out_dir = tmp_path / "exp_timeout"
    patch_file, snapshot_file = preserver.save_snapshot_and_patch(snapshot, out_dir, "timeout_run")

    assert patch_file.exists()
    assert snapshot_file.exists()
    assert len(snapshot["modified_files"]) == 1
    assert snapshot["modified_files"][0]["path"] == "code.py"
    assert len(snapshot["created_files"]) == 1
    assert snapshot["created_files"][0]["path"] == "temp_trace.txt"
    assert snapshot["metadata"]["timed_out"] is True


def test_resume_and_replay(tmp_path: Path):
    out_dir = tmp_path / "resume_exp"
    runner = AblationExperimentRunner(output_dir=out_dir, dry_run=True, repetitions=1)

    assert runner.patches_dir.exists()
    assert runner.snapshots_dir.exists()
    assert runner.manifests_dir.exists()
    assert runner.traj_dir.exists()
    assert runner.logs_dir.exists()

    # Create prior snapshot and patch
    run_id = "task1_simple_assertion_A_baseline_rep1"
    patch_file = runner.patches_dir / f"{run_id}.patch"
    patch_file.write_text("--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-1\n+2\n")
    snap_file = runner.snapshots_dir / f"{run_id}.snapshot.json"
    snap_file.write_text(json.dumps({
        "schema_version": "1.0",
        "metadata": {"task": "task1_simple_assertion"},
        "patch_content": patch_file.read_text(),
        "patch_sha256": compute_sha256(patch_file.read_bytes()),
        "initial_state": {"f.py": {"sha256": "123", "size": 2}},
        "final_state": {"f.py": {"sha256": "456", "size": 2}},
        "modified_files": [],
        "created_files": [],
        "deleted_files": [],
    }))

    found_patch = runner._find_prior_artifact("patches", "task1_simple_assertion", "A_baseline", 1, ".patch")
    found_snap = runner._find_prior_artifact("snapshots", "task1_simple_assertion", "A_baseline", 1, ".snapshot.json")
    assert found_patch == patch_file
    assert found_snap == snap_file


def test_small_isolated_calculator_task(tmp_path: Path):
    """Requirement 14: End-to-end smoke test on a small isolated calculator task."""
    ws = tmp_path / "calc_sandbox"
    ws.mkdir()

    calc_py = ws / "calculator.py"
    calc_py.write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def divide(a, b):\n"
        "    return a / b  # Bug: no zero check with ValueError\n"
    )

    test_py = ws / "test_calculator.py"
    test_py.write_text(
        "import pytest\n"
        "from calculator import add, divide\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5\n\n"
        "def test_divide_zero():\n"
        "    with pytest.raises(ValueError, match='Cannot divide by zero'):\n"
        "        divide(5, 0)\n"
    )

    # Initial test run fails
    init_run = subprocess.run([sys.executable, "-m", "pytest", "test_calculator.py"], cwd=ws, capture_output=True)
    assert init_run.returncode != 0

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)

    # Agent applies fix
    calc_py.write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('Cannot divide by zero')\n"
        "    return a / b\n"
    )

    # Capture and save snapshot/patch
    snapshot = preserver.capture_snapshot(
        ws,
        baseline,
        metadata={"task": "calculator_isolated", "test_file": "test_calculator.py"},
    )
    out_dir = tmp_path / "exp_calc"
    patch_file, snapshot_file = preserver.save_snapshot_and_patch(snapshot, out_dir, "calc_rep1")

    assert patch_file.exists()
    assert snapshot_file.exists()
    assert "Cannot divide by zero" in patch_file.read_text()

    # Replay offline and verify
    def pristine_setup(target_dir: Path):
        (target_dir / "calculator.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def divide(a, b):\n"
            "    return a / b  # Bug\n"
        )
        (target_dir / "test_calculator.py").write_text(test_py.read_text())

    res = preserver.replay_and_evaluate(
        snapshot_path=snapshot_file,
        setup_fn=pristine_setup,
        test_file="test_calculator.py",
        patch_path=patch_file,
    )

    assert res["status"] == "PASS"
    assert res["returncode"] == 0
    assert "External verification succeeded" in res["explanation"]


def test_runner_lock_present_at_snapshot_absent_at_replay_succeeds(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "main.py").write_text("v1\n")
    (ws / ".experiment.lock").write_text("1234\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    assert ".experiment.lock" not in baseline

    (ws / "main.py").write_text("v2\n")
    snapshot = preserver.capture_snapshot(ws, baseline)
    assert ".experiment.lock" not in snapshot["initial_state"]
    assert ".experiment.lock" not in snapshot["final_state"]
    assert all(m["path"] != ".experiment.lock" for m in snapshot["modified_files"])
    assert all(c["path"] != ".experiment.lock" for c in snapshot["created_files"])

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "main.py").write_text("v1\n")
    assert not (clone / ".experiment.lock").exists()
    preserver.apply_snapshot(clone, snapshot)
    assert (clone / "main.py").read_text() == "v2\n"
    assert not (clone / ".experiment.lock").exists()


def test_legitimate_unexpected_file_fails_replay(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "app.py").write_text("code\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    snapshot = preserver.capture_snapshot(ws, baseline)

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "app.py").write_text("code\n")
    (clone / "rogue.py").write_text("untracked code\n")

    with pytest.raises(ValueError, match="Unexpected extra file in workspace: rogue.py"):
        preserver.apply_snapshot(clone, snapshot)


def test_tampered_file_sha256_fails_replay(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "app.py").write_text("v1\n")

    preserver = WorkspacePreserver()
    baseline = preserver.record_baseline(ws)
    (ws / "app.py").write_text("v2\n")
    snapshot = preserver.capture_snapshot(ws, baseline)

    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "app.py").write_text("v1\n")

    snapshot["final_state"]["app.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Hash mismatch on final file app.py"):
        preserver.apply_snapshot(clone, snapshot)


def test_legacy_snapshot_with_runner_lock_verifies_genuine_files(tmp_path: Path):
    preserver = WorkspacePreserver()
    target = tmp_path / "target"
    target.mkdir()
    (target / "calc.py").write_text("fixed\n")

    legacy_snapshot = {
        "modified_files": [],
        "created_files": [],
        "deleted_files": [],
        "final_state": {
            "calc.py": {"sha256": compute_sha256(b"fixed\n"), "size": 6},
            ".experiment.lock": {"sha256": compute_sha256(b"lock\n"), "size": 5},
        },
    }
    preserver.apply_snapshot(target, legacy_snapshot)

    (target / "calc.py").unlink()
    with pytest.raises(ValueError, match="Expected final file missing: calc.py"):
        preserver.apply_snapshot(target, legacy_snapshot)


def test_existing_real_snapshot_compatibility():
    real_snapshot_path = Path(
        "experiments/phase2_test_feedback/runs/run_20261012_000341/snapshots/task1_simple_assertion_D_combined_rep1.snapshot.json"
    )
    if not real_snapshot_path.exists():
        pytest.skip("Real snapshot file not present")

    raw_snap = json.loads(real_snapshot_path.read_text())
    assert ".experiment.lock" in raw_snap["final_state"]

    patch_path = (
        real_snapshot_path.parent.parent
        / "patches"
        / real_snapshot_path.name.replace(".snapshot.json", ".patch")
    )
    task_info = TASKS["task1_simple_assertion"]

    preserver = WorkspacePreserver()
    res = preserver.replay_and_evaluate(
        snapshot_path=real_snapshot_path,
        setup_fn=task_info["setup_fn"],
        test_file=task_info["test_file"],
        patch_path=patch_path,
    )
    assert res["status"] == "PASS"
    assert res["returncode"] == 0
    assert json.loads(real_snapshot_path.read_text()) == raw_snap

