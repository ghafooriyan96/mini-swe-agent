"""Mandatory Task Validity Preflight System for Phase 3 Experimental Evaluation.

Runs independently BEFORE any agent execution:
1. Isolates a pristine workspace and verifies the task test genuinely fails.
2. Isolates a second workspace, applies the reference fix, and verifies tests pass.
3. Confirms reference fix modifies production code, not test code.
4. Confirms test determinism across repeated executions.
5. Rejects tasks with appropriate status:
   - VALID_TASK
   - INVALID_TASK_ALREADY_PASSING
   - INVALID_TASK_REFERENCE_FIX_FAILS
   - INVALID_TASK_TEST_ERROR
   - INVALID_TASK_NONDETERMINISTIC
"""

from __future__ import annotations

from enum import Enum
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class TaskPreflightStatus(str, Enum):
    VALID_TASK = "VALID_TASK"
    INVALID_TASK_ALREADY_PASSING = "INVALID_TASK_ALREADY_PASSING"
    INVALID_TASK_REFERENCE_FIX_FAILS = "INVALID_TASK_REFERENCE_FIX_FAILS"
    INVALID_TASK_TEST_ERROR = "INVALID_TASK_TEST_ERROR"
    INVALID_TASK_NONDETERMINISTIC = "INVALID_TASK_NONDETERMINISTIC"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_workspace_hashes(workspace_dir: Path) -> dict[str, str]:
    return {
        p.name: _hash_file(p)
        for p in workspace_dir.glob("*.py")
    }


def validate_single_task(
    task_id: str,
    task_meta: dict,
    pytest_bin: Path | None = None,
    temp_base: Path | None = None,
) -> dict:
    if pytest_bin is None:
        venv_pytest = Path(sys.executable).parent / "pytest"
        pytest_bin = venv_pytest if venv_pytest.exists() else Path("pytest")

    test_file = task_meta["test_file"]
    setup_fn = task_meta["setup_fn"]
    reference_fix_fn = task_meta.get("reference_fix_fn")

    # Step 1: Test pristine workspace
    with tempfile.TemporaryDirectory(dir=temp_base, prefix=f"preflight_pristine_{task_id}_") as w_pristine_str:
        w_pristine = Path(w_pristine_str)
        setup_fn(w_pristine)
        pristine_hashes = _get_workspace_hashes(w_pristine)

        # Run test 1
        cmd = [str(pytest_bin), test_file]
        proc1 = subprocess.run(cmd, cwd=w_pristine, capture_output=True, text=True)

        if proc1.returncode == 0:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_ALREADY_PASSING.value,
                "reason": f"Task '{task_id}' passed initially in pristine state",
                "pristine_stdout": proc1.stdout,
                "pristine_stderr": proc1.stderr,
            }

        # Check for pytest execution error (e.g. exit code 3 or 4, or missing dependencies)
        # In pytest: 1 = Tests were run and some tests failed; 2 = interrupted; 3 = internal error; 4 = usage error; 5 = no tests collected
        if proc1.returncode not in (1,):
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_TEST_ERROR.value,
                "reason": f"Test execution failed with non-test error (exit code {proc1.returncode}): {proc1.stderr}",
                "pristine_stdout": proc1.stdout,
                "pristine_stderr": proc1.stderr,
            }

        # Run test 2 on pristine to verify failure determinism
        proc2 = subprocess.run(cmd, cwd=w_pristine, capture_output=True, text=True)
        if proc2.returncode != proc1.returncode:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_NONDETERMINISTIC.value,
                "reason": f"Task '{task_id}' pristine test run was nondeterministic (first: {proc1.returncode}, second: {proc2.returncode})",
            }

    # Step 2: Test fixed workspace
    if reference_fix_fn is None:
        return {
            "task_id": task_id,
            "status": TaskPreflightStatus.INVALID_TASK_REFERENCE_FIX_FAILS.value,
            "reason": f"Task '{task_id}' has no reference fix function registered",
        }

    with tempfile.TemporaryDirectory(dir=temp_base, prefix=f"preflight_fixed_{task_id}_") as w_fixed_str:
        w_fixed = Path(w_fixed_str)
        setup_fn(w_fixed)
        reference_fix_fn(w_fixed)
        fixed_hashes = _get_workspace_hashes(w_fixed)

        # Verify production code was changed, not the test file
        changed_files = [f for f, h in fixed_hashes.items() if pristine_hashes.get(f) != h]
        prod_changed = [f for f in changed_files if f != test_file]

        if not prod_changed:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_REFERENCE_FIX_FAILS.value,
                "reason": f"Reference fix for '{task_id}' did not modify any production code files (changed: {changed_files})",
            }

        if test_file in changed_files:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_REFERENCE_FIX_FAILS.value,
                "reason": f"Reference fix for '{task_id}' modified test file '{test_file}' instead of production code",
            }

        # Run test 1 on fixed workspace
        fix_proc1 = subprocess.run(cmd, cwd=w_fixed, capture_output=True, text=True)
        if fix_proc1.returncode != 0:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_REFERENCE_FIX_FAILS.value,
                "reason": f"Reference fix failed to resolve tests (exit code {fix_proc1.returncode}): {fix_proc1.stdout}",
                "fixed_stdout": fix_proc1.stdout,
                "fixed_stderr": fix_proc1.stderr,
            }

        # Run test 2 on fixed workspace to verify determinism
        fix_proc2 = subprocess.run(cmd, cwd=w_fixed, capture_output=True, text=True)
        if fix_proc2.returncode != 0:
            return {
                "task_id": task_id,
                "status": TaskPreflightStatus.INVALID_TASK_NONDETERMINISTIC.value,
                "reason": f"Reference fix test run was nondeterministic across runs",
            }

    return {
        "task_id": task_id,
        "status": TaskPreflightStatus.VALID_TASK.value,
        "production_files_modified": prod_changed,
        "test_file": test_file,
    }


def run_all_preflights(
    tasks_dict: dict,
    pytest_bin: Path | None = None,
    temp_base: Path | None = None,
) -> dict[str, dict]:
    results = {}
    for task_id, meta in tasks_dict.items():
        results[task_id] = validate_single_task(task_id, meta, pytest_bin=pytest_bin, temp_base=temp_base)
    return results
