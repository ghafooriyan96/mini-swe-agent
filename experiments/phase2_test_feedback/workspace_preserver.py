import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

# Blacklisted directories and extensions
EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".hypothesis",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".whl",
    ".o",
    ".a",
    ".tar",
    ".gz",
    ".zip",
}

# Runner control and transient lock files (not part of the agent project workspace)
RUNNER_CONTROL_FILES = {
    ".experiment.lock",
}

# Regex to detect sensitive credential or secret files
SECRET_PATTERN = re.compile(
    r"(^\.env.*|\btoken\b|\bsecret\b|\bcredential\b|.*id_rsa.*|\.pem$|\.key$)",
    re.IGNORECASE,
)

MAX_PRESERVE_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class SecurityError(Exception):
    """Raised when an operation encounters an unsafe path or symlink escape."""


def is_binary_file(data: bytes) -> bool:
    """Return True if data appears to be binary by checking for null bytes."""
    return b"\x00" in data[:8192]


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


class WorkspacePreserver:
    """Captures, saves, and safely replays workspace patches and snapshots."""

    def __init__(
        self,
        excluded_dirs: set[str] | None = None,
        excluded_extensions: set[str] | None = None,
        excluded_files: set[str] | None = None,
        max_file_size_bytes: int = MAX_PRESERVE_FILE_SIZE_BYTES,
    ):
        self.excluded_dirs = excluded_dirs or EXCLUDED_DIR_NAMES
        self.excluded_extensions = excluded_extensions or EXCLUDED_EXTENSIONS
        self.excluded_files = excluded_files or RUNNER_CONTROL_FILES
        self.max_file_size = max_file_size_bytes

    def is_safe_rel_path(self, rel_path: Path | str) -> bool:
        """Validate relative path has no path traversal components."""
        p_str = str(rel_path).replace("\\", "/")
        if not p_str or p_str == "." or "\x00" in p_str:
            return False
        if p_str.startswith("/") or p_str.startswith("\\"):
            return False
        parts = Path(p_str).parts
        if ".." in parts:
            return False
        return True

    def validate_workspace_path(self, base_dir: Path, rel_path: Path | str) -> Path:
        """Resolve path and verify it stays strictly inside base_dir without symlink escapes."""
        if not self.is_safe_rel_path(rel_path):
            raise SecurityError(f"Path traversal detected in relative path: {rel_path}")

        base_resolved = base_dir.resolve()
        target = base_dir / rel_path

        # Check if target or any parent directory is a symlink pointing outside base_dir
        curr = target
        while curr != base_dir and curr != curr.parent:
            if curr.is_symlink():
                sym_target = curr.resolve()
                if not sym_target.is_relative_to(base_resolved):
                    raise SecurityError(f"Symlink escapes workspace root: {curr} -> {sym_target}")
            curr = curr.parent

        if target.exists():
            target_resolved = target.resolve()
            if not target_resolved.is_relative_to(base_resolved):
                raise SecurityError(f"Target path escapes workspace root: {target_resolved}")
        else:
            parent_resolved = target.parent.resolve()
            if not parent_resolved.is_relative_to(base_resolved):
                raise SecurityError(f"Target parent escapes workspace root: {parent_resolved}")

        return target

    def should_include_file(self, rel_path: Path) -> bool:
        """Check whether a file should be included according to exclusion rules."""
        name = rel_path.name
        if name in self.excluded_files or name.startswith(".experiment.lock"):
            return False

        parts = rel_path.parts
        for part in parts:
            if part in self.excluded_dirs:
                return False

        if SECRET_PATTERN.search(name):
            return False

        if rel_path.suffix.lower() in self.excluded_extensions:
            return False

        return True

    def scan_workspace(self, workspace_dir: Path) -> dict[str, dict[str, Any]]:
        """Scan workspace and compute SHA-256 hashes and content for text files."""
        files_state = {}
        if not workspace_dir.is_dir():
            return files_state

        for root, dirs, files in os.walk(workspace_dir):
            # Prune excluded dirs in-place
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            root_path = Path(root)

            for f in sorted(files):
                file_path = root_path / f
                rel_path = file_path.relative_to(workspace_dir)

                if not self.should_include_file(rel_path):
                    continue

                try:
                    self.validate_workspace_path(workspace_dir, rel_path)
                except SecurityError:
                    continue

                try:
                    stat = file_path.stat()
                    size = stat.st_size
                    if size > self.max_file_size:
                        continue

                    data = file_path.read_bytes()
                    sha = compute_sha256(data)
                    is_bin = is_binary_file(data)
                    content_str = data.decode("utf-8", errors="replace") if not is_bin else None

                    files_state[str(rel_path)] = {
                        "size": size,
                        "sha256": sha,
                        "is_binary": is_bin,
                        "content": content_str,
                    }
                except (OSError, PermissionError):
                    continue

        return files_state

    def record_baseline(self, workspace_dir: Path) -> dict[str, dict[str, Any]]:
        """Record initial file hashes and contents before an agent run."""
        return self.scan_workspace(workspace_dir)

    def capture_snapshot(
        self,
        workspace_dir: Path,
        baseline: dict[str, dict[str, Any]],
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Compare current workspace against baseline and create snapshot structure."""
        filtered_baseline = {
            k: v for k, v in baseline.items() if self.should_include_file(Path(k))
        }
        current = self.scan_workspace(workspace_dir)

        modified_files = []
        created_files = []
        deleted_files = []

        # Check modified and deleted
        for rel_str, b_info in sorted(filtered_baseline.items()):
            if rel_str in current:
                c_info = current[rel_str]
                if b_info["sha256"] != c_info["sha256"]:
                    modified_files.append({
                        "path": rel_str,
                        "initial_sha256": b_info["sha256"],
                        "final_sha256": c_info["sha256"],
                        "initial_content": b_info.get("content"),
                        "final_content": c_info.get("content"),
                        "is_binary": c_info.get("is_binary", False),
                    })
            else:
                deleted_files.append({
                    "path": rel_str,
                    "initial_sha256": b_info["sha256"],
                    "initial_content": b_info.get("content"),
                })

        # Check created
        for rel_str, c_info in sorted(current.items()):
            if rel_str not in filtered_baseline:
                created_files.append({
                    "path": rel_str,
                    "sha256": c_info["sha256"],
                    "content": c_info.get("content"),
                    "is_binary": c_info.get("is_binary", False),
                })

        # Generate unified diff
        patch_text = self._build_unified_patch(modified_files, created_files, deleted_files)
        patch_sha256 = compute_sha256(patch_text.encode("utf-8"))

        return {
            "schema_version": "1.0",
            "metadata": metadata or {},
            "initial_state": {
                p: {"sha256": d["sha256"], "size": d["size"]}
                for p, d in filtered_baseline.items()
            },
            "initial_files": {
                p: {
                    "sha256": d["sha256"],
                    "size": d["size"],
                    "content": d.get("content"),
                    "is_binary": d.get("is_binary", False),
                }
                for p, d in filtered_baseline.items()
            },
            "final_state": {
                p: {"sha256": d["sha256"], "size": d["size"]}
                for p, d in current.items()
            },
            "modified_files": modified_files,
            "created_files": created_files,
            "deleted_files": deleted_files,
            "patch_content": patch_text,
            "patch_sha256": patch_sha256,
        }

    def _build_unified_patch(
        self,
        modified: list[dict],
        created: list[dict],
        deleted: list[dict],
    ) -> str:
        """Build portable unified diff patch across all changes."""
        chunks = []
        for item in modified:
            if item.get("is_binary"):
                continue
            a_content = item.get("initial_content") or ""
            b_content = item.get("final_content") or ""
            diff = difflib.unified_diff(
                a_content.splitlines(keepends=True),
                b_content.splitlines(keepends=True),
                fromfile=f"a/{item['path']}",
                tofile=f"b/{item['path']}",
            )
            chunks.extend(diff)

        for item in created:
            if item.get("is_binary"):
                continue
            b_content = item.get("content") or ""
            diff = difflib.unified_diff(
                [],
                b_content.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{item['path']}",
            )
            chunks.extend(diff)

        for item in deleted:
            a_content = item.get("initial_content") or ""
            diff = difflib.unified_diff(
                a_content.splitlines(keepends=True),
                [],
                fromfile=f"a/{item['path']}",
                tofile="/dev/null",
            )
            chunks.extend(diff)

        return "".join(chunks)

    def save_snapshot_and_patch(
        self,
        snapshot: dict,
        output_dir: Path,
        run_id: str,
    ) -> tuple[Path, Path]:
        """Save patch and snapshot into patches/ and snapshots/ subdirectories."""
        patches_dir = output_dir / "patches"
        snapshots_dir = output_dir / "snapshots"
        patches_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        patch_file = patches_dir / f"{run_id}.patch"
        snapshot_file = snapshots_dir / f"{run_id}.snapshot.json"

        patch_text = snapshot.get("patch_content", "")
        patch_file.write_text(patch_text, encoding="utf-8")

        # Snapshot file records relative patch path and sha256
        saved_snapshot = dict(snapshot)
        saved_snapshot["patch_file"] = str(patch_file.relative_to(output_dir))
        snapshot_file.write_text(json.dumps(saved_snapshot, indent=2), encoding="utf-8")

        return patch_file, snapshot_file

    def apply_snapshot(self, target_dir: Path, snapshot: dict) -> None:
        """Apply snapshot changes into target_dir with strict path and hash validation."""
        target_resolved = target_dir.resolve()

        # 1. Apply deletions
        for d in snapshot.get("deleted_files", []):
            rel_path = Path(d["path"])
            dest = self.validate_workspace_path(target_dir, rel_path)
            if dest.exists() or dest.is_symlink():
                dest.unlink()

        # 2. Apply modifications
        for m in snapshot.get("modified_files", []):
            rel_path = Path(m["path"])
            dest = self.validate_workspace_path(target_dir, rel_path)
            content = m.get("final_content")
            if content is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = content.encode("utf-8")
                dest.write_bytes(data)
                written_sha = compute_sha256(data)
                expected_sha = m.get("final_sha256")
                if expected_sha and written_sha != expected_sha:
                    raise ValueError(
                        f"Integrity check failed for modified file {rel_path}: "
                        f"wrote {written_sha}, expected {expected_sha}"
                    )

        # 3. Apply creations
        for c in snapshot.get("created_files", []):
            rel_path = Path(c["path"])
            dest = self.validate_workspace_path(target_dir, rel_path)
            content = c.get("content")
            if content is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = content.encode("utf-8")
                dest.write_bytes(data)
                written_sha = compute_sha256(data)
                expected_sha = c.get("sha256")
                if expected_sha and written_sha != expected_sha:
                    raise ValueError(
                        f"Integrity check failed for created file {rel_path}: "
                        f"wrote {written_sha}, expected {expected_sha}"
                    )

        # 4. Final verification against snapshot final_state
        cur_files = self.scan_workspace(target_dir)
        final_state = snapshot.get("final_state", {})
        
        # Compatibility policy for existing snapshots:
        # Exclude runner-owned control files (e.g. .experiment.lock) recorded by older snapshots.
        # Genuine project files are strictly preserved and cryptographically verified.
        expected_final = {
            rel_str: f_meta
            for rel_str, f_meta in final_state.items()
            if self.should_include_file(Path(rel_str))
        }

        for rel_str, f_meta in expected_final.items():
            if rel_str not in cur_files:
                raise ValueError(f"Expected final file missing: {rel_str}")
            if cur_files[rel_str]["sha256"] != f_meta["sha256"]:
                raise ValueError(
                    f"Hash mismatch on final file {rel_str}: "
                    f"found {cur_files[rel_str]['sha256']}, expected {f_meta['sha256']}"
                )
        for rel_str in cur_files:
            if rel_str not in expected_final:
                raise ValueError(f"Unexpected extra file in workspace: {rel_str}")

    def replay_and_evaluate(
        self,
        snapshot_path: Path,
        setup_fn: Callable[[Path], None] | None = None,
        test_file: str | None = None,
        test_cmd: list[str] | None = None,
        patch_path: Path | None = None,
    ) -> dict[str, Any]:
        """Safely replay snapshot in disposable isolated workspace and evaluate external tests."""
        if not snapshot_path.exists():
            return {
                "status": "UNKNOWN",
                "returncode": -1,
                "explanation": f"Snapshot file not found: {snapshot_path}",
            }

        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {
                "status": "FAIL",
                "returncode": -1,
                "explanation": f"Corrupted snapshot JSON: {e}",
            }

        # Check patch integrity if patch_path is provided
        if patch_path and patch_path.exists():
            actual_patch_sha = compute_sha256(patch_path.read_bytes())
            expected_patch_sha = snapshot.get("patch_sha256")
            if expected_patch_sha and actual_patch_sha != expected_patch_sha:
                return {
                    "status": "FAIL",
                    "returncode": -1,
                    "explanation": (
                        f"Corrupted patch artifact: SHA-256 mismatch "
                        f"(found {actual_patch_sha}, expected {expected_patch_sha})"
                    ),
                }

        temp_dir = Path(tempfile.mkdtemp(prefix="offline_replay_"))
        try:
            # 1. Setup pristine baseline
            if setup_fn is not None:
                setup_fn(temp_dir)
            elif "initial_files" in snapshot:
                for rel_str, f_info in snapshot["initial_files"].items():
                    rel_p = Path(rel_str)
                    if not self.should_include_file(rel_p):
                        continue
                    dest = self.validate_workspace_path(temp_dir, rel_p)
                    if f_info.get("content") is not None:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_text(f_info["content"], encoding="utf-8")

            # 2. Apply snapshot
            try:
                self.apply_snapshot(temp_dir, snapshot)
            except (SecurityError, ValueError) as err:
                return {
                    "status": "FAIL",
                    "returncode": -1,
                    "explanation": f"Failed to apply snapshot: {err}",
                }

            # 3. Run external test command
            cmd = test_cmd
            if not cmd:
                py_exec = sys.executable
                if test_file:
                    cmd = [py_exec, "-m", "pytest", test_file]
                else:
                    cmd = [py_exec, "-m", "pytest"]

            proc = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30.0,
            )

            status = "PASS" if proc.returncode == 0 else "FAIL"
            explanation = (
                "External verification succeeded after applying patch"
                if proc.returncode == 0
                else f"External verification failed with exit code {proc.returncode}"
            )

            return {
                "status": status,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "explanation": explanation,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "FAIL",
                "returncode": -1,
                "explanation": "External test command timed out during replay",
            }
        except Exception as e:
            return {
                "status": "UNKNOWN",
                "returncode": -1,
                "explanation": f"Unexpected replay error: {e}",
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
