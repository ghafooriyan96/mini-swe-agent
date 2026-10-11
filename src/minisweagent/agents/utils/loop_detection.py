"""Loop detection utilities for mini-SWE-agent."""

import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class LoopDetectionConfig(BaseModel):
    enabled: bool = False
    repetition_threshold: int = 3
    intervention: str = "warn"
    warning_template: str = (
        "Repeated action detected without observable progress. "
        "Review the previous results and try a different approach."
    )


def normalize_command(command: str) -> str:
    """Normalize whitespace in command without altering arguments, flags, or quotes."""
    cmd = command.strip()
    try:
        return " ".join(shlex.split(cmd, posix=False))
    except ValueError:
        return re.sub(r"[ \t]+", " ", cmd)


def fingerprint_output(output: str) -> str:
    """Fingerprint output by normalizing timestamps, ephemeral runtimes, and whitespace."""
    if not output:
        return ""
    text = re.sub(r"\bin \d+(?:\.\d+)?s\b", "in Xs", output)
    text = re.sub(r"\b\d+(?:\.\d+)? (?:seconds|secs|ms)\b", "X time", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "TIMESTAMP", text)
    text = re.sub(r"\b0x[0-9a-fA-F]{6,16}\b", "0xHEX", text)
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())


def _hash_file_content(path: str) -> str:
    """Compute sha256 hash of file content. Handles large files efficiently."""
    h = hashlib.sha256()
    try:
        size = os.path.getsize(path)
        if size > 10 * 1024 * 1024:
            with open(path, "rb") as f:
                h.update(f.read(65536))
                f.seek(max(0, size - 65536))
                h.update(f.read(65536))
                h.update(str(size).encode())
        else:
            with open(path, "rb") as f:
                h.update(f.read())
    except OSError:
        return ""
    return h.hexdigest()


def get_repo_fingerprint(root_dir: str | Path | None, cache: dict[str, tuple[int, int, str]] | None = None) -> int:
    """Compute content-based hash across non-ephemeral files in a directory.

    Uses (mtime, size) to avoid re-reading unchanged files, but guarantees that
    mtime updates without content modifications do NOT alter the fingerprint.
    """
    if not root_dir:
        return 0
    p = Path(root_dir)
    if not p.is_dir():
        return 0

    ignored_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        ".mypy_cache",
        ".tox",
        ".eggs",
        ".ruff_cache",
        ".cache",
        "htmlcov",
        "build",
        "dist",
    }
    ignored_exts = (
        ".pyc",
        ".pyo",
        ".log",
        ".traj.json",
        ".swp",
        ".tmp",
        ".bak",
        ".coverage",
    )

    if cache is None:
        cache = {}

    entries: list[tuple[str, str]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.endswith(".egg-info")]
            rel_dir = os.path.relpath(dirpath, p)
            for fname in sorted(filenames):
                if fname.endswith(ignored_exts) or fname.startswith(".coverage"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                    cached = cache.get(fpath)
                    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                        content_hash = cached[2]
                    else:
                        content_hash = _hash_file_content(fpath)
                        cache[fpath] = (st.st_mtime_ns, st.st_size, content_hash)
                    entries.append((os.path.normpath(os.path.join(rel_dir, fname)), content_hash))
                except OSError:
                    pass
    except OSError:
        return 0
    return hash(tuple(entries))


class LoopDetector:
    """Detects unproductive repeated actions and injects corrective feedback."""

    def __init__(self, config: LoopDetectionConfig):
        self.config = config
        self.total_loop_detections = 0
        self.total_interventions = 0
        self.content_cache: dict[str, tuple[int, int, str]] = {}
        self.reset()

    def reset(self) -> None:
        """Reset consecutive detection state between independent runs."""
        self.last_command: str | None = None
        self.last_returncode: int | None = None
        self.last_output_fp: str | None = None
        self.last_repo_fp: int | None = None
        self.consecutive_count: int = 0

    def get_repo_state(self, env: Any = None) -> int:
        """Obtain repository fingerprint from environment or current working directory."""
        if hasattr(env, "get_repo_fingerprint"):
            return env.get_repo_fingerprint()
        cwd = getattr(getattr(env, "config", None), "cwd", "") or os.getcwd()
        return get_repo_fingerprint(cwd, cache=self.content_cache)

    def process(self, action: dict, output: dict, env: Any = None) -> dict:
        """Inspect action and output. If loop detected, inject corrective warning."""
        if not self.config.enabled:
            return output

        command = action.get("command", "")
        if not command:
            return output

        norm_cmd = normalize_command(command)
        returncode = output.get("returncode", 0)
        out_fp = fingerprint_output(output.get("output", ""))

        is_same_action = (
            self.last_command is not None
            and norm_cmd == self.last_command
            and returncode == self.last_returncode
            and out_fp == self.last_output_fp
        )

        if is_same_action:
            curr_repo_fp = self.get_repo_state(env)
            repo_changed = (self.last_repo_fp is not None) and (curr_repo_fp != self.last_repo_fp)
            if repo_changed:
                self.consecutive_count = 1
                self.last_repo_fp = curr_repo_fp
            else:
                self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_repo_fp = self.get_repo_state(env)

        self.last_command = norm_cmd
        self.last_returncode = returncode
        self.last_output_fp = out_fp

        if self.consecutive_count >= self.config.repetition_threshold:
            self.total_loop_detections += 1
            output = self._apply_intervention(output)
            self.consecutive_count = 0

        return output

    def _apply_intervention(self, output: dict) -> dict:
        """Inject warning into observation output and metadata."""
        self.total_interventions += 1
        warning = self.config.warning_template
        existing_text = output.get("output", "")
        output["output"] = existing_text + f"\n\n[WARNING] {warning}"

        extra = output.setdefault("extra", {})
        extra["loop_detected"] = True
        extra["loop_intervention"] = self.config.intervention
        extra["loop_warning"] = warning
        return output
