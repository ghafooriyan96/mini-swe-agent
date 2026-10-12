"""Adaptive Context Management for mini-SWE-agent."""

import copy
import hashlib
import json
import re
import time
from typing import Any

from pydantic import BaseModel


class AdaptiveContextConfig(BaseModel):
    __test__ = False
    enabled: bool = False
    """Whether adaptive context compaction is enabled."""
    recent_turns_protected: int = 3
    """Number of most recent interaction turns to keep completely uncompacted."""
    max_observation_lines: int = 30
    """Maximum lines allowed in historical tool observation outputs before folding."""
    max_observation_chars: int = 1500
    """Maximum characters allowed in historical tool observation outputs before folding."""
    head_lines: int = 10
    """Number of leading lines preserved when folding multiline output."""
    tail_lines: int = 10
    """Number of trailing lines preserved when folding multiline output."""
    head_chars: int = 500
    """Number of leading characters preserved when folding single-line output."""
    tail_chars: int = 500
    """Number of trailing characters preserved when folding single-line output."""
    preserve_test_failures: bool = True
    """Whether to protect active test failures, tracebacks, and syntax errors."""
    preserve_file_modifications: bool = True
    """Whether to protect outputs containing diffs, patches, or file writes."""


def is_critical_test_failure(text: str) -> bool:
    """Detect if output contains active test failures, syntax errors, or tracebacks."""
    if not text:
        return False
    if re.search(r"===\s*(?:FAILURES|ERRORS)\s*===", text):
        return True
    if re.search(r"\bFAILED\s*(?:\(|\[)", text):
        return True
    if re.search(r"\bSyntaxError:\s*", text):
        return True
    if "Traceback (most recent call last):" in text:
        return True
    if "=== TEST OUTPUT ANALYSIS ===" in text or "Structured Test Feedback" in text:
        return True
    if "FAILED TESTS DETECTED" in text:
        return True
    return False


def is_code_modification_output(text: str) -> bool:
    """Detect if output contains diffs, patch outputs, or file edit confirmations."""
    if not text:
        return False
    if "diff --git" in text or ("--- a/" in text and "+++ b/" in text):
        return True
    if "patching file " in text:
        return True
    return False


def fold_text(
    text: str,
    *,
    max_lines: int = 30,
    max_chars: int = 1500,
    head_lines: int = 10,
    tail_lines: int = 10,
    head_chars: int = 500,
    tail_chars: int = 500,
) -> tuple[str, bool, dict[str, Any]]:
    """Fold text if exceeding max_lines or max_chars. Returns (folded_text, was_folded, meta)."""
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    total_chars = len(text)

    # 1. Multiline folding
    if total_lines > max_lines and total_lines > (head_lines + tail_lines):
        kept_head = "".join(lines[:head_lines])
        kept_tail = "".join(lines[-tail_lines:])
        omitted_lines_list = lines[head_lines:-tail_lines]
        omitted_text = "".join(omitted_lines_list)
        omitted_lines = len(omitted_lines_list)
        omitted_chars = len(omitted_text)
        omitted_sha = hashlib.sha256(omitted_text.encode("utf-8")).hexdigest()[:8]

        marker = f"... [AdaptiveContext: Folded {omitted_lines} lines ({omitted_chars} chars); sha256: {omitted_sha}] ...\n"
        folded = kept_head + marker + kept_tail
        if len(folded) < total_chars:
            meta = {
                "folded": True,
                "strategy": "multiline",
                "original_lines": total_lines,
                "original_chars": total_chars,
                "folded_chars": len(folded),
                "omitted_lines": omitted_lines,
                "omitted_chars": omitted_chars,
                "omitted_sha": omitted_sha,
            }
            return folded, True, meta

    # 2. Single-line / few-line character folding
    if total_chars > max_chars and total_chars > (head_chars + tail_chars):
        kept_head = text[:head_chars]
        kept_tail = text[-tail_chars:]
        omitted_text = text[head_chars:-tail_chars]
        omitted_chars = len(omitted_text)
        omitted_sha = hashlib.sha256(omitted_text.encode("utf-8")).hexdigest()[:8]

        marker = f" ... [AdaptiveContext: Folded {omitted_chars} chars; sha256: {omitted_sha}] ... "
        folded = kept_head + marker + kept_tail
        if len(folded) < total_chars:
            meta = {
                "folded": True,
                "strategy": "single_line",
                "original_lines": total_lines,
                "original_chars": total_chars,
                "folded_chars": len(folded),
                "omitted_lines": 0,
                "omitted_chars": omitted_chars,
                "omitted_sha": omitted_sha,
            }
            return folded, True, meta

    return text, False, {"folded": False}


def compact_observation_content(
    content: str,
    config: AdaptiveContextConfig,
) -> tuple[str, bool, dict[str, Any]]:
    """Safely compact tool observation content, handling JSON-encoded strings,

    XML <output>...</output> structures, and plain text.
    """
    if not content:
        return content, False, {"folded": False, "reason": "empty"}

    stripped = content.strip()

    # Case A: JSON-encoded string
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "output" in parsed and isinstance(parsed["output"], str):
                raw_out = parsed["output"]
                if config.preserve_test_failures and is_critical_test_failure(raw_out):
                    return content, False, {"folded": False, "protected": "test_failure"}
                if config.preserve_file_modifications and is_code_modification_output(raw_out):
                    return content, False, {"folded": False, "protected": "code_modification"}

                folded_out, was_folded, meta = fold_text(
                    raw_out,
                    max_lines=config.max_observation_lines,
                    max_chars=config.max_observation_chars,
                    head_lines=config.head_lines,
                    tail_lines=config.tail_lines,
                    head_chars=config.head_chars,
                    tail_chars=config.tail_chars,
                )
                if was_folded:
                    updated_dict = dict(parsed)
                    updated_dict["output"] = folded_out
                    updated_dict["acm_folded"] = True
                    return json.dumps(updated_dict), True, meta
                return content, False, meta
        except Exception:
            return content, False, {"folded": False, "reason": "malformed_json"}

    # Case B: Template-rendered string with <output>...</output>
    xml_output_match = re.search(r"(<output>\n?)(.*?)(\n?</output>)", content, re.DOTALL)
    if xml_output_match:
        prefix = xml_output_match.group(1)
        raw_out = xml_output_match.group(2)
        suffix = xml_output_match.group(3)

        if config.preserve_test_failures and is_critical_test_failure(raw_out):
            return content, False, {"folded": False, "protected": "test_failure"}
        if config.preserve_file_modifications and is_code_modification_output(raw_out):
            return content, False, {"folded": False, "protected": "code_modification"}

        folded_out, was_folded, meta = fold_text(
            raw_out,
            max_lines=config.max_observation_lines,
            max_chars=config.max_observation_chars,
            head_lines=config.head_lines,
            tail_lines=config.tail_lines,
            head_chars=config.head_chars,
            tail_chars=config.tail_chars,
        )
        if was_folded:
            new_content = (
                content[: xml_output_match.start()]
                + prefix
                + folded_out
                + suffix
                + content[xml_output_match.end() :]
            )
            return new_content, True, meta
        return content, False, meta

    # Case C: Plain text
    if config.preserve_test_failures and is_critical_test_failure(content):
        return content, False, {"folded": False, "protected": "test_failure"}
    if config.preserve_file_modifications and is_code_modification_output(content):
        return content, False, {"folded": False, "protected": "code_modification"}

    folded_out, was_folded, meta = fold_text(
        content,
        max_lines=config.max_observation_lines,
        max_chars=config.max_observation_chars,
        head_lines=config.head_lines,
        tail_lines=config.tail_lines,
        head_chars=config.head_chars,
        tail_chars=config.tail_chars,
    )
    return folded_out, was_folded, meta


def estimate_tokens(messages: list[dict]) -> int:
    """Fast, deterministic heuristic token count (~4 chars per token)."""
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
        for tc in m.get("tool_calls", []) or []:
            func = tc.get("function", {})
            total_chars += len(func.get("arguments", ""))
    return max(1, total_chars // 4)


class AdaptiveContextManager:
    def __init__(self, config: AdaptiveContextConfig | None = None):
        self.config = config or AdaptiveContextConfig()
        self.total_compactions = 0
        self.total_observations_compacted = 0
        self.total_observations_protected = 0
        self.total_estimated_tokens_saved = 0
        self.total_compaction_time_ms = 0.0

    def reset(self) -> None:
        self.total_compactions = 0
        self.total_observations_compacted = 0
        self.total_observations_protected = 0
        self.total_estimated_tokens_saved = 0
        self.total_compaction_time_ms = 0.0

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "total_compactions": self.total_compactions,
            "total_observations_compacted": self.total_observations_compacted,
            "total_observations_protected": self.total_observations_protected,
            "total_estimated_tokens_saved": self.total_estimated_tokens_saved,
            "total_compaction_time_ms": round(self.total_compaction_time_ms, 3),
        }

    def compact(self, messages: list[dict]) -> tuple[list[dict], dict[str, Any]]:
        """Produce an ephemeral, compacted copy of messages for model dispatch.

        Guarantees self.messages is NEVER mutated.
        """
        t0 = time.perf_counter()
        if not self.config.enabled or len(messages) <= 2:
            orig_toks = estimate_tokens(messages)
            return messages, {
                "enabled": self.config.enabled,
                "compaction_performed": False,
                "original_estimated_tokens": orig_toks,
                "compacted_estimated_tokens": orig_toks,
                "estimated_reduction_tokens": 0,
                "observations_compacted": 0,
                "observations_protected": 0,
                "unsupported_skipped": 0,
                "duration_ms": 0.0,
            }

        orig_tokens = estimate_tokens(messages)

        # Group turns starting with assistant or user (after messages[0, 1])
        turns: list[list[int]] = []
        current_turn: list[int] = []
        for idx in range(2, len(messages)):
            msg = messages[idx]
            role = msg.get("role")
            if role in ("assistant", "user") and current_turn:
                turns.append(current_turn)
                current_turn = [idx]
            else:
                current_turn.append(idx)
        if current_turn:
            turns.append(current_turn)

        if len(turns) <= self.config.recent_turns_protected:
            dt_ms = (time.perf_counter() - t0) * 1000
            protected_count = sum(1 for m in messages[2:] if m.get("role") == "tool")
            self.total_observations_protected += protected_count
            return [copy.deepcopy(m) for m in messages], {
                "enabled": True,
                "compaction_performed": False,
                "original_estimated_tokens": orig_tokens,
                "compacted_estimated_tokens": orig_tokens,
                "estimated_reduction_tokens": 0,
                "observations_compacted": 0,
                "observations_protected": protected_count,
                "unsupported_skipped": 0,
                "duration_ms": round(dt_ms, 3),
            }

        cutoff_turn = len(turns) - self.config.recent_turns_protected
        protected_indices = set(range(2))
        for t in turns[cutoff_turn:]:
            protected_indices.update(t)

        compacted_messages = []
        observations_compacted = 0
        observations_protected = 0
        unsupported_skipped = 0

        for idx, msg in enumerate(messages):
            if idx in protected_indices:
                if msg.get("role") == "tool":
                    observations_protected += 1
                compacted_messages.append(copy.deepcopy(msg))
                continue

            if msg.get("role") != "tool":
                compacted_messages.append(copy.deepcopy(msg))
                continue

            content = msg.get("content", "")
            new_content, was_folded, meta = compact_observation_content(content, self.config)

            if was_folded:
                new_msg = copy.deepcopy(msg)
                new_msg["content"] = new_content
                new_msg.setdefault("extra", {})["acm_meta"] = meta
                compacted_messages.append(new_msg)
                observations_compacted += 1
            else:
                if meta.get("protected"):
                    observations_protected += 1
                elif meta.get("reason") in ("empty", "malformed_json"):
                    unsupported_skipped += 1
                compacted_messages.append(copy.deepcopy(msg))

        comp_tokens = estimate_tokens(compacted_messages)
        reduction = max(0, orig_tokens - comp_tokens)
        dt_ms = (time.perf_counter() - t0) * 1000

        if observations_compacted > 0:
            self.total_compactions += 1
            self.total_observations_compacted += observations_compacted
            self.total_estimated_tokens_saved += reduction

        self.total_observations_protected += observations_protected
        self.total_compaction_time_ms += dt_ms

        telemetry = {
            "enabled": True,
            "compaction_performed": observations_compacted > 0,
            "original_estimated_tokens": orig_tokens,
            "compacted_estimated_tokens": comp_tokens,
            "estimated_reduction_tokens": reduction,
            "observations_compacted": observations_compacted,
            "observations_protected": observations_protected,
            "unsupported_skipped": unsupported_skipped,
            "duration_ms": round(dt_ms, 3),
        }
        return compacted_messages, telemetry
