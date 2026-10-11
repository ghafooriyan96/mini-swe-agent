"""Structured test output analysis and feedback for mini-SWE-agent."""

import re
from typing import Any
from pydantic import BaseModel


class TestFeedbackConfig(BaseModel):
    __test__ = False
    enabled: bool = False
    frameworks: list[str] = ["pytest", "unittest"]
    max_feedback_chars: int = 4000
    preserve_original_output: bool = True


class ParsedTestResult(BaseModel):
    framework: str
    status: str
    failed_tests: list[str] = []
    error_type: str | None = None
    details: str | None = None
    expected: str | None = None
    actual: str | None = None
    file_line: str | None = None
    summary: str | None = None
    returncode: int = 0


def _is_pytest(command: str, output: str) -> bool:
    if re.search(r"\b(pytest|py\.test|python\s+(?:-m\s+)?pytest)\b", command):
        return True
    if "=== FAILURES ===" in output or "=== short test summary info ===" in output:
        return True
    if re.search(r"===\s*\d+\s+(?:passed|failed|error)", output):
        return True
    return False


def _is_unittest(command: str, output: str) -> bool:
    if re.search(r"\b(unittest|python\s+(?:-m\s+)?unittest)\b", command):
        return True
    if re.search(r"\bRan\s+\d+\s+tests?\s+in\s+[\d\.]+s\b", output):
        return True
    if re.search(r"\bFAILED\s*\((?:failures|errors)=\d+", output):
        return True
    return False


def parse_pytest_output(command: str, raw_output: str, returncode: int) -> ParsedTestResult | None:
    output = raw_output or ""
    if not _is_pytest(command, output):
        return None

    # Check for syntax error
    syntax_match = re.search(r"SyntaxError:\s*(.*)", output)
    if syntax_match:
        fl_match = re.search(r'File "([^"]+)", line (\d+)', output)
        fl = f"{fl_match.group(1)}:{fl_match.group(2)}" if fl_match else None
        return ParsedTestResult(
            framework="pytest",
            status="SYNTAX_ERROR",
            error_type="SyntaxError",
            details=syntax_match.group(1).strip(),
            file_line=fl,
            returncode=returncode,
        )

    # Check for missing dependency
    dep_match = re.search(r"(?:ModuleNotFoundError|ImportError):\s*(.*)", output)
    if dep_match:
        return ParsedTestResult(
            framework="pytest",
            status="MISSING_DEPENDENCY",
            error_type="ModuleNotFoundError",
            details=dep_match.group(1).strip(),
            returncode=returncode,
        )

    # Check for timeout
    if returncode == 124 or "timed out" in output.lower():
        return ParsedTestResult(
            framework="pytest",
            status="TIMEOUT",
            error_type="TimeoutError",
            details="Test command timed out before completion",
            returncode=returncode,
        )

    # Check collection errors
    if "=== ERRORS ===" in output and ("ERROR collecting" in output or "collection failure" in output.lower()):
        err_tests = re.findall(r"ERROR collecting\s+([^\s]+)", output)
        return ParsedTestResult(
            framework="pytest",
            status="COLLECTION_ERROR",
            failed_tests=err_tests,
            error_type="CollectionError",
            details="Pytest collection encountered errors",
            returncode=returncode,
        )

    # Summary regex
    summary_match = re.search(r"===+\s*([\d\w\s,]+(?:passed|failed|error|skipped)[\d\w\s,]*)\s*===+", output)
    summary_str = summary_match.group(1).strip() if summary_match else None

    # Check passed
    if returncode == 0 and ("passed" in output.lower() or "100%" in output) and "FAILED" not in output:
        return ParsedTestResult(
            framework="pytest",
            status="PASSED",
            summary=summary_str or "All tests passed",
            returncode=returncode,
        )

    # Parse failed tests
    failed_tests: list[str] = []
    # 1. From short test summary info
    short_matches = re.findall(r"^FAILED\s+([^\s\-]+)", output, flags=re.MULTILINE)
    if short_matches:
        failed_tests.extend(short_matches)
    else:
        # 2. From header blocks: ___ test_name ___
        head_matches = re.findall(r"_{3,}\s*(.+?)\s*_{3,}", output)
        if head_matches:
            failed_tests.extend(head_matches)

    # Deduplicate while preserving order
    seen: set[str] = set()
    failed_tests = [t for t in failed_tests if not (t in seen or seen.add(t))]

    # Error type and details
    error_type: str | None = None
    details: str | None = None
    err_match = re.search(r"^E\s+([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.*)", output, flags=re.MULTILINE)
    if err_match:
        error_type = err_match.group(1)
        details = err_match.group(2).strip()
    else:
        assert_err = re.search(r"^E\s+(assert\s+.*)", output, flags=re.MULTILINE)
        if assert_err:
            error_type = "AssertionError"
            details = assert_err.group(1).strip()

    # Expected and actual
    expected: str | None = None
    actual: str | None = None
    if details:
        eq_match = re.search(r"assert\s+(.+?)\s*==\s*(.+)", details)
        if eq_match:
            actual = eq_match.group(1).strip()
            expected = eq_match.group(2).strip()

    # File and line
    file_line: str | None = None
    fl_match = re.search(r"^([a-zA-Z0-9_\-\./\\+]+\.py):(\d+):", output, flags=re.MULTILINE)
    if fl_match:
        file_line = f"{fl_match.group(1)}:{fl_match.group(2)}"

    return ParsedTestResult(
        framework="pytest",
        status="FAILED",
        failed_tests=failed_tests,
        error_type=error_type,
        details=details,
        expected=expected,
        actual=actual,
        file_line=file_line,
        summary=summary_str,
        returncode=returncode,
    )


def parse_unittest_output(command: str, raw_output: str, returncode: int) -> ParsedTestResult | None:
    output = raw_output or ""
    if not _is_unittest(command, output):
        return None

    # Syntax error check
    syntax_match = re.search(r"SyntaxError:\s*(.*)", output)
    if syntax_match:
        fl_match = re.search(r'File "([^"]+)", line (\d+)', output)
        fl = f"{fl_match.group(1)}:{fl_match.group(2)}" if fl_match else None
        return ParsedTestResult(
            framework="unittest",
            status="SYNTAX_ERROR",
            error_type="SyntaxError",
            details=syntax_match.group(1).strip(),
            file_line=fl,
            returncode=returncode,
        )

    # Missing dependency check
    dep_match = re.search(r"(?:ModuleNotFoundError|ImportError):\s*(.*)", output)
    if dep_match:
        return ParsedTestResult(
            framework="unittest",
            status="MISSING_DEPENDENCY",
            error_type="ModuleNotFoundError",
            details=dep_match.group(1).strip(),
            returncode=returncode,
        )

    # Timeout check
    if returncode == 124 or "timed out" in output.lower():
        return ParsedTestResult(
            framework="unittest",
            status="TIMEOUT",
            error_type="TimeoutError",
            details="Test command timed out before completion",
            returncode=returncode,
        )

    # Passed check
    if returncode == 0 and ("\nOK" in output or output.strip().endswith("OK")):
        ran_match = re.search(r"Ran\s+(\d+\s+tests?)\s+in", output)
        summary = ran_match.group(1) + ", all passed" if ran_match else "All tests passed"
        return ParsedTestResult(
            framework="unittest",
            status="PASSED",
            summary=summary,
            returncode=returncode,
        )

    # Failed tests
    failed_tests: list[str] = []
    for m in re.finditer(r"^(?:FAIL|ERROR):\s*([^\s]+)\s*\(([^)]+)\)", output, flags=re.MULTILINE):
        failed_tests.append(f"{m.group(2)}.{m.group(1)}")

    # Error type and details
    error_type: str | None = None
    details: str | None = None
    err_match = re.search(r"^([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.*)", output, flags=re.MULTILINE)
    if err_match:
        error_type = err_match.group(1)
        details = err_match.group(2).strip()

    # Expected and actual
    expected: str | None = None
    actual: str | None = None
    if details:
        neq_match = re.search(r"^(.+?)\s*!=\s*(.+)", details)
        if neq_match:
            actual = neq_match.group(1).strip()
            expected = neq_match.group(2).strip()

    # File and line
    file_line: str | None = None
    fl_matches = list(re.finditer(r'File "([^"]+\.py)", line (\d+)', output))
    if fl_matches:
        last_fl = fl_matches[-1]
        file_line = f"{last_fl.group(1)}:{last_fl.group(2)}"

    summary_match = re.search(r"FAILED\s*\(([^)]+)\)", output)
    summary_str = summary_match.group(1).strip() if summary_match else None

    return ParsedTestResult(
        framework="unittest",
        status="FAILED",
        failed_tests=failed_tests,
        error_type=error_type,
        details=details,
        expected=expected,
        actual=actual,
        file_line=file_line,
        summary=summary_str,
        returncode=returncode,
    )


def format_structured_feedback(result: ParsedTestResult, max_chars: int = 4000) -> str:
    lines = ["[STRUCTURED TEST FEEDBACK]", f"Framework: {result.framework}", f"Status: {result.status}"]

    if result.failed_tests:
        if len(result.failed_tests) == 1:
            lines.append(f"Failed Test: {result.failed_tests[0]}")
        else:
            lines.append("Failed Tests:")
            for t in result.failed_tests[:10]:
                lines.append(f"  - {t}")
            if len(result.failed_tests) > 10:
                lines.append(f"  ... and {len(result.failed_tests) - 10} more")

    if result.error_type:
        lines.append(f"Error Type: {result.error_type}")
    if result.details:
        lines.append(f"Details: {result.details}")
    if result.expected is not None:
        lines.append(f"Expected: {result.expected}")
    if result.actual is not None:
        lines.append(f"Actual: {result.actual}")
    if result.file_line:
        lines.append(f"File & Line: {result.file_line}")
    if result.summary:
        lines.append(f"Test Summary: {result.summary}")
    lines.append(f"Returncode: {result.returncode}")

    if result.status in ("FAILED", "COLLECTION_ERROR", "SYNTAX_ERROR", "MISSING_DEPENDENCY"):
        lines.append("\nSuggested Focus:")
        if result.status == "SYNTAX_ERROR":
            lines.append("Review the syntax error at the referenced file and line.")
        elif result.status == "MISSING_DEPENDENCY":
            lines.append("Install the missing module or check import statements.")
        elif result.status == "COLLECTION_ERROR":
            lines.append("Resolve errors preventing test collection.")
        else:
            lines.append("Investigate the failing assertion and relevant traceback above.")

    feedback_text = "\n".join(lines)
    if len(feedback_text) > max_chars:
        feedback_text = feedback_text[:max_chars] + "\n... [TRUNCATED FEEDBACK]"
    return feedback_text


class TestOutputAnalyzer:
    """Deterministic analyzer and structured feedback generator for test outputs."""

    __test__ = False

    def __init__(self, config: TestFeedbackConfig):
        self.config = config
        self.total_feedbacks_generated = 0
        self.total_failed_tests_detected = 0

    def reset(self) -> None:
        self.total_feedbacks_generated = 0
        self.total_failed_tests_detected = 0

    def process(self, action: dict, output: dict) -> dict:
        if not self.config.enabled:
            return output

        # Prevent duplicate feedback on same output
        extra = output.setdefault("extra", {})
        if extra.get("test_feedback") is not None:
            return output

        command = action.get("command", "")
        raw_output = output.get("output", "")
        returncode = output.get("returncode", 0)

        result: ParsedTestResult | None = None
        if "pytest" in self.config.frameworks:
            result = parse_pytest_output(command, raw_output, returncode)
        if result is None and "unittest" in self.config.frameworks:
            result = parse_unittest_output(command, raw_output, returncode)

        if result is None:
            return output

        feedback_str = format_structured_feedback(result, self.config.max_feedback_chars)
        self.total_feedbacks_generated += 1
        self.total_failed_tests_detected += len(result.failed_tests)

        extra["test_feedback"] = result.model_dump()
        extra["structured_feedback"] = feedback_str

        # Prepend structured test feedback block
        if self.config.preserve_original_output:
            output["output"] = f"{feedback_str}\n\n{raw_output}"
        else:
            output["output"] = feedback_str

        return output
