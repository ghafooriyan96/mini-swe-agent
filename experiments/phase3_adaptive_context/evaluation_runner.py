"""Phase 3 Evaluation Runner — Paired Evaluation of Baseline G vs Compaction H.

Implements:
- Mandatory preflight validation before any agent execution
- Paired experimental design with alternating or randomized execution order
- Isolated workspace preservation and snapshot replay
- Comprehensive metric collection (provider usage, UNKNOWN cache handling, ACM telemetry)
- Structured JSON and Markdown evaluation reporting
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

REPO_DIR = Path(__file__).resolve().parent.parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiments.phase2_test_feedback.ablation_runner import (
    ALL_CONFIGS,
    ALL_TASKS,
    EXP_DIR,
    REPO_DIR,
    SANDBOX_DIR,
    VALID_TERMINAL_STATUSES,
    SandboxLock,
    WorkspacePreserver,
    atomic_write_json,
    check_test_failed,
    check_test_passed,
    extract_acm_metrics,
    extract_trajectory_test_metrics,
    parse_tool_observation,
    run_command_with_timeout,
    validate_trajectory,
)
from experiments.phase3_adaptive_context.benchmark_tasks import PHASE3_BENCHMARK_TASKS
from experiments.phase3_adaptive_context.preflight import (
    TaskPreflightStatus,
    run_all_preflights,
    validate_single_task,
)

PHASE3_CONFIGS = {
    "G_context_baseline": "config/research/antigravity_context_baseline.yaml",
    "H_context_compaction": "config/research/antigravity_context_compaction.yaml",
}


def build_paired_schedule(
    tasks: list[str],
    configs: list[str],
    repetitions: int,
    order_strategy: str = "alternating",
    seed: int = 42,
) -> list[dict]:
    schedule = []
    run_order = 1
    rng = random.Random(seed)

    for task_id in tasks:
        for rep in range(1, repetitions + 1):
            pair_id = f"pair_{task_id}_rep{rep}"
            pair_configs = list(configs)
            if len(pair_configs) == 2:
                cfg_a, cfg_b = pair_configs[0], pair_configs[1]
                if order_strategy == "alternating":
                    ordered_pair = [cfg_a, cfg_b] if rep % 2 == 1 else [cfg_b, cfg_a]
                elif order_strategy == "randomized":
                    ordered_pair = [cfg_a, cfg_b]
                    if rng.choice([True, False]):
                        ordered_pair.reverse()
                else:
                    ordered_pair = [cfg_a, cfg_b]
            else:
                ordered_pair = list(pair_configs)

            for order_idx, cfg_id in enumerate(ordered_pair, 1):
                schedule.append({
                    "run_order_index": run_order,
                    "pair_id": pair_id,
                    "task": task_id,
                    "config": cfg_id,
                    "rep": rep,
                    "order_within_pair": order_idx,
                    "order_strategy": order_strategy,
                })
                run_order += 1

    return schedule


def compute_median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    mid = len(sorted_v) // 2
    if len(sorted_v) % 2 == 1:
        return float(sorted_v[mid])
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def compute_std(values: list[float], mean_val: float) -> float | None:
    if len(values) <= 1:
        return None
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / (len(values) - 1))


def is_valid_token_observation(r: dict, token_key: str) -> bool:
    val = r.get(token_key)
    if val is None or not isinstance(val, (int, float)):
        return False
    if r.get("valid_observation") is False or r.get("is_valid") is False:
        return False
    if r.get("exit_status") in ("TIMEOUT", "FAILED", "ERROR") and r.get("total_steps", 0) == 0 and val == 0:
        return False
    return True


def generate_pilot_disclaimer(
    total_runs: int,
    total_complete_pairs: int,
    total_partial_pairs: int,
    num_tasks: int,
    num_configs: int,
) -> str:
    run_str = f"{total_runs} {'run' if total_runs == 1 else 'runs'}"
    verb = "provides" if total_runs == 1 else "provide"
    task_str = f"{num_tasks} {'task' if num_tasks == 1 else 'tasks'}"
    cfg_str = f"{num_configs} {'condition' if num_configs == 1 else 'conditions'}"
    pair_str = f"{total_complete_pairs} complete matched {'pair' if total_complete_pairs == 1 else 'pairs'}"

    if total_partial_pairs > 0:
        partial_str = f"{total_partial_pairs} partial {'pair' if total_partial_pairs == 1 else 'pairs'}"
        details = f"{pair_str}, {partial_str} across {task_str} and {cfg_str}"
    else:
        details = f"{pair_str} across {task_str} and {cfg_str}"

    return f"Sample size disclaimer: {run_str} ({details}) {verb} preliminary directional comparison."


class Phase3EvaluationRunner:
    """Orchestrates Phase 3 paired benchmark evaluations."""

    def __init__(
        self,
        output_dir: Path | None = None,
        sandbox_dir: Path | None = None,
        timeout_seconds: float = 180.0,
        resume: bool = False,
        resume_source: Path | None = None,
        repetitions: int = 2,
        step_limit: int = 15,
        dry_run: bool = False,
        task_filter: str | None = None,
        config_filter: str | None = None,
        order_strategy: str = "alternating",
        skip_preflight: bool = False,
    ):
        self.sandbox_dir = sandbox_dir or SANDBOX_DIR
        self.timeout_seconds = timeout_seconds
        self.resume = resume
        self.repetitions = repetitions
        self.step_limit = step_limit
        self.dry_run = dry_run
        self.task_filter = task_filter
        self.config_filter = config_filter
        self.order_strategy = order_strategy
        self.skip_preflight = skip_preflight

        # Available tasks: Phase 3 benchmark tasks + all registered tasks
        self.available_tasks = {**PHASE3_BENCHMARK_TASKS, **ALL_TASKS}
        if task_filter is not None:
            if task_filter not in self.available_tasks:
                raise ValueError(f"Invalid task filter '{task_filter}'. Available: {list(self.available_tasks.keys())}")
            self.active_tasks = {task_filter: self.available_tasks[task_filter]}
        else:
            self.active_tasks = dict(PHASE3_BENCHMARK_TASKS)

        # Available configs: G and H
        self.available_configs = {**PHASE3_CONFIGS, **ALL_CONFIGS}
        if config_filter is not None:
            if config_filter not in self.available_configs:
                raise ValueError(f"Invalid config filter '{config_filter}'. Available: {list(self.available_configs.keys())}")
            self.active_configs = {config_filter: self.available_configs[config_filter]}
        else:
            self.active_configs = dict(PHASE3_CONFIGS)

        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.output_dir = EXP_DIR / "runs" / f"run_phase3_{timestamp}"

        self.traj_dir = self.output_dir / "trajectories"
        self.logs_dir = self.output_dir / "logs"
        self.patches_dir = self.output_dir / "patches"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.manifests_dir = self.output_dir / "manifests"

        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_file = self.output_dir / "manifest.json"
        self.results_file = self.output_dir / "phase3_evaluation_report.json"
        self.resume_source = resume_source

        self.preserver = WorkspacePreserver()
        self.manifest: list[dict] = []
        self.raw_results: list[dict] = []
        self.preflight_results: dict[str, dict] = {}

    def run_preflight(self) -> bool:
        if self.skip_preflight:
            print("[PREFLIGHT] Preflight skipped by flag.")
            return True

        print("=" * 70)
        print("MANDATORY TASK VALIDITY PREFLIGHT")
        print("=" * 70)

        self.preflight_results = run_all_preflights(self.active_tasks)
        all_valid = True

        for task_id, res in self.preflight_results.items():
            status = res["status"]
            if status == TaskPreflightStatus.VALID_TASK.value:
                print(f"  [PASS] {task_id}: VALID_TASK (production modified: {res.get('production_files_modified')})")
            else:
                all_valid = False
                print(f"  [REJECT] {task_id}: {status} — {res.get('reason')}")

        preflight_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_valid": all_valid,
            "tasks": self.preflight_results,
        }
        atomic_write_json(self.output_dir / "preflight_results.json", preflight_record)

        if not all_valid:
            print("[PREFLIGHT ERROR] One or more tasks failed preflight validity checks. Agent launch blocked.")
            return False

        print("[PREFLIGHT] All benchmark tasks validated successfully. Proceeding with execution.\n")
        return True

    def execute_suite(self) -> dict:
        if not self.run_preflight():
            raise RuntimeError("Task preflight validation failed. Aborting experiment.")

        schedule = build_paired_schedule(
            tasks=list(self.active_tasks.keys()),
            configs=list(self.active_configs.keys()),
            repetitions=self.repetitions,
            order_strategy=self.order_strategy,
        )

        total_runs = len(schedule)
        print(f"Scheduled {total_runs} paired evaluation runs ({len(self.active_tasks)} tasks x {len(self.active_configs)} configs x {self.repetitions} reps, strategy={self.order_strategy})")

        with SandboxLock(self.sandbox_dir):
            for item in schedule:
                run_idx = item["run_order_index"]
                task_id = item["task"]
                cfg_id = item["config"]
                rep = item["rep"]
                pair_id = item["pair_id"]
                cfg_file = self.active_configs[cfg_id]
                task_meta = self.active_tasks[task_id]

                run_id = f"{task_id}_{cfg_id}_rep{rep}"
                target_traj_path = self.traj_dir / f"traj_{run_id}.json"
                stdout_log = self.logs_dir / f"{run_id}_stdout.log"
                stderr_log = self.logs_dir / f"{run_id}_stderr.log"

                # Check for resume
                if self.resume:
                    resumed = self._try_resume_run(run_idx, total_runs, task_id, cfg_id, rep, pair_id, item)
                    if resumed:
                        continue

                # Dry-run check
                if self.dry_run:
                    print(f"[{run_idx}/{total_runs}] [DRY RUN] Would execute {task_id} | {cfg_id} | rep {rep} (pair={pair_id})")
                    dry_entry = {
                        "run_order_index": run_idx,
                        "pair_id": pair_id,
                        "task": task_id,
                        "config": cfg_id,
                        "rep": rep,
                        "status": "DRY_RUN",
                        "resumed": False,
                    }
                    self.manifest.append(dry_entry)
                    atomic_write_json(self.manifests_dir / f"{run_id}.json", dry_entry)
                    continue

                # Live execution (guarded; not executed in dry-run or offline mode)
                print(f"[{run_idx}/{total_runs}] Executing {task_id} | {cfg_id} | rep {rep} (timeout={self.timeout_seconds}s)...", end="", flush=True)

                task_meta["setup_fn"](self.sandbox_dir)
                baseline = self.preserver.record_baseline(self.sandbox_dir)

                cmd = [
                    str(REPO_DIR / ".venv" / "bin" / "mini"),
                    "-c", cfg_file,
                    "-m", "openai/gemini-3.1-pro-high",
                    "-c", f"environment.cwd={self.sandbox_dir}",
                    "-c", f"agent.step_limit={self.step_limit}",
                    "-y", "--exit-immediately",
                    "-o", str(target_traj_path),
                    "-t", task_meta["prompt"],
                ]
                env = os.environ.copy()
                env["MSWEA_CONFIGURED"] = "true"
                env["PATH"] = f"{REPO_DIR / '.venv' / 'bin'}:{env.get('PATH', '')}"

                run_diag = run_command_with_timeout(
                    cmd=cmd,
                    cwd=REPO_DIR,
                    env=env,
                    timeout_seconds=self.timeout_seconds,
                    stdout_file=stdout_log,
                    stderr_file=stderr_log,
                )

                # Snapshot capture
                snapshot = self.preserver.capture_snapshot(
                    self.sandbox_dir,
                    baseline,
                    metadata={
                        "task": task_id,
                        "config": cfg_id,
                        "rep": rep,
                        "pair_id": pair_id,
                        "run_status": run_diag["status"],
                        "timed_out": run_diag["timed_out"],
                        "returncode": run_diag["returncode"],
                        "duration_seconds": run_diag["duration_seconds"],
                    },
                )
                patch_file, snapshot_file = self.preserver.save_snapshot_and_patch(snapshot, self.output_dir, run_id)

                # Independent pytest
                test_proc = subprocess.run(
                    [str(REPO_DIR / ".venv" / "bin" / "pytest"), task_meta["test_file"]],
                    cwd=self.sandbox_dir,
                    capture_output=True,
                    text=True,
                )
                resolved = (test_proc.returncode == 0)

                # Trajectory validation
                valid, parsed, val_reason = validate_trajectory(
                    target_traj_path,
                    task_id,
                    cfg_id,
                    rep,
                    independent_verified=resolved,
                )

                if parsed is None:
                    parsed = self._synthesize_fallback_result(target_traj_path, run_diag, resolved, task_id, cfg_id, rep)

                parsed["pair_id"] = pair_id
                parsed["run_order_index"] = run_idx
                parsed["order_within_pair"] = item["order_within_pair"]
                parsed["runtime_seconds"] = run_diag["duration_seconds"]
                parsed["patch_file"] = str(patch_file) if patch_file else None
                parsed["snapshot_file"] = str(snapshot_file) if snapshot_file else None

                self.raw_results.append(parsed)
                self.manifest.append(parsed)
                atomic_write_json(self.manifests_dir / f"{run_id}.json", parsed)
                self._save_state()

                res_str = "PASS" if resolved else "FAIL"
                print(f" {res_str} ({parsed.get('exit_status', 'UNKNOWN')}, {run_diag['duration_seconds']}s, prompt_tokens={parsed.get('prompt_tokens', 0)})")

        report = self.generate_final_report()
        return report

    def _try_resume_run(self, run_idx: int, total_runs: int, task_id: str, cfg_id: str, rep: int, pair_id: str, item: dict) -> bool:
        run_id = f"{task_id}_{cfg_id}_rep{rep}"
        traj_name = f"traj_{run_id}.json"
        target_traj = self.traj_dir / traj_name

        cand_traj = target_traj if target_traj.exists() else None
        if not cand_traj and self.resume_source:
            s_cand = self.resume_source / "trajectories" / traj_name
            if s_cand.exists():
                cand_traj = s_cand
            elif (self.resume_source / traj_name).exists():
                cand_traj = self.resume_source / traj_name

        if cand_traj and cand_traj.exists():
            valid, parsed, reason = validate_trajectory(cand_traj, task_id, cfg_id, rep)
            if valid and parsed:
                if cand_traj != target_traj:
                    shutil.copy2(cand_traj, target_traj)
                parsed["pair_id"] = pair_id
                parsed["run_order_index"] = run_idx
                parsed["order_within_pair"] = item["order_within_pair"]
                self.raw_results.append(parsed)
                self.manifest.append(parsed)
                atomic_write_json(self.manifests_dir / f"{run_id}.json", parsed)
                print(f"[{run_idx}/{total_runs}] [RESUMED] {task_id} | {cfg_id} | rep {rep} ({parsed.get('exit_status')})")
                return True
        return False

    def _synthesize_fallback_result(self, traj_path: Path, run_diag: dict, resolved: bool, task_id: str, cfg_id: str, rep: int) -> dict:
        exit_status = "TIMEOUT" if run_diag["timed_out"] else "FAILED"
        p_tok, c_tok, r_tok, t_tok, steps = None, None, None, None, 0
        cached_tok = None
        if traj_path.exists():
            try:
                data = json.loads(traj_path.read_text())
                msgs = data.get("messages", [])
                metrics = extract_trajectory_test_metrics(msgs)
                p_tok = metrics["prompt_tokens"]
                c_tok = metrics["completion_tokens"]
                r_tok = metrics["reasoning_tokens"]
                t_tok = metrics["total_tokens"]
                steps = metrics["total_steps"]
                exit_status = data.get("info", {}).get("exit_status") or exit_status
            except Exception:
                pass

        return {
            "trajectory": traj_path.name,
            "task": task_id,
            "config": cfg_id,
            "rep": rep,
            "exit_status": exit_status,
            "submitted": (exit_status == "Submitted"),
            "independent_resolved": resolved,
            "resolved": resolved,
            "total_steps": steps,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "reasoning_tokens": r_tok,
            "total_tokens": t_tok,
            "cached_tokens": cached_tok,
            "cached_tokens_status": "UNKNOWN" if cached_tok is None else "MEASURED",
            "acm_enabled": (cfg_id == "H_context_compaction"),
            "acm_status": "ENABLED_ACTIVE" if cfg_id == "H_context_compaction" else "DISABLED",
            "acm_compactions": 0,
            "acm_observations_compacted": 0,
            "acm_estimated_reduction_tokens": 0,
        }

    def _save_state(self) -> None:
        atomic_write_json(self.manifest_file, self.manifest)
        atomic_write_json(self.results_file, {"runs": self.raw_results})

    def generate_final_report(self) -> dict:
        report = summarize_paired_evaluation(self.raw_results, self.output_dir)
        atomic_write_json(self.results_file, report)
        md_content = generate_markdown_phase3_report(report)
        (self.output_dir / "phase3_evaluation_report.md").write_text(md_content)
        return report


def summarize_paired_evaluation(results: list[dict], output_dir: Path | str | None = None) -> dict:
    by_config: dict[str, list[dict]] = {}
    by_task: dict[str, list[dict]] = {}
    by_pair: dict[str, dict[str, dict]] = {}

    for idx, r in enumerate(results):
        cfg = r.get("config", "unknown")
        task = r.get("task", "unknown")
        pair_id = r.get("pair_id") or f"unpaired_{idx}"

        by_config.setdefault(cfg, []).append(r)
        by_task.setdefault(task, []).append(r)
        by_pair.setdefault(pair_id, {})[cfg] = r

    def summarize_runs(run_list: list[dict]) -> dict:
        n = len(run_list)
        if n == 0:
            return {}
        valid_p_toks = [float(r["prompt_tokens"]) for r in run_list if is_valid_token_observation(r, "prompt_tokens")]
        valid_t_toks = [float(r["total_tokens"]) for r in run_list if is_valid_token_observation(r, "total_tokens")]
        valid_runtimes = [
            float(r["runtime_seconds"])
            for r in run_list
            if r.get("runtime_seconds") is not None and isinstance(r["runtime_seconds"], (int, float))
        ]
        valid_steps = [
            int(r["total_steps"])
            for r in run_list
            if r.get("total_steps") is not None and isinstance(r["total_steps"], (int, float))
        ]
        res_count = sum(1 for r in run_list if r.get("independent_resolved") or r.get("resolved"))
        sub_count = sum(1 for r in run_list if r.get("submitted"))

        cached_vals = [r.get("cached_tokens") for r in run_list if r.get("cached_tokens") is not None]

        if valid_p_toks:
            m_ptok = sum(valid_p_toks) / len(valid_p_toks)
            med_ptok = compute_median(valid_p_toks)
            std_ptok = compute_std(valid_p_toks, m_ptok)
        else:
            m_ptok = 0.0
            med_ptok = 0.0
            std_ptok = None

        if valid_t_toks:
            m_ttok = sum(valid_t_toks) / len(valid_t_toks)
            med_ttok = compute_median(valid_t_toks)
            std_ttok = compute_std(valid_t_toks, m_ttok)
        else:
            m_ttok = 0.0
            med_ttok = 0.0
            std_ttok = None

        m_run = sum(valid_runtimes) / len(valid_runtimes) if valid_runtimes else 0.0

        return {
            "total_runs": n,
            "resolved_count": res_count,
            "resolved_rate_pct": round(res_count / n * 100, 1),
            "submitted_count": sub_count,
            "submitted_rate_pct": round(sub_count / n * 100, 1),
            "prompt_tokens_mean": round(m_ptok, 1),
            "prompt_tokens_median": round(med_ptok, 1) if isinstance(med_ptok, float) else med_ptok,
            "prompt_tokens_std": round(std_ptok, 1) if std_ptok is not None else None,
            "prompt_tokens_valid_count": len(valid_p_toks),
            "total_tokens_mean": round(m_ttok, 1),
            "total_tokens_median": round(med_ttok, 1) if isinstance(med_ttok, float) else med_ttok,
            "total_tokens_std": round(std_ttok, 1) if std_ttok is not None else None,
            "total_tokens_valid_count": len(valid_t_toks),
            "runtime_seconds_mean": round(m_run, 2),
            "runtime_seconds_median": compute_median(valid_runtimes) if valid_runtimes else 0.0,
            "steps_mean": round(sum(valid_steps) / len(valid_steps), 1) if valid_steps else 0.0,
            "cached_tokens_reported_count": len(cached_vals),
            "cached_tokens_unknown_count": n - len(cached_vals),
            "cached_tokens_mean": round(sum(cached_vals) / len(cached_vals), 1) if cached_vals else "UNKNOWN",
            "acm_compactions_total": sum(r.get("acm_compactions", 0) for r in run_list),
            "acm_estimated_reduction_tokens_total": sum(r.get("acm_estimated_reduction_tokens", 0) for r in run_list),
        }

    # Matched-pair differences
    pair_comparisons = []
    total_complete_pairs = 0
    total_partial_pairs = 0
    for pair_id, pair_runs in sorted(by_pair.items()):
        has_g = "G_context_baseline" in pair_runs
        has_h = "H_context_compaction" in pair_runs
        if has_g and has_h:
            total_complete_pairs += 1
            g_run = pair_runs["G_context_baseline"]
            h_run = pair_runs["H_context_compaction"]
            g_ptok = g_run.get("prompt_tokens") or 0
            h_ptok = h_run.get("prompt_tokens") or 0
            diff_prompt = h_ptok - g_ptok
            pair_comparisons.append({
                "pair_id": pair_id,
                "task": g_run.get("task"),
                "rep": g_run.get("rep"),
                "g_prompt_tokens": g_run.get("prompt_tokens", 0),
                "h_prompt_tokens": h_run.get("prompt_tokens", 0),
                "diff_prompt_tokens": diff_prompt,
                "g_resolved": g_run.get("independent_resolved", False),
                "h_resolved": h_run.get("independent_resolved", False),
                "h_compactions": h_run.get("acm_compactions", 0),
                "h_estimated_savings": h_run.get("acm_estimated_reduction_tokens", 0),
            })
        elif has_g or has_h:
            total_partial_pairs += 1

    summary_by_config = {cfg: summarize_runs(runs) for cfg, runs in by_config.items()}
    summary_by_task = {task: summarize_runs(runs) for task, runs in by_task.items()}

    disclaimer = generate_pilot_disclaimer(
        total_runs=len(results),
        total_complete_pairs=total_complete_pairs,
        total_partial_pairs=total_partial_pairs,
        num_tasks=len(by_task),
        num_configs=len(by_config),
    )

    return {
        "title": "Phase 3 Experimental Evaluation Report (Paired G/H Comparison)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir) if output_dir is not None else "",
        "total_runs": len(results),
        "total_complete_matched_pairs": total_complete_pairs,
        "total_partial_matched_pairs": total_partial_pairs,
        "summary_by_config": summary_by_config,
        "summary_by_task": summary_by_task,
        "matched_pair_comparisons": pair_comparisons,
        "data_quality_notes": [
            "Provider-reported tokens (prompt, completion, total, reasoning) reflect model API measurements.",
            "Provider-reported cached tokens are recorded as UNKNOWN when not provided by API response metadata.",
            "ACM estimated reduction tokens (~4 chars/token heuristic) reflect message content reduction, NOT provider currency.",
            disclaimer,
        ],
        "detailed_runs": results,
    }


def generate_markdown_phase3_report(report: dict) -> str:
    lines = [
        f"# {report['title']}",
        "",
        f"- **Timestamp**: {report['timestamp']}",
        f"- **Total Evaluated Runs**: {report['total_runs']}",
        f"- **Complete Matched Pairs**: {report.get('total_complete_matched_pairs', 0)}",
    ]
    if report.get("total_partial_matched_pairs", 0) > 0:
        lines.append(f"- **Partial Matched Pairs**: {report['total_partial_matched_pairs']}")
    lines.extend([
        "",
        "## 1. Overall Configuration Comparison (Baseline G vs Compaction H)",
        "",
        "| Configuration | Runs | Resolved Rate | Submitted Rate | Prompt Tokens (Mean ± Std) | Prompt Tokens (Median) | Runtime Mean | ACM Compactions | Est Savings |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for cfg, s in report["summary_by_config"].items():
        mean_p = s.get("prompt_tokens_mean")
        std_p = s.get("prompt_tokens_std")
        mean_str = f"{mean_p:,.1f}" if isinstance(mean_p, float) else f"{mean_p:,}" if mean_p is not None else "N/A"
        if std_p is None:
            prompt_tok_mean_std = f"{mean_str} ± N/A"
        elif isinstance(std_p, float):
            prompt_tok_mean_std = f"{mean_str} ± {std_p:,.1f}"
        else:
            prompt_tok_mean_std = f"{mean_str} ± {std_p:,}"

        median_p = s.get("prompt_tokens_median")
        median_str = f"{median_p:,.1f}" if isinstance(median_p, float) else f"{median_p:,}" if median_p is not None else "N/A"
        runtime_m = s.get("runtime_seconds_mean", 0)
        runtime_str = f"{runtime_m}s" if runtime_m is not None else "N/A"
        acm_comp = s.get("acm_compactions_total", 0)
        acm_est = s.get("acm_estimated_reduction_tokens_total", 0)

        lines.append(
            f"| `{cfg}` | {s.get('total_runs', 0)} | {s.get('resolved_rate_pct', 0)}% | {s.get('submitted_rate_pct', 0)}% | "
            f"{prompt_tok_mean_std} | {median_str} | "
            f"{runtime_str} | {acm_comp} | {acm_est:,} |"
        )

    lines.extend([
        "",
        "## 2. Matched Pair Differences (H - G)",
        "",
        "| Pair ID | Task | Rep | G Prompt Tokens | H Prompt Tokens | Prompt Diff (H - G) | G Resolved | H Resolved | H Compactions | H Est Savings |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for p in report["matched_pair_comparisons"]:
        lines.append(
            f"| `{p['pair_id']}` | `{p['task']}` | {p['rep']} | {p['g_prompt_tokens']:,} | {p['h_prompt_tokens']:,} | "
            f"{p['diff_prompt_tokens']:+,} | {p['g_resolved']} | {p['h_resolved']} | {p['h_compactions']} | {p['h_estimated_savings']:,} |"
        )

    lines.extend([
        "",
        "## 3. Data Quality & Pilot Disclaimers",
        "",
    ])
    for note in report["data_quality_notes"]:
        lines.append(f"- {note}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Evaluation Runner (Paired G/H Benchmark)")
    parser.add_argument("--task", type=str, default=None, help="Optional task filter")
    parser.add_argument("--config", type=str, default=None, help="Optional config filter")
    parser.add_argument("--repetitions", type=int, default=2, help="Repetitions per condition (default: 2)")
    parser.add_argument("--step-limit", type=int, default=15, help="Step limit per agent run (default: 15)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-run wall-clock timeout in seconds (default: 180)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate run checks without invoking agents")
    parser.add_argument("--preflight-only", action="store_true", help="Execute mandatory preflight checks and exit")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    parser.add_argument("--order-strategy", choices=["alternating", "randomized"], default="alternating", help="Execution order strategy")
    parser.add_argument("--resume", action="store_true", help="Resume from previously completed verified runs")
    parser.add_argument("--resume-source", type=Path, default=None, help="Directory to search for prior trajectories")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save run results")
    args = parser.parse_args()

    runner = Phase3EvaluationRunner(
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
        resume=args.resume,
        resume_source=args.resume_source,
        repetitions=args.repetitions,
        step_limit=args.step_limit,
        dry_run=args.dry_run,
        task_filter=args.task,
        config_filter=args.config,
        order_strategy=args.order_strategy,
        skip_preflight=args.skip_preflight,
    )

    if args.preflight_only:
        valid = runner.run_preflight()
        sys.exit(0 if valid else 1)

    runner.execute_suite()


if __name__ == "__main__":
    main()
