# Phase 2 Ablation Study — Reprocessed Trajectory Evaluation Report

- **Generated:** 2026-10-11T20:26:14.084963+00:00
- **Source Trajectories Directory:** `experiments/phase2_test_feedback/runs/run_20261011_183454`
- **Total Trajectories Evaluated:** 60

## Overall Executive Summary

| Metric | Count | Rate (%) | Description / Notes |
| :--- | :---: | :---: | :--- |
| **Total Experiments** | 60 | 100.0% | Complete Phase 2 ablation matrix (3 tasks x 4 configs x 5 reps) |
| **Process Completed** | 58 | 96.7% | Subprocess exited normally without timeout or crash |
| **Timed Out** | 2 | 3.3% | Wall-clock limit exceeded (180s watchdog killed child process group) |
| **Submitted by Agent** | 28 | 46.7% | Explicitly issued submission command (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) |
| **Observed Passing Test in Trajectory** | 42 | 70.0% | At least one passing pytest execution witnessed in trajectory tool observations |
| **Independent External Correctness (True)** | 23 | 57.5% (23/40 verified) | Verified externally on disk sandbox (**excludes UNKNOWN from denominator**) |
| **Independent External Correctness (False)** | 17 | 42.5% (17/40 verified) | External test evaluation failed on disk sandbox (**excludes UNKNOWN**) |
| **Independent External Correctness (UNKNOWN)** | 20 | 33.3% (20/60 unverified) | Historical Task 1 runs resumed without retained disk state |
| **Total Failed Test Executions** | 60 | — | pytest/unittest runs with non-zero exitcode |
| **Total Passed Test Executions** | 77 | — | pytest/unittest runs with exitcode 0 |
| **Mean Runtime (Seconds)** | 55.96s | — | Wall-clock duration recovered from manifest diagnostics & trajectory timestamps |

## Results by Configuration

| Configuration | Runs | Process Done | Submitted | Observed Pass | Indep. Pass (Verified Only) | Mean Runtime (s) | Mean Steps | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_baseline` | 15 | 100.0% | 46.7% | 66.7% | 60.0% (6/10) | 54.26s | 9.33 | 40964 |
| `B_loop_detection_only` | 15 | 100.0% | 53.3% | 66.7% | 50.0% (5/10) | 46.56s | 9.27 | 38653 |
| `C_test_feedback_only` | 15 | 93.3% | 40.0% | 66.7% | 50.0% (5/10) | 64.07s | 8.67 | 36539 |
| `D_combined` | 15 | 93.3% | 46.7% | 80.0% | 70.0% (7/10) | 58.94s | 9.27 | 40751 |

## Results by Task

| Task | Runs | Process Done | Submitted | Observed Pass | Indep. Pass (Verified Only) | Mean Failed Tests | Mean Passed Tests | Mean Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `task1_simple_assertion` | 20 | 100.0% | 100.0% | 100.0% | UNKNOWN | 1.0 | 1.9 | 16.13s |
| `task2_multi_failure` | 20 | 90.0% | 5.0% | 10.0% | 15.0% (3/20) | 0.95 | 0.15 | 87.54s |
| `task3_input_validation` | 20 | 100.0% | 35.0% | 100.0% | 100.0% (20/20) | 1.05 | 1.8 | 64.22s |

## Data Quality Limitations

- Historical runs 1-20 (Task 1: 20/20 runs) were resumed from an archived session where post-run sandbox disk state was not retained. Independent patch correctness is strictly marked as UNKNOWN, despite observed passing pytest runs in trajectories.
- Runs 21-60 were independently verified on disk immediately post-run. On Task 2, 3/20 runs achieved independent patch correctness (17/20 failed, 18/20 hit step limits, and 2/20 timed out). On Task 3, all 20/20 runs achieved observed passing tests in trajectory and 20/20 achieved independent patch correctness, but 13/20 runs exhausted step limit before submitting (submission rate: 7/20).
- Loop detection metrics were audited: all feature flags match configuration specifications (Config B and D have loop detection enabled). Across all 60 runs, 0 loop warnings were triggered because the agent never reached the repetition threshold of 3 consecutive identical actions (max consecutive identical actions observed was 1).
- Independent correctness rates strictly exclude UNKNOWN outcomes from their denominator (Task 1 is UNKNOWN: 0/0 verified; overall rate of verified runs is 23/40 = 57.5%).
- Observation return codes are decoded from JSON string content '{"returncode": ...}' with fallback to message extra metadata.

## Detailed Run Breakdown

| Run (Task / Config / Rep) | Process Done | Exit Status | Submitted | Observed Pass | Independent Resolved | Timed Out | Failed Tests | Passed Tests | Runtime (s) | Steps | Tokens |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `task1_simple_assertion` / `A_baseline` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 19.79s | 9 | 22107 |
| `task1_simple_assertion` / `A_baseline` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 18.63s | 8 | 18781 |
| `task1_simple_assertion` / `A_baseline` / rep `3` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.67s | 8 | 18572 |
| `task1_simple_assertion` / `A_baseline` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.0s | 8 | 18355 |
| `task1_simple_assertion` / `A_baseline` / rep `5` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.25s | 8 | 19750 |
| `task1_simple_assertion` / `B_loop_detection_only` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.98s | 8 | 19140 |
| `task1_simple_assertion` / `B_loop_detection_only` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.6s | 8 | 19296 |
| `task1_simple_assertion` / `B_loop_detection_only` / rep `3` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.29s | 8 | 19403 |
| `task1_simple_assertion` / `B_loop_detection_only` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 15.33s | 8 | 25618 |
| `task1_simple_assertion` / `B_loop_detection_only` / rep `5` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.82s | 8 | 18653 |
| `task1_simple_assertion` / `C_test_feedback_only` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 14.1s | 8 | 20404 |
| `task1_simple_assertion` / `C_test_feedback_only` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 1 | 11.51s | 7 | 16338 |
| `task1_simple_assertion` / `C_test_feedback_only` / rep `3` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 19.98s | 10 | 25026 |
| `task1_simple_assertion` / `C_test_feedback_only` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 16.05s | 8 | 20265 |
| `task1_simple_assertion` / `C_test_feedback_only` / rep `5` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 13.59s | 8 | 19144 |
| `task1_simple_assertion` / `D_combined` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 30.64s | 8 | 19467 |
| `task1_simple_assertion` / `D_combined` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 16.66s | 8 | 19939 |
| `task1_simple_assertion` / `D_combined` / rep `3` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 15.27s | 8 | 19707 |
| `task1_simple_assertion` / `D_combined` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 1 | 13.88s | 8 | 19936 |
| `task1_simple_assertion` / `D_combined` / rep `5` | ✓ | `Submitted` | ✓ | ✓ | `UNKNOWN` | ✗ | 1 | 2 | 15.48s | 8 | 20041 |
| `task2_multi_failure` / `A_baseline` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✗ | `True` | ✗ | 1 | 0 | 168.56s | 10 | 52581 |
| `task2_multi_failure` / `A_baseline` / rep `2` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 62.91s | 10 | 47815 |
| `task2_multi_failure` / `A_baseline` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 60.93s | 10 | 59602 |
| `task2_multi_failure` / `A_baseline` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 66.31s | 10 | 52264 |
| `task2_multi_failure` / `A_baseline` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 69.5s | 10 | 49191 |
| `task2_multi_failure` / `B_loop_detection_only` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 69.43s | 10 | 48154 |
| `task2_multi_failure` / `B_loop_detection_only` / rep `2` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 53.34s | 10 | 61454 |
| `task2_multi_failure` / `B_loop_detection_only` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 51.92s | 10 | 68917 |
| `task2_multi_failure` / `B_loop_detection_only` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 46.9s | 10 | 36521 |
| `task2_multi_failure` / `B_loop_detection_only` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 80.37s | 10 | 51690 |
| `task2_multi_failure` / `C_test_feedback_only` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 99.3s | 10 | 58320 |
| `task2_multi_failure` / `C_test_feedback_only` / rep `2` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 86.32s | 10 | 51004 |
| `task2_multi_failure` / `C_test_feedback_only` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 71.87s | 10 | 54248 |
| `task2_multi_failure` / `C_test_feedback_only` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 113.5s | 10 | 46863 |
| `task2_multi_failure` / `C_test_feedback_only` / rep `5` | ✗ | `TIMEOUT` | ✗ | ✗ | `False` | ✓ | 0 | 0 | 180.12s | 0 | 0 |
| `task2_multi_failure` / `D_combined` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 51.86s | 10 | 49139 |
| `task2_multi_failure` / `D_combined` / rep `2` | ✗ | `LimitsExceeded` | ✗ | ✗ | `False` | ✓ | 1 | 0 | 180.03s | 10 | 50508 |
| `task2_multi_failure` / `D_combined` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 92.39s | 10 | 65277 |
| `task2_multi_failure` / `D_combined` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 1 | 89.53s | 10 | 58014 |
| `task2_multi_failure` / `D_combined` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✗ | `False` | ✗ | 1 | 0 | 55.62s | 10 | 49544 |
| `task3_input_validation` / `A_baseline` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 54.74s | 10 | 49991 |
| `task3_input_validation` / `A_baseline` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 50.97s | 10 | 55679 |
| `task3_input_validation` / `A_baseline` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 1 | 92.99s | 10 | 67287 |
| `task3_input_validation` / `A_baseline` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 51.01s | 9 | 33807 |
| `task3_input_validation` / `A_baseline` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 54.71s | 10 | 48682 |
| `task3_input_validation` / `B_loop_detection_only` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 1 | 48.55s | 10 | 36795 |
| `task3_input_validation` / `B_loop_detection_only` / rep `2` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 57.39s | 9 | 37893 |
| `task3_input_validation` / `B_loop_detection_only` / rep `3` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 58.3s | 10 | 45815 |
| `task3_input_validation` / `B_loop_detection_only` / rep `4` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 56.73s | 10 | 43701 |
| `task3_input_validation` / `B_loop_detection_only` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 101.47s | 10 | 46758 |
| `task3_input_validation` / `C_test_feedback_only` / rep `1` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 46.17s | 9 | 39413 |
| `task3_input_validation` / `C_test_feedback_only` / rep `2` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 1 | 48.68s | 10 | 41585 |
| `task3_input_validation` / `C_test_feedback_only` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 73.97s | 10 | 44563 |
| `task3_input_validation` / `C_test_feedback_only` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 73.62s | 10 | 64287 |
| `task3_input_validation` / `C_test_feedback_only` / rep `5` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 92.34s | 10 | 46635 |
| `task3_input_validation` / `D_combined` / rep `1` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 2 | 2 | 67.74s | 10 | 44551 |
| `task3_input_validation` / `D_combined` / rep `2` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 62.66s | 10 | 45308 |
| `task3_input_validation` / `D_combined` / rep `3` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 1 | 74.48s | 10 | 58096 |
| `task3_input_validation` / `D_combined` / rep `4` | ✓ | `LimitsExceeded` | ✗ | ✓ | `True` | ✗ | 1 | 2 | 60.81s | 10 | 51965 |
| `task3_input_validation` / `D_combined` / rep `5` | ✓ | `Submitted` | ✓ | ✓ | `True` | ✗ | 1 | 2 | 56.98s | 9 | 39785 |
