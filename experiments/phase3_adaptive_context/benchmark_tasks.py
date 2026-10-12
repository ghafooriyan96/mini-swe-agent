"""Benchmark Tasks for Phase 3 Experimental Evaluation.

Defines three deterministic, offline, self-contained software repair tasks:
- Task E1 (task_e1_multifile): Multi-file dataflow bug across order, inventory, and pricing modules.
- Task E2 (task_e2_multifailure): Multiple realistic failing test cases with distinct failure modes.
- Task E3 (task_e3_verbose_investigation): Large-context repository with time-series aggregation boundary bug.

Each task provides:
- setup_fn(workspace_dir): Creates clean workspace with bug.
- reference_fix_fn(workspace_dir): Applies valid reference repair.
- test_file: Test file to run via pytest.
- prompt: Task prompt for agent.
"""

from __future__ import annotations

from pathlib import Path


def cleanup_workspace(workspace_dir: Path) -> None:
    for item in workspace_dir.glob("*.py"):
        item.unlink()


# ---------------------------------------------------------------------------
# Task E1: Multi-file bug repair (task_e1_multifile)
# ---------------------------------------------------------------------------

def setup_task_e1(workspace_dir: Path) -> None:
    cleanup_workspace(workspace_dir)

    (workspace_dir / "inventory.py").write_text(
        '"""Inventory management module."""\n\n'
        "class InventoryManager:\n"
        "    def __init__(self, stock: dict[str, int]):\n"
        "        self.stock = dict(stock)\n\n"
        "    def check_availability(self, item_id: str, quantity: int) -> bool:\n"
        "        return self.stock.get(item_id, 0) >= quantity\n\n"
        "    def deduct_stock(self, item_id: str, quantity: int) -> None:\n"
        "        if not isinstance(quantity, int) or quantity < 0:\n"
        '            raise ValueError(f"Quantity must be a positive integer, got {quantity}")\n'
        "        available = self.stock.get(item_id, 0)\n"
        "        if available < quantity:\n"
        '            raise RuntimeError(f"Insufficient stock for {item_id}: requested {quantity}, available {available}")\n'
        "        self.stock[item_id] = available - quantity\n"
    )

    (workspace_dir / "pricing.py").write_text(
        '"""Pricing and discount calculation module."""\n\n'
        "def calculate_unit_price(base_price: float, discount_percent: float) -> float:\n"
        "    if discount_percent < 0 or discount_percent > 100:\n"
        '        raise ValueError("Discount percent must be between 0 and 100")\n'
        "    return round(base_price * (1.0 - discount_percent / 100.0), 2)\n\n"
        "def apply_tax(amount: float, tax_rate: float) -> float:\n"
        "    if tax_rate < 0:\n"
        '        raise ValueError("Tax rate cannot be negative")\n'
        "    return round(amount * (1.0 + tax_rate), 2)\n"
    )

    (workspace_dir / "order_service.py").write_text(
        '"""Order processing and coordination service."""\n'
        "from inventory import InventoryManager\n"
        "from pricing import calculate_unit_price, apply_tax\n\n"
        "class OrderService:\n"
        "    def __init__(self, inventory: InventoryManager, tax_rate: float = 0.05):\n"
        "        self.inventory = inventory\n"
        "        self.tax_rate = tax_rate\n\n"
        "    def process_order(self, order: dict) -> dict:\n"
        '        item_id = order["item_id"]\n'
        '        quantity = order["quantity"]\n'
        '        base_price = order["base_price"]\n'
        '        discount = order.get("discount_percent", 0.0)\n\n'
        "        if not self.inventory.check_availability(item_id, quantity):\n"
        '            raise RuntimeError(f"Item {item_id} unavailable")\n\n'
        "        unit_price = calculate_unit_price(base_price, discount)\n"
        "        subtotal = round(unit_price * quantity, 2)\n"
        "        total_price = apply_tax(subtotal, self.tax_rate)\n\n"
        "        # Bug: passing total_price instead of quantity to deduct_stock\n"
        "        self.inventory.deduct_stock(item_id, total_price)\n\n"
        "        return {\n"
        '            "item_id": item_id,\n'
        '            "quantity": quantity,\n'
        '            "total_price": total_price,\n'
        '            "status": "CONFIRMED",\n'
        "        }\n"
    )

    (workspace_dir / "test_order_pipeline.py").write_text(
        "import pytest\n"
        "from inventory import InventoryManager\n"
        "from order_service import OrderService\n\n"
        "def test_order_stock_deduction_and_pricing():\n"
        '    inv = InventoryManager({"widget_a": 10, "gadget_b": 20})\n'
        "    service = OrderService(inv, tax_rate=0.10)\n"
        '    order = {"item_id": "widget_a", "quantity": 4, "base_price": 50.0, "discount_percent": 10.0}\n'
        "    result = service.process_order(order)\n"
        '    assert result["status"] == "CONFIRMED"\n'
        '    assert result["total_price"] == 198.0\n'
        '    assert inv.stock["widget_a"] == 6\n\n'
        "def test_insufficient_stock_prevents_order():\n"
        '    inv = InventoryManager({"widget_a": 2})\n'
        "    service = OrderService(inv)\n"
        "    with pytest.raises(RuntimeError, match='unavailable'):\n"
        '        service.process_order({"item_id": "widget_a", "quantity": 5, "base_price": 10.0})\n'
    )


def reference_fix_task_e1(workspace_dir: Path) -> None:
    order_file = workspace_dir / "order_service.py"
    content = order_file.read_text()
    fixed_content = content.replace(
        "self.inventory.deduct_stock(item_id, total_price)",
        "self.inventory.deduct_stock(item_id, quantity)",
    )
    order_file.write_text(fixed_content)


# ---------------------------------------------------------------------------
# Task E2: Multiple failing tests (task_e2_multifailure)
# ---------------------------------------------------------------------------

def setup_task_e2(workspace_dir: Path) -> None:
    cleanup_workspace(workspace_dir)

    (workspace_dir / "record_processor.py").write_text(
        '"""Data normalization and aggregation module."""\n\n'
        "def normalize_records(raw_records: list[dict]) -> list[dict]:\n"
        "    normalized = []\n"
        "    for r in raw_records:\n"
        "        item = {}\n"
        "        for k, v in r.items():\n"
        "            key = k.strip().lower()\n"
        "            # Bug A: calling .strip() without checking string type\n"
        "            item[key] = v.strip()\n"
        "        normalized.append(item)\n"
        "    return normalized\n\n"
        "def aggregate_scores_by_department(records: list[dict]) -> dict[str, dict]:\n"
        "    groups: dict[str, list[float]] = {}\n"
        "    for r in records:\n"
        '        dept = r.get("department", "unknown")\n'
        '        groups.setdefault(dept, [])\n'
        '        if "score" in r and r["score"] is not None:\n'
        '            groups[dept].append(float(r["score"]))\n\n'
        "    summary = {}\n"
        "    for dept, scores in groups.items():\n"
        "        # Bug B: ZeroDivisionError if department has no scores\n"
        "        summary[dept] = {\n"
        '            "count": len(scores),\n'
        '            "total": sum(scores),\n'
        '            "average": round(sum(scores) / len(scores), 2),\n'
        "        }\n"
        "    return summary\n"
    )

    (workspace_dir / "report_formatter.py").write_text(
        '"""Report presentation and table formatting module."""\n\n'
        "def format_summary_table(summary: dict[str, dict]) -> list[str]:\n"
        '    lines = ["DEPT | COUNT | AVG"]\n'
        "    for dept in sorted(summary.keys()):\n"
        "        data = summary[dept]\n"
        "        # Bug C: uppercase key lookup causing KeyError\n"
        "        lines.append(f\"{dept} | {data['COUNT']} | {data['average']:.2f}\")\n"
        "    return lines\n"
    )

    (workspace_dir / "test_report_pipeline.py").write_text(
        "import pytest\n"
        "from record_processor import normalize_records, aggregate_scores_by_department\n"
        "from report_formatter import format_summary_table\n\n"
        "def test_normalize_mixed_types():\n"
        '    raw = [{" name ": " Alice ", " AGE ": 30, " Score ": 95.5}]\n'
        "    res = normalize_records(raw)\n"
        '    assert res == [{"name": "Alice", "age": 30, "score": 95.5}]\n\n'
        "def test_aggregate_scores_with_empty_group():\n"
        '    records = [{"department": "eng", "score": 90.0}, {"department": "hr", "score": None}]\n'
        "    summary = aggregate_scores_by_department(records)\n"
        '    assert summary["eng"]["average"] == 90.0\n'
        '    assert summary["hr"]["count"] == 0\n'
        '    assert summary["hr"]["average"] == 0.0\n\n'
        "def test_format_summary_table():\n"
        "    summary = {\n"
        '        "eng": {"count": 2, "total": 180.0, "average": 90.0},\n'
        '        "sales": {"count": 1, "total": 85.0, "average": 85.0},\n'
        "    }\n"
        "    lines = format_summary_table(summary)\n"
        "    assert len(lines) == 3\n"
        '    assert "eng | 2 | 90.00" in lines\n'
        '    assert "sales | 1 | 85.00" in lines\n'
    )


def reference_fix_task_e2(workspace_dir: Path) -> None:
    rec_file = workspace_dir / "record_processor.py"
    rec_content = rec_file.read_text()
    rec_fixed = rec_content.replace(
        "item[key] = v.strip()",
        "item[key] = v.strip() if isinstance(v, str) else v",
    ).replace(
        '            "average": round(sum(scores) / len(scores), 2),\n',
        '            "average": round(sum(scores) / len(scores), 2) if scores else 0.0,\n',
    )
    rec_file.write_text(rec_fixed)

    rep_file = workspace_dir / "report_formatter.py"
    rep_content = rep_file.read_text()
    rep_fixed = rep_content.replace(
        "lines.append(f\"{dept} | {data['COUNT']} | {data['average']:.2f}\")",
        "lines.append(f\"{dept} | {data['count']} | {data['average']:.2f}\")",
    )
    rep_file.write_text(rep_fixed)


# ---------------------------------------------------------------------------
# Task E3: Verbose repository investigation (task_e3_verbose_investigation)
# ---------------------------------------------------------------------------

def setup_task_e3(workspace_dir: Path) -> None:
    cleanup_workspace(workspace_dir)

    schema_lines = [
        '"""Timeseries data schema and metric catalog."""',
        "from dataclasses import dataclass",
        "from typing import Optional",
        "",
        "@dataclass",
        "class DataPoint:",
        "    timestamp: int",
        "    metric: str",
        "    value: float",
        "    tag: str = 'default'",
        "",
        "VALID_METRICS = {'cpu_usage', 'memory_rss', 'network_rx', 'disk_write'}",
        "",
        "def validate_metric_name(metric: str) -> bool:",
        "    return metric in VALID_METRICS",
        "",
    ]
    for i in range(1, 35):
        schema_lines.append(
            f"def schema_helper_{i}(val: float) -> float:\n"
            f'    """Validation rule {i}."""\n'
            f"    return val * {1.0 + (i * 0.01):.2f}\n"
        )
    (workspace_dir / "timeseries_schema.py").write_text("\n".join(schema_lines))

    agg_lines = [
        '"""Window aggregation library for streaming timeseries data."""',
        "from timeseries_schema import DataPoint, validate_metric_name",
        "",
        "class TumblingWindowAggregator:",
        "    def __init__(self, window_size_seconds: int = 10):",
        "        if window_size_seconds <= 0:",
        '            raise ValueError("window_size_seconds must be positive")',
        "        self.window_size_seconds = window_size_seconds",
        "",
        "    def partition_into_windows(self, points: list[DataPoint]) -> dict[int, list[float]]:",
        "        windows: dict[int, list[float]] = {}",
        "        for p in points:",
        "            if not validate_metric_name(p.metric):",
        "                continue",
        "            # Boundary Bug: Exact boundary timestamps assigned to next window",
        "            if p.timestamp % self.window_size_seconds == 0:",
        "                window_id = (p.timestamp // self.window_size_seconds) + 1",
        "            else:",
        "                window_id = p.timestamp // self.window_size_seconds",
        "            windows.setdefault(window_id, []).append(p.value)",
        "        return windows",
        "",
        "    def calculate_window_means(self, points: list[DataPoint]) -> dict[int, float]:",
        "        partitions = self.partition_into_windows(points)",
        "        return {w_id: round(sum(vals) / len(vals), 2) for w_id, vals in partitions.items() if vals}",
        "",
    ]
    for i in range(1, 35):
        agg_lines.append(
            f"def aggregation_utility_{i}(data: list[float]) -> float:\n"
            f'    """Aggregation utility routine {i}.\"\"\"\n'
            f"    return sum(data) / len(data) if data else 0.0\n"
        )
    (workspace_dir / "window_aggregator.py").write_text("\n".join(agg_lines))

    (workspace_dir / "test_rolling_window.py").write_text(
        "from timeseries_schema import DataPoint\n"
        "from window_aggregator import TumblingWindowAggregator\n\n"
        "def test_tumbling_window_boundary_alignment():\n"
        "    agg = TumblingWindowAggregator(window_size_seconds=10)\n"
        "    points = [\n"
        "        DataPoint(timestamp=0, metric='cpu_usage', value=10.0),\n"
        "        DataPoint(timestamp=5, metric='cpu_usage', value=20.0),\n"
        "        DataPoint(timestamp=10, metric='cpu_usage', value=30.0),\n"
        "        DataPoint(timestamp=15, metric='cpu_usage', value=40.0),\n"
        "    ]\n"
        "    windows = agg.partition_into_windows(points)\n"
        "    # Window 0 should contain timestamps 0 and 5\n"
        "    assert 0 in windows\n"
        "    assert windows[0] == [10.0, 20.0]\n"
        "    # Window 1 should contain timestamps 10 and 15\n"
        "    assert 1 in windows\n"
        "    assert windows[1] == [30.0, 40.0]\n\n"
        "def test_window_means_accuracy():\n"
        "    agg = TumblingWindowAggregator(window_size_seconds=10)\n"
        "    points = [\n"
        "        DataPoint(timestamp=0, metric='cpu_usage', value=10.0),\n"
        "        DataPoint(timestamp=5, metric='cpu_usage', value=20.0),\n"
        "        DataPoint(timestamp=10, metric='cpu_usage', value=30.0),\n"
        "    ]\n"
        "    means = agg.calculate_window_means(points)\n"
        "    assert means[0] == 15.0\n"
        "    assert means[1] == 30.0\n"
    )


def reference_fix_task_e3(workspace_dir: Path) -> None:
    agg_file = workspace_dir / "window_aggregator.py"
    content = agg_file.read_text()
    old_code = (
        "            if p.timestamp % self.window_size_seconds == 0:\n"
        "                window_id = (p.timestamp // self.window_size_seconds) + 1\n"
        "            else:\n"
        "                window_id = p.timestamp // self.window_size_seconds"
    )
    new_code = "            window_id = p.timestamp // self.window_size_seconds"
    fixed_content = content.replace(old_code, new_code)
    agg_file.write_text(fixed_content)


# ---------------------------------------------------------------------------
# Task Registry for Phase 3
# ---------------------------------------------------------------------------

PHASE3_BENCHMARK_TASKS = {
    "task_e1_multifile": {
        "setup_fn": setup_task_e1,
        "reference_fix_fn": reference_fix_task_e1,
        "test_file": "test_order_pipeline.py",
        "prompt": (
            "There is a data flow bug across order_service.py, inventory.py, and pricing.py causing test_order_pipeline.py to fail. "
            "Please run pytest test_order_pipeline.py to inspect the failure, trace the data flow across the modules, "
            "fix the bug in order_service.py, verify by re-running pytest test_order_pipeline.py, and submit your work."
        ),
    },
    "task_e2_multifailure": {
        "setup_fn": setup_task_e2,
        "reference_fix_fn": reference_fix_task_e2,
        "test_file": "test_report_pipeline.py",
        "prompt": (
            "Multiple tests are failing in test_report_pipeline.py across record_processor.py and report_formatter.py. "
            "Please run pytest test_report_pipeline.py to inspect the errors, fix the functions in record_processor.py and report_formatter.py, "
            "verify by re-running pytest test_report_pipeline.py, and submit your work."
        ),
    },
    "task_e3_verbose_investigation": {
        "setup_fn": setup_task_e3,
        "reference_fix_fn": reference_fix_task_e3,
        "test_file": "test_rolling_window.py",
        "prompt": (
            "There is a window boundary alignment bug in window_aggregator.py causing test_rolling_window.py to fail. "
            "Please run pytest test_rolling_window.py to analyze the failure, inspect timeseries_schema.py and window_aggregator.py, "
            "fix the boundary calculation in window_aggregator.py, verify by re-running pytest test_rolling_window.py, and submit your work."
        ),
    },
}
