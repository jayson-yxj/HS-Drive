import ast
from pathlib import Path
import queue
from simulator.m5m_shadow import latest_put


def test_latest_queue_replaces_stale_work_without_blocking():
    q = queue.Queue(1)
    assert latest_put(q, 1)
    assert latest_put(q, 2)
    assert q.get_nowait() == 2


def test_worker_has_no_simulator_or_control_access():
    tree = ast.parse(Path("simulator/m5m_shadow.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name != "carla" for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "carla"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in [
                "tick",
                "apply_control",
                "load_world",
                "get_world",
            ]
    text = Path("simulator/m5m_shadow.py").read_text()
    assert "age > 0.25" in text and "total <= 0.25" in text
