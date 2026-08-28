from __future__ import annotations

import json
from pathlib import Path

from .config import SecretaryConfig
from .engine import SecretaryEngine
from .memory.store import MemoryStore


def run_benchmark(config: SecretaryConfig | None = None, output=None) -> int:
    root = (config or SecretaryConfig.from_environment()).project_root
    scenario_root = root / "scenarios"
    if not (scenario_root / "benchmark.json").is_file():
        scenario_root = Path(__file__).resolve().parents[2] / "scenarios"
    manifest_path = scenario_root / "benchmark.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for case in manifest.get("scenarios", []):
        name = str(case["name"])
        path = scenario_root / str(case["file"])
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
        engine = SecretaryEngine(config or SecretaryConfig.from_environment(root), store=MemoryStore(":memory:"))
        try:
            results = [engine.process(item) for item in items]
            actions = [result.decision.action.value for result in results]
            forbidden = set(case.get("forbidden", []))
            if forbidden.intersection(actions):
                failures.append(f"{name}: forbidden action {sorted(forbidden.intersection(actions))}")
            expected_any = set(case.get("expected_any", []))
            if expected_any and not expected_any.intersection(actions):
                failures.append(f"{name}: expected one of {sorted(expected_any)}, got {actions}")
            if case.get("expect_privacy") and not any(result.privacy_suppressed for result in results):
                failures.append(f"{name}: privacy event was not suppressed")
        finally:
            engine.close()
    target = output or __import__("sys").stdout
    if failures:
        print("CPU benchmark: FAIL", file=target)
        for failure in failures:
            print(f"- {failure}", file=target)
        return 1
    print(f"CPU benchmark: PASS ({len(manifest.get('scenarios', []))} scenarios)", file=target)
    return 0
