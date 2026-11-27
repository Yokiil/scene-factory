"""Run the local source release checks and bind results to the checked sources.

This does not start Isaac or certify Linux GPU deployment, model training, or
physical task success. Those require their separate measured acceptance reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def source_hashes() -> dict[str, str]:
    paths = [ROOT / "pyproject.toml"]
    for folder in ("scene_factory", "tests", "tools"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "schemas").glob("*.json"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError("use a new output directory to retain earlier checks")
    before = source_hashes()
    checks = [
        ("ruff", [sys.executable, "-m", "ruff", "check", "scene_factory", "tests", "tools"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "--tb=short"]),
        ("release", [sys.executable, "tools/check_release.py"]),
    ]
    report = {"schema_version": "scene_factory.source_release_checks.v1", "status": "running",
              "scope": "local_source_checks_not_GPU_task_or_Linux_acceptance",
              "source_sha256": before, "checks": []}

    def save():
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        temporary.replace(report_path)

    save()
    for name, command in checks:
        log_path = output / (name+".txt")
        started = time.perf_counter()
        with log_path.open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry = {"name": name, "command": command, "exit_code": result.returncode,
                 "elapsed_s": time.perf_counter()-started, "log": log_path.name,
                 "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}
        report["checks"].append(entry)
        save()
        print(json.dumps({"name": name, "exit_code": result.returncode,
                          "elapsed_s": entry["elapsed_s"]}), flush=True)
    after = source_hashes()
    report["source_unchanged"] = after == before
    report["changed_sources"] = sorted(name for name in before.keys() | after.keys()
                                      if before.get(name) != after.get(name))
    report["status"] = "passed" if after == before and all(
        item["exit_code"] == 0 for item in report["checks"]) else "failed"
    save()
    print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
