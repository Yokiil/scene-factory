"""Reject incompatible diagnostic options before allocating simulator resources."""
from __future__ import annotations

import sys

import pytest

from tools import run_precision_assembly


@pytest.mark.parametrize("options", [
    ["--prelift-grip-fraction", ".22"],
    ["--grasp-control", "timed", "--grasp-only"],
    ["--grasp-control", "timed", "--grasp-config", "unused.json"],
    ["--grasp-control", "timed", "--grasp-contact-details"],
])
def test_incompatible_grasp_options_fail_before_creating_trial(monkeypatch, tmp_path, options):
    output = tmp_path / "should_not_exist"
    monkeypatch.setattr(sys, "argv", ["run_precision_assembly.py", "unused_fixture",
        "--output", str(output), "--asset-root", "unused_assets", *options])
    with pytest.raises(SystemExit) as rejected:
        run_precision_assembly.main()
    assert rejected.value.code == 2
    assert not output.exists()
