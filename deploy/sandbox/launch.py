"""Container entry point; enforce explicit shared-service token configuration."""
from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("SCENE_FACTORY_SERVICE_TOKEN", "").strip():
        print("SCENE_FACTORY_SERVICE_TOKEN must be configured before starting the container service", file=sys.stderr)
        return 2
    arguments = sys.argv[1:]
    pool = os.environ.get("SCENE_FACTORY_MUJOCO_POOL", "0")
    if pool not in {"0", "1"}:
        raise ValueError("SCENE_FACTORY_MUJOCO_POOL must be 0 or 1")
    if pool == "1" and "--mujoco-worker-pool" not in arguments:
        arguments += ["--mujoco-worker-pool", "--pool-max-idle", os.environ.get("SCENE_FACTORY_POOL_MAX_IDLE", "1"),
                      "--pool-idle-seconds", os.environ.get("SCENE_FACTORY_POOL_IDLE_SECONDS", "60"),
                      "--pool-max-uses", os.environ.get("SCENE_FACTORY_POOL_MAX_USES", "100")]
    os.execv(sys.executable, [sys.executable, "-m", "scene_factory.cli", *arguments])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
