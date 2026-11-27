"""Read-only health probe; never prints the authentication token."""
from __future__ import annotations

import json
import os
import urllib.request


def main() -> int:
    token = os.environ.get("SCENE_FACTORY_SERVICE_TOKEN")
    headers = {"Authorization": "Bearer " + token} if token else {}
    request = urllib.request.Request("http://127.0.0.1:8765/healthz", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            result = json.load(response)
        return 0 if result.get("status") == "ok" else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
