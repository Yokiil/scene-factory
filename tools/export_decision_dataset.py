"""Export measured rollouts to strict, provenance-preserving behavior labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.agent_adapter import RegisteredSkill
from scene_factory.decision_dataset import export_decision_dataset, verify_decision_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_decision_dataset(args.output)))
        return 0
    if args.config is None:
        parser.error("--config is required for export")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"sources", "skills", "split_seed"}:
        raise ValueError("config requires exactly sources, skills and split_seed")
    sources = []
    for source in config["sources"]:
        path = Path(source["path"])
        sources.append({**source, "path": path if path.is_absolute() else args.config.resolve().parent / path})
    skills = []
    for skill in config["skills"]:
        if not isinstance(skill, dict) or set(skill) != {"skill_id", "description", "action"}:
            raise ValueError("skill requires exactly skill_id, description and action")
        skills.append(RegisteredSkill(**skill))
    manifest = export_decision_dataset(sources, args.output, skills=skills, split_seed=config["split_seed"])
    result = verify_decision_dataset(args.output)
    print(json.dumps({"sample_count": manifest["sample_count"], "episode_count": manifest["episode_count"],
                      "training_performed": manifest["training_performed"], "verification": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
