#!/usr/bin/env python
"""Re-run an experiment from a saved ``config_used.json`` alone (criterion I8).

Brief S18 and falsification question 13 require that a result be reproducible from its saved
configuration.  This script reads ``config_used.json``, reports every provenance difference
between then and now, and optionally re-executes the arm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaun.config import resolve
from amaun.utils.provenance import collect_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="a results/<run_mode>/<experiment_id> directory")
    parser.add_argument("--execute", action="store_true", help="actually re-run the arm")
    parser.add_argument("--strategy", default="amaun")
    parser.add_argument("--dev-tasks", nargs="*", default=None)
    parser.add_argument("--out-suffix", default="-repro")
    args = parser.parse_args()

    directory = Path(args.results_dir)
    config_path = directory / "config_used.json"
    if not config_path.exists():
        print(f"no config_used.json in {directory}")
        return 1

    saved = json.loads(config_path.read_text())
    print(f"experiment    {saved['experiment_id']}")
    print(f"run_mode      {saved['run_mode']} (reportable={saved['is_reportable']})")
    print(f"seed          {saved['seed']}")
    print(f"config hash   {saved['config_hash']}")
    print(f"backbone      {saved['backbone']['model_id']}")

    # Verify the config still resolves to the same hash under this code version.
    rebuilt = resolve(
        saved["config"], run_mode=saved["run_mode"],
        experiment_id=saved["experiment_id"], seed=saved["seed"],
        device=saved.get("device"),
    )
    match = rebuilt.config_hash == saved["config_hash"]
    print(f"\nconfig re-resolves to the same hash: {match}")
    if not match:
        print(f"  saved {saved['config_hash']}")
        print(f"  now   {rebuilt.config_hash}")
        print("  The config schema or a run-mode overlay changed since this run.")

    # Environment differences.
    then = saved.get("provenance", {})
    now = collect_provenance()
    print("\nenvironment differences (then -> now):")
    differences = 0
    for package, version in (then.get("packages") or {}).items():
        current = (now["packages"] or {}).get(package)
        if version != current:
            print(f"  {package:16s} {version} -> {current}")
            differences += 1
    then_git, now_git = then.get("git") or {}, now["git"]
    if then_git.get("commit") != now_git.get("commit"):
        print(f"  git commit       {then_git.get('commit')} -> {now_git.get('commit')}")
        differences += 1
    if then_git.get("diff_sha256") != now_git.get("diff_sha256"):
        print(f"  uncommitted diff {str(then_git.get('diff_sha256'))[:12]} -> "
              f"{str(now_git.get('diff_sha256'))[:12]}")
        differences += 1
    for key in ("platform", "machine"):
        if (then.get("hardware") or {}).get(key) != now["hardware"].get(key):
            print(f"  {key:16s} {(then.get('hardware') or {}).get(key)} -> {now['hardware'].get(key)}")
            differences += 1
    if differences == 0:
        print("  none -- the environment is identical")

    if then_git.get("dirty"):
        print("\nWARNING: the original run had an unclean git tree. Exact reproduction requires")
        print("         the same uncommitted changes (diff hash above).")

    if not args.execute:
        print("\nPass --execute to re-run. The command would be:")

    config_file = directory / "reproduce_config.yaml"
    import yaml

    config_file.write_text(yaml.safe_dump(saved["config"], sort_keys=False))
    command = [
        sys.executable, str(Path(__file__).parent / "run_continual_experiment.py"),
        "--config", str(config_file),
        "--run-mode", saved["run_mode"],
        "--seed", str(saved["seed"]),
        "--experiment-id", saved["experiment_id"] + args.out_suffix,
        "--strategy", args.strategy,
    ]
    if args.dev_tasks:
        command += ["--dev-tasks", *args.dev_tasks]
    print("  " + " ".join(command))

    if args.execute:
        print()
        return subprocess.run(command, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
