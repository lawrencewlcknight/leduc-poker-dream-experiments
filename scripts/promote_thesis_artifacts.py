"""Promote lightweight thesis artifacts from experiment outputs into the repo.

The script discovers experiment run directories by searching for
``experiment_metadata.json`` below one or more source paths, then copies only
curated, lightweight files into ``thesis_artifacts/<experiment>/<run>/``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INCLUDE_PATTERNS = [
    "*.png",
    "*.csv",
    "aggregate_summary.json",
    "paired_difference_summary.json",
    "paired_aggregate_summary.json",
    "best_checkpoint_summary.json",
    "experiment_metadata.json",
]

DEFAULT_EXCLUDE_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.npz",
    "*.log",
    "failed_seeds.json",
    "checkpoints/*",
    "snapshots/*",
    "traces/*",
]


def parse_globs(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def infer_experiment_name(run_dir: Path, metadata: Dict) -> str:
    experiment_config = metadata.get("experiment_config")
    if isinstance(experiment_config, dict):
        name = experiment_config.get("experiment_name")
        if name:
            return str(name)
    name = metadata.get("experiment_name")
    if name:
        return str(name)
    config = metadata.get("config")
    if isinstance(config, dict) and config.get("experiment_name"):
        return str(config["experiment_name"])
    return run_dir.name


def discover_run_dirs(sources: Sequence[Path]) -> List[Path]:
    run_dirs = set()
    for source in sources:
        source = source.expanduser()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            if source.name == "experiment_metadata.json":
                run_dirs.add(source.parent.resolve())
            continue
        direct_metadata = source / "experiment_metadata.json"
        if direct_metadata.exists():
            run_dirs.add(source.resolve())
        for metadata_path in source.rglob("experiment_metadata.json"):
            run_dirs.add(metadata_path.parent.resolve())
    return sorted(run_dirs)


def matches_pattern(rel_path: Path, pattern: str) -> bool:
    rel = rel_path.as_posix()
    name = rel_path.name
    return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern)


def classify_files(
    run_dir: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> Tuple[List[Path], List[Dict[str, str]]]:
    selected: List[Path] = []
    skipped: List[Dict[str, str]] = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir)
        include_match = any(matches_pattern(rel, pattern) for pattern in include_patterns)
        exclude_match = any(matches_pattern(rel, pattern) for pattern in exclude_patterns)
        if include_match and not exclude_match:
            selected.append(path)
        else:
            if exclude_match:
                reason = "excluded"
            else:
                reason = "not_included"
            skipped.append({"path": rel.as_posix(), "reason": reason})
    return selected, skipped


def copy_selected_files(
    selected_files: Iterable[Path],
    run_dir: Path,
    dest_run_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    copied: List[Dict[str, str]] = []
    skipped_existing: List[Dict[str, str]] = []
    for source_path in selected_files:
        rel = source_path.relative_to(run_dir)
        dest_path = dest_run_dir / rel
        record = {
            "source": str(source_path),
            "destination": str(dest_path),
            "relative_path": rel.as_posix(),
        }
        if dest_path.exists() and not overwrite:
            skipped_existing.append(
                {
                    "path": rel.as_posix(),
                    "reason": "destination_exists",
                    "destination": str(dest_path),
                }
            )
            continue
        copied.append(record)
        if dry_run:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
    return copied, skipped_existing


def write_manifest(dest_run_dir: Path, manifest: Dict, dry_run: bool) -> None:
    if dry_run:
        return
    dest_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_run_dir / "promotion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def promote_run(
    run_dir: Path,
    dest_root: Path,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
    overwrite: bool,
    dry_run: bool,
) -> Dict:
    metadata_path = run_dir / "experiment_metadata.json"
    metadata = load_json(metadata_path)
    experiment_name = infer_experiment_name(run_dir, metadata)
    dest_run_dir = dest_root / experiment_name / run_dir.name
    selected, skipped = classify_files(run_dir, include_patterns, exclude_patterns)
    copied, skipped_existing = copy_selected_files(selected, run_dir, dest_run_dir, overwrite, dry_run)
    skipped.extend(skipped_existing)
    manifest = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_paths": [str(run_dir)],
        "destination_paths": [str(dest_run_dir)],
        "source_run_dir": str(run_dir),
        "destination_run_dir": str(dest_run_dir),
        "experiment_name": experiment_name,
        "run_directory_name": run_dir.name,
        "include_patterns": list(include_patterns),
        "exclude_patterns": list(exclude_patterns),
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "selected_files": copied,
        "skipped_files": skipped,
    }
    write_manifest(dest_run_dir, manifest, dry_run)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Run, cloud job, or parent output directories.")
    parser.add_argument("--dest", type=Path, default=Path("thesis_artifacts"))
    parser.add_argument("--include", type=str, default="", help="Comma-separated extra include globs.")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated extra exclude globs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing promoted files.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be promoted without copying.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    include_patterns = DEFAULT_INCLUDE_PATTERNS + parse_globs(args.include)
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + parse_globs(args.exclude)
    run_dirs = discover_run_dirs(args.sources)
    if not run_dirs:
        print("No experiment run directories found.")
        return 1

    manifests = []
    for run_dir in run_dirs:
        manifest = promote_run(
            run_dir=run_dir,
            dest_root=args.dest,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        manifests.append(manifest)
        selected_count = len(manifest["selected_files"])
        skipped_count = len(manifest["skipped_files"])
        print(
            f"{run_dir} -> {manifest['destination_run_dir']} "
            f"({selected_count} selected, {skipped_count} skipped)"
        )

    print(f"Processed {len(manifests)} run directory/directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
