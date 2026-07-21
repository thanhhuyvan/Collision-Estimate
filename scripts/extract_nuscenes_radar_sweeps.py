"""Extract previous radar sweeps required by an existing nuScenes evaluation manifest."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=3)
    args = parser.parse_args()
    if args.sweeps < 1:
        raise ValueError("sweeps must be at least one")
    metadata = args.data_root / "v1.0-mini"
    sample_data = json.loads((metadata / "sample_data.json").read_text(encoding="utf-8"))
    by_filename = {row["filename"]: row for row in sample_data}
    by_token = {row["token"]: row for row in sample_data}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested: set[str] = set()
    for frame in manifest["frames"]:
        record = by_filename[frame["radar"]["file"]]
        for _ in range(args.sweeps):
            requested.add(record["filename"])
            if not record["prev"]:
                break
            record = by_token[record["prev"]]
    missing = [file for file in requested if not (args.data_root / file).is_file()]
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {member.name: member for member in archive if member.name in missing}
        unavailable = set(missing) - set(members)
        if unavailable:
            raise FileNotFoundError(f"missing from archive: {sorted(unavailable)[:3]}")
        for file in missing:
            archive.extract(members[file], args.data_root)
    print(f"sweeps={args.sweeps} requested={len(requested)} extracted={len(missing)}")


if __name__ == "__main__":
    main()
