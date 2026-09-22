#!/usr/bin/env python3
"""Read one existing submission until its terminal score and rank are available.

This watcher never pushes a Notebook or creates a competition submission.
Keep its output directory outside tracked source, for example under logs/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def submission_record(contents: str, reference: int) -> dict:
    rows = list(csv.DictReader(io.StringIO(contents)))
    matches = [row for row in rows if row.get("ref") == str(reference)]
    if len(matches) != 1:
        raise ValueError("The requested submission was not uniquely identified")
    row = matches[0]
    score = float(row["publicScore"]) if row["publicScore"].strip() else None
    if score is not None and not math.isfinite(score):
        raise ValueError("Nonfinite public score")
    return {"reference": reference, "status": row["status"].rsplit(".", 1)[-1],
            "public_score": score, "submitted_at": row["date"], "description": row["description"]}


def leaderboard_record(contents: str, team_id: int) -> dict:
    rows = list(csv.DictReader(io.StringIO(contents)))
    matches = [row for row in rows if row.get("TeamId") == str(team_id)]
    if len(matches) != 1:
        raise ValueError("The requested team was not uniquely identified")
    row = matches[0]
    rank, score = int(row["Rank"]), float(row["Score"])
    if not 1 <= rank <= len(rows) or not math.isfinite(score):
        raise ValueError("Invalid full-leaderboard result")
    cutoff = len(rows) // 10
    cutoff_row = next((item for item in rows if int(item["Rank"]) == cutoff), None)
    return {"team_id": team_id, "team_name": row["TeamName"], "rank": rank,
            "teams": len(rows), "rank_percent": 100.0 * rank / len(rows),
            "team_score": score, "top_10_percent_cutoff_rank": cutoff,
            "cutoff_score": float(cutoff_row["Score"]) if cutoff_row else None,
            "top_10_percent_reached": rank <= cutoff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--submission-ref", type=int, required=True)
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kaggle-bin", default="kaggle")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error("Poll intervals must be at least 60 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counter = 0

    def command(arguments):
        completed = subprocess.run([args.kaggle_bin, *arguments], capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(f"Kaggle command failed ({completed.returncode}): {completed.stderr.strip()[:1500]}")
        return completed.stdout

    def record(payload, final=False):
        payload = {"observed_at": datetime.now(timezone.utc).isoformat(), **payload}
        temporary = args.output_dir / "latest.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output_dir / "latest.json")
        with (args.output_dir / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        if final:
            (args.output_dir / "final.json").write_text(json.dumps(payload, indent=2) + "\n")
        (args.output_dir / "run_summary.md").write_text(
            "# Submission observation\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n")
        print(json.dumps(payload), flush=True)

    while True:
        try:
            contents = command(["competitions", "submissions", args.competition, "--csv", "--page-size", "100"])
            submission = submission_record(contents, args.submission_ref)
            if submission["status"] == "COMPLETE" and submission["public_score"] is not None:
                counter += 1
                board_dir = args.output_dir / f"leaderboard_{counter:03d}"
                board_dir.mkdir()
                command(["competitions", "leaderboard", args.competition, "--download", "--path", str(board_dir)])
                archives = list(board_dir.glob("*.zip"))
                if len(archives) != 1:
                    raise ValueError("Expected one full-leaderboard archive")
                with zipfile.ZipFile(archives[0]) as archive:
                    members = [name for name in archive.namelist() if name.endswith(".csv")]
                    if len(members) != 1:
                        raise ValueError("Expected one full-leaderboard CSV")
                    board = leaderboard_record(archive.read(members[0]).decode("utf-8-sig"), args.team_id)
                payload = {"submission": submission, "leaderboard": board,
                           "leaderboard_archive": str(archives[0]), "leaderboard_member": members[0]}
                if board["team_score"] + 1e-12 < submission["public_score"]:
                    record({**payload, "state": "waiting_for_leaderboard_refresh"})
                else:
                    record({**payload, "state": "complete"}, final=True)
                    return
            elif submission["status"] in {"ERROR", "CANCELLED", "CANCELED"}:
                record({"submission": submission, "state": "submission_failed"}, final=True)
                return
            else:
                record({"submission": submission, "state": "waiting_for_score"})
        except Exception as error:
            record({"state": "retrying_read_error", "error": f"{type(error).__name__}: {error}"})
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
