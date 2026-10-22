#!/usr/bin/env python3
"""Resample already-interpolated replay commands onto a uniform time grid.

The command streams in the replay artifacts are Atlas-evaluated samples (about
250 Hz).  This tool does not regenerate an Atlas trajectory.  It uses the
recorded command timestamps and linearly samples each joint onto one shared
100 Hz execution timeline.  State rows are copied unchanged so the output can
still be used for command/state alignment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


NUMERIC_COMMAND_COLUMNS = ("position", "velocity", "torque", "acceleration")
NULLABLE_COLUMNS = ("stamp_ns", "receive_ros_ns", "receive_monotonic_ns", "message_id")


def _rows(table: pa.Table) -> list[dict[str, Any]]:
    columns = table.column_names
    values = {name: table[name].to_pylist() for name in columns}
    return [{name: values[name][i] for name in columns} for i in range(table.num_rows)]


def _resample_one(input_rows: list[dict[str, Any]], target_hz: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command_rows = [r for r in input_rows if r.get("kind") == "command"]
    state_rows = [r for r in input_rows if r.get("kind") != "command"]
    if not command_rows:
        raise ValueError("input contains no kind=command rows")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in command_rows:
        grouped[str(row["joint"])].append(row)
    joints = sorted(grouped)
    streams: dict[str, list[dict[str, Any]]] = {}
    arrays: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for joint in joints:
        stream = grouped[joint]
        stream.sort(key=lambda r: float(r["time"]))
        if len(stream) < 2:
            raise ValueError(f"command stream {joint} has fewer than two samples")
        streams[joint] = stream
        ts = np.asarray([float(r["time"]) for r in stream], dtype=np.float64)
        arrays[joint] = (ts, {
            col: np.asarray([np.nan if r.get(col) is None else float(r[col]) for r in stream], dtype=np.float64)
            for col in NUMERIC_COMMAND_COLUMNS if col in stream[0]
        })

    # Use the common overlap of all streams.  This avoids inventing values at
    # either edge when individual ROS receive timestamps differ by a few ms.
    starts = np.array([float(s[0]["time"]) for s in streams.values()])
    ends = np.array([float(s[-1]["time"]) for s in streams.values()])
    start = float(starts.max())
    end = float(ends.min())
    if end <= start:
        raise ValueError(f"command streams have no common time interval: {start}..{end}")
    period = 1.0 / float(target_hz)
    count = int(np.floor((end - start) / period + 1e-9)) + 1
    target_times = start + np.arange(count, dtype=np.float64) * period

    output_commands: list[dict[str, Any]] = []
    for joint in joints:
        stream = streams[joint]
        ts, numeric = arrays[joint]
        interpolated = {}
        for col, vals in numeric.items():
            finite = np.isfinite(vals)
            interpolated[col] = np.interp(target_times, ts[finite], vals[finite]) if finite.any() else None
        source = stream[0]
        for k, t in enumerate(target_times):
            row: dict[str, Any] = dict(source)
            row["kind"] = "command"
            row["time"] = float(t)
            # These are synthetic samples, so do not present fabricated ROS
            # receive/header timestamps as hardware observations.
            for col in NULLABLE_COLUMNS:
                if col in row:
                    row[col] = None
            for col in NUMERIC_COMMAND_COLUMNS:
                if col not in row:
                    continue
                values = interpolated[col]
                row[col] = float(values[k]) if values is not None else None
            row["message_id"] = int(k)
            output_commands.append(row)

    output_rows = state_rows + output_commands
    output_rows.sort(key=lambda r: (float(r["time"]), 0 if r.get("kind") == "state" else 1, str(r.get("joint", ""))))
    original_count = len(command_rows)
    original_per_joint = {j: len(s) for j, s in streams.items()}
    report = {
        "target_frequency_hz": float(target_hz),
        "method": "per-joint linear sampling from recorded command timestamps onto a shared 1/frequency grid; no Atlas regeneration",
        "timeline": "common overlap: max(first command time) to min(last command time), endpoint included when on grid",
        "command_joint_count": len(joints),
        "original_command_rows": original_count,
        "resampled_command_rows": len(output_commands),
        "original_samples_per_joint": original_per_joint,
        "resampled_samples_per_joint": count,
        "time_start_s": start,
        "time_end_s": float(target_times[-1]),
        "time_span_s": float(target_times[-1] - target_times[0]),
        "source_command_union_start_s": float(starts.min()),
        "source_command_union_end_s": float(ends.max()),
        "source_command_common_start_s": start,
        "source_command_common_end_s": end,
        "leading_crop_from_union_start_s": float(start - starts.min()),
        "trailing_crop_from_union_end_s": float(ends.max() - target_times[-1]),
        "trailing_crop_from_common_end_s": float(end - target_times[-1]),
        "state_rows_copied_unchanged": len(state_rows),
        "command_provenance": {
            "numeric_fields": "position, velocity, torque, and acceleration linearly interpolated from recorded commands",
            "all_null_numeric_fields": "remain null; resampling does not infer missing command fields",
            "synthetic_header_fields_null": ["stamp_ns", "receive_ros_ns", "receive_monotonic_ns"],
            "message_id": "synthetic frame index shared by all joints at each target time; not an original message ID",
            "copied_fields": "part, joint, source_name, topic, and mode retain source stream labels; no synthetic sample was observed on that topic",
        },
        "duration_semantics": "command interval is cropped to the common overlap and full grid steps; exact source duration equality is not claimed; all original state rows remain",
        "units": {"time": "s", "position": "artifact units (rad except gripper percent)", "velocity": "artifact units/s", "torque": "artifact torque units"},
    }
    return output_rows, report


def process_one(input_dir: Path, output_dir: Path, target_hz: float) -> dict[str, Any]:
    input_parquet = input_dir / "samples.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(input_parquet)
    table = pq.read_table(input_parquet)
    rows, report = _resample_one(_rows(table), target_hz)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "samples.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output_parquet, compression="zstd")
    source_metadata = input_dir / "metadata.json"
    source_metadata_sha256 = None
    if source_metadata.exists():
        source_metadata_bytes = source_metadata.read_bytes()
        source_metadata_sha256 = hashlib.sha256(source_metadata_bytes).hexdigest()
        (output_dir / "source_metadata.json").write_bytes(source_metadata_bytes)
        source_meta = json.loads(source_metadata_bytes)
    else:
        source_meta = {}
    state_counts = Counter(row.get("joint") for row in rows if row.get("kind") == "state")
    output_metadata = {
        "artifact_type": "resampled_replay_commands",
        "source_artifact": str(input_dir),
        "source_metadata_file": "source_metadata.json" if source_metadata_sha256 else None,
        "source_metadata_sha256": source_metadata_sha256,
        "source_samples_sha256": hashlib.sha256(input_parquet.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(output_parquet.read_bytes()).hexdigest(),
        "target_frequency_hz": float(target_hz),
        "resampling_method": report["method"],
        "timeline": report["timeline"],
        "command_provenance": report["command_provenance"],
        "command_time_semantics": "synthetic resampling of observed ROS topic command times; not proven actuator execution times",
        "state_time_basis": source_meta.get("time_basis"),
        "state_origin_ros_ns": source_meta.get("origin_ros_ns"),
        "feedback_source": source_meta.get("feedback_source"),
        "time_start_s": report["time_start_s"],
        "time_end_s": report["time_end_s"],
        "time_span_s": report["time_span_s"],
        "units": source_meta.get("units", {}),
        "torque_comparable": False,
        "torque_note": "Resampling does not establish torque comparability; original flags and acquisition metadata are preserved in source_metadata.json.",
        "coverage": {
            joint: {
                "synthetic_command_samples": report["resampled_samples_per_joint"],
                "state_samples_copied_unchanged": state_counts[joint],
                "unit": source_meta.get("units", {}).get(joint),
                "torque_comparable": False,
            }
            for joint in report["original_samples_per_joint"]
        },
        "source_coverage": "See source_metadata.json; its original command counts refer to source samples, not synthetic resampled commands.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "input_artifact": str(input_dir),
        "input_parquet": str(input_parquet),
        "output_artifact": str(output_dir),
        "output_parquet": str(output_parquet),
        "source_metadata": str(source_metadata) if source_metadata_sha256 else None,
        "source_metadata_sha256": source_metadata_sha256,
        "source_samples_sha256": output_metadata["source_samples_sha256"],
        "output_samples_sha256": output_metadata["samples_sha256"],
        "output_metadata": str(output_dir / "metadata.json"),
        "preserved_source_metadata": str(output_dir / "source_metadata.json") if source_metadata_sha256 else None,
        "torque_comparable": False,
        **report,
    }
    (output_dir / "resample_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="artifact directories containing samples.parquet")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/resampled_100hz"))
    parser.add_argument("--frequency-hz", type=float, default=100.0)
    args = parser.parse_args()
    if args.frequency_hz <= 0:
        parser.error("--frequency-hz must be positive")
    manifests = []
    for input_dir in args.inputs:
        output_dir = args.output_root / input_dir.name
        manifests.append(process_one(input_dir, output_dir, args.frequency_hz))
        print(f"{input_dir} -> {output_dir}: {manifests[-1]['resampled_samples_per_joint']} samples/joint, "
              f"{manifests[-1]['time_span_s']:.6f}s")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps({"target_frequency_hz": args.frequency_hz, "artifacts": manifests}, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
