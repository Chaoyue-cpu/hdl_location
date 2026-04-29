#!/usr/bin/env python3
"""
Build an XY slice map for ours-map from per-frame scans and a pose file.

Pipeline:
1) Read each frame PCD from a directory.
2) Keep points whose local-frame z is within [z_min, z_max].
3) Match the frame timestamp to a TUM pose.
4) Transform the filtered points into the global frame.
5) Drop z and export a merged XY PCD (z is written as 0 in the output PCD).

Input pose format (TUM):
  timestamp tx ty tz qx qy qz qw
"""

import argparse
import bisect
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_header(path: str) -> Tuple[Dict[str, List[str]], int]:
    header: Dict[str, List[str]] = {}
    header_len = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD without DATA line: {path}")
            header_len += len(line)
            s = line.decode("utf-8", errors="ignore").strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
    return header, header_len


def structured_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    dt_fields = []
    for name, sz, ty, cnt in zip(fields, sizes, types, counts):
        if ty != "F":
            raise ValueError(f"Only float PCD fields are supported: field={name}, type={ty}")
        if sz == 4:
            base = np.float32
        elif sz == 8:
            base = np.float64
        else:
            raise ValueError(f"Unsupported float byte size={sz} for field={name}")

        if cnt == 1:
            dt_fields.append((name, base))
        else:
            for i in range(cnt):
                dt_fields.append((f"{name}_{i}", base))
    return np.dtype(dt_fields)


def load_pcd_xyz(path: str) -> np.ndarray:
    header, header_len = parse_header(path)

    fields = header.get("FIELDS", [])
    sizes = list(map(int, header.get("SIZE", [])))
    types = header.get("TYPE", [])
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    points = int(header.get("POINTS", ["0"])[0])
    data_mode = header.get("DATA", [""])[0].lower()

    if not fields or len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
        raise ValueError(f"Invalid PCD header: {path}")
    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise ValueError(f"PCD missing xyz fields: {path}")

    with open(path, "rb") as f:
        f.seek(header_len)
        if data_mode == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            col = {name: i for i, name in enumerate(fields)}
            return arr[:, [col["x"], col["y"], col["z"]]].astype(np.float64, copy=False)

        if data_mode == "binary":
            dt = structured_dtype(fields, sizes, types, counts)
            raw = f.read()
            expected = points * dt.itemsize
            if len(raw) < expected:
                raise ValueError(f"PCD binary payload too short: {path}")
            rec = np.frombuffer(raw[:expected], dtype=dt, count=points)
            return np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64, copy=False)

        raise ValueError(f"Unsupported PCD DATA mode '{data_mode}' in {path}")


def load_poses_tum(path: str) -> Tuple[List[float], List[np.ndarray]]:
    times: List[float] = []
    transforms: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            a = s.split()
            if len(a) < 8:
                continue
            t = float(a[0])
            tx, ty, tz = map(float, a[1:4])
            qx, qy, qz, qw = map(float, a[4:8])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            times.append(t)
            transforms.append(T)

    if not times:
        raise RuntimeError(f"No valid poses found in {path}")

    order = np.argsort(np.asarray(times))
    sorted_times = [times[i] for i in order]
    sorted_transforms = [transforms[i] for i in order]
    return sorted_times, sorted_transforms


def match_pose_index(ts: float, pose_times: List[float], tolerance_sec: float) -> int:
    idx = bisect.bisect_left(pose_times, ts)
    candidates = []
    if idx < len(pose_times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return -1

    best = min(candidates, key=lambda i: abs(pose_times[i] - ts))
    if abs(pose_times[best] - ts) > tolerance_sec:
        return -1
    return best


def write_pcd_header(f, n_points: int) -> None:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z\n")
    f.write("SIZE 4 4 4\n")
    f.write("TYPE F F F\n")
    f.write("COUNT 1 1 1\n")
    f.write(f"WIDTH {n_points}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {n_points}\n")
    f.write("DATA ascii\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ours-map XY PCD from frame scans and poses.")
    parser.add_argument(
        "--pcd_dir",
        default="/media/scy/新加卷/数据集/井下/0805/hdl_filtered_points",
        help="Directory containing per-frame PCD scans.",
    )
    parser.add_argument(
        "--pose_file",
        default="/media/scy/新加卷/数据集/井下/0805/0805_optimized-2",
        help="TUM pose file path.",
    )
    parser.add_argument(
        "--out_prefix",
        default="/home/scy/catkin_ws_hdl_location/中轴/ours-map/ours_map_from_scans_pose_zmax3p5",
        help="Output prefix, without file suffix.",
    )
    parser.add_argument("--z_min", type=float, default=-1e9, help="Min local z kept before transforming.")
    parser.add_argument("--z_max", type=float, default=3.5, help="Max local z kept before transforming.")
    parser.add_argument("--pose_tolerance_sec", type=float, default=0.05, help="Nearest pose match tolerance in seconds.")
    parser.add_argument("--progress_every", type=int, default=200, help="Print progress every N frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    pcd_files = [
        os.path.join(args.pcd_dir, fn)
        for fn in sorted(os.listdir(args.pcd_dir))
        if fn.lower().endswith(".pcd")
    ]
    if not pcd_files:
        raise FileNotFoundError(f"No PCD files found in {args.pcd_dir}")

    pose_times, pose_transforms = load_poses_tum(args.pose_file)

    tmp_body = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        prefix="ours_map_xy_body_",
        suffix=".txt",
    )
    tmp_csv = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        prefix="ours_map_xy_csv_",
        suffix=".csv",
    )
    tmp_csv.write("x,y\n")

    scanned = 0
    used = 0
    no_pose = 0
    total_input_points = 0
    total_kept_points = 0

    xy_min = np.array([np.inf, np.inf], dtype=np.float64)
    xy_max = np.array([-np.inf, -np.inf], dtype=np.float64)

    for path in pcd_files:
        scanned += 1
        stem = os.path.splitext(os.path.basename(path))[0]
        ts = float(stem)
        pose_idx = match_pose_index(ts, pose_times, args.pose_tolerance_sec)
        if pose_idx < 0:
            no_pose += 1
            continue

        xyz = load_pcd_xyz(path)
        total_input_points += len(xyz)
        z = xyz[:, 2]
        mask = (z >= args.z_min) & (z <= args.z_max)
        xyz = xyz[mask]
        if len(xyz) == 0:
            continue

        T = pose_transforms[pose_idx]
        global_xyz = (T[:3, :3] @ xyz.T).T + T[:3, 3]
        xy = global_xyz[:, :2]

        xy_min = np.minimum(xy_min, np.min(xy, axis=0))
        xy_max = np.maximum(xy_max, np.max(xy, axis=0))

        for x, y in xy:
            tmp_body.write(f"{x:.6f} {y:.6f} 0.000000\n")
            tmp_csv.write(f"{x:.6f},{y:.6f}\n")

        total_kept_points += len(xy)
        used += 1

        if scanned % args.progress_every == 0:
            print(
                f"[PROGRESS] scanned={scanned} used={used} no_pose={no_pose} "
                f"kept_points={total_kept_points}",
                flush=True,
            )

    tmp_body.flush()
    tmp_body.close()
    tmp_csv.flush()
    tmp_csv.close()

    if total_kept_points == 0:
        raise RuntimeError("No points remained after z filtering and pose matching")

    out_pcd = f"{args.out_prefix}_xy.pcd"
    out_csv = f"{args.out_prefix}_xy.csv"
    out_report = f"{args.out_prefix}_summary.txt"

    with open(out_pcd, "w", encoding="utf-8") as f_out:
        write_pcd_header(f_out, total_kept_points)
        with open(tmp_body.name, "r", encoding="utf-8") as f_in:
            while True:
                chunk = f_in.read(1 << 20)
                if not chunk:
                    break
                f_out.write(chunk)

    os.replace(tmp_csv.name, out_csv)
    os.unlink(tmp_body.name)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write(f"pcd_dir={args.pcd_dir}\n")
        f.write(f"pose_file={args.pose_file}\n")
        f.write(f"z_min={args.z_min}\n")
        f.write(f"z_max={args.z_max}\n")
        f.write(f"pose_tolerance_sec={args.pose_tolerance_sec}\n")
        f.write(f"scanned={scanned}\n")
        f.write(f"used={used}\n")
        f.write(f"no_pose={no_pose}\n")
        f.write(f"total_input_points={total_input_points}\n")
        f.write(f"total_kept_points={total_kept_points}\n")
        f.write(f"x_min={xy_min[0]:.6f}\n")
        f.write(f"x_max={xy_max[0]:.6f}\n")
        f.write(f"y_min={xy_min[1]:.6f}\n")
        f.write(f"y_max={xy_max[1]:.6f}\n")

    print(f"[DONE] scanned={scanned} used={used} no_pose={no_pose}", flush=True)
    print(f"[DONE] total_input_points={total_input_points}", flush=True)
    print(f"[DONE] total_kept_points={total_kept_points}", flush=True)
    print(f"[DONE] pcd={out_pcd}", flush=True)
    print(f"[DONE] csv={out_csv}", flush=True)
    print(f"[DONE] summary={out_report}", flush=True)
    print(
        f"[STAT] xy_range x=[{xy_min[0]:.3f},{xy_max[0]:.3f}] "
        f"y=[{xy_min[1]:.3f},{xy_max[1]:.3f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
