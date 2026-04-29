#!/usr/bin/env python3
"""
Voxel-downsample an ASCII XY PCD and write another ASCII XY PCD.

Input is expected to be a PCD with fields x y z.
Output keeps x y and writes z=0.
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_header(path: str) -> Tuple[List[str], int]:
    header_lines: List[str] = []
    header_len = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD without DATA line: {path}")
            header_len += len(line)
            s = line.decode("utf-8", errors="ignore")
            header_lines.append(s)
            if s.strip().upper().startswith("DATA"):
                break
    return header_lines, header_len


def write_xy_pcd(path: str, xy: np.ndarray) -> None:
    n = xy.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for x, y in xy:
            f.write(f"{x:.6f} {y:.6f} 0.000000\n")


def write_xy_csv(path: str, xy: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in xy:
            f.write(f"{x:.6f},{y:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxel downsample an ASCII XY PCD.")
    parser.add_argument("--input", required=True, help="Input ASCII PCD path.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without suffix.")
    parser.add_argument("--voxel", type=float, default=0.3, help="Voxel size in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    if args.voxel <= 0:
        raise ValueError("voxel must be positive")

    _, header_len = parse_header(args.input)
    sums: Dict[Tuple[int, int], np.ndarray] = {}
    counts: Dict[Tuple[int, int], int] = {}
    total = 0

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(header_len)
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            a = s.split()
            if len(a) < 2:
                continue
            x = float(a[0])
            y = float(a[1])
            ix = int(math.floor(x / args.voxel))
            iy = int(math.floor(y / args.voxel))
            key = (ix, iy)
            if key not in sums:
                sums[key] = np.array([x, y], dtype=np.float64)
                counts[key] = 1
            else:
                sums[key] += np.array([x, y], dtype=np.float64)
                counts[key] += 1
            total += 1

    xy = np.zeros((len(sums), 2), dtype=np.float64)
    for i, key in enumerate(sorted(sums.keys())):
        xy[i] = sums[key] / float(counts[key])

    out_pcd = f"{args.output_prefix}_xy.pcd"
    out_csv = f"{args.output_prefix}_xy.csv"
    out_summary = f"{args.output_prefix}_summary.txt"

    write_xy_pcd(out_pcd, xy)
    write_xy_csv(out_csv, xy)

    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"input={args.input}\n")
        f.write(f"voxel={args.voxel}\n")
        f.write(f"input_points={total}\n")
        f.write(f"output_points={len(xy)}\n")
        if len(xy) > 0:
            f.write(f"x_min={xy[:,0].min():.6f}\n")
            f.write(f"x_max={xy[:,0].max():.6f}\n")
            f.write(f"y_min={xy[:,1].min():.6f}\n")
            f.write(f"y_max={xy[:,1].max():.6f}\n")

    print(f"[DONE] input_points={total}")
    print(f"[DONE] output_points={len(xy)}")
    print(f"[DONE] pcd={out_pcd}")
    print(f"[DONE] csv={out_csv}")
    print(f"[DONE] summary={out_summary}")


if __name__ == "__main__":
    main()
