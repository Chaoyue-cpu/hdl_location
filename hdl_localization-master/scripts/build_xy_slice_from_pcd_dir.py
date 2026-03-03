#!/usr/bin/env python3
"""
Merge PCD tiles, slice by Z range, and project to XY for centerline drawing.

Supports PCD DATA formats:
- ascii
- binary (uncompressed)
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


def _parse_header(path: str) -> Tuple[Dict[str, object], int]:
    header: Dict[str, object] = {}
    header_len = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD (no DATA line): {path}")
            header_len += len(line)
            s = line.decode("utf-8", errors="ignore").strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            key = parts[0].upper()
            vals = parts[1:]
            header[key] = vals
            if key == "DATA":
                break
    return header, header_len


def _structured_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    dt_fields = []
    for name, sz, ty, cnt in zip(fields, sizes, types, counts):
        if ty != "F":
            raise ValueError(f"Only float fields are supported, got type={ty} for field={name}")
        if sz == 4:
            base = np.float32
        elif sz == 8:
            base = np.float64
        else:
            raise ValueError(f"Unsupported float byte size: {sz}")

        if cnt == 1:
            dt_fields.append((name, base))
        else:
            for i in range(cnt):
                dt_fields.append((f"{name}_{i}", base))
    return np.dtype(dt_fields)


def load_pcd_xyz(path: str) -> np.ndarray:
    header, header_len = _parse_header(path)

    fields = header.get("FIELDS", [])
    sizes = list(map(int, header.get("SIZE", [])))
    types = header.get("TYPE", [])
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    points = int(header.get("POINTS", ["0"])[0])
    data_mode = header.get("DATA", [""])[0].lower()

    if not fields or len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
        raise ValueError(f"Invalid PCD header in {path}")
    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise ValueError(f"PCD missing xyz fields: {path}")

    with open(path, "rb") as f:
        f.seek(header_len)
        if data_mode == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            col = {name: i for i, name in enumerate(fields)}
            xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float64, copy=False)
            return xyz

        if data_mode == "binary":
            dt = _structured_dtype(fields, sizes, types, counts)
            raw = f.read()
            expected = points * dt.itemsize
            if len(raw) < expected:
                raise ValueError(f"PCD binary payload too short: {path}")
            rec = np.frombuffer(raw[:expected], dtype=dt, count=points)
            xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64, copy=False)
            return xyz

        if data_mode == "binary_compressed":
            raise ValueError(f"Unsupported DATA binary_compressed: {path}")

        raise ValueError(f"Unsupported DATA mode '{data_mode}' in {path}")


def save_xy_csv(path: str, xy: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for p in xy:
            f.write(f"{p[0]:.6f},{p[1]:.6f}\n")


def save_xy_pcd(path: str, xy: np.ndarray) -> None:
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
        for p in xy:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} 0.000000\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PCD tiles and build XY slice map")
    parser.add_argument("--pcd-dir", required=True, help="Directory containing *.pcd tiles")
    parser.add_argument("--z-min", type=float, default=1.0, help="Min z for slicing")
    parser.add_argument("--z-max", type=float, default=1.5, help="Max z for slicing")
    parser.add_argument("--out-prefix", default="/tmp/xy_slice_map", help="Output file prefix")
    parser.add_argument("--ext", default=".pcd", help="PCD file extension")
    args = parser.parse_args()

    if args.z_min > args.z_max:
        raise ValueError("z-min must be <= z-max")

    pcd_files = []
    for root, _, files in os.walk(args.pcd_dir):
        for fn in files:
            if fn.lower().endswith(args.ext.lower()):
                pcd_files.append(os.path.join(root, fn))
    pcd_files.sort()

    if not pcd_files:
        raise FileNotFoundError(f"No PCD files found in {args.pcd_dir}")

    clouds = []
    for p in pcd_files:
        xyz = load_pcd_xyz(p)
        clouds.append(xyz)

    all_xyz = np.concatenate(clouds, axis=0)
    z = all_xyz[:, 2]
    mask = (z >= args.z_min) & (z <= args.z_max)
    sliced = all_xyz[mask]
    xy = sliced[:, :2]

    out_csv = f"{args.out_prefix}_xy.csv"
    out_pcd = f"{args.out_prefix}_xy.pcd"
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    save_xy_csv(out_csv, xy)
    save_xy_pcd(out_pcd, xy)

    print(f"[OK] merged_tiles={len(pcd_files)}")
    print(f"[OK] total_points={len(all_xyz)}")
    print(f"[OK] sliced_points={len(sliced)} in z=[{args.z_min:.3f}, {args.z_max:.3f}]")
    print(f"[OK] csv={out_csv}")
    print(f"[OK] pcd={out_pcd}")
    if len(sliced) > 0:
        print(
            "[STAT] xy_range "
            f"x=[{xy[:,0].min():.3f},{xy[:,0].max():.3f}] "
            f"y=[{xy[:,1].min():.3f},{xy[:,1].max():.3f}]"
        )


if __name__ == "__main__":
    main()

