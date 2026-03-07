#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build wall_inner points and prior voxel layer for tunnel localization.

Outputs:
1) wall_inner_global_merged.pcd
2) wall_contour_global_merged_for_prior.pcd (optional reference export)
3) prior_voxel_layer.csv
4) per-tile sidecar CSV: <tile_name>.prior.csv

Prior schema per voxel:
  class, p_static, p_contour, p_inner, conf_prior
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_poses_tum(path: str) -> Dict[str, np.ndarray]:
    poses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
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
            poses[f"{t:.6f}"] = T
    return poses


def median_smooth(vals: np.ndarray, win: int) -> np.ndarray:
    if len(vals) < 3:
        return vals.copy()
    k = max(3, win)
    if k % 2 == 0:
        k += 1
    r = k // 2
    out = vals.copy()
    for i in range(len(vals)):
        s = max(0, i - r)
        e = min(len(vals), i + r + 1)
        out[i] = float(np.median(vals[s:e]))
    return out


def parse_metadata(path: str) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """
    Parse simple autoware metadata yaml-like file:
      x_resolution: 300
      y_resolution: 300
      tile_name.pcd: [x, y]
    """
    x_res = None
    y_res = None
    tiles: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("x_resolution:"):
                x_res = float(s.split(":", 1)[1].strip())
                continue
            if s.startswith("y_resolution:"):
                y_res = float(s.split(":", 1)[1].strip())
                continue
            # tile entry
            if ":" in s and "[" in s and "]" in s:
                key, val = s.split(":", 1)
                key = key.strip()
                arr = val.strip().lstrip("[").rstrip("]")
                xs = [x.strip() for x in arr.split(",")]
                if len(xs) >= 2:
                    tiles[key] = (float(xs[0]), float(xs[1]))

    if x_res is None or y_res is None:
        raise RuntimeError(f"Invalid metadata file: missing x_resolution/y_resolution -> {path}")
    return x_res, y_res, tiles


@dataclass
class VoxelStat:
    total: int = 0
    contour: int = 0
    inner: int = 0
    time_bins: set = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wall_inner and prior voxel layer from raw scans.")
    parser.add_argument(
        "--pcd_dir",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/original-scans",
        help="Per-frame raw scans directory.",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/localization_vehicle_pose.tum",
        help="TUM pose file (t tx ty tz qx qy qz qw).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="/media/scy/新加卷/数据集/autoware_map_output/公司地图/pointcloud_map_metadata.yaml",
        help="Tile metadata file used by map server.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/scy/catkin_ws_hdl_location/墙平面0416/prior_layer",
        help="Output directory.",
    )
    parser.add_argument("--frame_stride", type=int, default=2, help="Use every N-th frame.")
    parser.add_argument("--time_bin_sec", type=float, default=1.0, help="Temporal bin width for p_static.")

    # NDT/grid parameters
    parser.add_argument("--ndt_resolution", type=float, default=2.0, help="Prior voxel size (should match ndt_resolution).")

    # local preprocess
    parser.add_argument("--wall_z_min", type=float, default=0.5)
    parser.add_argument("--wall_z_max", type=float, default=3.0)
    parser.add_argument("--per_frame_voxel", type=float, default=0.07)
    parser.add_argument("--normal_radius", type=float, default=0.55)
    parser.add_argument("--normal_max_nn", type=int, default=24)
    parser.add_argument("--abs_nz_max", type=float, default=0.50)

    # vehicle crop in local frame
    parser.add_argument("--veh_x_min", type=float, default=-0.91)
    parser.add_argument("--veh_x_max", type=float, default=2.12)
    parser.add_argument("--veh_y_min", type=float, default=-0.72)
    parser.add_argument("--veh_y_max", type=float, default=0.69)
    parser.add_argument("--veh_z_min", type=float, default=1.88)
    parser.add_argument("--veh_z_max", type=float, default=2.12)

    # contour envelope (same family as branch-recall)
    parser.add_argument("--x_bin_size", type=float, default=0.45)
    parser.add_argument("--q_left", type=float, default=0.90)
    parser.add_argument("--q_right", type=float, default=0.10)
    parser.add_argument("--q_left_ext", type=float, default=0.97)
    parser.add_argument("--q_right_ext", type=float, default=0.03)
    parser.add_argument("--min_pts_per_bin", type=int, default=14)
    parser.add_argument("--smooth_win", type=int, default=7)
    parser.add_argument("--envelope_tol", type=float, default=0.22)
    parser.add_argument("--tail_tol", type=float, default=0.35)
    parser.add_argument("--min_side_gap_ratio", type=float, default=0.10)
    parser.add_argument("--min_wall_width", type=float, default=0.65)
    parser.add_argument("--inner_margin", type=float, default=0.15, help="Margin from envelope to define inner zone.")

    # prior scoring
    parser.add_argument("--static_ref_hits", type=int, default=6, help="p_static saturates when hits >= this value.")
    parser.add_argument("--class_contour_thresh", type=float, default=0.55)
    parser.add_argument("--class_inner_thresh", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tiles_out = os.path.join(args.output_dir, "prior_tiles")
    os.makedirs(tiles_out, exist_ok=True)

    poses = load_poses_tum(args.pose_file)
    files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.pcd")))
    if not files:
        raise RuntimeError(f"No pcd files found in {args.pcd_dir}")
    x_res, y_res, tiles = parse_metadata(args.metadata_file)

    print(f"[INFO] files={len(files)} poses={len(poses)} stride={args.frame_stride}", flush=True)
    print(f"[INFO] metadata tiles={len(tiles)} x_res={x_res} y_res={y_res}", flush=True)

    t0 = None
    scanned = used = no_pose = 0
    contour_global = []
    inner_global = []
    voxels: Dict[Tuple[int, int, int], VoxelStat] = defaultdict(VoxelStat)

    for p in files[:: args.frame_stride]:
        scanned += 1
        stem = os.path.splitext(os.path.basename(p))[0]
        T = poses.get(stem)
        if T is None:
            no_pose += 1
            continue

        ts = float(stem)
        if t0 is None:
            t0 = ts
        tb = int((ts - t0) / args.time_bin_sec)

        pc = o3d.io.read_point_cloud(p)
        pts = np.asarray(pc.points)
        if len(pts) < 200:
            continue

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep_z = (z >= args.wall_z_min) & (z <= args.wall_z_max)
        veh = (
            (x >= args.veh_x_min)
            & (x <= args.veh_x_max)
            & (y >= args.veh_y_min)
            & (y <= args.veh_y_max)
            & (z >= args.veh_z_min)
            & (z <= args.veh_z_max)
        )
        pts = pts[keep_z & (~veh)]
        if len(pts) < 120:
            continue

        p0 = o3d.geometry.PointCloud()
        p0.points = o3d.utility.Vector3dVector(pts)
        p0 = p0.voxel_down_sample(voxel_size=args.per_frame_voxel)
        pts = np.asarray(p0.points)
        if len(pts) < 100:
            continue

        p0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=args.normal_max_nn)
        )
        nrm = np.asarray(p0.normals)
        pv = pts[np.abs(nrm[:, 2]) < args.abs_nz_max]
        if len(pv) < 80:
            continue

        x0 = float(np.min(pv[:, 0]))
        xb = np.floor((pv[:, 0] - x0) / args.x_bin_size).astype(np.int32)
        bins = np.unique(xb)

        b_valid, yL, yR, yLe, yRe = [], [], [], [], []
        for b in bins:
            ys = pv[xb == b, 1]
            if len(ys) < args.min_pts_per_bin:
                continue
            yl = float(np.quantile(ys, args.q_left))
            yr = float(np.quantile(ys, args.q_right))
            yle = float(np.quantile(ys, args.q_left_ext))
            yre = float(np.quantile(ys, args.q_right_ext))
            if yl - yr < args.min_wall_width:
                continue
            b_valid.append(int(b))
            yL.append(yl)
            yR.append(yr)
            yLe.append(yle)
            yRe.append(yre)
        if len(b_valid) < 5:
            continue

        yL = median_smooth(np.array(yL), args.smooth_win)
        yR = median_smooth(np.array(yR), args.smooth_win)
        yLe = median_smooth(np.array(yLe), args.smooth_win)
        yRe = median_smooth(np.array(yRe), args.smooth_win)

        L = {b: float(v) for b, v in zip(b_valid, yL)}
        Rr = {b: float(v) for b, v in zip(b_valid, yR)}
        Le = {b: float(v) for b, v in zip(b_valid, yLe)}
        Re = {b: float(v) for b, v in zip(b_valid, yRe)}

        contour_local = []
        inner_local = []
        contour_flags = []

        for pt in pv:
            b = int(np.floor((pt[0] - x0) / args.x_bin_size))
            if b not in L:
                continue

            yl, yr = L[b], Rr[b]
            yle, yre = Le[b], Re[b]
            w = yl - yr
            if w < args.min_wall_width:
                continue

            yv = float(pt[1])
            ym = 0.5 * (yl + yr)
            side_gap = abs(yv - ym)

            standard = (side_gap >= args.min_side_gap_ratio * w) and (
                min(abs(yv - yl), abs(yv - yr)) <= args.envelope_tol
            )
            tail = (abs(yv - yle) <= args.tail_tol) or (abs(yv - yre) <= args.tail_tol)
            is_contour = standard or tail

            # inner zone: inside envelope with a margin, but not contour
            in_corridor = (yv <= yl - args.inner_margin) and (yv >= yr + args.inner_margin)
            is_inner = (not is_contour) and in_corridor

            if is_contour:
                contour_local.append(pt)
                contour_flags.append(1)
            if is_inner:
                inner_local.append(pt)
                contour_flags.append(0)

        if len(contour_local) + len(inner_local) < 50:
            continue

        contour_local = np.asarray(contour_local) if contour_local else np.empty((0, 3), dtype=np.float64)
        inner_local = np.asarray(inner_local) if inner_local else np.empty((0, 3), dtype=np.float64)

        Rg = T[:3, :3]
        tg = T[:3, 3]
        cg = (Rg @ contour_local.T).T + tg if len(contour_local) > 0 else np.empty((0, 3), dtype=np.float64)
        ig = (Rg @ inner_local.T).T + tg if len(inner_local) > 0 else np.empty((0, 3), dtype=np.float64)

        if len(cg) > 0:
            contour_global.append(cg)
            k = np.floor(cg / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = voxels[key]
                st.total += 1
                st.contour += 1
                st.time_bins.add(tb)

        if len(ig) > 0:
            inner_global.append(ig)
            k = np.floor(ig / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = voxels[key]
                st.total += 1
                st.inner += 1
                st.time_bins.add(tb)

        used += 1
        if scanned % 150 == 0:
            print(f"[PROGRESS] scanned={scanned} used={used} voxels={len(voxels)}", flush=True)

    if not voxels:
        raise RuntimeError("No voxels generated. Please relax thresholds.")

    # merged pcd exports
    contour_merged = np.concatenate(contour_global, axis=0) if contour_global else np.empty((0, 3), dtype=np.float64)
    inner_merged = np.concatenate(inner_global, axis=0) if inner_global else np.empty((0, 3), dtype=np.float64)

    contour_pcd_path = os.path.join(args.output_dir, "wall_contour_global_merged_for_prior.pcd")
    inner_pcd_path = os.path.join(args.output_dir, "wall_inner_global_merged.pcd")

    if len(contour_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(contour_merged)
        o3d.io.write_point_cloud(contour_pcd_path, pc, write_ascii=False, compressed=False)
    if len(inner_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(inner_merged)
        o3d.io.write_point_cloud(inner_pcd_path, pc, write_ascii=False, compressed=False)

    # build prior voxel table
    rows = []
    for (ix, iy, iz), st in voxels.items():
        total = max(1, st.total)
        p_contour = float(st.contour) / float(total)
        p_inner = float(st.inner) / float(total)
        hits = len(st.time_bins)
        p_static = min(1.0, float(hits) / float(max(1, args.static_ref_hits)))

        if p_contour >= args.class_contour_thresh:
            cls = "contour"
            cls_id = 2
        elif p_inner >= args.class_inner_thresh:
            cls = "inner"
            cls_id = 1
        else:
            cls = "uncertain"
            cls_id = 0

        # one scalar prior confidence (higher for static + contour dominance)
        conf_prior = p_static * (0.15 + 0.85 * max(p_contour, p_inner))
        if cls == "uncertain":
            conf_prior *= 0.7

        cx = (ix + 0.5) * args.ndt_resolution
        cy = (iy + 0.5) * args.ndt_resolution
        cz = (iz + 0.5) * args.ndt_resolution

        rows.append(
            {
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "x": cx,
                "y": cy,
                "z": cz,
                "total": st.total,
                "contour": st.contour,
                "inner": st.inner,
                "time_hits": hits,
                "class_id": cls_id,
                "class": cls,
                "p_static": p_static,
                "p_contour": p_contour,
                "p_inner": p_inner,
                "conf_prior": conf_prior,
            }
        )

    prior_csv = os.path.join(args.output_dir, "prior_voxel_layer.csv")
    fieldnames = [
        "ix",
        "iy",
        "iz",
        "x",
        "y",
        "z",
        "total",
        "contour",
        "inner",
        "time_hits",
        "class_id",
        "class",
        "p_static",
        "p_contour",
        "p_inner",
        "conf_prior",
    ]
    with open(prior_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # tile sidecar export
    # tile area: [ox, ox+x_res), [oy, oy+y_res)
    tile_counts = {}
    for tile_name, (ox, oy) in tiles.items():
        out_file = os.path.join(tiles_out, f"{tile_name}.prior.csv")
        cnt = 0
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                x = r["x"]
                y = r["y"]
                if (x >= ox) and (x < ox + x_res) and (y >= oy) and (y < oy + y_res):
                    w.writerow(r)
                    cnt += 1
        tile_counts[tile_name] = cnt

    summary = {
        "files_total": len(files),
        "files_used": used,
        "files_scanned": scanned,
        "files_no_pose": no_pose,
        "voxels_total": len(rows),
        "contour_points_total": int(len(contour_merged)),
        "inner_points_total": int(len(inner_merged)),
        "ndt_resolution": args.ndt_resolution,
        "time_bin_sec": args.time_bin_sec,
        "metadata_file": args.metadata_file,
        "tiles_total": len(tiles),
        "tile_nonempty": int(sum(1 for v in tile_counts.values() if v > 0)),
    }
    with open(os.path.join(args.output_dir, "prior_voxel_layer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] scanned={scanned} used={used} no_pose={no_pose}", flush=True)
    print(f"[DONE] contour_points={len(contour_merged)} inner_points={len(inner_merged)}", flush=True)
    print(f"[DONE] voxels={len(rows)}", flush=True)
    print(f"[DONE] prior_csv={prior_csv}", flush=True)
    print(f"[DONE] tile_sidecar_dir={tiles_out}", flush=True)


if __name__ == "__main__":
    main()

