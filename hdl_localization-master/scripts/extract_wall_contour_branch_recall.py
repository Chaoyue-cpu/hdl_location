#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wall contour extraction (branch-recall version).

This script corresponds to the selected final contour version:
  wall_contour_global_merged_before_multipeak.pcd

Method summary:
1) Local filtering (z-range, vehicle crop, voxel downsample, vertical-normal filter)
2) Per x-bin envelope:
   - standard envelope by q90/q10
   - branch tail envelope by q97/q03
3) Keep points near standard envelope OR tail envelope
4) Transform to global and apply temporal vote
5) Final light denoise + cluster pruning
"""

import argparse
import glob
import os
import shutil
from collections import defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_poses_tum(path: str):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Extract wall contour with branch-recall envelope.")
    parser.add_argument(
        "--pcd_dir",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/original-scans",
        help="Directory of per-frame raw scans (.pcd).",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/localization_vehicle_pose.tum",
        help="TUM pose file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/scy/catkin_ws_hdl_location/墙平面0416",
        help="Output directory.",
    )
    parser.add_argument("--frame_stride", type=int, default=2)

    # local pre-filter
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

    # envelope
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

    # temporal vote
    parser.add_argument("--vote_voxel", type=float, default=0.20)
    parser.add_argument("--vote_min_frames", type=int, default=2)

    # final cleanup
    parser.add_argument("--final_voxel", type=float, default=0.07)
    parser.add_argument("--radius_nb_points", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.55)
    parser.add_argument("--dbscan_eps", type=float, default=0.55)
    parser.add_argument("--dbscan_min_points", type=int, default=10)
    parser.add_argument("--cluster_min_size", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    final_path = os.path.join(args.output_dir, "wall_contour_global_merged.pcd")
    extra_path = os.path.join(args.output_dir, "wall_contour_global_merged_branch_recall.pcd")
    backup_path = os.path.join(args.output_dir, "wall_contour_global_merged_before_branch_recall_script.pcd")
    report_path = os.path.join(args.output_dir, "wall_contour_branch_recall_report.txt")

    poses = load_poses_tum(args.pose_file)
    files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.pcd")))
    if not files:
        raise RuntimeError(f"No pcd files found in: {args.pcd_dir}")

    print(f"[INFO] files={len(files)} poses={len(poses)} stride={args.frame_stride}", flush=True)

    all_global = []
    vote = defaultdict(int)
    scanned = used = no_pose = 0

    for p in files[:: args.frame_stride]:
        scanned += 1
        stem = os.path.splitext(os.path.basename(p))[0]
        T = poses.get(stem)
        if T is None:
            no_pose += 1
            continue

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
        if len(pts) < 150:
            continue

        p0 = o3d.geometry.PointCloud()
        p0.points = o3d.utility.Vector3dVector(pts)
        p0 = p0.voxel_down_sample(voxel_size=args.per_frame_voxel)
        pts = np.asarray(p0.points)
        if len(pts) < 120:
            continue

        p0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.normal_radius,
                max_nn=args.normal_max_nn,
            )
        )
        nrm = np.asarray(p0.normals)
        pv = pts[np.abs(nrm[:, 2]) < args.abs_nz_max]
        if len(pv) < 100:
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
            if yl - yr < 0.65:
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

        sel = []
        for pt in pv:
            b = int(np.floor((pt[0] - x0) / args.x_bin_size))
            if b not in L:
                continue

            yl, yr = L[b], Rr[b]
            yle, yre = Le[b], Re[b]
            w = yl - yr
            if w < 0.65:
                continue

            ym = 0.5 * (yl + yr)
            side_gap = abs(float(pt[1]) - ym)

            standard = (side_gap >= args.min_side_gap_ratio * w) and (
                min(abs(float(pt[1]) - yl), abs(float(pt[1]) - yr)) <= args.envelope_tol
            )
            tail = (abs(float(pt[1]) - yle) <= args.tail_tol) or (abs(float(pt[1]) - yre) <= args.tail_tol)

            if standard or tail:
                sel.append(pt)

        if len(sel) < 60:
            continue

        sel = np.asarray(sel)
        g = (T[:3, :3] @ sel.T).T + T[:3, 3]

        kv = np.floor(g / args.vote_voxel).astype(np.int32)
        uniq = set((int(a), int(b), int(c)) for a, b, c in kv)
        for k in uniq:
            vote[k] += 1

        all_global.append(g)
        used += 1

        if scanned % 150 == 0:
            print(f"[PROGRESS] scanned={scanned} used={used} vote_vox={len(vote)}", flush=True)

    if not all_global:
        raise RuntimeError("No contour points extracted.")

    cont = np.concatenate(all_global, axis=0)
    keys_all = np.floor(cont / args.vote_voxel).astype(np.int32)
    mask = np.array([vote[(int(a), int(b), int(c))] >= args.vote_min_frames for a, b, c in keys_all], dtype=bool)
    cont = cont[mask]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cont)
    pc = pc.voxel_down_sample(voxel_size=args.final_voxel)
    pc, _ = pc.remove_radius_outlier(nb_points=args.radius_nb_points, radius=args.radius)

    labels = np.array(pc.cluster_dbscan(eps=args.dbscan_eps, min_points=args.dbscan_min_points, print_progress=False))
    pts = np.asarray(pc.points)
    if len(labels) == len(pts) and (labels >= 0).any():
        cnt = np.bincount(labels[labels >= 0])
        good = set(np.where(cnt >= args.cluster_min_size)[0].tolist())
        keep = np.array([(lb in good) if lb >= 0 else False for lb in labels], dtype=bool)
        pts = pts[keep]

    out_pc = o3d.geometry.PointCloud()
    out_pc.points = o3d.utility.Vector3dVector(pts)

    tmp_out = os.path.join("/tmp", "wall_contour_branch_recall_tmp.pcd")
    o3d.io.write_point_cloud(tmp_out, out_pc, write_ascii=False, compressed=False)

    shutil.copy2(tmp_out, extra_path)
    if os.path.exists(final_path):
        shutil.copy2(final_path, backup_path)
    shutil.copy2(tmp_out, final_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"files={len(files)} stride={args.frame_stride} scanned={scanned} used={used} no_pose={no_pose}\n")
        f.write(f"output_points={len(pts)}\n")
        f.write("mode=branch_recall\n")
        f.write(
            "params: standard_band(q90/q10)+tail_band(q97/q03), "
            f"vote_min_frames={args.vote_min_frames}, cluster_min={args.cluster_min_size}\n"
        )

    print(f"[DONE] scanned={scanned} used={used} no_pose={no_pose}", flush=True)
    print(f"[DONE] output_points={len(pts)}", flush=True)
    print(f"[DONE] saved_default={final_path}", flush=True)
    print(f"[DONE] saved_extra={extra_path}", flush=True)
    print(f"[DONE] backup={backup_path}", flush=True)
    print(f"[DONE] report={report_path}", flush=True)


if __name__ == "__main__":
    main()

