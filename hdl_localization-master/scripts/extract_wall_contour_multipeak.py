#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract wall contour points from raw LiDAR scans using a multi-peak envelope model.

Core idea:
1) Per x-bin, detect top-K peaks on y histogram (instead of only q90/q10).
2) Link peaks across neighboring bins to form multiple wall tracks.
3) Keep contour points near any valid track.
4) Adaptive temporal voting:
   - main tracks: min_frames = main_vote_min_frames
   - branch tracks: min_frames = branch_vote_min_frames

This script is designed for tunnel scenes with branches/junctions where dual-side
envelope (left/right only) often suppresses side tunnel extensions.
"""

import argparse
import glob
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


@dataclass
class Track:
    """One connected y-track across x-bins."""

    tid: int
    samples: List[Tuple[int, float, int]] = field(default_factory=list)  # (x_bin, y_peak, peak_count)
    peak_type: str = "branch"  # "main" or "branch"

    def add(self, x_bin: int, y_peak: float, peak_count: int) -> None:
        self.samples.append((x_bin, y_peak, peak_count))

    @property
    def length_bins(self) -> int:
        return len(self.samples)

    @property
    def score(self) -> float:
        if not self.samples:
            return 0.0
        # favor long and dense tracks
        return float(self.length_bins) * float(np.mean([s[2] for s in self.samples]))


def load_poses_tum(path: str) -> Dict[str, np.ndarray]:
    """Load TUM pose file into {timestamp_str_6f: 4x4 transform}."""
    poses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            arr = s.split()
            if len(arr) < 8:
                continue
            t = float(arr[0])
            tx, ty, tz = map(float, arr[1:4])
            qx, qy, qz, qw = map(float, arr[4:8])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses[f"{t:.6f}"] = T
    return poses


def median_smooth(vals: np.ndarray, win: int) -> np.ndarray:
    """Robust 1D smoothing."""
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


def find_topk_hist_peaks(
    ys: np.ndarray,
    k: int,
    y_bin_size: float,
    min_peak_count: int,
    smooth_win: int,
) -> List[Tuple[float, int]]:
    """
    Find top-K local maxima on y histogram.
    Return list of (y_peak, peak_count), sorted by peak_count desc.
    """
    if len(ys) < max(10, min_peak_count):
        return []

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max - y_min < max(0.2, y_bin_size):
        return []

    n_bins = int(np.ceil((y_max - y_min) / y_bin_size)) + 1
    hist, edges = np.histogram(ys, bins=n_bins, range=(y_min, y_max))
    hist = hist.astype(np.float64)
    hist = median_smooth(hist, smooth_win)

    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= min_peak_count:
            yc = 0.5 * (edges[i] + edges[i + 1])
            peaks.append((yc, int(hist[i])))

    if not peaks:
        return []
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:k]


def link_tracks(
    peaks_per_bin: Dict[int, List[Tuple[float, int]]],
    max_link_dy: float,
) -> List[Track]:
    """Greedy nearest-neighbor linking of peaks across consecutive x-bins."""
    tracks: List[Track] = []
    next_tid = 0
    active: Dict[int, Tuple[int, float]] = {}  # tid -> (last_bin, last_y)

    for xb in sorted(peaks_per_bin.keys()):
        peaks = peaks_per_bin[xb]
        used = [False] * len(peaks)

        # try to link active tracks first
        for tid, (last_bin, last_y) in list(active.items()):
            if xb - last_bin > 1:
                del active[tid]
                continue

            best_i = -1
            best_d = 1e9
            for i, (py, _) in enumerate(peaks):
                if used[i]:
                    continue
                d = abs(py - last_y)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= max_link_dy:
                py, pc = peaks[best_i]
                tracks[tid].add(xb, py, pc)
                active[tid] = (xb, py)
                used[best_i] = True
            else:
                del active[tid]

        # unassigned peaks start new tracks
        for i, (py, pc) in enumerate(peaks):
            if used[i]:
                continue
            tr = Track(tid=next_tid)
            tr.add(xb, py, pc)
            tracks.append(tr)
            active[next_tid] = (xb, py)
            next_tid += 1

    return tracks


def build_bin_track_lookup(tracks: List[Track]) -> Dict[int, List[Tuple[float, str, int]]]:
    """
    Build lookup:
      x_bin -> [(y_track, track_type(main/branch), tid), ...]
    """
    lookup: Dict[int, List[Tuple[float, str, int]]] = defaultdict(list)
    for tr in tracks:
        for xb, yv, _ in tr.samples:
            lookup[xb].append((yv, tr.peak_type, tr.tid))
    return lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract wall contour with multi-peak envelope and adaptive temporal voting."
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/original-scans",
        help="Directory of per-frame raw PCD scans.",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default="/media/scy/新加卷/数据集/井下/融合定位数据20240416/建图/localization_vehicle_pose.tum",
        help="TUM pose file. Format: t tx ty tz qx qy qz qw.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/scy/catkin_ws_hdl_location/墙平面0416",
        help="Output directory for contour PCD and report.",
    )
    parser.add_argument("--frame_stride", type=int, default=2, help="Use every N-th frame.")

    # local preprocessing
    parser.add_argument("--wall_z_min", type=float, default=0.5, help="Min z for wall candidate in local frame.")
    parser.add_argument("--wall_z_max", type=float, default=3.0, help="Max z for wall candidate in local frame.")
    parser.add_argument("--per_frame_voxel", type=float, default=0.07, help="Per-frame voxel downsampling.")
    parser.add_argument("--normal_radius", type=float, default=0.55, help="Normal estimation radius.")
    parser.add_argument("--normal_max_nn", type=int, default=24, help="Normal estimation max neighbors.")
    parser.add_argument("--abs_nz_max", type=float, default=0.50, help="Keep points with |normal_z| < this.")

    # vehicle crop in local frame
    parser.add_argument("--veh_x_min", type=float, default=-0.91)
    parser.add_argument("--veh_x_max", type=float, default=2.12)
    parser.add_argument("--veh_y_min", type=float, default=-0.72)
    parser.add_argument("--veh_y_max", type=float, default=0.69)
    parser.add_argument("--veh_z_min", type=float, default=1.88)
    parser.add_argument("--veh_z_max", type=float, default=2.12)

    # multi-peak envelope
    parser.add_argument("--x_bin_size", type=float, default=0.45, help="x-bin size in local frame.")
    parser.add_argument("--y_bin_size", type=float, default=0.12, help="y histogram bin size for peak finding.")
    parser.add_argument("--top_k_peaks", type=int, default=3, help="Top-K peaks per x-bin.")
    parser.add_argument("--min_peak_count", type=int, default=6, help="Minimum count to accept a y peak.")
    parser.add_argument("--peak_smooth_win", type=int, default=5, help="Median smooth window for y histogram.")
    parser.add_argument("--max_link_dy", type=float, default=0.45, help="Max |dy| when linking peaks cross bins.")
    parser.add_argument("--track_min_bins", type=int, default=3, help="Minimum bins for a valid track.")
    parser.add_argument("--main_track_count", type=int, default=2, help="How many tracks are treated as main walls.")
    parser.add_argument("--envelope_tol", type=float, default=0.24, help="Distance threshold to keep point near track.")

    # temporal vote
    parser.add_argument("--vote_voxel", type=float, default=0.20, help="Global vote voxel size.")
    parser.add_argument("--main_vote_min_frames", type=int, default=2, help="Min hit frames for main-track voxels.")
    parser.add_argument("--branch_vote_min_frames", type=int, default=1, help="Min hit frames for branch-track voxels.")

    # final cleanup
    parser.add_argument("--final_voxel", type=float, default=0.07, help="Final voxel downsampling.")
    parser.add_argument("--radius_nb_points", type=int, default=4, help="Radius outlier min neighbors.")
    parser.add_argument("--radius", type=float, default=0.55, help="Radius outlier radius.")
    parser.add_argument("--dbscan_eps", type=float, default=0.55, help="DBSCAN eps.")
    parser.add_argument("--dbscan_min_points", type=int, default=10, help="DBSCAN min points.")
    parser.add_argument("--cluster_min_size", type=int, default=40, help="Minimum cluster size kept in output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    final_path = os.path.join(args.output_dir, "wall_contour_global_merged.pcd")
    extra_path = os.path.join(args.output_dir, "wall_contour_global_merged_multipeak.pcd")
    backup_path = os.path.join(args.output_dir, "wall_contour_global_merged_before_multipeak.pcd")
    report_path = os.path.join(args.output_dir, "wall_contour_multipeak_report.txt")

    files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.pcd")))
    if not files:
        raise RuntimeError(f"No pcd files found in {args.pcd_dir}")
    poses = load_poses_tum(args.pose_file)
    print(f"[INFO] files={len(files)} poses={len(poses)} stride={args.frame_stride}", flush=True)

    # Adaptive voxel vote maps (main/branch tracked separately)
    vote_main = defaultdict(int)
    vote_branch = defaultdict(int)
    all_global = []

    scanned = 0
    used = 0
    no_pose = 0

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

        # local preprocessing
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

        # keep vertical-ish points as wall candidates
        p0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.normal_radius, max_nn=args.normal_max_nn
            )
        )
        nrm = np.asarray(p0.normals)
        pv = pts[np.abs(nrm[:, 2]) < args.abs_nz_max]
        if len(pv) < 80:
            continue

        # per x-bin multi peaks
        x0 = float(np.min(pv[:, 0]))
        xb = np.floor((pv[:, 0] - x0) / args.x_bin_size).astype(np.int32)
        bins = np.unique(xb)

        peaks_per_bin: Dict[int, List[Tuple[float, int]]] = {}
        for b in bins:
            ys = pv[xb == b, 1]
            peaks = find_topk_hist_peaks(
                ys=ys,
                k=args.top_k_peaks,
                y_bin_size=args.y_bin_size,
                min_peak_count=args.min_peak_count,
                smooth_win=args.peak_smooth_win,
            )
            if peaks:
                peaks_per_bin[int(b)] = peaks
        if not peaks_per_bin:
            continue

        tracks = link_tracks(peaks_per_bin, max_link_dy=args.max_link_dy)
        tracks = [t for t in tracks if t.length_bins >= args.track_min_bins]
        if not tracks:
            continue

        # define main tracks: top-N by score
        tracks_sorted = sorted(tracks, key=lambda t: t.score, reverse=True)
        main_ids = set(t.tid for t in tracks_sorted[: args.main_track_count])
        for t in tracks:
            t.peak_type = "main" if t.tid in main_ids else "branch"

        lookup = build_bin_track_lookup(tracks)

        # select contour points near multi tracks
        contour_pts = []
        contour_types = []  # "main" / "branch"
        for pt, b in zip(pv, xb):
            cands = lookup.get(int(b), [])
            if not cands:
                continue

            best_d = 1e9
            best_type = "branch"
            for yt, tp, _ in cands:
                d = abs(float(pt[1]) - yt)
                if d < best_d:
                    best_d = d
                    best_type = tp

            if best_d <= args.envelope_tol:
                contour_pts.append(pt)
                contour_types.append(best_type)

        if len(contour_pts) < 50:
            continue

        contour_pts = np.asarray(contour_pts)
        contour_types = np.asarray(contour_types)

        # local -> global
        g = (T[:3, :3] @ contour_pts.T).T + T[:3, 3]
        all_global.append(g)
        used += 1

        # adaptive temporal vote
        # use per-frame unique voxel keys to reduce density bias
        keys = np.floor(g / args.vote_voxel).astype(np.int32)
        main_keys = keys[contour_types == "main"]
        branch_keys = keys[contour_types == "branch"]

        if len(main_keys) > 0:
            uniq_main = set((int(a), int(b), int(c)) for a, b, c in np.unique(main_keys, axis=0))
            for k in uniq_main:
                vote_main[k] += 1
        if len(branch_keys) > 0:
            uniq_branch = set((int(a), int(b), int(c)) for a, b, c in np.unique(branch_keys, axis=0))
            for k in uniq_branch:
                vote_branch[k] += 1

        if scanned % 150 == 0:
            print(f"[PROGRESS] scanned={scanned} used={used} vote_vox={len(vote_main) + len(vote_branch)}", flush=True)

    if not all_global:
        raise RuntimeError("No contour points extracted")

    cont = np.concatenate(all_global, axis=0)
    keys_all = np.floor(cont / args.vote_voxel).astype(np.int32)

    # adaptive threshold by voxel dominant class
    mask = np.zeros(len(keys_all), dtype=bool)
    for i, (a, b, c) in enumerate(keys_all):
        key = (int(a), int(b), int(c))
        hm = vote_main.get(key, 0)
        hb = vote_branch.get(key, 0)
        if hm >= hb:
            mask[i] = hm >= args.main_vote_min_frames
        else:
            mask[i] = hb >= args.branch_vote_min_frames
    cont = cont[mask]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cont)
    pc = pc.voxel_down_sample(voxel_size=args.final_voxel)
    pc, _ = pc.remove_radius_outlier(nb_points=args.radius_nb_points, radius=args.radius)

    labels = np.array(
        pc.cluster_dbscan(eps=args.dbscan_eps, min_points=args.dbscan_min_points, print_progress=False)
    )
    pts = np.asarray(pc.points)
    if len(labels) == len(pts) and (labels >= 0).any():
        cnt = np.bincount(labels[labels >= 0])
        good = set(np.where(cnt >= args.cluster_min_size)[0].tolist())
        keep = np.array([(lb in good) if lb >= 0 else False for lb in labels], dtype=bool)
        pts = pts[keep]

    out_pc = o3d.geometry.PointCloud()
    out_pc.points = o3d.utility.Vector3dVector(pts)

    # write output files
    tmp_out = os.path.join("/tmp", "wall_contour_global_merged_multipeak_tmp.pcd")
    o3d.io.write_point_cloud(tmp_out, out_pc, write_ascii=False, compressed=False)
    shutil.copy2(tmp_out, extra_path)
    if os.path.exists(final_path):
        shutil.copy2(final_path, backup_path)
    shutil.copy2(tmp_out, final_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"files={len(files)} stride={args.frame_stride} scanned={scanned} used={used} no_pose={no_pose}\n")
        f.write(f"output_points={len(pts)}\n")
        f.write("mode=multipeak_envelope_branch_recall\n")
        f.write(
            f"params: top_k={args.top_k_peaks}, track_min_bins={args.track_min_bins}, "
            f"main_vote={args.main_vote_min_frames}, branch_vote={args.branch_vote_min_frames}\n"
        )

    print(f"[DONE] scanned={scanned} used={used} no_pose={no_pose}", flush=True)
    print(f"[DONE] output_points={len(pts)}", flush=True)
    print(f"[DONE] saved_default={final_path}", flush=True)
    print(f"[DONE] saved_extra={extra_path}", flush=True)
    print(f"[DONE] backup={backup_path}", flush=True)
    print(f"[DONE] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
