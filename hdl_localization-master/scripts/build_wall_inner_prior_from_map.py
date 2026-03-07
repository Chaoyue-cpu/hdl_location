#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build wall_inner + prior voxel layer directly from offline map tiles.

This script does NOT use test raw frames.
It reads pointcloud map tiles and metadata, then builds:
  - wall_contour_core_global_merged_from_map.pcd
  - wall_contour_band_global_merged_from_map.pcd
  - wall_contour_global_merged_from_map.pcd
  - wall_inner_global_merged_from_map.pcd
  - prior_voxel_layer_from_map.csv
  - per-tile sidecar: <tile_name>.prior.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import open3d as o3d


def parse_metadata(path: str) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
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
            if ":" in s and "[" in s and "]" in s:
                key, val = s.split(":", 1)
                key = key.strip()
                arr = val.strip().lstrip("[").rstrip("]")
                xs = [x.strip() for x in arr.split(",")]
                if len(xs) >= 2:
                    tiles[key] = (float(xs[0]), float(xs[1]))
    if x_res is None or y_res is None:
        raise RuntimeError(f"Invalid metadata file: {path}")
    return x_res, y_res, tiles


@dataclass
class VoxelStat:
    total: int = 0
    core: int = 0
    band: int = 0
    inner: int = 0
    uncertain: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wall_inner/prior directly from map tiles.")
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="/media/scy/新加卷/数据集/autoware_map_output/公司地图/pointcloud_map_metadata.yaml",
        help="Autoware map metadata yaml.",
    )
    parser.add_argument(
        "--tile_map_dir",
        type=str,
        default="/media/scy/新加卷/数据集/autoware_map_output/公司地图/pointcloud_map.pcd",
        help="Directory containing tile pcd files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/scy/catkin_ws_hdl_location/墙平面0416/prior_layer_map",
        help="Output directory.",
    )
    parser.add_argument("--ndt_resolution", type=float, default=2.0, help="Prior voxel size.")

    # wall candidate filtering
    parser.add_argument("--wall_z_min", type=float, default=0.5)
    parser.add_argument("--wall_z_max", type=float, default=3.0)
    parser.add_argument("--downsample", type=float, default=0.08, help="Tile downsample before normal.")
    parser.add_argument("--normal_radius", type=float, default=0.6)
    parser.add_argument("--normal_max_nn", type=int, default=28)
    parser.add_argument("--abs_nz_max", type=float, default=0.5, help="Keep vertical-ish points.")

    # PCA envelope split (main wall) + branch recall
    parser.add_argument("--u_bin", type=float, default=1.0, help="Longitudinal bin size in PCA frame.")
    parser.add_argument("--q_low", type=float, default=0.08, help="Lower quantile for wall envelope.")
    parser.add_argument("--q_high", type=float, default=0.92, help="Upper quantile for wall envelope.")
    parser.add_argument("--envelope_band", type=float, default=0.28, help="Distance to envelope to be contour.")
    parser.add_argument(
        "--contour_band_width",
        type=float,
        default=0.9,
        help="Distance to envelope to be contour band (outside core).",
    )
    parser.add_argument("--envelope_min_points", type=int, default=28, help="Min points per u-bin to compute envelope.")
    parser.add_argument(
        "--envelope_min_width",
        type=float,
        default=2.6,
        help="Min (v_right - v_left) to accept a u-bin as valid two-wall observation.",
    )
    parser.add_argument("--smooth_bins", type=int, default=5, help="Median filter window (odd) for envelope.")
    parser.add_argument(
        "--inner_wall_reject_band",
        type=float,
        default=0.55,
        help="Points within this distance to either wall envelope are excluded from inner.",
    )
    parser.add_argument(
        "--inner_between_margin",
        type=float,
        default=0.35,
        help="Inner points must be at least this far inside from both envelopes.",
    )
    parser.add_argument("--branch_recall", action="store_true", help="Enable branch wall recall from sparse side peaks.")
    parser.add_argument("--branch_edge_grid", type=float, default=0.45, help="Grid size for branch edge detection in (u,v).")
    parser.add_argument("--branch_edge_min_neighbors", type=int, default=5, help="Edge criterion in (u,v) grid.")
    parser.add_argument("--branch_offset", type=float, default=1.2, help="Min |v-center| for branch candidates.")
    parser.add_argument("--branch_min_component_cells", type=int, default=12, help="Drop tiny branch components.")
    parser.add_argument("--peak_v_bin", type=float, default=0.25, help="v-axis histogram bin size for wall peak detection.")
    parser.add_argument("--peak_min_count", type=int, default=8, help="Minimum count for a valid wall peak.")
    parser.add_argument("--peak_window", type=float, default=0.45, help="Local v-window for wall peak support stats.")
    parser.add_argument("--pair_width_min", type=float, default=2.8, help="Minimum valid wall-pair width in v.")
    parser.add_argument("--pair_width_max", type=float, default=12.0, help="Maximum valid wall-pair width in v.")
    parser.add_argument("--pair_max_interp_gap", type=int, default=3, help="Max bin gap allowed for trusted interpolation.")
    parser.add_argument("--band_min", type=float, default=0.35, help="Minimum adaptive band width.")
    parser.add_argument("--band_max", type=float, default=1.2, help="Maximum adaptive band width.")
    parser.add_argument("--band_width_ratio", type=float, default=0.08, help="Adaptive band gain from wall-pair width.")
    parser.add_argument("--band_roughness_gain", type=float, default=0.35, help="Adaptive band gain from wall roughness.")
    parser.add_argument("--band_visibility_gain", type=float, default=0.25, help="Adaptive band gain from low visibility.")
    parser.add_argument("--visibility_ref_count", type=float, default=120.0, help="Reference per-bin count for visibility.")
    parser.add_argument(
        "--contour_vertical_support",
        action="store_true",
        help="Require local vertical support for contour cells (nearby z has upper/lower spread).",
    )
    parser.add_argument("--support_xy_grid", type=float, default=0.35, help="XY grid for vertical support check.")
    parser.add_argument("--support_min_count", type=int, default=3, help="Min points in support cell.")
    parser.add_argument("--support_min_z_span", type=float, default=0.9, help="Min z-span in support cell.")
    parser.add_argument(
        "--support_min_z_max",
        type=float,
        default=2.2,
        help="Contour support cell must reach at least this z_max (suppress low objects/fences).",
    )
    parser.add_argument(
        "--band_support_min_z_span",
        type=float,
        default=0.6,
        help="Relaxed z-span support for band cells.",
    )
    parser.add_argument(
        "--band_support_min_z_max",
        type=float,
        default=2.0,
        help="Relaxed z_max support for band cells.",
    )

    # prior class thresholds
    parser.add_argument("--class_core_thresh", type=float, default=0.50)
    parser.add_argument("--class_band_thresh", type=float, default=0.50)
    parser.add_argument("--class_inner_thresh", type=float, default=0.55)
    return parser.parse_args()


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0 or k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    h = k // 2
    y = np.empty_like(x)
    for i in range(len(x)):
        a = max(0, i - h)
        b = min(len(x), i + h + 1)
        y[i] = np.median(x[a:b])
    return y


def _edge_mask_2d(coords: np.ndarray, grid: float, edge_min_neighbors: int) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros((0,), dtype=bool)
    gx = np.floor(coords[:, 0] / grid).astype(np.int32)
    gy = np.floor(coords[:, 1] / grid).astype(np.int32)
    keys = np.stack([gx, gy], axis=1)
    occ = set((int(a), int(b)) for a, b in keys)
    edge_cells = set()
    for cx, cy in occ:
        n_occ = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (cx + dx, cy + dy) in occ:
                    n_occ += 1
        if n_occ <= edge_min_neighbors:
            edge_cells.add((cx, cy))
    return np.array([(int(a), int(b)) in edge_cells for a, b in keys], dtype=bool)


def _filter_by_component_cells(coords: np.ndarray, grid: float, min_cells: int) -> np.ndarray:
    """Keep only points whose grid cell belongs to a sufficiently large connected component."""
    if len(coords) == 0:
        return np.zeros((0,), dtype=bool)

    gx = np.floor(coords[:, 0] / grid).astype(np.int32)
    gy = np.floor(coords[:, 1] / grid).astype(np.int32)
    keys = np.stack([gx, gy], axis=1)
    occ = set((int(a), int(b)) for a, b in keys)

    visited = set()
    keep_cells = set()

    for c in occ:
        if c in visited:
            continue
        q = deque([c])
        visited.add(c)
        comp = [c]
        while q:
            cx, cy = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in occ and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
                        comp.append(nb)
        if len(comp) >= min_cells:
            keep_cells.update(comp)

    return np.array([(int(a), int(b)) in keep_cells for a, b in keys], dtype=bool)


def filter_contour_by_vertical_support(
    points: np.ndarray,
    grid: float,
    min_count: int,
    min_z_span: float,
    min_z_max: float,
) -> np.ndarray:
    """
    Keep contour points only if their local XY cell has enough vertical spread.
    This suppresses fence/vehicle-like strips that do not form true wall-height structure.
    """
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)

    gx = np.floor(points[:, 0] / grid).astype(np.int32)
    gy = np.floor(points[:, 1] / grid).astype(np.int32)
    z = points[:, 2]

    cell_stats: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0, 1e9, -1e9])  # count, zmin, zmax
    for i in range(len(points)):
        key = (int(gx[i]), int(gy[i]))
        st = cell_stats[key]
        st[0] += 1
        if z[i] < st[1]:
            st[1] = float(z[i])
        if z[i] > st[2]:
            st[2] = float(z[i])

    good_cells = set()
    for k, (cnt, zmin, zmax) in cell_stats.items():
        if int(cnt) >= min_count and (float(zmax) - float(zmin)) >= min_z_span and float(zmax) >= min_z_max:
            good_cells.add(k)

    keep = np.array([(int(a), int(b)) in good_cells for a, b in zip(gx, gy)], dtype=bool)
    return keep


def classify_contour_inner_pca_envelope(
    points: np.ndarray,
    u_bin: float,
    q_low: float,
    q_high: float,
    envelope_band: float,
    contour_band_width: float,
    envelope_min_points: int,
    envelope_min_width: float,
    smooth_bins: int,
    inner_wall_reject_band: float,
    inner_between_margin: float,
    branch_recall: bool,
    branch_edge_grid: float,
    branch_edge_min_neighbors: int,
    branch_offset: float,
    branch_min_component_cells: int,
    peak_v_bin: float,
    peak_min_count: int,
    peak_window: float,
    pair_width_min: float,
    pair_width_max: float,
    pair_max_interp_gap: int,
    band_min: float,
    band_max: float,
    band_width_ratio: float,
    band_roughness_gain: float,
    band_visibility_gain: float,
    visibility_ref_count: float,
    support_min_z_span: float,
    support_min_z_max: float,
):
    """
    Split wall points into contour/inner in a tile-local PCA frame.
    - Main walls: quantile envelope in each u-bin.
    - Branch recall: sparse side edge cells, with center-offset + component filtering.
    """
    n = len(points)
    if n == 0:
        z = np.zeros((0,), dtype=bool)
        return z, z, z

    xy = points[:, :2]
    mu = np.mean(xy, axis=0)
    c = xy - mu
    cov = np.cov(c.T)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    e_u = eigvec[:, order[0]]
    e_v = eigvec[:, order[1]]

    u = c @ e_u
    v = c @ e_v

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if (u_max - u_min) < max(2.0 * u_bin, 0.5):
        # Degenerate tile: fallback to simple edge rule in PCA plane.
        edge = _edge_mask_2d(np.stack([u, v], axis=1), grid=branch_edge_grid, edge_min_neighbors=branch_edge_min_neighbors)
        band = np.zeros_like(edge)
        inner = ~edge
        return edge, band, inner

    nb = int(np.floor((u_max - u_min) / u_bin)) + 1
    bidx = np.floor((u - u_min) / u_bin).astype(np.int32)
    bidx = np.clip(bidx, 0, nb - 1)

    v_left = np.full((nb,), np.nan, dtype=np.float64)
    v_right = np.full((nb,), np.nan, dtype=np.float64)
    v_center = np.full((nb,), np.nan, dtype=np.float64)

    bin_counts = np.zeros((nb,), dtype=np.float64)
    pair_valid = np.zeros((nb,), dtype=bool)

    for bi in range(nb):
        m = bidx == bi
        cnt = int(np.count_nonzero(m))
        bin_counts[bi] = cnt
        if cnt < envelope_min_points:
            continue
        vv = v[m]
        zz = points[m, 2]

        # Detect candidate wall peaks on v-histogram, then pair two walls adaptively.
        v_min = float(np.min(vv))
        v_max = float(np.max(vv))
        if (v_max - v_min) > peak_v_bin * 3.0:
            edges = np.arange(v_min - peak_v_bin, v_max + 2.0 * peak_v_bin, peak_v_bin)
            hist, h_edges = np.histogram(vv, bins=edges)
            centers = 0.5 * (h_edges[:-1] + h_edges[1:])
            peak_ids = []
            for i in range(1, len(hist) - 1):
                if hist[i] < peak_min_count:
                    continue
                if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
                    peak_ids.append(i)

            candidates = []
            for pid in peak_ids:
                vp = centers[pid]
                pm = np.abs(vv - vp) <= peak_window
                pcnt = int(np.count_nonzero(pm))
                if pcnt < peak_min_count:
                    continue
                z_local = zz[pm]
                z_span = float(np.max(z_local) - np.min(z_local))
                z_max = float(np.max(z_local))
                if z_span < support_min_z_span or z_max < support_min_z_max:
                    continue
                candidates.append((vp, pcnt))

            best = None
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    vi, ci = candidates[i]
                    vj, cj = candidates[j]
                    width = abs(vj - vi)
                    if width < pair_width_min or width > pair_width_max:
                        continue
                    score = ci + cj
                    if best is None or score > best[0]:
                        best = (score, min(vi, vj), max(vi, vj))

            if best is not None:
                _score, l, r = best
                v_left[bi] = l
                v_right[bi] = r
                v_center[bi] = 0.5 * (l + r)
                pair_valid[bi] = True
                continue

        # Fallback: quantile envelope if no reliable pair found.
        ql = np.quantile(vv, q_low)
        qh = np.quantile(vv, q_high)
        if (qh - ql) < envelope_min_width:
            continue
        v_left[bi] = ql
        v_right[bi] = qh
        v_center[bi] = 0.5 * (ql + qh)
        pair_valid[bi] = True

    valid = np.isfinite(v_left) & np.isfinite(v_right)
    if int(np.count_nonzero(valid)) < 4:
        edge = _edge_mask_2d(np.stack([u, v], axis=1), grid=branch_edge_grid, edge_min_neighbors=branch_edge_min_neighbors)
        band = np.zeros_like(edge)
        inner = ~edge
        return edge, band, inner

    idx = np.arange(nb)
    v_left = np.interp(idx, idx[valid], v_left[valid])
    v_right = np.interp(idx, idx[valid], v_right[valid])

    c_valid = np.isfinite(v_center)
    if int(np.count_nonzero(c_valid)) >= 2:
        v_center = np.interp(idx, idx[c_valid], v_center[c_valid])
    else:
        v_center[:] = 0.5 * (v_left + v_right)

    v_left = _median_filter_1d(v_left, smooth_bins)
    v_right = _median_filter_1d(v_right, smooth_bins)
    v_center = _median_filter_1d(v_center, smooth_bins)

    # Trust bins by distance to nearest valid wall pair bin.
    dist = np.full((nb,), 10**9, dtype=np.int32)
    last = -10**9
    for i in range(nb):
        if pair_valid[i]:
            last = i
        dist[i] = i - last
    last = 10**9
    for i in range(nb - 1, -1, -1):
        if pair_valid[i]:
            last = i
        d = last - i
        if d < dist[i]:
            dist[i] = d
    trusted_bin = dist <= int(max(0, pair_max_interp_gap))

    dv_left = np.abs(v - v_left[bidx])
    dv_right = np.abs(v - v_right[bidx])
    d_wall = np.minimum(dv_left, dv_right)
    widths = np.maximum(0.5, v_right - v_left)
    dl = np.abs(np.gradient(v_left))
    dr = np.abs(np.gradient(v_right))
    rough = 0.5 * (dl + dr)
    ref = np.quantile(rough[valid], 0.90) if np.any(valid) else 1.0
    rough_n = np.clip(rough / max(1e-3, ref), 0.0, 1.0)
    visibility = np.clip(bin_counts / max(1e-3, visibility_ref_count), 0.0, 1.0)
    adaptive_band = (
        envelope_band
        + band_width_ratio * widths
        + band_roughness_gain * rough_n
        + band_visibility_gain * (1.0 - visibility)
    )
    adaptive_band = np.clip(adaptive_band, band_min, band_max)

    trusted_pt = trusted_bin[bidx]
    contour_main = (d_wall <= envelope_band) & trusted_pt
    # band radius should be at least inner reject band to ensure inner is far from walls.
    band_radius = np.maximum(adaptive_band[bidx], max(contour_band_width, inner_wall_reject_band))
    near_wall = (d_wall <= band_radius) & trusted_pt

    core = contour_main.copy()
    if branch_recall:
        uv = np.stack([u, v], axis=1)
        edge_sparse = _edge_mask_2d(uv, grid=branch_edge_grid, edge_min_neighbors=branch_edge_min_neighbors)
        side_far = np.abs(v - v_center[bidx]) >= branch_offset
        branch = edge_sparse & side_far & (~contour_main)
        if np.any(branch):
            keep = _filter_by_component_cells(uv[branch], grid=branch_edge_grid, min_cells=branch_min_component_cells)
            branch_idx = np.where(branch)[0]
            core[branch_idx[keep]] = True

    between_walls = (v >= (v_left[bidx] + inner_between_margin)) & (v <= (v_right[bidx] - inner_between_margin)) & trusted_pt
    band = near_wall & (~core)
    inner = (~near_wall) & (~core) & between_walls
    return core, band, inner


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tile_sidecar_dir = os.path.join(args.output_dir, "prior_tiles")
    os.makedirs(tile_sidecar_dir, exist_ok=True)

    x_res, y_res, tiles = parse_metadata(args.metadata_file)
    print(f"[INFO] metadata tiles={len(tiles)} x_res={x_res} y_res={y_res}", flush=True)

    vox: Dict[Tuple[int, int, int], VoxelStat] = defaultdict(VoxelStat)
    core_all: List[np.ndarray] = []
    band_all: List[np.ndarray] = []
    inner_all: List[np.ndarray] = []
    uncertain_all: List[np.ndarray] = []

    tiles_scanned = 0
    tiles_used = 0

    for tile_name, _origin in tiles.items():
        tiles_scanned += 1
        tile_path = os.path.join(args.tile_map_dir, tile_name)
        if not os.path.exists(tile_path):
            continue

        pc = o3d.io.read_point_cloud(tile_path)
        pts = np.asarray(pc.points)
        if len(pts) < 100:
            continue

        # z range wall candidates
        z = pts[:, 2]
        pts = pts[(z >= args.wall_z_min) & (z <= args.wall_z_max)]
        if len(pts) < 80:
            continue

        # downsample + vertical normal filter
        p0 = o3d.geometry.PointCloud()
        p0.points = o3d.utility.Vector3dVector(pts)
        p0 = p0.voxel_down_sample(voxel_size=args.downsample)
        pts = np.asarray(p0.points)
        if len(pts) < 60:
            continue

        p0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.normal_radius,
                max_nn=args.normal_max_nn,
            )
        )
        nrm = np.asarray(p0.normals)
        pts = pts[np.abs(nrm[:, 2]) < args.abs_nz_max]
        if len(pts) < 40:
            continue

        core_mask, band_mask, inner_mask = classify_contour_inner_pca_envelope(
            pts,
            u_bin=args.u_bin,
            q_low=args.q_low,
            q_high=args.q_high,
            envelope_band=args.envelope_band,
            contour_band_width=args.contour_band_width,
            envelope_min_points=args.envelope_min_points,
            envelope_min_width=args.envelope_min_width,
            smooth_bins=args.smooth_bins,
            inner_wall_reject_band=args.inner_wall_reject_band,
            inner_between_margin=args.inner_between_margin,
            branch_recall=args.branch_recall,
            branch_edge_grid=args.branch_edge_grid,
            branch_edge_min_neighbors=args.branch_edge_min_neighbors,
            branch_offset=args.branch_offset,
            branch_min_component_cells=args.branch_min_component_cells,
            peak_v_bin=args.peak_v_bin,
            peak_min_count=args.peak_min_count,
            peak_window=args.peak_window,
            pair_width_min=args.pair_width_min,
            pair_width_max=args.pair_width_max,
            pair_max_interp_gap=args.pair_max_interp_gap,
            band_min=args.band_min,
            band_max=args.band_max,
            band_width_ratio=args.band_width_ratio,
            band_roughness_gain=args.band_roughness_gain,
            band_visibility_gain=args.band_visibility_gain,
            visibility_ref_count=args.visibility_ref_count,
            support_min_z_span=args.support_min_z_span,
            support_min_z_max=args.support_min_z_max,
        )
        core_pts = pts[core_mask]
        band_pts = pts[band_mask]
        inner_pts = pts[inner_mask]
        uncertain_mask = ~(core_mask | band_mask | inner_mask)
        uncertain_pts = pts[uncertain_mask]

        if args.contour_vertical_support and len(core_pts) > 0:
            keep_c = filter_contour_by_vertical_support(
                core_pts,
                grid=args.support_xy_grid,
                min_count=args.support_min_count,
                min_z_span=args.support_min_z_span,
                min_z_max=args.support_min_z_max,
            )
            dropped = core_pts[~keep_c]
            core_pts = core_pts[keep_c]
            if len(dropped) > 0:
                # Downgrade unsupported core to band (still near-wall, but lower confidence).
                if len(band_pts) > 0:
                    band_pts = np.concatenate([band_pts, dropped], axis=0)
                else:
                    band_pts = dropped
        if args.contour_vertical_support and len(band_pts) > 0:
            keep_b = filter_contour_by_vertical_support(
                band_pts,
                grid=args.support_xy_grid,
                min_count=args.support_min_count,
                min_z_span=args.band_support_min_z_span,
                min_z_max=args.band_support_min_z_max,
            )
            dropped_b = band_pts[~keep_b]
            band_pts = band_pts[keep_b]
            if len(dropped_b) > 0:
                # Unsupported near-wall points are ambiguous; push to uncertain,
                # not inner, to avoid contaminating inner with contour-like strips.
                if len(uncertain_pts) > 0:
                    uncertain_pts = np.concatenate([uncertain_pts, dropped_b], axis=0)
                else:
                    uncertain_pts = dropped_b

        if len(core_pts) > 0:
            core_all.append(core_pts)
            k = np.floor(core_pts / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = vox[key]
                st.total += 1
                st.core += 1
        if len(band_pts) > 0:
            band_all.append(band_pts)
            k = np.floor(band_pts / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = vox[key]
                st.total += 1
                st.band += 1
        if len(inner_pts) > 0:
            inner_all.append(inner_pts)
            k = np.floor(inner_pts / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = vox[key]
                st.total += 1
                st.inner += 1
        if len(uncertain_pts) > 0:
            uncertain_all.append(uncertain_pts)
            k = np.floor(uncertain_pts / args.ndt_resolution).astype(np.int32)
            for a, b, c in k:
                key = (int(a), int(b), int(c))
                st = vox[key]
                st.total += 1
                st.uncertain += 1

        tiles_used += 1
        print(f"[PROGRESS] tile={tile_name} points={len(pts)} voxels={len(vox)}", flush=True)

    if not vox:
        raise RuntimeError("No voxel generated from map tiles.")

    core_merged = np.concatenate(core_all, axis=0) if core_all else np.empty((0, 3), dtype=np.float64)
    band_merged = np.concatenate(band_all, axis=0) if band_all else np.empty((0, 3), dtype=np.float64)
    contour_merged = (
        np.concatenate([core_merged, band_merged], axis=0)
        if len(core_merged) + len(band_merged) > 0
        else np.empty((0, 3), dtype=np.float64)
    )
    inner_merged = np.concatenate(inner_all, axis=0) if inner_all else np.empty((0, 3), dtype=np.float64)
    uncertain_merged = np.concatenate(uncertain_all, axis=0) if uncertain_all else np.empty((0, 3), dtype=np.float64)

    contour_core_pcd = os.path.join(args.output_dir, "wall_contour_core_global_merged_from_map.pcd")
    contour_band_pcd = os.path.join(args.output_dir, "wall_contour_band_global_merged_from_map.pcd")
    contour_pcd = os.path.join(args.output_dir, "wall_contour_global_merged_from_map.pcd")
    inner_pcd = os.path.join(args.output_dir, "wall_inner_global_merged_from_map.pcd")
    uncertain_pcd = os.path.join(args.output_dir, "wall_uncertain_global_merged_from_map.pcd")
    if len(core_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(core_merged)
        o3d.io.write_point_cloud(contour_core_pcd, pc, write_ascii=False, compressed=False)
    if len(band_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(band_merged)
        o3d.io.write_point_cloud(contour_band_pcd, pc, write_ascii=False, compressed=False)
    if len(contour_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(contour_merged)
        o3d.io.write_point_cloud(contour_pcd, pc, write_ascii=False, compressed=False)
    if len(inner_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(inner_merged)
        o3d.io.write_point_cloud(inner_pcd, pc, write_ascii=False, compressed=False)
    if len(uncertain_merged) > 0:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(uncertain_merged)
        o3d.io.write_point_cloud(uncertain_pcd, pc, write_ascii=False, compressed=False)

    # prior rows
    rows = []
    for (ix, iy, iz), st in vox.items():
        total = max(1, st.total)
        p_core = float(st.core) / float(total)
        p_band = float(st.band) / float(total)
        p_contour = float(st.core + st.band) / float(total)
        p_inner = float(st.inner) / float(total)
        p_uncertain = float(st.uncertain) / float(total)
        p_static = 1.0  # map-based prior is treated as static baseline

        if p_core >= args.class_core_thresh:
            cls = "core"
            cls_id = 3
        elif p_band >= args.class_band_thresh:
            cls = "band"
            cls_id = 2
        elif p_inner >= args.class_inner_thresh:
            cls = "inner"
            cls_id = 1
        else:
            cls = "uncertain"
            cls_id = 0

        if cls == "core":
            conf_prior = p_static * (0.30 + 0.70 * p_core)
        elif cls == "band":
            conf_prior = p_static * (0.22 + 0.68 * p_band)
        elif cls == "inner":
            conf_prior = p_static * (0.12 + 0.58 * p_inner)
        else:
            conf_prior = p_static * (0.10 + 0.25 * (1.0 - p_uncertain))

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
                "core": st.core,
                "band": st.band,
                "contour": st.core + st.band,  # backward-compatible alias
                "inner": st.inner,
                "uncertain": st.uncertain,
                "time_hits": 1,  # map-only baseline
                "class_id": cls_id,
                "class": cls,
                "p_static": p_static,
                "p_core": p_core,
                "p_band": p_band,
                "p_contour": p_contour,
                "p_inner": p_inner,
                "p_uncertain": p_uncertain,
                "conf_prior": conf_prior,
            }
        )

    fieldnames = [
        "ix",
        "iy",
        "iz",
        "x",
        "y",
        "z",
        "total",
        "core",
        "band",
        "contour",
        "inner",
        "uncertain",
        "time_hits",
        "class_id",
        "class",
        "p_static",
        "p_core",
        "p_band",
        "p_contour",
        "p_inner",
        "p_uncertain",
        "conf_prior",
    ]

    prior_csv = os.path.join(args.output_dir, "prior_voxel_layer_from_map.csv")
    with open(prior_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # tile sidecar
    tile_counts = {}
    for tile_name, (ox, oy) in tiles.items():
        out_file = os.path.join(tile_sidecar_dir, f"{tile_name}.prior.csv")
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
        "tiles_scanned": tiles_scanned,
        "tiles_used": tiles_used,
        "tiles_total": len(tiles),
        "voxels_total": len(rows),
        "core_points_total": int(len(core_merged)),
        "band_points_total": int(len(band_merged)),
        "contour_points_total": int(len(contour_merged)),
        "inner_points_total": int(len(inner_merged)),
        "uncertain_points_total": int(len(uncertain_merged)),
        "ndt_resolution": args.ndt_resolution,
        "metadata_file": args.metadata_file,
        "tile_map_dir": args.tile_map_dir,
        "tile_nonempty": int(sum(1 for v in tile_counts.values() if v > 0)),
        "mode": "map_only_prior",
    }
    with open(os.path.join(args.output_dir, "prior_voxel_layer_from_map_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] tiles_scanned={tiles_scanned} tiles_used={tiles_used}", flush=True)
    print(
        f"[DONE] core_points={len(core_merged)} band_points={len(band_merged)} "
        f"contour_points={len(contour_merged)} inner_points={len(inner_merged)} "
        f"uncertain_points={len(uncertain_merged)}",
        flush=True,
    )
    print(f"[DONE] voxels={len(rows)}", flush=True)
    print(f"[DONE] prior_csv={prior_csv}", flush=True)
    print(f"[DONE] tile_sidecar_dir={tile_sidecar_dir}", flush=True)


if __name__ == "__main__":
    main()
