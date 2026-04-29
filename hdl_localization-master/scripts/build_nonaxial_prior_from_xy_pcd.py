#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从单个 XY 地图点云中提取“非轴向轮廓先验体素”。

设计目标：
1. 输入是一张已经投影到 XY 平面的地图点云（z 可全为 0）。
2. 不依赖测试帧，不依赖 tile metadata。
3. 在整图 PCA 坐标系中，把点云按主方向切成很多 u-bin。
4. 对每个 u-bin，在横向 v 方向估计左右轮廓包络。
5. 根据点到包络的距离，把点划分为：
   - core: 轮廓核心
   - band: 轮廓外带
   - inner: 轮廓之间的内部区域
   - uncertain: 其余不确定区域
6. 最后按 ndt_resolution 体素化，输出 prior csv。

输出文件默认放在输入点云同目录：
  - <stem>_nonaxial_contour_core.pcd
  - <stem>_nonaxial_contour_band.pcd
  - <stem>_nonaxial_contour_all.pcd
  - <stem>_nonaxial_inner.pcd
  - <stem>_nonaxial_uncertain.pcd
  - <stem>_nonaxial_prior_voxels.csv
  - <stem>_nonaxial_prior_summary.json
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


@dataclass
class VoxelStat:
    total: int = 0
    core: int = 0
    band: int = 0
    inner: int = 0
    uncertain: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nonaxial contour prior voxels from one XY PCD.")
    parser.add_argument(
        "--input_pcd",
        type=str,
        default="/home/scy/catkin_ws_hdl_location/中轴/ours-map/ours_map_from_scans_pose_zmax3p5_voxel0p3_xy.pcd",
        help="输入 XY 点云。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="输出目录。为空时默认写到输入点云同目录。",
    )
    parser.add_argument("--ndt_resolution", type=float, default=2.0, help="先验体素大小。")
    parser.add_argument("--u_bin", type=float, default=1.0, help="沿主方向的分箱宽度。")
    parser.add_argument("--q_low", type=float, default=0.08, help="左/右轮廓包络的低分位数。")
    parser.add_argument("--q_high", type=float, default=0.92, help="左/右轮廓包络的高分位数。")
    parser.add_argument("--min_pts_per_bin", type=int, default=24, help="每个 u-bin 至少多少点才参与包络估计。")
    parser.add_argument("--min_wall_width", type=float, default=2.0, help="左右包络最小宽度。")
    parser.add_argument("--smooth_bins", type=int, default=5, help="包络中值滤波窗口。")
    parser.add_argument("--core_band", type=float, default=0.35, help="核心轮廓距离阈值。")
    parser.add_argument("--outer_band", type=float, default=0.9, help="轮廓外带距离阈值。")
    parser.add_argument("--inner_margin", type=float, default=0.35, help="点被视作 inner 时，距左右包络至少这么远。")
    parser.add_argument("--downsample", type=float, default=0.2, help="输入点云预下采样分辨率。")
    return parser.parse_args()


def median_filter_1d(arr: np.ndarray, win: int) -> np.ndarray:
    """简单 1D 中值滤波，用来平滑左右包络。"""
    if len(arr) == 0 or win <= 1:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    radius = win // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - radius)
        hi = min(len(arr), i + radius + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def write_xyz_pcd(path: str, points: np.ndarray) -> None:
    """把 numpy 点集写成 PCD。"""
    if len(points) == 0:
        points = np.empty((0, 3), dtype=np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(path, pc, write_ascii=False, compressed=False)


def classify_nonaxial_regions(
    points: np.ndarray,
    u_bin: float,
    q_low: float,
    q_high: float,
    min_pts_per_bin: int,
    min_wall_width: float,
    smooth_bins: int,
    core_band: float,
    outer_band: float,
    inner_margin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    核心逻辑：
    1. 对整图 XY 做 PCA，得到主方向 u 和横向 v。
    2. 在每个 u-bin 中，用 v 的分位数估计左右轮廓。
    3. 根据点到左右轮廓的距离划分 core / band / inner / uncertain。
    """
    xy = points[:, :2]
    mean_xy = np.mean(xy, axis=0)
    centered = xy - mean_xy

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e_u = eigvecs[:, order[0]]
    e_v = eigvecs[:, order[1]]

    u = centered @ e_u
    v = centered @ e_v

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    num_bins = int(np.floor((u_max - u_min) / u_bin)) + 1
    bidx = np.floor((u - u_min) / u_bin).astype(np.int32)
    bidx = np.clip(bidx, 0, num_bins - 1)

    v_left = np.full(num_bins, np.nan, dtype=np.float64)
    v_right = np.full(num_bins, np.nan, dtype=np.float64)

    for bi in range(num_bins):
        mask = bidx == bi
        if int(np.count_nonzero(mask)) < min_pts_per_bin:
            continue
        vv = v[mask]
        ql = float(np.quantile(vv, q_low))
        qh = float(np.quantile(vv, q_high))
        if (qh - ql) < min_wall_width:
            continue
        v_left[bi] = ql
        v_right[bi] = qh

    valid = np.isfinite(v_left) & np.isfinite(v_right)
    if int(np.count_nonzero(valid)) < 4:
        raise RuntimeError("可用的包络 bin 太少，无法稳定生成非轴向轮廓先验。")

    idx = np.arange(num_bins)
    v_left = np.interp(idx, idx[valid], v_left[valid])
    v_right = np.interp(idx, idx[valid], v_right[valid])
    v_left = median_filter_1d(v_left, smooth_bins)
    v_right = median_filter_1d(v_right, smooth_bins)

    dv_left = np.abs(v - v_left[bidx])
    dv_right = np.abs(v - v_right[bidx])
    d_wall = np.minimum(dv_left, dv_right)

    # core: 非轴向轮廓最靠近左右边界的位置
    core = d_wall <= core_band

    # band: 靠近轮廓，但不在核心带里
    band = (d_wall > core_band) & (d_wall <= outer_band)

    # inner: 夹在左右轮廓之间，并且离左右边界都足够远
    inner = (
        (v >= (v_left[bidx] + inner_margin))
        & (v <= (v_right[bidx] - inner_margin))
        & (d_wall > outer_band)
    )

    uncertain = ~(core | band | inner)

    debug = {
        "u_min": u_min,
        "u_max": u_max,
        "num_bins": num_bins,
        "valid_bins": int(np.count_nonzero(valid)),
        "pca_axis_x": float(e_u[0]),
        "pca_axis_y": float(e_u[1]),
    }
    return core, band, inner, uncertain, debug


def build_prior_rows(
    points: np.ndarray,
    core_mask: np.ndarray,
    band_mask: np.ndarray,
    inner_mask: np.ndarray,
    uncertain_mask: np.ndarray,
    ndt_resolution: float,
) -> Tuple[list, Dict[str, int]]:
    """按体素聚合各类别计数，并转成 CSV 行。"""
    vox: Dict[Tuple[int, int, int], VoxelStat] = defaultdict(VoxelStat)

    def accumulate(mask: np.ndarray, field: str) -> None:
        pts = points[mask]
        if len(pts) == 0:
            return
        keys = np.floor(pts / ndt_resolution).astype(np.int32)
        for ix, iy, iz in keys:
            key = (int(ix), int(iy), int(iz))
            stat = vox[key]
            stat.total += 1
            setattr(stat, field, getattr(stat, field) + 1)

    accumulate(core_mask, "core")
    accumulate(band_mask, "band")
    accumulate(inner_mask, "inner")
    accumulate(uncertain_mask, "uncertain")

    rows = []
    class_counter = {"core": 0, "band": 0, "inner": 0, "uncertain": 0}
    for (ix, iy, iz), st in vox.items():
        total = max(st.total, 1)
        p_core = st.core / total
        p_band = st.band / total
        p_contour = (st.core + st.band) / total
        p_inner = st.inner / total
        p_uncertain = st.uncertain / total

        if p_core >= 0.50:
            cls = "core"
            cls_id = 3
            conf_prior = 0.30 + 0.70 * p_core
        elif p_band >= 0.50:
            cls = "band"
            cls_id = 2
            conf_prior = 0.22 + 0.68 * p_band
        elif p_inner >= 0.55:
            cls = "inner"
            cls_id = 1
            conf_prior = 0.12 + 0.58 * p_inner
        else:
            cls = "uncertain"
            cls_id = 0
            conf_prior = 0.10 + 0.25 * (1.0 - p_uncertain)

        class_counter[cls] += 1
        rows.append(
            {
                "ix": ix,
                "iy": iy,
                "iz": iz,
                "x": (ix + 0.5) * ndt_resolution,
                "y": (iy + 0.5) * ndt_resolution,
                "z": (iz + 0.5) * ndt_resolution,
                "total": st.total,
                "core": st.core,
                "band": st.band,
                "contour": st.core + st.band,
                "inner": st.inner,
                "uncertain": st.uncertain,
                "time_hits": 1,
                "class_id": cls_id,
                "class": cls,
                "p_static": 1.0,
                "p_core": p_core,
                "p_band": p_band,
                "p_contour": p_contour,
                "p_inner": p_inner,
                "p_uncertain": p_uncertain,
                "conf_prior": conf_prior,
            }
        )
    return rows, class_counter


def main() -> None:
    args = parse_args()
    input_pcd = os.path.abspath(args.input_pcd)
    if not os.path.exists(input_pcd):
        raise FileNotFoundError(f"输入点云不存在: {input_pcd}")

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(input_pcd)
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_pcd))[0]
    pc = o3d.io.read_point_cloud(input_pcd)
    points = np.asarray(pc.points, dtype=np.float64)
    if len(points) < 100:
        raise RuntimeError("输入点太少，无法生成先验。")

    # 对输入图再做一次轻量下采样，降低局部重复点对分位数包络的扰动。
    if args.downsample > 0:
        pc = pc.voxel_down_sample(args.downsample)
        points = np.asarray(pc.points, dtype=np.float64)

    core_mask, band_mask, inner_mask, uncertain_mask, debug = classify_nonaxial_regions(
        points=points,
        u_bin=args.u_bin,
        q_low=args.q_low,
        q_high=args.q_high,
        min_pts_per_bin=args.min_pts_per_bin,
        min_wall_width=args.min_wall_width,
        smooth_bins=args.smooth_bins,
        core_band=args.core_band,
        outer_band=args.outer_band,
        inner_margin=args.inner_margin,
    )

    rows, class_counter = build_prior_rows(
        points=points,
        core_mask=core_mask,
        band_mask=band_mask,
        inner_mask=inner_mask,
        uncertain_mask=uncertain_mask,
        ndt_resolution=args.ndt_resolution,
    )

    # 保存分类点云，方便可视化核查。
    write_xyz_pcd(os.path.join(output_dir, f"{stem}_nonaxial_contour_core.pcd"), points[core_mask])
    write_xyz_pcd(os.path.join(output_dir, f"{stem}_nonaxial_contour_band.pcd"), points[band_mask])
    write_xyz_pcd(os.path.join(output_dir, f"{stem}_nonaxial_contour_all.pcd"), points[core_mask | band_mask])
    write_xyz_pcd(os.path.join(output_dir, f"{stem}_nonaxial_inner.pcd"), points[inner_mask])
    write_xyz_pcd(os.path.join(output_dir, f"{stem}_nonaxial_uncertain.pcd"), points[uncertain_mask])

    prior_csv = os.path.join(output_dir, f"{stem}_nonaxial_prior_voxels.csv")
    fieldnames = [
        "ix", "iy", "iz", "x", "y", "z",
        "total", "core", "band", "contour", "inner", "uncertain",
        "time_hits", "class_id", "class", "p_static",
        "p_core", "p_band", "p_contour", "p_inner", "p_uncertain", "conf_prior",
    ]
    with open(prior_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "input_pcd": input_pcd,
        "output_dir": output_dir,
        "input_points_after_downsample": int(len(points)),
        "core_points": int(np.count_nonzero(core_mask)),
        "band_points": int(np.count_nonzero(band_mask)),
        "contour_points": int(np.count_nonzero(core_mask | band_mask)),
        "inner_points": int(np.count_nonzero(inner_mask)),
        "uncertain_points": int(np.count_nonzero(uncertain_mask)),
        "voxels_total": int(len(rows)),
        "class_counter": class_counter,
        "ndt_resolution": args.ndt_resolution,
        "u_bin": args.u_bin,
        "core_band": args.core_band,
        "outer_band": args.outer_band,
        "inner_margin": args.inner_margin,
        "debug": debug,
    }
    summary_path = os.path.join(output_dir, f"{stem}_nonaxial_prior_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] prior csv: {prior_csv}")
    print(f"[OK] summary: {summary_path}")
    print(
        "[STAT] "
        f"points={len(points)} core={np.count_nonzero(core_mask)} band={np.count_nonzero(band_mask)} "
        f"inner={np.count_nonzero(inner_mask)} uncertain={np.count_nonzero(uncertain_mask)} voxels={len(rows)}"
    )


if __name__ == "__main__":
    main()
