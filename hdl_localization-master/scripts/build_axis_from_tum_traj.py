#!/usr/bin/env python3
"""
Build axis centerline/profile directly from a TUM trajectory.

Input TUM format:
  timestamp tx ty tz qx qy qz qw

Outputs:
  <prefix>_centerline.csv
  <prefix>_profile.csv
  <prefix>_transform.txt
"""

import argparse
import math
import os
from typing import Tuple

import numpy as np


def load_tum(path: str) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 8:
        raise ValueError(f"invalid TUM file: {path}")
    return arr


def dedup_traj(xy: np.ndarray, z: np.ndarray, min_step: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > min_step
    return xy[keep], z[keep]


def cumulative_s(poly: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(poly[1:] - poly[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def interp_polyline(poly: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    s_src = cumulative_s(poly)
    x = np.interp(s_query, s_src, poly[:, 0])
    y = np.interp(s_query, s_src, poly[:, 1])
    return np.column_stack([x, y])


def smooth_series(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")


def project_to_polyline(points_xy: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s_prefix = np.concatenate([[0.0], np.cumsum(seg_len)])

    best_d2 = np.full(len(points_xy), np.inf, dtype=float)
    best_s = np.zeros(len(points_xy), dtype=float)

    for i in range(len(seg)):
        a = poly[i]
        ab = seg[i]
        l2 = float(np.dot(ab, ab))
        if l2 < 1e-12:
            continue
        ap = points_xy - a[None, :]
        t = np.clip((ap @ ab) / l2, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        d2 = np.sum((points_xy - proj) ** 2, axis=1)
        better = d2 < best_d2
        best_d2[better] = d2[better]
        best_s[better] = s_prefix[i] + t[better] * seg_len[i]

    return best_s, np.sqrt(best_d2)


def robust_profile(s_vals: np.ndarray, z_vals: np.ndarray, total_s: float, bin_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nbins = int(math.floor(total_s / bin_size)) + 1
    s_centers = (np.arange(nbins, dtype=float) + 0.5) * bin_size
    z_median = np.full(nbins, np.nan, dtype=float)
    z_std = np.full(nbins, np.nan, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    idx = np.clip((s_vals / bin_size).astype(int), 0, nbins - 1)
    for b in range(nbins):
        mask = idx == b
        if not np.any(mask):
            continue
        zb = z_vals[mask]
        z_median[b] = float(np.median(zb))
        z_std[b] = float(np.std(zb))
        counts[b] = int(np.sum(mask))

    valid = np.isfinite(z_median)
    if np.sum(valid) >= 2:
        z_median = np.interp(s_centers, s_centers[valid], z_median[valid])
        fill_std = float(np.nanmedian(z_std[valid])) if np.any(valid) else 0.0
        z_std[np.isnan(z_std)] = fill_std
    elif np.sum(valid) == 1:
        z_median[:] = z_median[valid][0]
        z_std[np.isnan(z_std)] = 0.0
    else:
        z_median[:] = 0.0
        z_std[:] = 0.0

    return s_centers, z_median, np.nan_to_num(z_std), counts


def save_centerline(path: str, s_vals: np.ndarray, xy: np.ndarray, z_centerline: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,x,y,z\n")
        for s, (x, y), z in zip(s_vals, xy, z_centerline):
            f.write(f"{s:.6f},{x:.6f},{y:.6f},{z:.6f}\n")


def save_profile(path: str, s_centers: np.ndarray, z_median: np.ndarray, z_centerline: np.ndarray, counts: np.ndarray, z_std: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,z_vehicle_median,z_centerline,count,z_std\n")
        for s, z_med, z_ctr, cnt, z_sd in zip(s_centers, z_median, z_centerline, counts, z_std):
            f.write(f"{s:.6f},{z_med:.6f},{z_ctr:.6f},{int(cnt)},{z_sd:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build axis centerline/profile from TUM trajectory")
    parser.add_argument("--tum", required=True, help="Input TUM trajectory")
    parser.add_argument("--output-prefix", required=True, help="Output prefix")
    parser.add_argument("--resample-step", type=float, default=0.2, help="Uniform resampling step in meters before smoothing")
    parser.add_argument("--smooth-window-m", type=float, default=2.0, help="Moving-average window size in meters")
    parser.add_argument("--centerline-step", type=float, default=1.0, help="Centerline output sampling step in meters")
    parser.add_argument("--profile-bin-size", type=float, default=1.0, help="Profile bin size in meters")
    parser.add_argument("--z-centerline-absolute", type=float, default=1.75, help="Absolute centerline z")
    args = parser.parse_args()

    tum = load_tum(args.tum)
    raw_xy = tum[:, 1:3]
    raw_z = tum[:, 3]
    raw_xy, raw_z = dedup_traj(raw_xy, raw_z)

    raw_s = cumulative_s(raw_xy)
    total_s = float(raw_s[-1])
    if total_s <= 0.0:
        raise ValueError("degenerate trajectory")

    s_dense = np.arange(0.0, total_s + 1e-9, args.resample_step)
    dense_xy = interp_polyline(raw_xy, s_dense)

    window_samples = max(1, int(round(args.smooth_window_m / max(args.resample_step, 1e-6))))
    smooth_x = smooth_series(dense_xy[:, 0], window_samples)
    smooth_y = smooth_series(dense_xy[:, 1], window_samples)
    smooth_xy = np.column_stack([smooth_x, smooth_y])

    centerline_s = np.arange(0.0, total_s + 1e-9, args.centerline_step)
    centerline_xy = interp_polyline(smooth_xy, centerline_s)
    centerline_z = np.full(len(centerline_s), float(args.z_centerline_absolute), dtype=float)

    proj_s, lateral = project_to_polyline(raw_xy, centerline_xy)
    s_centers, z_median, z_std, counts = robust_profile(proj_s, raw_z, float(centerline_s[-1]), args.profile_bin_size)
    profile_z_centerline = np.full(len(s_centers), float(args.z_centerline_absolute), dtype=float)

    prefix_dir = os.path.dirname(args.output_prefix)
    if prefix_dir:
        os.makedirs(prefix_dir, exist_ok=True)

    centerline_csv = f"{args.output_prefix}_centerline.csv"
    profile_csv = f"{args.output_prefix}_profile.csv"
    transform_txt = f"{args.output_prefix}_transform.txt"

    save_centerline(centerline_csv, centerline_s, centerline_xy, centerline_z)
    save_profile(profile_csv, s_centers, z_median, profile_z_centerline, counts, z_std)
    with open(transform_txt, "w", encoding="utf-8") as f:
        f.write("# SE2 transform applied to input trajectory-derived centerline\n")
        f.write("r00=1.000000000 r01=0.000000000\n")
        f.write("r10=0.000000000 r11=1.000000000\n")
        f.write("tx=0.000000000 ty=0.000000000\n")

    print(f"[OK] centerline: {centerline_csv}")
    print(f"[OK] profile: {profile_csv}")
    print(f"[OK] transform: {transform_txt}")
    print(f"[STAT] poses={len(tum)} total_length={total_s:.3f} centerline_length={float(centerline_s[-1]):.3f}")
    print(f"[STAT] lateral_median={float(np.median(lateral)):.4f} lateral_p95={float(np.percentile(lateral, 95)):.4f}")


if __name__ == "__main__":
    main()
