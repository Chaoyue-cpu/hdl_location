#!/usr/bin/env python3
"""
Build a 1D axis-height profile from a TUM pose file.

Input pose format:
  timestamp tx ty tz qx qy qz qw

Main outputs:
  1) <prefix>_profile.csv
  2) <prefix>_centerline.csv
  3) <prefix>_transform.txt
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
        raise ValueError(f"Invalid TUM format: {path}")
    return arr


def polyline_lengths(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg = poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s_prefix = np.concatenate([[0.0], np.cumsum(seg_len)])
    return seg_len, s_prefix


def sample_polyline(poly: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    seg_len, s_prefix = polyline_lengths(poly)
    total = float(s_prefix[-1])
    if total <= 0:
        raise ValueError("Degenerate polyline")

    s_vals = np.arange(0.0, total + 1e-9, step)
    pts = np.zeros((len(s_vals), 2), dtype=float)

    for i, s in enumerate(s_vals):
        seg_idx = int(np.searchsorted(s_prefix, s, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(seg_len) - 1))
        ds = s - s_prefix[seg_idx]
        ratio = 0.0 if seg_len[seg_idx] < 1e-9 else ds / seg_len[seg_idx]
        pts[i] = poly[seg_idx] + ratio * (poly[seg_idx + 1] - poly[seg_idx])

    return s_vals, pts


def principal_angle(points: np.ndarray) -> float:
    c = points - points.mean(axis=0)
    cov = (c.T @ c) / max(len(points) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    return float(math.atan2(axis[1], axis[0]))


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def nearest_indices(src: np.ndarray, dst: np.ndarray, chunk: int = 512) -> np.ndarray:
    idx = np.zeros(len(src), dtype=int)
    for i in range(0, len(src), chunk):
        j = min(i + chunk, len(src))
        block = src[i:j]
        d2 = np.sum((block[:, None, :] - dst[None, :, :]) ** 2, axis=2)
        idx[i:j] = np.argmin(d2, axis=1)
    return idx


def kabsch_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    xs = src - cs
    xd = dst - cd
    h = xs.T @ xd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[1, :] *= -1.0
        r = vt.T @ u.T
    t = cd - r @ cs
    return r, t


def icp_align_polyline_to_traj(poly_samples: np.ndarray, traj_xy: np.ndarray, iters: int) -> Tuple[np.ndarray, np.ndarray, float]:
    theta = principal_angle(traj_xy) - principal_angle(poly_samples)
    r = rot2(theta)
    t = traj_xy.mean(axis=0) - (poly_samples @ r.T).mean(axis=0)

    rmse = 0.0
    for _ in range(iters):
        src = poly_samples @ r.T + t
        nn_idx = nearest_indices(src, traj_xy)
        dst = traj_xy[nn_idx]
        dr, dt = kabsch_2d(src, dst)
        r = dr @ r
        t = dr @ t + dt
        src2 = poly_samples @ r.T + t
        rmse = float(np.sqrt(np.mean(np.sum((src2 - dst) ** 2, axis=1))))

    return r, t, rmse


def project_to_polyline(points_xy: np.ndarray, poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg_len, s_prefix = polyline_lengths(poly)
    n = len(points_xy)
    best_d2 = np.full(n, np.inf, dtype=float)
    best_s = np.zeros(n, dtype=float)

    for i in range(len(seg_len)):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
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


def robust_profile(s: np.ndarray, z: np.ndarray, s_total: float, bin_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nb = int(math.floor(s_total / bin_size)) + 1
    centers = (np.arange(nb, dtype=float) + 0.5) * bin_size
    z_median = np.full(nb, np.nan, dtype=float)
    z_std = np.full(nb, np.nan, dtype=float)
    counts = np.zeros(nb, dtype=int)

    idx = np.clip((s / bin_size).astype(int), 0, nb - 1)
    for b in range(nb):
        m = idx == b
        if not np.any(m):
            continue
        zb = z[m]
        z_median[b] = float(np.median(zb))
        z_std[b] = float(np.std(zb))
        counts[b] = int(np.sum(m))

    valid = np.isfinite(z_median)
    if np.sum(valid) >= 2:
        z_median = np.interp(centers, centers[valid], z_median[valid])
        z_std[np.isnan(z_std)] = float(np.nanmedian(z_std[valid])) if np.any(valid) else 0.0
    elif np.sum(valid) == 1:
        z_median[:] = z_median[valid][0]
        z_std[np.isnan(z_std)] = 0.0
    else:
        z_median[:] = 0.0
        z_std[:] = 0.0

    return centers, z_median, np.nan_to_num(z_std), counts


def save_profile_csv(path: str, s_centers: np.ndarray, z_med: np.ndarray, z_center: np.ndarray, z_std: np.ndarray, counts: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,z_vehicle_median,z_centerline,count,z_std\n")
        for i in range(len(s_centers)):
            f.write(f"{s_centers[i]:.6f},{z_med[i]:.6f},{z_center[i]:.6f},{int(counts[i])},{z_std[i]:.6f}\n")


def save_centerline_csv(path: str, s_samples: np.ndarray, xy_samples: np.ndarray, z_samples: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("s,x,y,z\n")
        for i in range(len(s_samples)):
            f.write(f"{s_samples[i]:.6f},{xy_samples[i,0]:.6f},{xy_samples[i,1]:.6f},{z_samples[i]:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1D centerline z(s) from TUM poses")
    parser.add_argument("--tum", required=True, help="Path to TUM file")
    parser.add_argument(
        "--point",
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        required=True,
        help="Centerline control point in XY, can repeat --point",
    )
    parser.add_argument("--output-prefix", default="/tmp/axis_profile", help="Output prefix path")
    parser.add_argument("--z-offset", type=float, default=1.5, help="Centerline height offset relative to vehicle pose")
    parser.add_argument(
        "--z-centerline-absolute",
        type=float,
        default=None,
        help="Absolute z for centerline. If set, overrides --z-offset and keeps z_centerline fixed.",
    )
    parser.add_argument("--bin-size", type=float, default=1.0, help="z(s) bin size in meters")
    parser.add_argument("--sample-step", type=float, default=0.5, help="Polyline sampling step for alignment")
    parser.add_argument("--centerline-step", type=float, default=1.0, help="Centerline output sampling step")
    parser.add_argument("--max-lateral-dist", type=float, default=8.0, help="Use poses within this distance to axis for z profile")
    parser.add_argument("--auto-se2", action="store_true", help="Auto align centerline XY to trajectory XY with SE2 ICP")
    parser.add_argument("--icp-iters", type=int, default=12, help="ICP iterations for auto-se2")
    args = parser.parse_args()

    tum = load_tum(args.tum)
    traj_xy = tum[:, 1:3]
    traj_z = tum[:, 3]
    ctrl = np.array([[float(x), float(y)] for x, y in args.point], dtype=float)
    if len(ctrl) < 2:
        raise ValueError("At least 2 centerline points are required")

    r = np.eye(2, dtype=float)
    t = np.zeros(2, dtype=float)
    rmse = float("nan")
    if args.auto_se2:
        _, ctrl_samp = sample_polyline(ctrl, args.sample_step)
        r, t, rmse = icp_align_polyline_to_traj(ctrl_samp, traj_xy, args.icp_iters)
        ctrl = ctrl @ r.T + t

    s_traj, d_traj = project_to_polyline(traj_xy, ctrl)
    _, s_prefix = polyline_lengths(ctrl)
    s_total = float(s_prefix[-1])
    use = d_traj <= args.max_lateral_dist
    if np.sum(use) < 10:
        print(f"[WARN] Only {int(np.sum(use))} poses within {args.max_lateral_dist:.2f}m of axis. "
              "Check centerline points or enable --auto-se2.")

    s_centers, z_med, z_std, counts = robust_profile(s_traj[use], traj_z[use], s_total, args.bin_size)
    if args.z_centerline_absolute is not None:
        z_center = np.full_like(z_med, float(args.z_centerline_absolute))
    else:
        z_center = z_med + args.z_offset

    s_out, xy_out = sample_polyline(ctrl, args.centerline_step)
    z_out = np.interp(s_out, s_centers, z_center)

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    profile_csv = f"{args.output_prefix}_profile.csv"
    centerline_csv = f"{args.output_prefix}_centerline.csv"
    tf_txt = f"{args.output_prefix}_transform.txt"

    save_profile_csv(profile_csv, s_centers, z_med, z_center, z_std, counts)
    save_centerline_csv(centerline_csv, s_out, xy_out, z_out)
    with open(tf_txt, "w", encoding="utf-8") as f:
        f.write("# SE2 transform applied to input centerline points\n")
        f.write(f"r00={r[0,0]:.9f} r01={r[0,1]:.9f}\n")
        f.write(f"r10={r[1,0]:.9f} r11={r[1,1]:.9f}\n")
        f.write(f"tx={t[0]:.9f} ty={t[1]:.9f}\n")
        if np.isfinite(rmse):
            f.write(f"icp_rmse={rmse:.6f}\n")

    print(f"[OK] profile: {profile_csv}")
    print(f"[OK] centerline: {centerline_csv}")
    print(f"[OK] transform: {tf_txt}")
    print(f"[STAT] poses={len(tum)} used={int(np.sum(use))} lateral_all_median={float(np.median(d_traj)):.3f} "
          f"lateral_all_p95={float(np.percentile(d_traj, 95)):.3f}")
    if np.any(use):
        print(f"[STAT] used_lateral_median={float(np.median(d_traj[use])):.3f} "
              f"used_lateral_p95={float(np.percentile(d_traj[use], 95)):.3f}")
    if np.isfinite(rmse):
        print(f"[STAT] auto-se2 icp_rmse={rmse:.3f}")


if __name__ == "__main__":
    main()
