# Tunnel 3-Segment Map-Based Localization Results (2026-03-29)

## Setup

- Map slices:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_map_refresh_20260328_215935/map_slices/pointcloud_map_metadata.yaml`
- Localization bag:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_localization_varspeed_20260328_210412/bag/tunnel_3seg_localization_varspeed_no_cam.bag`
- GT:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_localization_varspeed_20260328_210412/output/gt_lidar_at_velodyne.tum`

## Current comparable results

| Experiment | Description | Matched poses | APE RMSE | APE mean | APE median | APE max | APE aligned RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0-FAST | frame-to-map FAST_GICP only | 1826 | 1.454301 m | 0.599745 m | 0.008438 m | 4.538794 m | 1.333265 m |
| E2 | map NDT + f2f FAST_GICP, isotropic fusion | 1645 | 2.246801 m | 1.093834 m | 0.100161 m | 6.009495 m | 1.965415 m |
| E3-old-axis | map NDT + f2f FAST_GICP, anisotropic fusion, old 0416 axis files | 1686 | 2.909069 m | 1.485919 m | 0.117537 m | 7.092376 m | 2.502712 m |
| E3-reaxis | same E3 settings, but using tunnel-specific axis files regenerated from current GT | 1751 | 3.297360 m | 1.788198 m | 0.137819 m | 7.369884 m | 2.771898 m |

## Result files

- E0-FAST:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_mapbased_E0FAST_20260328_235639/localization/output/summary.txt`
- E2:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_mapbased_E2_20260328_233855/localization/output/summary.txt`
- E3-old-axis:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_mapbased_E3_20260329_095342/localization/output/summary.txt`
- E3-reaxis:
  - `/home/scy/velodyne_simulator/experiments/tunnel3seg_mapbased_E3_reaxis_20260329_102402/localization/output/summary.txt`

## Axis check

The old axis files used by the robot-sim launch presets were:

- `/home/scy/catkin_ws_hdl_location/中轴/axis_0416_new5_noalign_centerline.csv`
- `/home/scy/catkin_ws_hdl_location/中轴/axis_0416_new5_noalign_profile.csv`

These files do not match the current tunnel world trajectory geometry.

A new tunnel-specific axis pair was generated from the current localization GT and current tunnel centerline:

- `/home/scy/catkin_ws_hdl_location/中轴/axis_tunnel3seg_localization_to_junction_20260329_centerline.csv`
- `/home/scy/catkin_ws_hdl_location/中轴/axis_tunnel3seg_localization_to_junction_20260329_profile.csv`

Key observations:

- GT to old 0416 axis lateral distance:
  - median: 218.006 m
  - p95: 249.590 m
  - max: 254.295 m
- GT to current tunnel axis lateral distance:
  - median: 0.001 m
  - p95: 0.337 m
  - max: 0.687 m
- Re-running E3 with the corrected axis did not improve the result; it became worse.

## Interpretation

- For the current tunnel simulation setup, E0-FAST is still the strongest result among the tested variants.
- E2 improves over E1, but is still weaker than E0-FAST.
- E3 does not currently outperform E2.
- The E3 degradation is not explained only by the old axis-file mismatch; even after replacing the axis files with tunnel-specific ones, E3 remained worse.
