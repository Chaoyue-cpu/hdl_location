# Tunnel3seg Axis-Adaptive Results (2026-03-29)

This folder archives the parameter files and result artifacts used for the latest tunnel3seg axial-adaptive fusion comparison.

## Data and map context
- Localization bag: `tunnel_3seg_localization_varspeed_no_cam.bag`
- GT: `gt_lidar_at_velodyne.tum`
- Centerline: `axis_tunnel3seg_localization_to_junction_20260329_centerline.csv`
- Same map slices and same localization bag were used across the compared runs below.

## Included experiments
- `E2_gain010.launch` and `E2_gain010_summary.txt`
  - Isotropic fusion baseline, `f2f_score_confidence_gain=0.1`
- `E3_lowf2f_recheck_beta01.launch`
  - Current-code recheck of the previous best E3-style setup
  - `enable_axis_anisotropic_fusion=true`
  - `enable_inc_static_axial_adaptation=true`
  - `inc_static_beta=0.1`
- `E3_lowf2f_beta005.launch`
  - Same setup as above, but `inc_static_beta=0.05`
- `E3_nostatic.launch` and `E3_nostatic_summary.txt`
  - Same anisotropic setup with `enable_inc_static_axial_adaptation=false`, used to verify whether the static-related axial adaptation branch is helpful

## Key axial-adaptive parameters
Shared key parameters for the two E3 runs:
- `f2f_score_confidence_gain=0.1`
- `enable_axis_anisotropic_fusion=true`
- `f2f_axial_conf_gain=1.0`
- `f2f_nonaxial_conf_gain=0.25`
- `map_axial_conf_gain=0.3`
- `map_nonaxial_conf_gain=1.0`
- `enable_wall_r_axial_adaptation=false`
- `enable_inc_static_axial_adaptation=true`
- `inc_static_r_deadzone=0.25`
- `inc_static_scale_min=1.0`
- `inc_static_scale_max=1.5`
- `map_prior_enable=false`
- `enable_map_prior_nonaxial_adaptation=false`

Parameter difference under test:
- Recheck baseline: `inc_static_beta=0.1`
- Tuned result: `inc_static_beta=0.05`

## Main results
### E2 isotropic baseline
- APE rmse: `3.025042 m`
- mean: `1.562234 m`
- median: `0.116999 m`
- aligned rmse: `2.593310 m`

### E3 current-code recheck (`beta=0.1`)
- APE rmse: `3.174847 m`
- mean: `1.702159 m`
- median: `0.112028 m`
- aligned rmse: `2.682618 m`

### E3 tuned (`beta=0.05`)
- APE rmse: `2.095693 m`
- mean: `0.990411 m`
- median: `0.066314 m`
- aligned rmse: `1.850746 m`

### E3 no-static ablation
- APE rmse: `3.247422 m`
- mean: `1.751967 m`
- median: `0.127178 m`
- aligned rmse: `2.736837 m`

## Interpretation
- Reducing the static-related axial adaptation strength from `0.1` to `0.05` significantly improved the axial-adaptive E3 run.
- Compared with the current-code E3 recheck (`beta=0.1`), the tuned `beta=0.05` run improved APE rmse by about `33.99%`.
- Compared with the isotropic E2 baseline, the tuned `beta=0.05` run improved APE rmse by about `30.72%`.
- The associated `reg_debug.csv` files show that the tuned run eliminated the large axial f2f takeover frames (`w_f2f_ax >= 0.2`), while map confidence stayed high throughout the bag.
- Directly disabling the static-related axial adaptation branch made the anisotropic result worse, so the branch is not useless; the issue was that its ratio was previously too strong.

## Extra analysis
- `axis_decomposition.txt` contains the pure frame-to-frame axial/nonaxial drift analysis used to interpret why overly strong axial f2f weighting hurts in this tunnel scene.
