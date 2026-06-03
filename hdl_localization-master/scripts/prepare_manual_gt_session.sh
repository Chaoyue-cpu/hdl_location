#!/usr/bin/env bash
set -euo pipefail

label="${1:-manual_gt_$(date +%Y-%m-%d_%H-%M-%S)}"
workspace_root="/home/scy/catkin_ws_hdl_location"
repo_root="/home/scy/catkin_ws_hdl_location/src"
pkg_root="${repo_root}/hdl_localization-master"
session_dir="${workspace_root}/帧间融合但不分解日志/${label}"

mkdir -p "${session_dir}"

cp "${pkg_root}/launch/hdl_localization-0416-gt.launch" "${session_dir}/hdl_localization-0416-gt.launch.snapshot"
cp "${pkg_root}/rviz/hdl_localization.rviz" "${session_dir}/hdl_localization.rviz.snapshot"

git -C "${repo_root}" branch --show-current > "${session_dir}/git_branch.txt"
git -C "${repo_root}" rev-parse HEAD > "${session_dir}/git_commit.txt"
git -C "${repo_root}" status --short > "${session_dir}/git_status.txt"

rg -n \
  "f2f_score_confidence_gain|enable_axis_anisotropic_fusion|f2f_axial_conf_gain|f2f_nonaxial_conf_gain|map_axial_conf_gain|map_nonaxial_conf_gain|enable_inc_static_axial_adaptation|ndt_resolution|ndt_step_size|ndt_score_good|ndt_score_bad|map_prior_enable|enable_map_prior_nonaxial_adaptation|reg_debug_csv_path" \
  "${pkg_root}/launch/hdl_localization-0416-gt.launch" \
  > "${session_dir}/launch_key_params.txt"

cat > "${session_dir}/README.txt" <<EOF
Session label: ${label}
Prepared at: $(date '+%F %T %Z')

Manual playback bag command:
  cd "/media/scy/新加卷/数据集/井下/融合定位数据20240416"
  rosbag play --clock output_imu_lidar.bag -r 0.9 --start=900

Launch command:
  source /opt/ros/noetic/setup.bash
  source /home/scy/catkin_ws_hdl_location/devel/setup.bash
  roslaunch hdl_localization hdl_localization-0416-gt.launch

RViz command:
  source /opt/ros/noetic/setup.bash
  source /home/scy/catkin_ws_hdl_location/devel/setup.bash
  rviz -d /home/scy/catkin_ws_hdl_location/src/hdl_localization-master/rviz/hdl_localization.rviz

Record /odom bag:
  source /opt/ros/noetic/setup.bash
  source /home/scy/catkin_ws_hdl_location/devel/setup.bash
  rosbag record -O "${session_dir}/odom_manual.bag" /odom

Record /odom csv:
  source /opt/ros/noetic/setup.bash
  source /home/scy/catkin_ws_hdl_location/devel/setup.bash
  rostopic echo -p /odom > "${session_dir}/odom_manual.csv"
EOF

echo "${session_dir}"
