#include <hdl_localization/pose_estimator.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pclomp/ndt_omp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <hdl_localization/pose_system.hpp>
#include <hdl_localization/odom_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

namespace {
struct VoxelKey {
  int x;
  int y;
  int z;

  bool operator==(const VoxelKey& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t hx = static_cast<std::size_t>(std::hash<int>{}(key.x));
    std::size_t hy = static_cast<std::size_t>(std::hash<int>{}(key.y));
    std::size_t hz = static_cast<std::size_t>(std::hash<int>{}(key.z));
    return hx ^ (hy << 1) ^ (hz << 2);
  }
};

VoxelKey to_voxel_key(const Eigen::Vector3f& p, double voxel_size) {
  const float inv = 1.0f / static_cast<float>(voxel_size);
  return VoxelKey{static_cast<int>(std::floor(p.x() * inv)), static_cast<int>(std::floor(p.y() * inv)), static_cast<int>(std::floor(p.z() * inv))};
}
}  // namespace

/**
 * @brief constructor
 * @param registration        registration method
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
// PoseEstimator构造函数
// 输入：
// registration              —— 点云配准方法（例如ICP或NDT）
// pos                       —— 初始位置（3维）
// quat                      —— 初始姿态（四元数）
// cool_time_duration        —— “冷却时间”，用于控制何时更新位姿
PoseEstimator::PoseEstimator(
  pcl::Registration<PointT, PointT>::Ptr& registration,
  const Eigen::Vector3f& pos,
  const Eigen::Quaternionf& quat,
  double cool_time_duration,
  bool enable_frame2frame_ndt,
  int frame_to_frame_reg_num_threads,
  const std::string& frame2frame_reg_method,
  bool enable_score_weighted_fusion,
  double ndt_score_good,
  double ndt_score_bad,
  double ndt_score_min_confidence,
  double f2f_score_confidence_gain,
  bool enable_axis_anisotropic_fusion,
  double f2f_axial_conf_gain,
  double f2f_nonaxial_conf_gain,
  bool enable_f2f_confidence_filter,
  bool enable_f2f_dynamic_filter,
  double f2f_wall_y_threshold,
  double f2f_wall_z_min,
  double f2f_wall_z_max,
  double f2f_wall_keep_ratio,
  double f2f_dynamic_voxel_size,
  double f2f_dynamic_keep_ratio,
  int f2f_min_filtered_points,
  bool enable_axis_prior,
  const std::string& axis_centerline_csv,
  const std::string& axis_profile_csv,
  double axis_search_window,
  double axis_lateral_weight,
  double axis_vertical_weight,
  double axis_smooth_weight,
  double axis_temporal_weight,
  double axis_max_lateral,
  double axis_max_delta_s,
  bool axis_prealign_only_when_degenerate,
  bool axis_postalign_only_when_degenerate)
: registration(registration),
  cool_time_duration(cool_time_duration),
  enable_frame2frame_ndt(enable_frame2frame_ndt),
  map_trans_noise(0.3),
  map_rot_noise(0.1),
  f2f_trans_noise(0.9),
  f2f_rot_noise(0.3),
  enable_score_weighted_fusion(enable_score_weighted_fusion),
  ndt_score_good(ndt_score_good),
  ndt_score_bad(ndt_score_bad),
  ndt_score_min_confidence(ndt_score_min_confidence),
  f2f_score_confidence_gain(std::max(0.0, f2f_score_confidence_gain)),
  enable_axis_anisotropic_fusion(enable_axis_anisotropic_fusion),
  f2f_axial_conf_gain(std::max(0.0, f2f_axial_conf_gain)),
  f2f_nonaxial_conf_gain(std::max(0.0, f2f_nonaxial_conf_gain)),
  enable_f2f_confidence_filter(enable_f2f_confidence_filter),
  enable_f2f_dynamic_filter(enable_f2f_dynamic_filter),
  f2f_wall_y_threshold(std::max(0.0, f2f_wall_y_threshold)),
  f2f_wall_z_min(std::min(f2f_wall_z_min, f2f_wall_z_max)),
  f2f_wall_z_max(std::max(f2f_wall_z_min, f2f_wall_z_max)),
  f2f_wall_keep_ratio(std::max(0.0, std::min(1.0, f2f_wall_keep_ratio))),
  f2f_dynamic_voxel_size(std::max(0.05, f2f_dynamic_voxel_size)),
  f2f_dynamic_keep_ratio(std::max(0.0, std::min(1.0, f2f_dynamic_keep_ratio))),
  f2f_min_filtered_points(std::max(50, f2f_min_filtered_points)),
  enable_axis_prior(enable_axis_prior),
  axis_search_window(std::max(1.0, axis_search_window)),
  axis_lateral_weight(std::max(0.0, axis_lateral_weight)),
  axis_vertical_weight(std::max(0.0, axis_vertical_weight)),
  axis_smooth_weight(std::max(0.0, axis_smooth_weight)),
  axis_temporal_weight(std::max(0.0, axis_temporal_weight)),
  axis_max_lateral(std::max(0.5, axis_max_lateral)),
  axis_max_delta_s(std::max(0.01, axis_max_delta_s)),
  axis_prealign_only_when_degenerate(axis_prealign_only_when_degenerate),
  axis_postalign_only_when_degenerate(axis_postalign_only_when_degenerate),
  last_map_degenerate(false),
  prev_map_pose(Eigen::Matrix4f::Identity()) {
  // 初始化最后一次观测的变换矩阵（4x4单位矩阵）
  last_observation = Eigen::Matrix4f::Identity();

  // 设置旋转部分为初始四元数对应的旋转矩阵
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();

  // 设置平移部分为初始位置向量
  last_observation.block<3, 1>(0, 3) = pos;
  prev_map_pose = last_observation;

  // 设置过程噪声协方差矩阵（16x16），用于UKF的状态预测
  process_noise = Eigen::MatrixXf::Identity(16, 16);

  // 各部分过程噪声配置（根据状态向量结构设定）：
  // 状态结构：pos(3)+vel(3)+quat(4)+acc_bias(3)+gyro_bias(3)

  // 位置（前3行）和速度（第4~6行）过程噪声设为 1.0
  process_noise.middleRows(0, 3) *= 1.0;
  process_noise.middleRows(3, 3) *= 1.0;

  // 姿态（四元数，第7~10行）过程噪声设为 0.5,不能设置过大,比如设置1.5,自由度调节过大了
  process_noise.middleRows(6, 4) *= 0.5;

  // 加速度偏差和陀螺仪偏差（最后6行）设为极小值，表示其缓慢变化
  process_noise.middleRows(10, 3) *= 1e-6;
  process_noise.middleRows(13, 3) *= 1e-6;

  // 初始化观测噪声协方差矩阵（7x7），对应观测是 位置+姿态（3+4）
  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);

  // 位置观测噪声设为0.01，姿态观测噪声设为0.001
  measurement_noise.middleRows(0, 3) *= 0.01;
  measurement_noise.middleRows(3, 4) *= 0.001;

  // 初始化状态向量（16维）均值,默认列向量
  Eigen::VectorXf mean(16);

  // 前3维：位置，初始化为输入位置
  mean.middleRows(0, 3) = pos;

  // 中间3维：速度初始化为0
  mean.middleRows(3, 3).setZero();

  // 姿态四元数（第6~9行），归一化后设为输入姿态
  mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z()).normalized();

  // 加速度偏差与陀螺仪偏差都初始化为0
  mean.middleRows(10, 3).setZero();
  mean.middleRows(13, 3).setZero();

  // 初始协方差矩阵（16x16），整体设为较小值 0.01，表示初始较确信
  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  // 创建系统模型（预测模型），用于UKF中状态传播
  PoseSystem system;

  // 构造UKF滤波器（16维状态，6维控制，7维观测），构造函数定义在ukf头文件里
  ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(
    system,             // 系统模型
    16,                 // 状态维度
    6,                  // 控制输入维度（未用，可设为0或6）
    7,                  // 观测维度（位置3 + 姿态4）
    process_noise,      // 过程噪声协方差
    measurement_noise,  // 观测噪声协方差
    mean,               // 初始状态均值
    cov                 // 初始协方差
    ));

  std::string f2f_method_upper = frame2frame_reg_method;
  std::transform(f2f_method_upper.begin(), f2f_method_upper.end(), f2f_method_upper.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

  if (f2f_method_upper == "FAST_GICP") {
    fast_gicp::FastGICP<PointT, PointT>::Ptr f2f_fast_gicp(new fast_gicp::FastGICP<PointT, PointT>());
    f2f_fast_gicp->setNumThreads(std::max(1, frame_to_frame_reg_num_threads));
    f2f_fast_gicp->setTransformationEpsilon(1e-3);
    f2f_fast_gicp->setMaximumIterations(64);
    f2f_fast_gicp->setMaxCorrespondenceDistance(2.0);
    f2f_fast_gicp->setCorrespondenceRandomness(20);
    frame2frame_registration = f2f_fast_gicp;
    ROS_INFO_STREAM("frame-to-frame registration: FAST_GICP");
  } else {
    if (f2f_method_upper != "NDT_OMP") {
      ROS_WARN_STREAM("unknown frame2frame_reg_method: " << frame2frame_reg_method << ", fallback to NDT_OMP");
    }
    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    ndt->setResolution(1.0);
    ndt->setTransformationEpsilon(1e-3);
    ndt->setMaximumIterations(30);
    ndt->setNumThreads(std::max(1, frame_to_frame_reg_num_threads));
    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    frame2frame_registration = ndt;
    ROS_INFO_STREAM("frame-to-frame registration: NDT_OMP");
  }

  if (enable_axis_prior) {
    const bool ok_centerline = load_axis_centerline_csv(axis_centerline_csv);
    const bool ok_profile = load_axis_profile_csv(axis_profile_csv);
    if (!ok_centerline) {
      ROS_WARN_STREAM("axis prior disabled: failed to load centerline csv: " << axis_centerline_csv);
      this->enable_axis_prior = false;
    } else if (!ok_profile) {
      ROS_WARN_STREAM("axis profile csv missing or invalid, axis prior will use XY only: " << axis_profile_csv);
    } else {
      ROS_INFO_STREAM("axis prior loaded: centerline_samples=" << axis_centerline.size() << " profile_samples=" << axis_profile.size());
      double init_s = 0.0;
      double init_lateral = 0.0;
      if (project_to_axis(pos, &init_s, &init_lateral)) {
        last_axis_s = init_s;
        ROS_INFO_STREAM("axis prior init from pose: s=" << init_s << " lateral=" << init_lateral);
      } else {
        ROS_WARN_STREAM("axis prior init failed to project initial pose to centerline");
      }
    }
  }
}

PoseEstimator::~PoseEstimator() {}

void PoseEstimator::set_registration(const pcl::Registration<PointT, PointT>::Ptr& reg) {
  std::lock_guard<std::mutex> lk(reg_mtx_);
  registration = reg;  // O(1) 指针交换
}
pcl::Registration<PoseEstimator::PointT, PoseEstimator::PointT>::Ptr PoseEstimator::getRegistration() const {
  std::lock_guard<std::mutex> lk(reg_mtx_);
  return registration;  // 返回 shared_ptr 副本（O(1)）
}

static double now_thread_cpu_ms() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

bool PoseEstimator::is_wall_point(const PointT& pt) const {
  if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
    return false;
  }
  return std::abs(pt.y) >= f2f_wall_y_threshold && pt.z >= f2f_wall_z_min && pt.z <= f2f_wall_z_max;
}

bool PoseEstimator::keep_point_by_ratio(const PointT& pt, double keep_ratio, int salt) const {
  if (keep_ratio >= 1.0) {
    return true;
  }
  if (keep_ratio <= 0.0) {
    return false;
  }

  const int qx = static_cast<int>(std::lround(pt.x * 100.0f));
  const int qy = static_cast<int>(std::lround(pt.y * 100.0f));
  const int qz = static_cast<int>(std::lround(pt.z * 100.0f));

  std::uint32_t h = 2166136261u;
  h = (h ^ static_cast<std::uint32_t>(qx + 374761393 + salt * 17)) * 16777619u;
  h = (h ^ static_cast<std::uint32_t>(qy + 668265263 + salt * 31)) * 16777619u;
  h = (h ^ static_cast<std::uint32_t>(qz + 2246822519u + salt * 43)) * 16777619u;

  const double u = static_cast<double>(h) / static_cast<double>(std::numeric_limits<std::uint32_t>::max());
  return u <= keep_ratio;
}

bool PoseEstimator::load_axis_centerline_csv(const std::string& path) {
  axis_centerline.clear();
  if (path.empty()) {
    return false;
  }

  std::ifstream ifs(path);
  if (!ifs.good()) {
    return false;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.find("s,") != std::string::npos) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> cols;
    while (std::getline(ss, cell, ',')) {
      cols.emplace_back(cell);
    }
    if (cols.size() < 4) {
      continue;
    }

    AxisSample sample;
    try {
      sample.s = std::stod(cols[0]);
      sample.p.x() = static_cast<float>(std::stod(cols[1]));
      sample.p.y() = static_cast<float>(std::stod(cols[2]));
      sample.p.z() = static_cast<float>(std::stod(cols[3]));
    } catch (...) {
      continue;
    }
    axis_centerline.emplace_back(sample);
  }

  if (axis_centerline.size() < 2) {
    axis_centerline.clear();
    return false;
  }

  std::sort(axis_centerline.begin(), axis_centerline.end(), [](const AxisSample& a, const AxisSample& b) { return a.s < b.s; });
  return true;
}

bool PoseEstimator::load_axis_profile_csv(const std::string& path) {
  axis_profile.clear();
  if (path.empty()) {
    return false;
  }

  std::ifstream ifs(path);
  if (!ifs.good()) {
    return false;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.find("s,") != std::string::npos) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> cols;
    while (std::getline(ss, cell, ',')) {
      cols.emplace_back(cell);
    }
    if (cols.size() < 3) {
      continue;
    }

    AxisZSample sample;
    try {
      sample.s = std::stod(cols[0]);
      sample.z = std::stod(cols[2]);  // z_centerline
    } catch (...) {
      continue;
    }
    axis_profile.emplace_back(sample);
  }

  if (axis_profile.empty()) {
    return false;
  }

  std::sort(axis_profile.begin(), axis_profile.end(), [](const AxisZSample& a, const AxisZSample& b) { return a.s < b.s; });
  return true;
}

bool PoseEstimator::interpolate_axis_sample(double s, Eigen::Vector3f* point, Eigen::Vector3f* tangent) const {
  if (axis_centerline.size() < 2 || !point || !tangent) {
    return false;
  }

  if (s <= axis_centerline.front().s) {
    const Eigen::Vector3f t = (axis_centerline[1].p - axis_centerline[0].p).normalized();
    *point = axis_centerline.front().p;
    *tangent = t;
    return true;
  }
  if (s >= axis_centerline.back().s) {
    const std::size_t n = axis_centerline.size();
    const Eigen::Vector3f t = (axis_centerline[n - 1].p - axis_centerline[n - 2].p).normalized();
    *point = axis_centerline.back().p;
    *tangent = t;
    return true;
  }

  const auto it = std::lower_bound(axis_centerline.begin(), axis_centerline.end(), s, [](const AxisSample& a, double v) { return a.s < v; });
  const std::size_t hi = static_cast<std::size_t>(it - axis_centerline.begin());
  const std::size_t lo = hi - 1;
  const double ds = std::max(axis_centerline[hi].s - axis_centerline[lo].s, 1e-6);
  const double r = (s - axis_centerline[lo].s) / ds;
  *point = static_cast<float>(1.0 - r) * axis_centerline[lo].p + static_cast<float>(r) * axis_centerline[hi].p;
  *tangent = (axis_centerline[hi].p - axis_centerline[lo].p).normalized();
  return true;
}

bool PoseEstimator::interpolate_axis_z(double s, double* z) const {
  if (!z || axis_profile.empty()) {
    return false;
  }

  if (s <= axis_profile.front().s) {
    *z = axis_profile.front().z;
    return true;
  }
  if (s >= axis_profile.back().s) {
    *z = axis_profile.back().z;
    return true;
  }

  const auto it = std::lower_bound(axis_profile.begin(), axis_profile.end(), s, [](const AxisZSample& a, double v) { return a.s < v; });
  const std::size_t hi = static_cast<std::size_t>(it - axis_profile.begin());
  const std::size_t lo = hi - 1;
  const double ds = std::max(axis_profile[hi].s - axis_profile[lo].s, 1e-6);
  const double r = (s - axis_profile[lo].s) / ds;
  *z = (1.0 - r) * axis_profile[lo].z + r * axis_profile[hi].z;
  return true;
}

bool PoseEstimator::project_to_axis(const Eigen::Vector3f& p, double* s, double* lateral_distance) const {
  if (axis_centerline.size() < 2 || !s || !lateral_distance) {
    return false;
  }

  double best_d2 = std::numeric_limits<double>::max();
  double best_s = axis_centerline.front().s;

  for (std::size_t i = 0; i + 1 < axis_centerline.size(); ++i) {
    const Eigen::Vector3f a = axis_centerline[i].p;
    const Eigen::Vector3f b = axis_centerline[i + 1].p;
    Eigen::Vector2f ab = (b - a).head<2>();
    const float l2 = ab.squaredNorm();
    if (l2 < 1e-9f) {
      continue;
    }

    Eigen::Vector2f ap = (p - a).head<2>();
    float t = ap.dot(ab) / l2;
    t = std::max(0.0f, std::min(1.0f, t));
    const Eigen::Vector2f proj = a.head<2>() + t * ab;
    const Eigen::Vector2f diff = p.head<2>() - proj;
    const double d2 = static_cast<double>(diff.squaredNorm());
    if (d2 < best_d2) {
      best_d2 = d2;
      const double seg_s = axis_centerline[i].s + static_cast<double>(t) * (axis_centerline[i + 1].s - axis_centerline[i].s);
      best_s = seg_s;
    }
  }

  *s = best_s;
  *lateral_distance = std::sqrt(std::max(0.0, best_d2));
  return true;
}

Eigen::Matrix4f PoseEstimator::apply_axis_prior(const Eigen::Matrix4f& pose, const char* stage) {
  if (!enable_axis_prior || axis_centerline.size() < 2) {
    return pose;
  }

  Eigen::Matrix4f adjusted = pose;
  const Eigen::Vector3f p = pose.block<3, 1>(0, 3);

  double s_ref = 0.0;
  double lateral = 0.0;
  if (!project_to_axis(p, &s_ref, &lateral)) {
    return pose;
  }
  if (lateral > axis_max_lateral) {
    ROS_WARN_STREAM_THROTTLE(1.0, "[AXIS] skip prior at " << stage << " due to large lateral distance: " << lateral);
    return pose;
  }

  const double s_min = std::max(axis_centerline.front().s, s_ref - axis_search_window);
  const double s_max = std::min(axis_centerline.back().s, s_ref + axis_search_window);

  double best_s = s_ref;
  double best_cost = std::numeric_limits<double>::max();
  for (const auto& sample : axis_centerline) {
    if (sample.s < s_min || sample.s > s_max) {
      continue;
    }

    const Eigen::Vector2f dxy = p.head<2>() - sample.p.head<2>();
    double cost = axis_lateral_weight * static_cast<double>(dxy.squaredNorm());
    const double ds_ref = sample.s - s_ref;
    cost += axis_smooth_weight * ds_ref * ds_ref;

    if (!axis_profile.empty()) {
      double z_model = sample.p.z();
      interpolate_axis_z(sample.s, &z_model);
      const double dz = static_cast<double>(p.z()) - z_model;
      cost += axis_vertical_weight * dz * dz;
    }

    if (last_axis_s) {
      const double ds_t = sample.s - *last_axis_s;
      cost += axis_temporal_weight * ds_t * ds_t;
    }

    if (cost < best_cost) {
      best_cost = cost;
      best_s = sample.s;
    }
  }

  const double raw_delta_s = best_s - s_ref;
  const double delta_s = std::max(-axis_max_delta_s, std::min(axis_max_delta_s, raw_delta_s));
  Eigen::Vector3f axis_point;
  Eigen::Vector3f axis_tangent;
  if (!interpolate_axis_sample(s_ref, &axis_point, &axis_tangent)) {
    return pose;
  }
  adjusted.block<3, 1>(0, 3) = p + static_cast<float>(delta_s) * axis_tangent;
  last_axis_s = s_ref + delta_s;

  ROS_INFO_STREAM_THROTTLE(
    0.5,
    "[AXIS] " << stage << " s_ref=" << s_ref << " s_best=" << best_s << " raw_delta_s=" << raw_delta_s << " delta_s=" << delta_s << " lateral=" << lateral
              << " cost=" << best_cost);
  return adjusted;
}

bool PoseEstimator::compute_f2f_absolute_pose(
  const pcl::PointCloud<PointT>::ConstPtr& cloud,
  const Eigen::Matrix4f& init_guess,
  Eigen::Matrix4f* f2f_absolute_pose,
  double* f2f_fitness_score) {
  if (!enable_frame2frame_ndt || !frame2frame_registration || !prev_cloud || !f2f_absolute_pose) {
    return false;
  }

  const Eigen::Matrix4f init_rel = prev_map_pose.inverse() * init_guess;
  pcl::PointCloud<PointT> aligned;
  auto run_align = [&](const pcl::PointCloud<PointT>::ConstPtr& target, const pcl::PointCloud<PointT>::ConstPtr& source) {
    frame2frame_registration->setInputTarget(target);
    frame2frame_registration->setInputSource(source);
    aligned.clear();
    frame2frame_registration->align(aligned, init_rel);
    return frame2frame_registration->hasConverged();
  };

  bool converged = false;
  if (enable_f2f_confidence_filter) {
    pcl::PointCloud<PointT>::Ptr filtered_target(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr filtered_source(new pcl::PointCloud<PointT>());
    filtered_target->reserve(prev_cloud->size());
    filtered_source->reserve(cloud->size());

    std::unordered_set<VoxelKey, VoxelKeyHash> target_voxels;
    if (enable_f2f_dynamic_filter && f2f_dynamic_voxel_size > 0.0) {
      target_voxels.reserve(prev_cloud->size());
      for (const auto& pt : prev_cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
          continue;
        }
        target_voxels.insert(to_voxel_key(Eigen::Vector3f(pt.x, pt.y, pt.z), f2f_dynamic_voxel_size));
      }
    }

    for (std::size_t i = 0; i < prev_cloud->size(); ++i) {
      const auto& pt = prev_cloud->points[i];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }

      const double keep_ratio = is_wall_point(pt) ? f2f_wall_keep_ratio : 1.0;
      if (keep_point_by_ratio(pt, keep_ratio, static_cast<int>(i))) {
        filtered_target->push_back(pt);
      }
    }

    for (std::size_t i = 0; i < cloud->size(); ++i) {
      const auto& pt = cloud->points[i];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }

      const bool wall_point = is_wall_point(pt);
      bool dynamic_point = false;
      if (enable_f2f_dynamic_filter && !target_voxels.empty()) {
        const Eigen::Vector4f p_curr(pt.x, pt.y, pt.z, 1.0f);
        const Eigen::Vector4f p_prev = init_rel * p_curr;
        const VoxelKey key = to_voxel_key(p_prev.head<3>(), f2f_dynamic_voxel_size);
        dynamic_point = target_voxels.find(key) == target_voxels.end();
      }

      double keep_ratio = 1.0;
      if (wall_point) {
        keep_ratio *= f2f_wall_keep_ratio;
      }
      if (dynamic_point) {
        keep_ratio *= f2f_dynamic_keep_ratio;
      }

      if (keep_point_by_ratio(pt, keep_ratio, static_cast<int>(i) + 1337)) {
        filtered_source->push_back(pt);
      }
    }

    const bool enough_filtered_points =
      filtered_target->size() >= static_cast<std::size_t>(f2f_min_filtered_points) && filtered_source->size() >= static_cast<std::size_t>(f2f_min_filtered_points);
    if (enough_filtered_points) {
      converged = run_align(filtered_target, filtered_source);
    }
  }

  if (!converged) {
    converged = run_align(prev_cloud, cloud);
  }

  if (!converged) {
    return false;
  }

  *f2f_absolute_pose = prev_map_pose * frame2frame_registration->getFinalTransformation();
  if (f2f_fitness_score) {
    *f2f_fitness_score = frame2frame_registration->getFitnessScore();
  }
  return true;
}

double PoseEstimator::score_to_confidence(double score) const {
  const double min_conf = std::min(std::max(ndt_score_min_confidence, 0.0), 1.0);
  if (!std::isfinite(score)) {
    return min_conf;
  }
  if (ndt_score_bad <= ndt_score_good + 1e-9) {
    return 1.0;
  }

  const double normalized = std::max(0.0, std::min(1.0, (ndt_score_bad - score) / (ndt_score_bad - ndt_score_good)));
  return min_conf + (1.0 - min_conf) * normalized;
}

Eigen::Matrix4f PoseEstimator::fuse_map_and_f2f_pose(
  const Eigen::Matrix4f& map_pose,
  const Eigen::Matrix4f& f2f_pose,
  double map_fitness_score,
  double f2f_fitness_score,
  double* out_map_conf,
  double* out_f2f_conf,
  double* out_w_f2f_trans,
  double* out_w_f2f_rot,
  double* out_w_f2f_axial,
  double* out_w_f2f_nonaxial) const {
  const double map_trans_var = std::max(map_trans_noise * map_trans_noise, 1e-9);
  const double f2f_trans_var = std::max(f2f_trans_noise * f2f_trans_noise, 1e-9);
  const double map_rot_var = std::max(map_rot_noise * map_rot_noise, 1e-9);
  const double f2f_rot_var = std::max(f2f_rot_noise * f2f_rot_noise, 1e-9);

  double map_conf = 1.0;
  double f2f_conf = 1.0;
  if (enable_score_weighted_fusion) {
    map_conf = score_to_confidence(map_fitness_score);
    f2f_conf = score_to_confidence(f2f_fitness_score) * f2f_score_confidence_gain;
  }

  const double inv_map_trans = map_conf / map_trans_var;
  const double inv_f2f_trans = f2f_conf / f2f_trans_var;
  const double inv_map_rot = map_conf / map_rot_var;
  const double inv_f2f_rot = f2f_conf / f2f_rot_var;

  const double w_f2f_trans = inv_f2f_trans / std::max(inv_map_trans + inv_f2f_trans, 1e-9);
  const double w_f2f_rot = inv_f2f_rot / std::max(inv_map_rot + inv_f2f_rot, 1e-9);

  const Eigen::Vector3f p_map = map_pose.block<3, 1>(0, 3);
  const Eigen::Vector3f p_f2f = f2f_pose.block<3, 1>(0, 3);
  double w_f2f_axial = w_f2f_trans;
  double w_f2f_nonaxial = w_f2f_trans;
  Eigen::Vector3f p_fused = (1.0 - w_f2f_trans) * p_map + w_f2f_trans * p_f2f;

  if (enable_axis_anisotropic_fusion && enable_axis_prior && axis_centerline.size() >= 2) {
    double s_ref = 0.0;
    double lateral = 0.0;
    if (project_to_axis(p_map, &s_ref, &lateral)) {
      Eigen::Vector3f axis_point = Eigen::Vector3f::Zero();
      Eigen::Vector3f axis_tangent = Eigen::Vector3f::UnitX();
      if (interpolate_axis_sample(s_ref, &axis_point, &axis_tangent) && axis_tangent.norm() > 1e-6f) {
        axis_tangent.normalize();
        const Eigen::Vector3f delta = p_f2f - p_map;
        const Eigen::Vector3f delta_axial = delta.dot(axis_tangent) * axis_tangent;
        const Eigen::Vector3f delta_nonaxial = delta - delta_axial;

        const double inv_f2f_axial = (f2f_conf * f2f_axial_conf_gain) / f2f_trans_var;
        const double inv_f2f_nonaxial = (f2f_conf * f2f_nonaxial_conf_gain) / f2f_trans_var;
        w_f2f_axial = inv_f2f_axial / std::max(inv_map_trans + inv_f2f_axial, 1e-9);
        w_f2f_nonaxial = inv_f2f_nonaxial / std::max(inv_map_trans + inv_f2f_nonaxial, 1e-9);

        p_fused = p_map + static_cast<float>(w_f2f_axial) * delta_axial + static_cast<float>(w_f2f_nonaxial) * delta_nonaxial;
      }
    }
  }

  Eigen::Quaternionf q_map(map_pose.block<3, 3>(0, 0));
  Eigen::Quaternionf q_f2f(f2f_pose.block<3, 3>(0, 0));
  if (q_map.coeffs().dot(q_f2f.coeffs()) < 0.0f) {
    q_f2f.coeffs() *= -1.0f;
  }
  const Eigen::Quaternionf q_fused = q_map.slerp(w_f2f_rot, q_f2f).normalized();

  Eigen::Matrix4f fused = Eigen::Matrix4f::Identity();
  fused.block<3, 1>(0, 3) = p_fused;
  fused.block<3, 3>(0, 0) = q_fused.toRotationMatrix();

  if (out_map_conf) {
    *out_map_conf = map_conf;
  }
  if (out_f2f_conf) {
    *out_f2f_conf = f2f_conf;
  }
  if (out_w_f2f_trans) {
    *out_w_f2f_trans = w_f2f_trans;
  }
  if (out_w_f2f_rot) {
    *out_w_f2f_rot = w_f2f_rot;
  }
  if (out_w_f2f_axial) {
    *out_w_f2f_axial = w_f2f_axial;
  }
  if (out_w_f2f_nonaxial) {
    *out_w_f2f_nonaxial = w_f2f_nonaxial;
  }

  return fused;
}
/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp) {
  // 初始化时刻还没设定，就用当前时间设为初始时间
  if (init_stamp.is_zero()) {
    init_stamp = stamp;
  }

  // 如果当前时刻距离初始化时刻太短（在冷却时间内）或者时间戳无效或重复，则不进行预测,cool_time_duration为系统初始启动时不稳定的冷却
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  // 计算时间差 dt（单位秒）
  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  // 更新 process noise covariance（过程噪声协方差矩阵），乘上时间差
  ukf->setProcessNoiseCov(process_noise * dt);

  // 设置系统模型中的时间差
  ukf->system.dt = dt;

  // 执行预测（不使用控制输入）
  ukf->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
  // 如果初始时间未设定，记录下当前时间
  if (init_stamp.is_zero()) {
    init_stamp = stamp;
  }

  // 冷却时间没过，或者时间戳非法或重复，就退出，不做预测
  if ((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  // 计算时间差
  double dt = (stamp - prev_stamp).toSec();
  prev_stamp = stamp;

  // 设置过程噪声与时间差成比例
  ukf->setProcessNoiseCov(process_noise * dt);

  // 设置系统模型时间差
  ukf->system.dt = dt;

  // 构造控制向量 u = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  // 如果加速度过大可以滤波
  // std::cout << "acc_original_predict: " << acc.transpose() << std::endl;

  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;
  // std::cout << "control: " << control.transpose() << std::endl;

  // 使用带控制输入的预测
  ukf->predict(control);
}

/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
  if (!odom_ukf) {
    Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
    Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

    Eigen::VectorXf odom_mean(7);
    odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
    odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]);
    Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

    OdomSystem odom_system;
    odom_ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>(odom_system, 7, 7, 7, odom_process_noise, odom_measurement_noise, odom_mean, odom_cov));
  }

  // invert quaternion if the rotation axis is flipped
  Eigen::Quaternionf quat(odom_delta.block<3, 3>(0, 0));
  if (odom_quat().coeffs().dot(quat.coeffs()) < 0.0) {
    quat.coeffs() *= -1.0f;
  }

  Eigen::VectorXf control(7);
  control.middleRows(0, 3) = odom_delta.block<3, 1>(0, 3);
  control.middleRows(3, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());

  Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(7, 7);
  process_noise.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * odom_delta.block<3, 1>(0, 3).norm() + Eigen::Matrix3f::Identity() * 1e-3;
  process_noise.bottomRightCorner(4, 4) = Eigen::Matrix4f::Identity() * (1 - std::abs(quat.w())) + Eigen::Matrix4f::Identity() * 1e-3;

  odom_ukf->setProcessNoiseCov(process_noise);
  odom_ukf->predict(control);
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
// 使用当前点云观测对UKF进行修正，并返回配准后的点云
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
  pcl::Registration<PointT, PointT>::Ptr reg;
  {
    std::lock_guard<std::mutex> lk(reg_mtx_);
    reg = registration;  // 只锁这一下，极短
  }
  // 后面 reg->align() 不要再锁

  // 如果这是第一次调用，则设置初始时间戳
  if (init_stamp.is_zero()) {
    init_stamp = stamp;
  }

  // 记录当前修正时间戳
  last_correction_stamp = stamp;

  // 保存几个用于评估误差的初始估计
  Eigen::Matrix4f no_guess = last_observation;  // 不使用预测，仅使用上次观测作为猜测
  Eigen::Matrix4f imu_guess;
  Eigen::Matrix4f odom_guess;
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // 初始化配准的初始猜测

  // 如果没有启用odom_ukf，说明只使用IMU，那么初始猜测直接来自IMU（ukf）
  if (!odom_ukf) {
    // 初始为输入的位姿
    init_guess = imu_guess = matrix();  // 从ukf当前状态提取IMU预测
  } else {
    // 同时使用IMU和里程计
    imu_guess = matrix();        // 从IMU相关ukf获取预测,mean就是预测值
    odom_guess = odom_matrix();  // 从里程计ukf获取预测

    // --- 联合融合 IMU + Odom 的状态均值和协方差 ---

    // 构造IMU状态的7维观测（位置3 + 姿态4）
    Eigen::VectorXf imu_mean(7);
    // mean.block<3, 1>(0, 0)取3行一列的值,从0行0列开始
    imu_mean.block<3, 1>(0, 0) = ukf->mean.block<3, 1>(0, 0);  // 位置,其实就等于上面imu_guess即matrix()的返回值
    imu_mean.block<4, 1>(3, 0) = ukf->mean.block<4, 1>(6, 0);  // 姿态四元数,中间三维速度

    // 提取IMU协方差的相关子块,初始化IMU的置信度,
    Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
    imu_cov.block<3, 3>(0, 0) = ukf->cov.block<3, 3>(0, 0);  // 提取3行3列
    imu_cov.block<3, 4>(0, 3) = ukf->cov.block<3, 4>(0, 6);
    imu_cov.block<4, 3>(3, 0) = ukf->cov.block<4, 3>(6, 0);
    imu_cov.block<4, 4>(3, 3) = ukf->cov.block<4, 4>(6, 6);

    // 里程计的状态均值和协方差（也是假设为位置+姿态）
    Eigen::VectorXf odom_mean = odom_ukf->mean;
    Eigen::MatrixXf odom_cov = odom_ukf->cov;

    // 如果IMU和odom的四元数方向相反（内积小于0），统一符号
    if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0) {
      odom_mean.tail<4>() *= -1.0;
    }

    // 使用协方差加权融合（经典信息融合公式）
    Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
    Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();
    Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
    Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

    // 构造融合后的初始猜测矩阵 init_guess
    init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);                                                    // 位置
    init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();  // 姿态
  }

  // 先用帧间配准提供地图配准初值（若不可用则退回IMU/Odom初值）
  Eigen::Matrix4f map_init_guess = init_guess;
  Eigen::Matrix4f f2f_init_pose = Eigen::Matrix4f::Identity();
  double f2f_init_score = std::numeric_limits<double>::quiet_NaN();
  const bool has_f2f_init = compute_f2f_absolute_pose(cloud, init_guess, &f2f_init_pose, &f2f_init_score);
  if (has_f2f_init) {
    map_init_guess = f2f_init_pose;
  }
  const bool use_axis_prealign = enable_axis_prior && (!axis_prealign_only_when_degenerate || last_map_degenerate);
  if (use_axis_prealign) {
    map_init_guess = apply_axis_prior(map_init_guess, "prealign");
  }

  // 配准对齐点云 cloud 到地图坐标系，使用 map_init_guess 作为初始估计
  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  if (!reg->getInputTarget()) {
    ROS_WARN_STREAM_THROTTLE(1.0, "[REG] target map is not ready, skip correction");
    pcl::transformPointCloud(*cloud, *aligned, map_init_guess);
    aligned->header = cloud->header;
    return aligned;
  }

  // 当前帧雷达点云
  reg->setInputSource(cloud);
  // ====== 开始计时 ======
  auto t0w = std::chrono::steady_clock::now();
  double t0c = now_thread_cpu_ms();

  reg->align(*aligned, map_init_guess);

  // ====== 结束计时 ======
  auto t1w = std::chrono::steady_clock::now();
  double t1c = now_thread_cpu_ms();

  double wall = std::chrono::duration<double, std::milli>(t1w - t0w).count();
  double cpu = t1c - t0c;
  std::cout << "[REG] wall=" << wall << " ms, cpu=" << cpu << " ms\n";
  // // solver.iterations()
  const bool map_converged = reg->hasConverged();
  std::cout << " conv=" << map_converged << std::endl;

  if (!map_converged) {
    ROS_WARN_STREAM_THROTTLE(1.0, "[REG] map registration did not converge, but still use map observation result");
  }

  // 提取帧到地图位姿
  Eigen::Matrix4f map_pose = reg->getFinalTransformation();
  const double map_fitness_score = reg->getFitnessScore();
  const bool map_degenerate = (!map_converged) || !std::isfinite(map_fitness_score) || map_fitness_score >= ndt_score_bad;

  // 帧间与地图观测融合（按分数与噪声给权重）
  Eigen::Matrix4f selected_pose = map_pose;
  std::string pose_source = "map_only";
  double map_conf = 1.0;
  double f2f_conf = 0.0;
  double w_f2f_trans = 0.0;
  double w_f2f_rot = 0.0;
  double w_f2f_axial = 0.0;
  double w_f2f_nonaxial = 0.0;
  if (has_f2f_init) {
    selected_pose =
      fuse_map_and_f2f_pose(map_pose, f2f_init_pose, map_fitness_score, f2f_init_score, &map_conf, &f2f_conf, &w_f2f_trans, &w_f2f_rot, &w_f2f_axial, &w_f2f_nonaxial);
    pose_source = "fused_map_f2f";
  }

  Eigen::Matrix4f final_pose = selected_pose;
  const bool use_axis_postalign = enable_axis_prior && (!axis_postalign_only_when_degenerate || map_degenerate);
  if (use_axis_postalign) {
    final_pose = apply_axis_prior(selected_pose, map_degenerate ? "postalign_degenerate" : "postalign");
  }
  last_map_degenerate = map_degenerate;
  // std::cout << "[TRANSFORM] Current frame pose:" << std::endl;
  // std::cout << trans << std::endl;

  Eigen::Vector3f p = final_pose.block<3, 1>(0, 3);    // 平移
  Eigen::Quaternionf q(final_pose.block<3, 3>(0, 0));  // 姿态

  // 确保四元数方向一致（避免跳变）
  if (quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  ROS_INFO_STREAM(
    "[REG] final pose p=[" << p.x() << ", " << p.y() << ", " << p.z() << "] "
                           << "q=[w " << q.w() << ", x " << q.x() << ", y " << q.y() << ", z " << q.z() << "] "
                           << "source=" << pose_source << " f2f_init_used=" << (has_f2f_init ? "true" : "false") << " map_converged=" << (map_converged ? "true" : "false")
                           << " map_score=" << map_fitness_score << " f2f_init_score=" << f2f_init_score << " map_conf=" << map_conf << " f2f_conf=" << f2f_conf
                           << " w_f2f_t=" << w_f2f_trans << " w_f2f_r=" << w_f2f_rot << " w_f2f_ax=" << w_f2f_axial << " w_f2f_nonax=" << w_f2f_nonaxial);

  // 构造观测向量 observation（位置+四元数，共7维）,已 修正四元数正负
  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

  // 保存最新观测结果（map配准）
  last_observation = final_pose;

  // 记录预测误差（基于map观测）
  wo_pred_error = no_guess.inverse() * final_pose;

  // 执行UKF状态更新（修正）
  ukf->correct(observation);

  // 记录IMU预测误差
  imu_pred_error = imu_guess.inverse() * final_pose;

  // 如果有odom_ukf，进行里程计UKF的修正
  if (odom_ukf) {
    // 确保四元数方向一致
    if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
      odom_ukf->mean.tail<4>() *= -1.0;
    }

    // 修正里程计UKF
    odom_ukf->correct(observation);

    // 记录里程计预测误差
    odom_pred_error = odom_guess.inverse() * final_pose;
    // 记录预测初值误差
    imu_odom_pred_error = init_guess.inverse() * final_pose;
  }

  prev_cloud = cloud;
  prev_map_pose = final_pose;

  // 返回配准后的点云
  return aligned;
}

//   // 如果没有启用odom_ukf，说明只使用IMU，那么初始猜测直接来自IMU（ukf）
//   if (!odom_ukf) {
//     // 初始为输入的位姿
//     init_guess = imu_guess = matrix();  // 从ukf当前状态提取IMU预测
//   } else {
//     // 同时使用IMU和里程计
//     imu_guess = matrix();        // 从IMU相关ukf获取预测,mean就是预测值
//     odom_guess = odom_matrix();  // 从里程计ukf获取预测

//     // --- 联合融合 IMU + Odom 的状态均值和协方差 ---

//     // 构造IMU状态的7维观测（位置3 + 姿态4）
//     Eigen::VectorXf imu_mean(7);
//     // mean.block<3, 1>(0, 0)取3行一列的值,从0行0列开始
//     imu_mean.block<3, 1>(0, 0) = ukf->mean.block<3, 1>(0, 0);  // 位置,其实就等于上面imu_guess即matrix()的返回值
//     imu_mean.block<4, 1>(3, 0) = ukf->mean.block<4, 1>(6, 0);  // 姿态四元数,中间三维速度

//     // 提取IMU协方差的相关子块,初始化IMU的置信度,
//     Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
//     imu_cov.block<3, 3>(0, 0) = ukf->cov.block<3, 3>(0, 0);  // 提取3行3列
//     imu_cov.block<3, 4>(0, 3) = ukf->cov.block<3, 4>(0, 6);
//     imu_cov.block<4, 3>(3, 0) = ukf->cov.block<4, 3>(6, 0);
//     imu_cov.block<4, 4>(3, 3) = ukf->cov.block<4, 4>(6, 6);

//     // 里程计的状态均值和协方差（也是假设为位置+姿态）
//     Eigen::VectorXf odom_mean = odom_ukf->mean;
//     Eigen::MatrixXf odom_cov = odom_ukf->cov;

//     // 如果IMU和odom的四元数方向相反（内积小于0），统一符号
//     if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0) {
//       odom_mean.tail<4>() *= -1.0;
//     }

//     // 使用协方差加权融合（经典信息融合公式）
//     Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
//     Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();
//     Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
//     Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

//     // 构造融合后的初始猜测矩阵 init_guess
//     init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);                                                    // 位置
//     init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();  // 姿态
//   }

//   // 配准对齐点云 cloud 到地图坐标系，使用 init_guess 作为初始估计
//   pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
//   // 当前帧雷达点云
//   reg->setInputSource(cloud);
//   // ====== 开始计时 ======
//   // auto t0w = std::chrono::steady_clock::now();
//   // double t0c = now_thread_cpu_ms();

//   reg->align(*aligned, init_guess);

//   // ====== 结束计时 ======
//   // auto t1w = std::chrono::steady_clock::now();
//   // double t1c = now_thread_cpu_ms();

//   // double wall = std::chrono::duration<double, std::milli>(t1w - t0w).count();
//   // double cpu = t1c - t0c;
//   // std::cout << "[GICP] wall=" << wall << " ms, cpu=" << cpu << " ms\n";
//   // // solver.iterations()
//   std::cout << " conv=" << reg->hasConverged() << std::endl;

//   // 提取最终变换矩阵
//   Eigen::Matrix4f trans = reg->getFinalTransformation();
//   // std::cout << "[TRANSFORM] Current frame pose:" << std::endl;
//   // std::cout << trans << std::endl;

//   Eigen::Vector3f p = trans.block<3, 1>(0, 3);    // 平移
//   Eigen::Quaternionf q(trans.block<3, 3>(0, 0));  // 姿态

//   // 确保四元数方向一致（避免跳变）
//   if (quat().coeffs().dot(q.coeffs()) < 0.0f) {
//     q.coeffs() *= -1.0f;
//   }

//   // 构造观测向量 observation（位置+四元数，共7维）,已 修正四元数正负
//   Eigen::VectorXf observation(7);
//   observation.middleRows(0, 3) = p;
//   observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

//   // 保存最新的点云配准观测结果
//   last_observation = trans;

//   // 记录预测误差：无预测时的误差,registration->getFinalTransformation()就是trans
//   wo_pred_error = no_guess.inverse() * reg->getFinalTransformation();

//   // 执行UKF状态更新（修正）
//   ukf->correct(observation);

//   // 记录IMU预测误差
//   imu_pred_error = imu_guess.inverse() * reg->getFinalTransformation();

//   // 如果有odom_ukf，进行里程计UKF的修正
//   if (odom_ukf) {
//     // 确保四元数方向一致
//     if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
//       odom_ukf->mean.tail<4>() *= -1.0;
//     }

//     // 修正里程计UKF
//     odom_ukf->correct(observation);

//     // 记录里程计预测误差
//     odom_pred_error = odom_guess.inverse() * reg->getFinalTransformation();
//     // 记录融合误差
//     imu_odom_pred_error = init_guess.inverse() * reg->getFinalTransformation();
//   }

//   // 返回配准后的点云
//   return trans;
// }

/* getters */
ros::Time PoseEstimator::last_correction_time() const {
  return last_correction_stamp;
}
// 返回状态量
Eigen::Vector3f PoseEstimator::pos() const {
  return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
}

Eigen::Vector3f PoseEstimator::vel() const {
  return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
}

Eigen::Quaternionf PoseEstimator::quat() const {
  return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
}
// 返回位姿,最终位姿获取是pose_estimator->matrix()获取
Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}
// 返回里程计（轮速计）相关值：
Eigen::Vector3f PoseEstimator::odom_pos() const {
  return Eigen::Vector3f(odom_ukf->mean[0], odom_ukf->mean[1], odom_ukf->mean[2]);
}

Eigen::Quaternionf PoseEstimator::odom_quat() const {
  return Eigen::Quaternionf(odom_ukf->mean[3], odom_ukf->mean[4], odom_ukf->mean[5], odom_ukf->mean[6]).normalized();
}

Eigen::Matrix4f PoseEstimator::odom_matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = odom_pos();
  return m;
}
// 返回预测误差
const boost::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
  return wo_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
  return imu_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
  return odom_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_odom_prediction_error() const {
  return imu_odom_pred_error;
}

}  // namespace hdl_localization
