#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <deque>
#include <boost/optional.hpp>
#include <mutex>

#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace kkl {
namespace alg {
template <typename T, class System>
class UnscentedKalmanFilterX;
}
}  // namespace kkl

namespace hdl_localization {

class PoseSystem;
class OdomSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;
  using RegPtr = pcl::Registration<PointT, PointT>::Ptr;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(
    pcl::Registration<PointT, PointT>::Ptr& registration,
    const Eigen::Vector3f& pos,
    const Eigen::Quaternionf& quat,
    double cool_time_duration = 1.0,
    bool enable_frame2frame_ndt = true,
    int frame_to_frame_reg_num_threads = 6,
    const std::string& frame2frame_reg_method = "NDT_OMP",
    bool enable_score_weighted_fusion = true,
    double ndt_score_good = 0.15,
    double ndt_score_bad = 1.5,
    double ndt_score_min_confidence = 0.05,
    double f2f_score_confidence_gain = 1.0,
    bool enable_axis_anisotropic_fusion = false,
    double f2f_axial_conf_gain = 2.0,
    double f2f_nonaxial_conf_gain = 0.5,
    double map_axial_conf_gain = 0.3,
    double map_nonaxial_conf_gain = 1.0,
    bool enable_wall_r_axial_adaptation = false,
    double wall_r_axial_beta = 1.0,
    double wall_r_axial_scale_min = 1.0,
    double wall_r_axial_scale_max = 2.5,
    bool enable_inc_static_axial_adaptation = false,
    int inc_static_window_size = 10,
    int inc_static_min_hits = 3,
    int inc_static_sample_step = 4,
    int inc_static_min_unknown_voxels = 12,
    double inc_static_beta = 1.0,
    double inc_static_r_deadzone = 0.3,
    double inc_static_scale_min = 1.0,
    double inc_static_scale_max = 2.0,
    bool enable_f2f_confidence_filter = false,
    bool enable_f2f_dynamic_filter = false,
    double f2f_wall_y_threshold = 3.0,
    double f2f_wall_z_min = -2.0,
    double f2f_wall_z_max = 3.0,
    double f2f_wall_keep_ratio = 0.25,
    double f2f_dynamic_voxel_size = 0.5,
    double f2f_dynamic_keep_ratio = 0.25,
    int f2f_min_filtered_points = 600,
    bool enable_axis_prior = false,
    const std::string& axis_centerline_csv = "",
    const std::string& axis_profile_csv = "",
    double axis_search_window = 20.0,
    double axis_lateral_weight = 1.0,
    double axis_vertical_weight = 1.0,
    double axis_smooth_weight = 0.05,
    double axis_temporal_weight = 0.2,
    double axis_max_lateral = 20.0,
    double axis_max_delta_s = 0.15,
    bool axis_prealign_only_when_degenerate = true,
    bool axis_postalign_only_when_degenerate = true,
    bool enable_map_prior_layer = false,
    bool enable_map_prior_nonaxial_adaptation = true,
    const std::string& map_prior_csv = "",
    double map_prior_voxel_size = 2.0,
    int map_prior_sample_step = 4,
    int map_prior_min_hits = 120,
    double map_prior_core_gain = 1.0,
    double map_prior_band_gain = 0.7,
    double map_prior_inner_gain = 0.25,
    double map_prior_uncertain_gain = 0.15,
    double map_prior_conf_floor = 0.2,
    const std::string& inc_static_reference_csv = "",
    double inc_static_reference_voxel_size = 2.0,
    bool enable_reg_debug_csv = false,
    const std::string& reg_debug_csv_path = "");
  ~PoseEstimator();

  /**
   * @brief predict
   * @param stamp    timestamp
   */
  void predict(const ros::Time& stamp);

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro);

  pcl::Registration<PoseEstimator::PointT, PoseEstimator::PointT>::Ptr getRegistration() const;

  void set_registration(const pcl::Registration<PointT, PointT>::Ptr& reg);

  /**
   * @brief update the state of the odomety-based pose estimation
   */
  void predict_odom(const Eigen::Matrix4f& odom_delta);

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud);

  /* getters */
  ros::Time last_correction_time() const;
  // 位置速度旋转
  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;

  Eigen::Vector3f odom_pos() const;
  Eigen::Quaternionf odom_quat() const;
  Eigen::Matrix4f odom_matrix() const;

  const boost::optional<Eigen::Matrix4f>& wo_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& imu_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& odom_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& imu_odom_prediction_error() const;

private:
  bool is_wall_point(const PointT& pt) const;
  bool keep_point_by_ratio(const PointT& pt, double keep_ratio, int salt) const;
  bool compute_f2f_absolute_pose(const pcl::PointCloud<PointT>::ConstPtr& cloud, const Eigen::Matrix4f& init_guess, Eigen::Matrix4f* f2f_absolute_pose, double* f2f_fitness_score);
  Eigen::Matrix4f fuse_map_and_f2f_pose(
    const Eigen::Matrix4f& map_pose,
    const Eigen::Matrix4f& f2f_pose,
    bool map_converged,
    double map_fitness_score,
    double f2f_fitness_score,
    double map_nonaxial_prior_multiplier = 1.0,
    double* out_map_conf = nullptr,
    double* out_f2f_conf = nullptr,
    double* out_w_f2f_trans = nullptr,
    double* out_w_f2f_rot = nullptr,
    double* out_w_f2f_axial = nullptr,
    double* out_w_f2f_nonaxial = nullptr) const;
  double score_to_confidence(double score) const;
  bool load_axis_centerline_csv(const std::string& path);
  bool load_axis_profile_csv(const std::string& path);
  bool interpolate_axis_sample(double s, Eigen::Vector3f* point, Eigen::Vector3f* tangent) const;
  bool interpolate_axis_z(double s, double* z) const;
  bool project_to_axis(const Eigen::Vector3f& p, double* s, double* lateral_distance) const;
  Eigen::Matrix4f apply_axis_prior(const Eigen::Matrix4f& pose, const char* stage);
  bool load_map_prior_csv(const std::string& path);
  double compute_map_prior_conf_multiplier(
    const pcl::PointCloud<PointT>::ConstPtr& cloud,
    const Eigen::Matrix4f& map_pose,
    double* out_hit_ratio = nullptr,
    double* out_avg_prior_conf = nullptr,
    double* out_core_hit_ratio = nullptr,
    double* out_band_hit_ratio = nullptr,
    double* out_inner_hit_ratio = nullptr) const;
  void compute_wall_observability_metrics(
    const pcl::PointCloud<PointT>::ConstPtr& cloud,
    const Eigen::Matrix4f& map_pose,
    double* out_ratio_inner = nullptr,
    double* out_relief_inner = nullptr,
    double* out_r_wall = nullptr) const;
  double compute_wall_r_axial_scale(double wall_r, bool map_converged, double map_fitness_score) const;
  void compute_inc_static_metrics(
    const pcl::PointCloud<PointT>::ConstPtr& cloud,
    const Eigen::Matrix4f& map_pose,
    double* out_unknown_ratio = nullptr,
    double* out_stable_ratio = nullptr,
    double* out_r_inc_static = nullptr);
  double compute_inc_static_axial_scale(double r_inc_static) const;
  void write_reg_debug_row(
    const ros::Time& stamp,
    const std::string& pose_source,
    bool has_f2f_init,
    bool map_converged,
    bool map_degenerate,
    double map_fitness_score,
    double f2f_fitness_score,
    double map_conf_raw,
    double map_conf,
    double f2f_conf,
    double w_f2f_trans,
    double w_f2f_rot,
    double w_f2f_axial,
    double w_f2f_nonaxial,
    double map_prior_mult,
    double map_prior_hit_ratio,
    double map_prior_avg_conf,
    double map_prior_core_hit_ratio,
    double map_prior_band_hit_ratio,
    double map_prior_inner_hit_ratio,
    double wall_ratio_inner,
    double wall_relief_inner,
    double wall_r,
    double wall_r_axial_scale,
    double inc_unknown_ratio,
    double inc_stable_ratio,
    double r_inc_static,
    double inc_static_axial_scale,
    const Eigen::Vector3f& map_p,
    const Eigen::Vector3f& f2f_p,
    const Eigen::Vector3f& map_to_f2f_delta,
    double map_to_f2f_delta_norm,
    double map_to_f2f_delta_axial_signed,
    double map_to_f2f_delta_nonaxial_norm,
    double fused_from_map_delta_norm,
    double fused_from_map_delta_axial_signed,
    double fused_from_map_delta_nonaxial_norm,
    const Eigen::Vector3f& p,
    const Eigen::Quaternionf& q);

  ros::Time init_stamp;  // when the estimator was initialized
  ros::Time prev_stamp;  // when the estimator was updated last time
  // 校正步骤（correction step） 的数学本质是通过融合预测值与观测值，得到最优估计（Minimum Mean Square Error Estimate）。
  // 当前时刻的最优状态估计值，可直接输出使用。
  // 若系统持续运行（如SLAM），该结果会作为下一时刻预测的初始值+运动模型，形成“预测-校正”循环。
  ros::Time last_correction_stamp;  // when the estimator performed the correction step
  double cool_time_duration;        //

  Eigen::MatrixXf process_noise;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>> odom_ukf;

  Eigen::Matrix4f last_observation;
  boost::optional<Eigen::Matrix4f> wo_pred_error;
  boost::optional<Eigen::Matrix4f> imu_pred_error;
  boost::optional<Eigen::Matrix4f> odom_pred_error;
  boost::optional<Eigen::Matrix4f> imu_odom_pred_error;

  bool enable_frame2frame_ndt;
  double map_trans_noise;
  double map_rot_noise;
  double f2f_trans_noise;
  double f2f_rot_noise;
  bool enable_score_weighted_fusion;
  double ndt_score_good;
  double ndt_score_bad;
  double ndt_score_min_confidence;
  double f2f_score_confidence_gain;
  bool enable_axis_anisotropic_fusion;
  double f2f_axial_conf_gain;
  double f2f_nonaxial_conf_gain;
  double map_axial_conf_gain;
  double map_nonaxial_conf_gain;
  bool enable_wall_r_axial_adaptation;
  double wall_r_axial_beta;
  double wall_r_axial_scale_min;
  double wall_r_axial_scale_max;
  bool enable_inc_static_axial_adaptation;
  int inc_static_window_size;
  int inc_static_min_hits;
  int inc_static_sample_step;
  int inc_static_min_unknown_voxels;
  double inc_static_beta;
  double inc_static_r_deadzone;
  double inc_static_scale_min;
  double inc_static_scale_max;
  bool enable_f2f_confidence_filter;
  bool enable_f2f_dynamic_filter;
  double f2f_wall_y_threshold;
  double f2f_wall_z_min;
  double f2f_wall_z_max;
  double f2f_wall_keep_ratio;
  double f2f_dynamic_voxel_size;
  double f2f_dynamic_keep_ratio;
  int f2f_min_filtered_points;

  struct AxisSample {
    double s;
    Eigen::Vector3f p;
  };

  struct AxisZSample {
    double s;
    double z;
  };

  bool enable_axis_prior;
  double axis_search_window;
  double axis_lateral_weight;
  double axis_vertical_weight;
  double axis_smooth_weight;
  double axis_temporal_weight;
  double axis_max_lateral;
  double axis_max_delta_s;
  bool axis_prealign_only_when_degenerate;
  bool axis_postalign_only_when_degenerate;
  bool last_map_degenerate;
  std::vector<AxisSample> axis_centerline;
  std::vector<AxisZSample> axis_profile;
  boost::optional<double> last_axis_s;

  pcl::Registration<PointT, PointT>::Ptr frame2frame_registration;
  pcl::PointCloud<PointT>::ConstPtr prev_cloud;
  Eigen::Matrix4f prev_map_pose;

  struct PriorVoxelCell {
    uint8_t class_id = 0;     // 0 uncertain, 1 inner, 2 band, 3 core
    float conf_prior = 0.0f;  // [0,1]
  };
  struct PriorKey {
    int x;
    int y;
    int z;
    bool operator==(const PriorKey& other) const { return x == other.x && y == other.y && z == other.z; }
  };
  struct PriorKeyHash {
    std::size_t operator()(const PriorKey& key) const {
      std::size_t hx = static_cast<std::size_t>(std::hash<int>{}(key.x));
      std::size_t hy = static_cast<std::size_t>(std::hash<int>{}(key.y));
      std::size_t hz = static_cast<std::size_t>(std::hash<int>{}(key.z));
      return hx ^ (hy << 1) ^ (hz << 2);
    }
  };
  bool load_prior_csv_into(
    const std::string& path,
    std::unordered_map<PriorKey, PriorVoxelCell, PriorKeyHash>* out_cells) const;
  bool enable_map_prior_layer;
  bool enable_map_prior_nonaxial_adaptation;
  double map_prior_voxel_size;
  int map_prior_sample_step;
  int map_prior_min_hits;
  double map_prior_core_gain;
  double map_prior_band_gain;
  double map_prior_inner_gain;
  double map_prior_uncertain_gain;
  double map_prior_conf_floor;
  std::unordered_map<PriorKey, PriorVoxelCell, PriorKeyHash> map_prior_cells;
  std::string inc_static_reference_csv;
  double inc_static_reference_voxel_size;
  std::unordered_map<PriorKey, PriorVoxelCell, PriorKeyHash> inc_static_reference_cells;
  bool enable_reg_debug_csv;
  std::string reg_debug_csv_path;
  std::ofstream reg_debug_csv_stream;
  std::uint64_t reg_debug_seq;
  double last_wall_r;
  double last_r_inc_static;
  std::deque<std::vector<PriorKey>> inc_static_unknown_history;
  std::unordered_map<PriorKey, int, PriorKeyHash> inc_static_unknown_hit_counts;
  mutable std::mutex reg_debug_csv_mutex;

  pcl::Registration<PointT, PointT>::Ptr registration;
  mutable std::mutex reg_mtx_;
};

}  // namespace hdl_localization

#endif  // POSE_ESTIMATOR_HPP
