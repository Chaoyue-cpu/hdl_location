#include <hdl_localization/pose_estimator.hpp>

#include <pcl/filters/voxel_grid.h>
#include <hdl_localization/pose_system.hpp>
#include <hdl_localization/odom_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

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
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration)
: registration(registration),
  cool_time_duration(cool_time_duration) {
  // 初始化最后一次观测的变换矩阵（4x4单位矩阵）
  last_observation = Eigen::Matrix4f::Identity();

  // 设置旋转部分为初始四元数对应的旋转矩阵
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();

  // 设置平移部分为初始位置向量
  last_observation.block<3, 1>(0, 3) = pos;

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
}

PoseEstimator::~PoseEstimator() {}

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

  // 配准对齐点云 cloud 到地图坐标系，使用 init_guess 作为初始估计
  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  // 当前帧雷达点云
  registration->setInputSource(cloud);
  registration->align(*aligned, init_guess);

  // 提取最终变换矩阵
  Eigen::Matrix4f trans = registration->getFinalTransformation();
  // std::cout << "[TRANSFORM] Current frame pose:" << std::endl;
  // std::cout << trans << std::endl;

  Eigen::Vector3f p = trans.block<3, 1>(0, 3);    // 平移
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0));  // 姿态

  // 确保四元数方向一致（避免跳变）
  if (quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  // 构造观测向量 observation（位置+四元数，共7维）,已 修正四元数正负
  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

  // 保存最新的点云配准观测结果
  last_observation = trans;

  // 记录预测误差：无预测时的误差,registration->getFinalTransformation()就是trans
  wo_pred_error = no_guess.inverse() * registration->getFinalTransformation();

  // 执行UKF状态更新（修正）
  ukf->correct(observation);

  // 记录IMU预测误差
  imu_pred_error = imu_guess.inverse() * registration->getFinalTransformation();

  // 如果有odom_ukf，进行里程计UKF的修正
  if (odom_ukf) {
    // 确保四元数方向一致
    if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
      odom_ukf->mean.tail<4>() *= -1.0;
    }

    // 修正里程计UKF
    odom_ukf->correct(observation);

    // 记录里程计预测误差
    odom_pred_error = odom_guess.inverse() * registration->getFinalTransformation();
    // 记录融合误差
    imu_odom_pred_error = init_guess.inverse() * registration->getFinalTransformation();
  }

  // 返回配准后的点云
  return aligned;
}

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
