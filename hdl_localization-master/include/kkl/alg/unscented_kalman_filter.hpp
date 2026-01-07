/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template <typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

public:
  /**
   * @brief constructor 构造函数
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  /**
   * @brief 无迹卡尔曼滤波器的构造函数
   * @param system               待估计的系统模型
   * @param state_dim            状态向量的维度
   * @param input_dim            输入向量的维度
   * @param measurement_dim      测量向量的维度
   * @param process_noise        过程噪声协方差矩阵 (state_dim x state_dim)
   * @param measurement_noise    测量噪声协方差矩阵 (measurement_dim x measurement_dim)
   * @param mean                 初始状态向量的均值
   * @param cov                  初始状态向量的协方差矩阵
   */
  UnscentedKalmanFilterX(
    const System& system,
    int state_dim,
    int input_dim,
    int measurement_dim,
    const MatrixXt& process_noise,
    const MatrixXt& measurement_noise,
    const VectorXt& mean,
    const MatrixXt& cov)
  : state_dim(state_dim),                  // 初始化状态向量维度
    input_dim(input_dim),                  // 初始化输入向量维度
    measurement_dim(measurement_dim),      // 初始化测量向量维度
    N(state_dim),                          // N 表示状态向量维度
    M(input_dim),                          // M 表示输入向量维度
    K(measurement_dim),                    // K 表示测量向量维度
    S(2 * state_dim + 1),                  // S 表示 Sigma 点的数量，公式为 2n + 1
    mean(mean),                            // 初始化状态向量的均值
    cov(cov),                              // 初始化状态向量的协方差矩阵
    system(system),                        // 初始化系统模型
    process_noise(process_noise),          // 初始化过程噪声协方差矩阵
    measurement_noise(measurement_noise),  // 初始化测量噪声协方差矩阵
    lambda(1),                             // 初始化无迹变换的缩放参数 lambda
    normal_dist(0.0, 1.0)                  // 初始化正态分布，均值为 0，标准差为 1
  {
    // 调整权重向量的大小，用于计算 Sigma 点的加权和
    weights.resize(S, 1);
    // 调整 Sigma 点矩阵的大小，行数为 Sigma 点数量，列数为状态向量维度
    sigma_points.resize(S, N);
    // 调整扩展状态空间的权重向量大小，用于计算包含误差方差的扩展状态空间的加权和
    ext_weights.resize(2 * (N + K) + 1, 1);
    // 调整扩展状态空间的 Sigma 点矩阵大小
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    // 调整预期测量值矩阵的大小，用于存储通过 Sigma 点预测的测量值
    expected_measurements.resize(2 * (N + K) + 1, K);

    // 初始化无迹滤波的权重
    // 第一个 Sigma 点的权重
    weights[0] = lambda / (N + lambda);
    // 初始化其余 Sigma 点的权重
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // 初始化包含误差方差的扩展状态空间的权重
    // 扩展状态空间的第一个 Sigma 点的权重
    ext_weights[0] = lambda / (N + K + lambda);
    // 初始化扩展状态空间其余 Sigma 点的权重
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict() {
    // 确保协方差矩阵的所有元素都是有效数（非 NaN、非负、非无穷）
    ensurePositiveFinite(cov);

    // 1. 从当前状态均值 mean 和协方差 cov 计算 sigma 点
    computeSigmaPoints(mean, cov, sigma_points);

    // 2. 所有 sigma 点分别输入状态转移函数 f 传播一次
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i));
    }

    // 3. 获取过程噪声协方差
    const auto& R = process_noise;

    // 4. 初始化预测结果的均值与协方差
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();

    // 5. 计算预测后的均值
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);  // 加权平均
    }

    // 6. 计算预测后的协方差
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();  // 加权 outer product
    }

    // 7. 加上过程噪声
    cov_pred += R;

    // 8. 更新滤波器的当前状态与协方差
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // 校验协方差合法性
    ensurePositiveFinite(cov);

    // 从均值和协方差生成 sigma 点
    computeSigmaPoints(mean, cov, sigma_points);

    // 所有 sigma 点输入带控制量的状态转移函数 f(x, u)
    for (int i = 0; i < S; i++) {
      // 预测函数system.f，predict
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }

    // 后续与无控制版本完全一致：
    const auto& R = process_noise;

    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();

    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }

    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }

    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   * 校正函数
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    // N：原始状态维度（您的系统是16），K：观测维度（您的系统是7），这里表示的是观测噪声，扩展是将观测噪声加入状态空间中，作为一部分，白噪声均值是0
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    // 扩展状态协方差矩阵，维度 = (N + K) × (N + K) = 23×23
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    // mean是预测完的预测值，填充原始状态部分，cov在预测中改变过
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    // measurement_noise参数写死了，在构造函数中初始化
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;
    // 检查协方差矩阵是否正定、有限，防止数值问题导致滤波发散，长时间运行时可能需要
    ensurePositiveFinite(ext_cov_pred);
    // 生成 2*(N+K)+1 个Sigma点，ext_sigma_points矩阵每一行代表一个点，维度是N+K，代表一个状态
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // unscented transform
    // system.h(x) 的输入是“状态 sigma 点”，输出是“该状态在观测空间下的预测观测值”
    // ext_sigma_points[i].topLeftCorner(N) 是 第 i 个状态 sigma 点
    // z = h(x)+v，拿预测的观测值生成理想预测观测，对于第i个Sigma点，计算其对应的预测观测值。
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }
    // 加权平均
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    // 计算预测观测的协方差
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      // 对角线元素 dᵢ*dᵢ：第i维观测的方差，非对角线元素 dᵢ*dⱼ：第i维和第j维观测的协方差（相关性）
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    // diffA(N+K)×1：第i个Sigma点与扩展状态均值的偏差
    // diffB（K×1）：第i个预测观测与观测均值的偏差
    // diffA * diffBᵀ：这两者偏差的相关性
    // sigma：加权平均的相关性 = 状态变化如何影响观测变化
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;
    // 残差，measurement直接雷达观测值，expected_measurement_mean点预测值，中间计算较多的主要是卡尔曼增益K
    // K×残差为更新量ext_mean，measurement输入量只在这里用到了
    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) {
    mean = m;
    return *this;
  }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) {
    cov = s;
    return *this;
  }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) {
    process_noise = p;
    return *this;
  }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) {
    measurement_noise = m;
    return *this;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;      //
  MatrixXt measurement_noise;  //

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  /**
   * @brief 计算无迹卡尔曼滤波所需的 Sigma 点
   * @param mean          状态向量的均值
   * @param cov           状态向量的协方差矩阵
   * @param sigma_points  用于存储计算得到的 Sigma 点的矩阵
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    // 获取均值向量的维度，该维度同时也是状态向量的维度
    const int n = mean.size();
    // 确保协方差矩阵是 n x n 的方阵，与均值向量维度匹配
    assert(cov.rows() == n && cov.cols() == n);

    // 创建一个 Cholesky 分解对象，使用 LLT 分解方法（即 A = LL^T）
    Eigen::LLT<MatrixXt> llt;
    // 对 (n + lambda) * cov 进行 Cholesky 分解，其中 lambda 是无迹变换的缩放参数
    llt.compute((n + lambda) * cov);
    // 获取 Cholesky 分解得到的下三角矩阵 L
    MatrixXt l = llt.matrixL();

    // 将第一个 Sigma 点设置为均值向量
    sigma_points.row(0) = mean;
    // 循环生成剩余的 2n 个 Sigma 点
    for (int i = 0; i < n; i++) {
      // 生成第 2i + 1 个 Sigma 点，等于均值向量加上下三角矩阵 L 的第 i 列
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      // 生成第 2i + 2 个 Sigma 点，等于均值向量减去下三角矩阵 L 的第 i 列
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  /**
   * @brief 确保协方差矩阵为正定且有限的矩阵。
   *        在实际计算中，由于数值误差，协方差矩阵可能失去正定性，
   *        此函数通过特征值分解修正协方差矩阵，使其满足正定性要求。
   * @param cov 待处理的协方差矩阵，函数会直接修改该矩阵的值。
   * 长时间运行	数值误差累积	数小时~数天后
    剧烈运动	高速旋转、急加速导致数值不稳定	特定动作时
    传感器失效	GICP匹配失败，观测噪声突增	匹配失败时
    参数不当	噪声矩阵设置过小	参数调试期间
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    // 注意：当前函数直接返回，后续修正协方差矩阵的代码不会执行。
    // 若要启用协方差矩阵修正功能，需移除下面的 return 语句。
    return;

    // 定义一个极小的正数，作为特征值的下限阈值。
    // 若特征值小于该阈值，将其修正为该阈值，以确保矩阵的正定性。
    const double eps = 1e-9;

    // 使用 Eigen 库的特征值求解器对协方差矩阵进行特征值分解。
    // 特征值分解将协方差矩阵分解为特征值矩阵 D 和特征向量矩阵 V。
    Eigen::EigenSolver<MatrixXt> solver(cov);

    // 获取特征值矩阵 D，该矩阵为对角矩阵，对角元素为协方差矩阵的特征值。
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    // 获取特征向量矩阵 V，每一列对应一个特征向量。
    MatrixXt V = solver.pseudoEigenvectors();

    // 遍历特征值矩阵的对角元素
    for (int i = 0; i < D.rows(); i++) {
      // 若特征值小于设定的阈值 eps
      if (D(i, i) < eps) {
        // 将该特征值修正为阈值 eps
        D(i, i) = eps;
      }
    }

    // 利用修正后的特征值矩阵 D 和特征向量矩阵 V 重构协方差矩阵。
    // 重构公式为 cov = V * D * V^(-1)
    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

}  // namespace alg
}  // namespace kkl

#endif
