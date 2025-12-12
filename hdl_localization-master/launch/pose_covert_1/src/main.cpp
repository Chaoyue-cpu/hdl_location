#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 从YAML配置中提取的参数
constexpr double PI = 3.141592653589793;

// IMU相对于arch的位姿
const Eigen::Vector3d imu_position(0.0, 0.147, 0.165);
const Eigen::Vector3d imu_orientation_rpy(0.0, 0.0, PI/2); // Roll, Pitch, Yaw

// 雷达相对于arch的位姿
const Eigen::Vector3d lidar_position(0.038, 0.0, 0.405);
const Eigen::Vector3d lidar_orientation_rpy(0.0, 0.013, ((PI/4.0)-0.004));

// RPY转旋转矩阵
Eigen::Matrix3d rpyToRotationMatrix(const Eigen::Vector3d& rpy) {
    // Eigen::AngleAxisd roll(rpy(0), Eigen::Vector3d::UnitX());
    // Eigen::AngleAxisd pitch(rpy(1), Eigen::Vector3d::UnitY());
    // Eigen::AngleAxisd yaw(rpy(2), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q = Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX());
    q.normalize();
    std::cout << "q.x= \n" << q.x()<<"\n"<<"q.y= \n"<<q.y()<<"\n"<<"q.z= \n"<<q.z()<<"\n"<<"q.w= \n"<<q.w() << std::endl;
    Eigen::Matrix3d rotation_matrix1 = q.toRotationMatrix();
    return rotation_matrix1;
}

// 构建齐次变换矩阵
Eigen::Matrix4d buildTransformMatrix(const Eigen::Vector3d& position, const Eigen::Matrix3d R) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = position;
    return T;
}

int main() {
    std::cout <<"lidar_orientation_rpy: "<<lidar_orientation_rpy.transpose()<<std::endl;

    // 1. 构建 T_arch_to_imu 和 T_arch_to_lidar
    Eigen::Matrix4d T_imu_to_arch = buildTransformMatrix(imu_position, rpyToRotationMatrix(imu_orientation_rpy));
    Eigen::Matrix4d T_lidar_to_arch = buildTransformMatrix(lidar_position, rpyToRotationMatrix(lidar_orientation_rpy));
std::cout <<"T_imu_to_arch: "<<std::endl<<T_imu_to_arch<<std::endl;
std::cout <<"T_lidar_to_arch: "<<std::endl<<T_lidar_to_arch<<std::endl;
    // 2. 计算 T_imu_to_lidar = T_arch_to_lidar * T_imu_to_arch = T_arch_to_lidar * T_arch_to_imu.inverse()
    Eigen::Matrix4d T_imu_to_lidar = T_imu_to_arch * T_lidar_to_arch.inverse();

    
    // 打印结果
    std::cout << "Transform Matrix from IMU to Velodyne (T_imu_to_lidar):\n" 
              << T_imu_to_lidar << std::endl;
    //核对四元数形式
    std::cout << "核对四元数\n" 
              << std::endl;
    Eigen::Quaterniond q_imu_to_arch(0.7071068,0, 0, 0.7071068);
    Eigen::Quaterniond q_lidar_to_arch(0.9246235,0.0024754, 0.0060101, 0.3808269);
    q_imu_to_arch.normalize();
    q_lidar_to_arch.normalize();
    Eigen::Matrix4d T_imu_to_arch_1 = buildTransformMatrix(imu_position,q_imu_to_arch.toRotationMatrix());
    Eigen::Matrix4d T_lidar_to_arch_1 = buildTransformMatrix(lidar_position,q_lidar_to_arch.toRotationMatrix());
     //核对imu_to_arch
std::cout<<"T_imu_to_arch_1\n"<<T_imu_to_arch_1<<std::endl;
std::cout<<"T_lidar_to_arch_1\n"<<T_lidar_to_arch_1<<std::endl;

     Eigen::Matrix4d T_imu_to_lidar_1 = T_imu_to_arch_1 * T_lidar_to_arch_1.inverse();
    std::cout << "T_imu_to_lidar_1:\n" 
              << T_imu_to_lidar_1 << std::endl;

   Eigen::Matrix4d T_verify = T_imu_to_lidar * T_imu_to_lidar_1.inverse();
   std::cout << "四元数和rpy结果核对:\n" 
              << T_verify << std::endl;
    //位姿矩阵转四元数
    // === 提取 R ===
    Eigen::Matrix3d R_imu_to_lidar = T_imu_to_lidar.block<3,3>(0,0);
    Eigen::Matrix3d R_imu_to_lidar_1 = T_imu_to_lidar_1.block<3,3>(0,0);


    // === R 转四元数 ===
    Eigen::Quaterniond q_imu_to_lidar(R_imu_to_lidar);
    std::cout << "Quaternion [w, x, y, z]: ["
    << q_imu_to_lidar.w() << ", "
    << q_imu_to_lidar.x() << ", "
    << q_imu_to_lidar.y() << ", "
    << q_imu_to_lidar.z() << "]\n";

    Eigen::Quaterniond q_imu_to_lidar_1(R_imu_to_lidar_1);
    std::cout << "Quaternion [w, x, y, z]: ["
    << q_imu_to_lidar_1.w() << ", "
    << q_imu_to_lidar_1.x() << ", "
    << q_imu_to_lidar_1.y() << ", "
    << q_imu_to_lidar_1.z() << "]\n";


    return 0;
}