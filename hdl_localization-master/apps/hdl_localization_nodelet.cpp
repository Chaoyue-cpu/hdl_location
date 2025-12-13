#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

// 确保包含FastVGICP的头文件（根据fast_gicp版本可能不同）
#include <fast_gicp/gicp/fast_vgicp.hpp>            // 常规版本
#include <fast_gicp/gicp/impl/fast_vgicp_impl.hpp>  // 模板实现

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>
// 经典GICP (PCL原生实现)
#include <pcl/registration/gicp.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// GICP的多线程版本 (需pclomp)
// #include <pclomp/gicp.h>

// FAST_GICP (需fast_gicp库)
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>

// 添加这些头文件
#include <yaml-cpp/yaml.h>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
// 确保在 hdl_localization_nodelet.cpp 开头添加：
#include <thread>  // 用于 std::thread
#include <hdl_localization/TileRequest.h>

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {}
  virtual ~HdlLocalizationNodelet() {
    // 停止地图线程
    if (map_thread_.joinable()) {
      // 先停止callback queue
      map_queue_.clear();
      map_queue_.disable();

      // 等待线程结束
      map_thread_.join();
    }

    // 清理其他资源
    if (registration) {
      registration.reset();
    }

    NODELET_INFO("Nodelet cleanup completed");
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    // 动态地图参数
    tile_request_pub = nh.advertise<hdl_localization::TileRequest>("/map_request/pcd", 1);

    tile_radius = private_nh.param<int>("tile_radius", 3);  // 默认3×3

    std::string metadata_file = private_nh.param<std::string>("metadata_file", "/home/scy/autoware_map_output/pointcloud_map_metadata.yaml");

    YAML::Node config = YAML::LoadFile(metadata_file);
    x_res = config["x_resolution"].as<double>();
    y_res = config["y_resolution"].as<double>();

    for (auto it = config.begin(); it != config.end(); ++it) {
      std::string name = it->first.as<std::string>();
      if (name == "x_resolution" || name == "y_resolution") continue;

      auto vals = it->second.as<std::vector<double>>();
      tile_map[name] = Eigen::Vector2f(vals[0], vals[1]);
    }

    NODELET_INFO_STREAM("Tile metadata loaded: " << tile_map.size());

    // IMU_to_Base_link

    private_nh.param("q_imu_to_base/x", qx, 0.0);
    private_nh.param("q_imu_to_base/y", qy, 0.0);
    private_nh.param("q_imu_to_base/z", qz, 0.0);
    private_nh.param("q_imu_to_base/w", qw, 1.0);
    q_imu_to_base = Eigen::Quaternionf(qw, qx, qy, qz).normalized();  // 注意参数的顺序：实部 w 系数在前，而内部存储顺序为：[x, y, z, w]。

    // 平移
    private_nh.param("t_imu_to_base/x", tx, 0.0);
    private_nh.param("t_imu_to_base/y", ty, 0.0);
    private_nh.param("t_imu_to_base/z", tz, 0.0);
    t_imu_to_base = Eigen::Vector3f(tx, ty, tz);
    // 时间偏移补偿
    private_nh.param("time_offset_lidar_to_imu", time_offset_lidar_to_imu, 0.0);

    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if (use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }

    // ----------------- 启动地图更新线程 -----------------
    // map_thread_ = std::thread([this]() {
    //   // NODELET_WARN("Map thread started! Thread ID: %ld", std::this_thread::get_id());
    //   // ros::NodeHandle nh_map(getPrivateNodeHandle());
    //   ros::NodeHandle nh_map;  // 默认构造函数，全局命名空间
    //   NODELET_WARN("NodeHandle namespace before setCallbackQueue: %s", nh_map.getNamespace().c_str());
    //   nh_map.setCallbackQueue(&map_queue_);
    //   NODELET_WARN("NodeHandle namespace after setCallbackQueue: %s", nh_map.getNamespace().c_str());
    //   globalmap_sub_ = nh_map.subscribe("/globalmap", 3, &HdlLocalizationNodelet::globalmapCallback, this);
    //   // 立即检查订阅状态
    //   NODELET_WARN("Subscriber created. Topic: %s", globalmap_sub_.getTopic().c_str());
    //   NODELET_WARN("Is subscriber valid? %s", globalmap_sub_ ? "YES" : "NO");
    //   for (int i = 0; i < 10 && ros::ok(); ++i) {
    //     int num_pubs = globalmap_sub_.getNumPublishers();
    //     NODELET_WARN("Publisher count for /globalmap: %d (check %d/10)", num_pubs, i + 1);
    //     // ...
    //   }
    //   ros::Rate rate(200);  // 用高频非阻塞轮询，不退出
    //   while (ros::ok()) {
    //     map_queue_.callAvailable();
    //     rate.sleep();
    //   }
    // });
    map_thread_ = std::thread(&HdlLocalizationNodelet::mapThreadFunc, this);

    // updateDynamicTiles(pose_estimator->pos());
    // 延迟调用，确保地图线程已启动
    if (pose_estimator) {
      Eigen::Vector3f pos = pose_estimator->pos();
      NODELET_WARN("Initial position for tile request: (%.2f, %.2f, %.2f)", pos.x(), pos.y(), pos.z());
      updateDynamicTiles(pos);
    } else {
      NODELET_WARN("Pose estimator not initialized yet, will update tiles after first scan");
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if (reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      // 创建了一个指针，指向new的对象
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      }

      else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    } else if (reg_method == "GICP") {
      NODELET_INFO("GICP (Generalized ICP) is selected");
      pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
      gicp->setTransformationEpsilon(0.01);
      gicp->setMaximumIterations(64);
      gicp->setMaxCorrespondenceDistance(2.5);
      gicp->setCorrespondenceRandomness(20);
      gicp->setMaximumOptimizerIterations(20);
      return gicp;
    }
    // else if (reg_method == "GICP_OMP") {
    //   NODELET_INFO("GICP_OMP (Multi-threaded GICP) is selected");
    //   pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());
    //   gicp->setTransformationEpsilon(0.01);
    //   gicp->setMaximumIterations(64);
    //   gicp->setMaxCorrespondenceDistance(2.5);
    //   gicp->setCorrespondenceRandomness(20);
    //   gicp->setMaximumOptimizerIterations(20);
    //   return gicp;
    // }
    else if (reg_method == "FAST_GICP") {
      NODELET_INFO("FAST_GICP is selected");
      std::cout << "registration: FAST_GICP" << std::endl;

      fast_gicp::FastGICP<PointT, PointT>::Ptr fast_gicp(new fast_gicp::FastGICP<PointT, PointT>());
      fast_gicp->setNumThreads(8);                   // 可配置参数
      fast_gicp->setTransformationEpsilon(0.0001);   // 若两次估计间差异小于0.0001，则认为已收敛
      fast_gicp->setMaximumIterations(64);           // 最大迭代次数
      fast_gicp->setMaxCorrespondenceDistance(2.5);  // 最大搜索半径
      fast_gicp->setCorrespondenceRandomness(20);    // 每个点求局部协方差时的最近邻个数,k 越大，局部面特征估计更稳定, 一般5~20
      // fast_gicp->setMaximumOptimizerIterations(20);//fast_gicp比普通gicp快一个数量级
      return fast_gicp;
    } else if (reg_method == "FAST_VGICP") {
      // 理论上更适合隧道
      std::cout << "registration: FAST_VGICP" << std::endl;
      fast_gicp::FastVGICP<PointT, PointT>::Ptr vgicp(new fast_gicp::FastVGICP<PointT, PointT>());
      // setNumThreads（0）自动根据硬件线程数选择并行线程数。
      // 原参数
      //  vgicp->setNumThreads(pnh.param<int>("reg_num_threads", 0));
      //  vgicp->setResolution(pnh.param<double>("reg_resolution", 1.0));
      //  vgicp->setTransformationEpsilon(pnh.param<double>("reg_transformation_epsilon", 0.01));
      //  vgicp->setMaximumIterations(pnh.param<int>("reg_maximum_iterations", 64));
      //  vgicp->setCorrespondenceRandomness(pnh.param<int>("reg_correspondence_randomness", 20));

      // vgicp->setNumThreads(0);
      // vgicp->setResolution(1.0);
      // vgicp->setTransformationEpsilon(0.01);
      // vgicp->setMaximumIterations( 64);
      // vgicp->setCorrespondenceRandomness(20);

      // gpt推荐参数
      vgicp->setNumThreads(8);                 // 自动使用硬件线程数，保留
      vgicp->setResolution(0.4);               // 分辨率建议降低（更细粒度的 voxel 支持更准确的高斯建模）
      vgicp->setTransformationEpsilon(1e-4);   // 收敛阈值适当收紧，避免过早收敛
      vgicp->setMaximumIterations(100);        // 通常 64 是够的，也可以调到 100 查看差异
      vgicp->setCorrespondenceRandomness(20);  // 对退化环境，减少随机邻域点数，有助于抑制噪声影响

      return vgicp;
      /* code */
    }

    // 没有返回的对象就出错，因为匹配方法不对
    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void mapThreadFunc() {
    ros::NodeHandle nh_map;
    nh_map.setCallbackQueue(&map_queue_);

    NODELET_INFO("Map thread started!!");

    globalmap_sub_ = nh_map.subscribe("/globalmap", 10, &HdlLocalizationNodelet::globalmapCallback, this);

    // 等待latched消息
    ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/globalmap", nh_map);

    ros::Rate rate(100);
    while (ros::ok()) {
      map_queue_.callAvailable(ros::WallDuration(0.01));
      rate.sleep();
    }
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    if (private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(
        registration,
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(
          private_nh.param<double>("init_ori_w", 1.0),
          private_nh.param<double>("init_ori_x", 0.0),
          private_nh.param<double>("init_ori_y", 0.0),
          private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)));
    }
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    {
      std::unique_lock<std::mutex> lock(map_mutex);

      if (has_staging_map) {
        // 检查指针是否为null和空
        if (!staging_map || staging_map->empty()) {
          NODELET_WARN("staging_map is null");
          return;
        }
        active_map.swap(staging_map);
        has_staging_map = false;
        if (!active_map || active_map->empty()) {
          NODELET_WARN("No active map yet, skip frame.");
          return;
        }

        // 要确保 scan thread 不在注册过程中使用旧 target
        registration->setInputTarget(active_map);
        ROS_INFO("Active map updated.");
      }
    }

    // ------- 2. 点云转换 -------
    // 原始
    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);
    // std::cout<<"pcl_cloud坐标系:"<<pcl_cloud->header.frame_id<<std::endl;
    if (pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }
    // ------- 3. 动态 tile 计算 -------
    // 动态地图
    // 找到当前 tile → 扩展 tile_radius → 生成 tile 列表
    updateDynamicTiles(pose_estimator->pos());

    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    // 准备修改
    // ros::Duration 对象指定了 canTransform 函数最多等待 0.1 秒钟
    // 能将点云转换到指定的imu下

    if (this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, ros::Time(0), ros::Duration(0.1), &tfError)) {
      // 输入点云，类型为 pcl::PointCloud<T>，原始点云，坐标系是 pcl_cloud->header.frame_id（例如 "base_link"）。
      // *cloud	输出点云，类型同上，转换后的点云将存储在这里，坐标系为 odom_child_frame_id。
      if (!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
        NODELET_ERROR("point cloud cannot be transformed into target frame!!");
        return;
      }
    } else {
      NODELET_ERROR(tfError.c_str());
      return;
    }

    auto filtered = downsample(cloud);
    last_scan = filtered;

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }
    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      // 在这段代码中，我们给预测函数提供的是 IMU 数据的时间戳（控制向量），而不是点云（观测值）的时间戳，
      // 并且“if(stamp < (*imu_iter)->header.stamp)”这一条件语句防止了在当前观测值时间戳之后的 IMU 数据被输入。因此，只有在观测值时间戳之前的 IMU 数据被用于预测 UKF
      // 状态。您明白了吗？ 如果imu时间戳小于雷达的，就往下执行，一直执行到大于雷达时间戳的imu数据，那么旧的imu数据已经使用过删除
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        // 如果当前观测值时间戳小于IMU数据时间戳，则跳出循环
        const ros::Time imu_time_corr = (*imu_iter)->header.stamp + ros::Duration(time_offset_lidar_to_imu);

        if (stamp < imu_time_corr) {
          break;
        }
        const auto& imu_acc = (*imu_iter)->linear_acceleration;
        // std::cout << "acc_original: " << Eigen::Vector3f(acc.x, acc.y, acc.z).transpose() << std::endl;

        const auto& imu_gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;

        // 2. 转换为Eigen类型并打印原始值
        Eigen::Vector3f acc_row(imu_acc.x, imu_acc.y, imu_acc.z);
        Eigen::Vector3f gyro_row(imu_gyro.x, imu_gyro.y, imu_gyro.z);
        // std::cout << "acc_original: " << acc_row.transpose() << std::endl;

        // 修改,将imu数据转换到baselink
        //  旋转到 base_link 坐标系

        Eigen::Vector3f acc = q_imu_to_base * acc_row;
        Eigen::Vector3f gyro = q_imu_to_base * gyro_row;

        pose_estimator->predict(imu_time_corr, acc_sign * acc, gyro_sign * gyro);
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    // 如果启用了基于机器人里程计的预测功能，并且上次校正时间不为零
    if (private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;  // 定义一个存储里程计变化的变量

      // 尝试查找从上次校正时间到当前时间，odom_child_frame_id 和 robot_odom_frame_id 之间的变换关系
      if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.1))) {
        // 如果可以找到变换关系，则获取该变换
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        // 如果第一种方式失败，尝试另一种方式查找变换关系
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      // 如果未成功获取变换关系（时间戳为零），发出警告
      if (odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        // 将变换关系转换为Eigen矩阵形式
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        // 使用里程计变化进行姿态预测
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if (aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    if (status_pub.getNumSubscribers()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }
    // 发布里程计
    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());
  }

  void updateDynamicTiles(const Eigen::Vector3f& position) {
    // NODELET_INFO("updateDynamicTiles - Position: (%.3f, %.3f, %.3f)", position.x(), position.y(), position.z());

    // 1. 计算需要的瓦片
    std::vector<std::string> need_tiles = computeRequiredTiles(position);
    // 2. 如果瓦片集合发生变化，发布请求
    if (need_tiles != last_tiles) {
      publishTileRequest(need_tiles);
      last_tiles = need_tiles;
      NODELET_INFO("Number of tiles needed: %zu", need_tiles.size());
    }
  }

  /**
   * @brief 计算当前位置需要的动态地图瓦片
   * @param position 当前位置 (x, y, z)
   * @return 需要加载的瓦片文件名列表
   */
  std::vector<std::string> computeRequiredTiles(const Eigen::Vector3f& position) {
    // 计算当前坐标所在的瓦片网格
    int current_tile_x = floor(position.x() / x_res) * x_res;  // 例如: 123.4 → 100
    int current_tile_y = floor(position.y() / y_res) * y_res;  // 例如: -178.9 → -200
    // NODELET_INFO("Current tile grid: (%d, %d)", current_tile_x, current_tile_y);

    std::vector<std::string> need_tiles;

    // 遍历周围瓦片区域
    for (int dx = -tile_radius; dx <= tile_radius; dx++) {
      for (int dy = -tile_radius; dy <= tile_radius; dy++) {
        // 目标瓦片的左下角坐标
        float target_x = current_tile_x + dx * x_res;
        float target_y = current_tile_y + dy * y_res;

        // NODELET_INFO("  Checking tile at: (%.1f, %.1f)", target_x, target_y);

        // 查找匹配的瓦片
        for (auto& kv : tile_map) {
          auto origin = kv.second;
          if (fabs(origin.x() - target_x) < 1 && fabs(origin.y() - target_y) < 1) {
            need_tiles.push_back(kv.first);  // 文件名压入
            break;                           // 找到就跳出内层循环
          }
        }
      }
    }

    return need_tiles;
  }

  /**
   * @brief 发布瓦片请求
   * @param tiles 需要请求的瓦片列表
  //  */
  // void publishTileRequest(const std::vector<std::string>& tiles) {
  //   if (tiles.empty()) {
  //     return;
  //   }

  //   std_msgs::String msg;
  //   std::stringstream ss;

  //   for (size_t i = 0; i < tiles.size(); ++i) {
  //     ss << tiles[i];
  //     if (i < tiles.size() - 1) {
  //       ss << ",";
  //     }
  //   }

  //   msg.data = ss.str();
  //   tile_request_pub.publish(msg);

  //   NODELET_INFO_STREAM("Requesting " << tiles.size() << " tiles.");
  // }
  void publishTileRequest(const std::vector<std::string>& tiles) {
    if (tiles.empty()) {
      return;
    }

    hdl_localization::TileRequest msg;
    msg.tiles = tiles;
    tile_request_pub.publish(msg);

    NODELET_INFO("Requesting %zu tiles.", tiles.size());
  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);
    NODELET_INFO("Globalmap frame_id: %s", points_msg->header.frame_id.c_str());

    if (use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if (!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  // 新线程地图
  void globalmapCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    NODELET_WARN("globalmapCallback received");
    pcl::PointCloud<PointT>::Ptr new_map(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *new_map);

    if (new_map->empty()) {
      NODELET_WARN("Received empty globalmap tile");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(map_mutex);
      staging_map = new_map;
      has_staging_map = true;
      map_ready = false;
    }

    NODELET_INFO_STREAM("Global map updated, points=" << staging_map->size());
    // globalmap空指针未初始化导致的内存崩溃
    // NODELET_INFO_STREAM("Global map updated, points=" << globalmap->size());

    // 函数结束，lock_guard离开作用域，自动解锁！
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(
      new hdl_localization::PoseEstimator(registration, pose.translation(), Eigen::Quaternionf(pose.linear()), private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   * 初始化rviz中点击的位姿估计
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      Eigen::Vector3f(p.x, p.y, p.z),
      Eigen::Quaternionf(q.w, q.x, q.y, q.z),
      private_nh.param<double>("cool_time_duration", 0.5)));
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry发布当前位姿的TF变换和里程计信息
   * @param stamp  timestamp
   * @param pose   odometry pose to be published当前的位姿（4x4变换矩阵）
   */

  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // 如果tf树中可以查询到 robot_odom_frame_id 到 odom_child_frame_id 的变换
    if (tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      // 1. 先计算 map 到 odom_child_frame 的变换（注意这里pose是 base_link 在 map 下的位姿，需取逆）
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      // 2. 查找 robot_odom_frame_id 到 odom_child_frame_id 的变换（如 base_link->odom）
      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));

      // 3. 将 map -> odom_child_frame 的变换转换到 map -> odom 的变换（即连乘）
      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      // 4. 计算 odom -> map 的变换（即map->odom的逆）
      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      // 5. 发布 odom -> map 的变换（用于构建TF树）
      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id;

      tf_broadcaster.sendTransform(odom_trans);
    } else {
      // 如果无法获得 tf 树中 frame 的变换信息，则直接发布 map -> odom_child_frame 的变换
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;
      tf_broadcaster.sendTransform(odom_trans);
    }

    // 6. 构造并发布 nav_msgs::Odometry 消息
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";  // 里程计在 map 坐标系下的位姿

    // 使用 tf 工具将 Eigen 位姿转换为 ROS 消息类型
    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;  // 子坐标系（一般为 base_link）
    // std::cout<<"odom.child_frame_id: "<<odom.child_frame_id<<std::endl;

    // 速度信息设置为 0（若需要速度信息，应从运动估计器获得）
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    // 发布 odometry 信息
    pose_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status information
   */
  /**
   * @brief 发布扫描匹配的状态信息，包括配准是否收敛、匹配误差、内点比例等
   * @param header   点云对应的ROS时间戳与frame_id信息
   * @param aligned  当前帧与地图配准后的点云
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    // 初始化状态消息
    ScanMatchingStatus status;
    status.header = header;

    // 配准是否收敛
    status.has_converged = registration->hasConverged();
    status.matching_error = 0.0;

    // 获取私有参数（可在launch文件或yaml中配置）
    const double max_correspondence_dist = private_nh.param<double>("status_max_correspondence_dist", 0.5);  // 最大对应点距离
    const double max_valid_point_dist = private_nh.param<double>("status_max_valid_point_dist", 25.0);       // 有效点的最大距离

    int num_inliers = 0;            // 满足距离阈值的点数
    int num_valid_points = 0;       // 满足最大距离要求的有效点数
    std::vector<int> k_indices;     // 最近邻搜索的结果索引
    std::vector<float> k_sq_dists;  // 最近邻距离的平方

    // 遍历配准后的点云
    for (int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);

      // 距离太远的点不参与计算
      if (pt.getVector3fMap().norm() > max_valid_point_dist) {
        continue;
      }
      num_valid_points++;

      // 搜索目标点云中与当前点最近的一个点
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);

      // 若最近邻距离在阈值内，视为内点，累计误差
      if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        status.matching_error += k_sq_dists[0];
        num_inliers++;
      }
    }

    // 平均匹配误差（只对内点求均值）
    status.matching_error /= num_inliers;

    // 内点比例（避免除0，用 max(1, num_valid_points)）
    status.inlier_fraction = static_cast<float>(num_inliers) / std::max(1, num_valid_points);

    // 最终的位姿变换（配准结果）
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    // 预测误差标签与值
    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    // 收集不同预测模型的误差（与配准估计结果的差异）
    if (pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";  // 无预测
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";  // 使用IMU或运动模型预测
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";  // 使用里程计
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }
    if (pose_estimator->imu_odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "imu_odom";  // 使用里程计
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_odom_prediction_error().get().cast<double>())).transform);
    }

    // 发布状态信息
    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  // 修改
  //  TF: imu_link → base_link 的旋转（你提供的值）
  // 在类定义中添加成员变量（替换原来的函数定义）
  Eigen::Quaternionf q_imu_to_base;
  Eigen::Vector3f t_imu_to_base;
  double qx, qy, qz, qw;
  double tx, ty, tz;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  // std::unique_ptr 是 C++ 标准库中的一种智能指针，用于管理动态分配的对象，并确保只有一个指针可以指向该对象。
  // 当 std::unique_ptr 被销毁时，它所管理的对象也会被自动删除（调用析构函数并释放内存），从而避免内存泄漏。
  // pose_estimator.reset()）时，它所指向的 PoseEstimator 对象将被自动删除。
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;

  // 动态地图
  std::map<std::string, Eigen::Vector2f> tile_map;
  std::mutex map_mutex;
  pcl::PointCloud<PointT>::Ptr current_map{nullptr};  // 主线程读
  pcl::PointCloud<PointT>::Ptr pending_map{nullptr};  // 地图线程写

  double x_res, y_res;
  int tile_radius;  // 1 => 3x3, 2 => 5x5

  std::vector<std::string> last_tiles;

  ros::Publisher tile_request_pub;

  double time_offset_lidar_to_imu;  // 时间偏移量

  // --- 新增：地图更新专用线程 ---
  std::thread map_thread_;
  ros::CallbackQueue map_queue_;   // 专用 callback queue
  ros::Subscriber globalmap_sub_;  // 地图订阅（在独立线程中）
  std::mutex map_mutex_;           // 用于 registration 替换
  bool map_ready = false;

  pcl::PointCloud<PointT>::Ptr active_map;   // 当前定位用地图
  pcl::PointCloud<PointT>::Ptr staging_map;  // 缓存新地图
  bool has_staging_map = false;
  std::atomic<bool> map_switch_requested{false};  // 标记请求加载
  std::mutex map_request_mutex;
  // pcl::PointCloud<PointT>::Ptr globalmap{nullptr};
  std::set<std::string> loaded_tiles;

  // --- 新增：主动使用的 registration（主线程用） ---
  std::shared_ptr<pcl::Registration<PointT, PointT>> active_registration_;

  // --- 新增：地图线程中使用的临时 registration ---
  std::shared_ptr<pcl::Registration<PointT, PointT>> updating_registration_;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
