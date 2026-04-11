#include <mutex>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

#include <yaml-cpp/yaml.h>  // 用于YAML解析
#include <thread>           // 用于std::thread
#include <fstream>          // 用于文件读取
#include <sstream>          // 用于字符串流
#include <vector>
#include <string>
#include <hdl_localization/TileRequest.h>

namespace hdl_localization {

namespace {
bool ensure_directory_exists(const std::string& dir_path) {
  if (dir_path.empty()) {
    return true;
  }

  std::string current;
  if (dir_path.front() == '/') {
    current = "/";
  }

  std::stringstream ss(dir_path);
  std::string part;
  while (std::getline(ss, part, '/')) {
    if (part.empty()) {
      continue;
    }
    if (!current.empty() && current.back() != '/') {
      current += "/";
    }
    current += part;
    if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

bool ensure_parent_dir_for_file(const std::string& file_path) {
  const std::size_t slash = file_path.find_last_of('/');
  if (slash == std::string::npos) {
    return true;
  }
  return ensure_directory_exists(file_path.substr(0, slash));
}
}  // namespace

class GlobalmapServerNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet() {}
  virtual ~GlobalmapServerNodelet() {
    if (tile_loading_csv_stream_.is_open()) {
      tile_loading_csv_stream_.flush();
      tile_loading_csv_stream_.close();
    }
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    // initialize_params();

    // publish globalmap with "latched" publisher
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);

    // 动态地图
    map_update_sub = nh.subscribe("/map_request/pcd", 10, &GlobalmapServerNodelet::map_update_callback, this);
    init_tile_loading_csv();

    // globalmap_pub_timer = nh.createWallTimer(ros::WallDuration(1.0), &GlobalmapServerNodelet::pub_once_cb, this, true, true);

    std::string metadata_file = private_nh.param<std::string>("metadata_file", "/home/scy/autoware_map_output/pointcloud_map_metadata.yaml");

    tile_dir = private_nh.param<std::string>("tile_map_dir", "/home/scy/autoware_map_output/pointcloud_map");

    std::ifstream metadata_ifs(metadata_file);
    if (!metadata_ifs.good()) {
      NODELET_FATAL_STREAM("metadata_file not found or unreadable: " << metadata_file);
      return;
    }

    YAML::Node config;
    try {
      config = YAML::LoadFile(metadata_file);
    } catch (const YAML::Exception& e) {
      NODELET_FATAL_STREAM("failed to parse metadata_file: " << metadata_file << ", error: " << e.what());
      return;
    }

    x_res = config["x_resolution"].as<double>();
    y_res = config["y_resolution"].as<double>();

    for (auto it = config.begin(); it != config.end(); ++it) {
      std::string name = it->first.as<std::string>();
      if (name == "x_resolution" || name == "y_resolution") continue;

      auto vals = it->second.as<std::vector<double>>();
      tile_map[name] = Eigen::Vector2f(vals[0], vals[1]);
    }

    NODELET_INFO_STREAM("Loaded " << tile_map.size() << " tiles from metadata.");
  }

private:
  // void initialize_params() {
  //   // read globalmap from a pcd file
  //   std::string globalmap_pcd = private_nh.param<std::string>("globalmap_pcd", "");
  //   globalmap.reset(new pcl::PointCloud<PointT>());
  //   pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
  //   globalmap->header.frame_id = "map";

  //   std::ifstream utm_file(globalmap_pcd + ".utm");
  //   if (utm_file.is_open() && private_nh.param<bool>("convert_utm_to_local", true)) {
  //     double utm_easting;
  //     double utm_northing;
  //     double altitude;
  //     utm_file >> utm_easting >> utm_northing >> altitude;
  //     for (auto& pt : globalmap->points) {
  //       pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
  //     }
  //     ROS_INFO_STREAM("Global map offset by UTM reference coordinates (x = " << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
  //   }

  //   // downsample globalmap,修改只有当downsample_resolution大于0时才进行下采样
  //   double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
  //   if (downsample_resolution > 0.0) {
  //     boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
  //     voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  //     voxelgrid->setInputCloud(globalmap);

  //     pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  //     voxelgrid->filter(*filtered);
  //     globalmap = filtered;
  //   }
  // }

  void pub_once_cb(const ros::WallTimerEvent& event) { globalmap_pub.publish(globalmap); }

  void init_tile_loading_csv() {
    tile_loading_csv_enabled_ = private_nh.param<bool>("enable_tile_loading_csv", true);
    tile_loading_csv_path_ = private_nh.param<std::string>("tile_loading_csv_path", std::string(""));
    if (!tile_loading_csv_enabled_) {
      return;
    }

    if (tile_loading_csv_path_.empty()) {
      tile_loading_csv_path_ = "/tmp/tile_server_loading_metrics.csv";
    }

    if (!ensure_parent_dir_for_file(tile_loading_csv_path_)) {
      tile_loading_csv_enabled_ = false;
      NODELET_WARN_STREAM("tile loading csv disabled: failed to create parent directory for path: " << tile_loading_csv_path_);
      return;
    }

    tile_loading_csv_stream_.open(tile_loading_csv_path_, std::ios::out | std::ios::trunc);
    if (!tile_loading_csv_stream_.is_open()) {
      tile_loading_csv_enabled_ = false;
      NODELET_WARN_STREAM("tile loading csv disabled: failed to open path: " << tile_loading_csv_path_);
      return;
    }

    tile_loading_csv_stream_ << "stamp,tile_count,load_ms,merged_points,tiles\n";
    NODELET_INFO_STREAM("tile loading csv enabled: " << tile_loading_csv_path_);
  }

  void log_tile_loading_event(const ros::Time& stamp, std::size_t tile_count, double load_ms, std::size_t merged_points, const std::vector<std::string>& tiles) {
    if (!tile_loading_csv_enabled_ || !tile_loading_csv_stream_.is_open()) {
      return;
    }

    std::ostringstream joined_tiles;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
      if (i) joined_tiles << "|";
      joined_tiles << tiles[i];
    }

    tile_loading_csv_stream_ << std::fixed << std::setprecision(9) << stamp.toSec() << "," << tile_count << "," << load_ms << "," << merged_points << "," << joined_tiles.str() << "\n";
    tile_loading_csv_stream_.flush();
  }

  // void map_update_callback(const std_msgs::String& msg) {
  //   ROS_INFO_STREAM("Received map request, map path : " << msg.data);
  //   std::string globalmap_pcd = msg.data;
  //   globalmap.reset(new pcl::PointCloud<PointT>());
  //   pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
  //   globalmap->header.frame_id = "map";

  //   // downsample globalmap
  //   double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
  //   boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
  //   voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  //   voxelgrid->setInputCloud(globalmap);

  //   pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  //   voxelgrid->filter(*filtered);

  //   globalmap = filtered;
  //   globalmap_pub.publish(globalmap);
  // }

  // 地图加载
  void load_tiles(const std::vector<std::string>& tiles) {
    pcl::PointCloud<PointT>::Ptr merged(new pcl::PointCloud<PointT>());
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& name : tiles) {
      std::string path = tile_dir + "/" + name;

      pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
      if (pcl::io::loadPCDFile(path, *cloud) < 0) {
        NODELET_WARN_STREAM("Failed to load: " << path);
        continue;
      }

      double downsample_resolution = private_nh.param<double>("downsample_resolution", 20);
      if (downsample_resolution > 0.0) {
        boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
        voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
        voxelgrid->setInputCloud(cloud);

        pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
        voxelgrid->filter(*filtered);

        *merged += *filtered;

      } else {
        *merged += *cloud;
      }
    }

    merged->header.frame_id = "map";

    globalmap = merged;
    globalmap_pub.publish(globalmap);
    const auto t1 = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    log_tile_loading_event(ros::Time::now(), tiles.size(), load_ms, merged->size(), tiles);
    NODELET_INFO_STREAM("globalmap loaded in globalmap_noderet");
  }

  /**
   * @brief 地图更新回调函数，处理接收到的地图瓦片列表消息
   * @param msg 包含地图瓦片文件名列表的字符串消息，格式为"file1.pcd,file2.pcd,file3.pcd"
   */
  void map_update_callback(const hdl_localization::TileRequestConstPtr& msg) {
    // 清空当前瓦片列表
    current_tiles.clear();

    // 清空当前瓦片列表
    current_tiles = msg->tiles;

    NODELET_INFO_STREAM("Loading " << current_tiles.size() << " tiles.");

    // 加载点云瓦片
    load_tiles(current_tiles);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  ros::Publisher globalmap_pub;
  ros::Subscriber map_update_sub;

  ros::WallTimer globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;

  // 动态地图变量
  std::map<std::string, Eigen::Vector2f> tile_map;
  std::string tile_dir;

  double x_res, y_res;

  std::vector<std::string> current_tiles;
  bool tile_loading_csv_enabled_ = false;
  std::string tile_loading_csv_path_;
  std::ofstream tile_loading_csv_stream_;
};

}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
