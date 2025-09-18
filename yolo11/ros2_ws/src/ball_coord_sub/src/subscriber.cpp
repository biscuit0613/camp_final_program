#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../kmfilter/KF.hpp"
#include <Eigen/Dense>
#include <unordered_map>

/*
这个功能是坐标转化+卡尔曼滤波。
本来想整可视化的，但是qt6在conda里面会报错，
这里选择了在sub中再建一个pub把数据传给专门负责可视化的python文件。
*/


class BallCoordSub : public rclcpp::Node {
public:
  BallCoordSub() : Node("ball_coord_sub") {
    // 从camera_calibration.json读取相机内参
    std::ifstream ifs("../src/camera_calibration.json");
    nlohmann::json j;
    ifs >> j;
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        camera_matrix_.at<double>(i, k) = j["camera_matrix"][i][k];
    dist_coeffs_ = cv::Mat(1, j["distortion_coefficients"].size(), CV_64F);
    for (size_t i = 0; i < j["distortion_coefficients"].size(); ++i)
      dist_coeffs_.at<double>(0, i) = j["distortion_coefficients"][i];

    // 声明参数
    this->declare_parameter("fps", 30.0);
    auto param_fps = this->get_parameter("fps");
    fps_ = param_fps.as_double();
    RCLCPP_INFO(this->get_logger(), "初始 FPS: %.2f", fps_);

    // 订阅 FPS
    sub_fps_ = this->create_subscription<std_msgs::msg::Float32>(
      "/ball/fps",
      10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        fps_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "更新 FPS: %.2f", fps_);
      });


    //跟pub那边一样，也建立一个qos
    rclcpp::QoS qos_profile(100);
    //qos_profile（）里面的数是队列深度
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    //pub里面用了reliable这里也要用

    //这下面就是创建订阅，接收pub轨迹点信息的代码了
    sub_center_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/ball/center_px", //topic的名称
      qos_profile,//qos,就是在pub里面定义的那个
      // 下面这一大坨是回调函数，就是收到消息了该怎么办。用了lambda表达式，比较简洁
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg){
        last_cx_ = msg->point.x;
        last_cy_ = msg->point.y;
        last_w_ = msg->point.z; // 直接用z存宽度
        last_stamp_ = msg->header.stamp;
        have_center_ = true;
        have_width_ = true;
        printIfReady(msg);
        //显然这里面没有获取id，因为id和上面获取的信息在 ROS 消息结构中的位置和用途不一样
        //坐标是放在point里面的，是浮点数，可以直接拿来用
      });
    //下面加一个pub,发卡尔曼滤波信息给visualize.py,用python实现可视化
    kf_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/ball/kf_pos", 100);
  }

  ~BallCoordSub() {
   
  }


private:
  std::vector<cv::Point3d> raw_points_;
  std::vector<cv::Point3d> kf_points_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_center_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_fps_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr kf_pub_;
  rclcpp::Time last_stamp_;
  std::unordered_map<int, KF> kf_map_; // 整了一个字典，key是目标的id,值是对应的卡尔曼滤波实例，这样就可以不同目标不同滤波。
  std::unordered_map<int, int> last_frame_num_map_; // 各目标最后处理（发布）的帧号
  std::unordered_map<int, double> last_time_map_;    // 各目标最后时间（秒）
  double fps_; // 从订阅获取

  void printIfReady(const geometry_msgs::msg::PointStamped::SharedPtr& msg) {
    // 解析frame_id，获取obj_id和frame_num
    std::stringstream ss(msg->header.frame_id);
    int obj_id, frame_num;
    char delim;
    ss >> obj_id >> delim >> frame_num;
    RCLCPP_INFO(this->get_logger(), "收到篮球id: %d, frame: %d", obj_id, frame_num);

    if (obj_id == -1) {
      // 空帧：对所有活跃目标在该帧时刻进行一次预测并发布
      double time = frame_num / fps_;
      for (auto& pair : kf_map_) {
        int id = pair.first;
        pair.second.predict(time);
        Eigen::Vector3d pred = pair.second.getPosition();
        geometry_msgs::msg::PointStamped kf_msg;
        kf_msg.header.stamp = msg->header.stamp;
        kf_msg.header.frame_id = std::to_string(id) + "_" + std::to_string(frame_num);
        kf_msg.point.x = pred[0];
        kf_msg.point.y = pred[1];
        kf_msg.point.z = pred[2];
        kf_pub_->publish(kf_msg);
        last_frame_num_map_[id] = frame_num; // 推进各自的最后帧号
        last_time_map_[id] = time;
        RCLCPP_INFO(this->get_logger(), "空帧预测: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)", id, frame_num, pred[0], pred[1], pred[2]);
      }
      return;
    }

    if (have_center_ && have_width_) {
      // solvePnP方法：用球心和球面上下左右五点做PnP解算，直接带参数就行，不用套公式了
      double cx = last_cx_;
      double cy = last_cy_;
      double w = last_w_;
      double h = last_w_; 
      double D = 0.246; // 球实际直径，单位：米
      double r = D / 2.0;
      std::vector<cv::Point2f> imagePoints = {
          {cx, cy},// 球心
          {cx, cy - h/2},// 上
          {cx, cy + h/2},// 下
          {cx - w/2, cy},// 左
          {cx + w/2, cy} // 右
      };
      std::vector<cv::Point3f> objectPoints = {
          {0, 0, 0},// 球心
          {0,  r, 0},// 上
          {0, -r, 0},// 下
          {-r, 0, 0},// 左
          { r, 0, 0} // 右
      };
      cv::Mat rvec, tvec;
      bool pnp_ok = cv::solvePnP(objectPoints, imagePoints, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
      if (pnp_ok) {
        Eigen::Vector3d obs(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        double time_cur = frame_num / fps_;  // 使用帧号时间同步

        // 插值缺失帧（基于该目标上一次的帧号），最多插值 5 帧避免误差累积
        int last_frame_for_id = last_frame_num_map_.count(obj_id) ? last_frame_num_map_[obj_id] : (frame_num - 1);
        int missing = frame_num - last_frame_for_id ;
        if (missing > 1 && missing <= 5) {
          for (int i = 1; i <= missing; ++i) {
            int frame_interp = last_frame_for_id + i;
            double t_interp = static_cast<double>(frame_interp) / fps_;
            kf_map_[obj_id].predict(t_interp);
            Eigen::Vector3d pred = kf_map_[obj_id].getPosition();
            geometry_msgs::msg::PointStamped kf_msg;
            kf_msg.header.stamp = msg->header.stamp;
            kf_msg.header.frame_id = std::to_string(obj_id) + "_" + std::to_string(frame_interp);
            kf_msg.point.x = pred[0];
            kf_msg.point.y = pred[1];
            kf_msg.point.z = pred[2];
            kf_pub_->publish(kf_msg);
            last_frame_num_map_[obj_id] = frame_interp;
            last_time_map_[obj_id] = t_interp;
            RCLCPP_INFO(this->get_logger(), "插值点: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)", obj_id, frame_interp, pred[0], pred[1], pred[2]);
          }
        } else if (missing > 5) {
          RCLCPP_WARN(this->get_logger(), "缺失帧过多 (%d)，跳过插值", missing);
        }

        // 使用当前观测更新滤波器
        kf_map_[obj_id].update(obs, time_cur);
        raw_points_.emplace_back(obs[0], obs[1], obs[2]);
        kf_points_.emplace_back(kf_map_[obj_id].getPosition()[0], kf_map_[obj_id].getPosition()[1], kf_map_[obj_id].getPosition()[2]);

        // 发布当前帧的卡尔曼滤波结果
        geometry_msgs::msg::PointStamped kf_msg;
        kf_msg.header.stamp = last_stamp_;
        kf_msg.header.frame_id = std::to_string(obj_id) + "_" + std::to_string(frame_num);
        kf_msg.point.x = kf_map_[obj_id].getPosition()[0];
        kf_msg.point.y = kf_map_[obj_id].getPosition()[1];
        kf_msg.point.z = kf_map_[obj_id].getPosition()[2];
        kf_pub_->publish(kf_msg);
        last_frame_num_map_[obj_id] = frame_num;
        last_time_map_[obj_id] = time_cur;
        RCLCPP_INFO(this->get_logger(), "PnP 解算后的相机坐标系下的球中心点(3d): (%.3f, %.3f, %.3f) [m] id=%d", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), obj_id);
      } else {
        RCLCPP_WARN(this->get_logger(), "solvePnP 没成功");
      }

      // 公式法（备用）
      /*
      double fx = camera_matrix_.at<double>(0, 0);
      double fy = camera_matrix_.at<double>(1, 1);
      double cx0 = camera_matrix_.at<double>(0, 2);
      double cy0 = camera_matrix_.at<double>(1, 2);
      double Z = fx * D / w;
      std::vector<cv::Point2f> pts = {cv::Point2f(last_cx_, last_cy_)};
      std::vector<cv::Point2f> undistorted;
      cv::undistortPoints(pts, undistorted, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
      double cx_undist = undistorted[0].x;
      double cy_undist = undistorted[0].y;
      double X = (cx_undist - cx0) * Z / fx;
      double Y = (cy_undist - cy0) * Z / fy;
      RCLCPP_INFO(this->get_logger(), "Camera coords: (%.3f, %.3f, %.3f) [m]", X, Y, Z);
      */
      have_center_ = have_width_ = false;
      /*
      和同学pnp算的比对了一下，不知道为什么，公式法的结果误差看起来比pnp更大
      */
    }
  }
  double last_cx_{0}, last_cy_{0}, last_w_{0};
  bool have_center_{false}, have_width_{false};

  cv::Mat camera_matrix_, dist_coeffs_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BallCoordSub>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
