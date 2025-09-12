#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../kmfilter/KF.hpp"
#include <Eigen/Dense>

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

    //跟pub那边一样，也建立一个qos
    rclcpp::QoS qos_profile(100);
    //qos_profile（）里面的数是队列深度
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    //pub里面用了reliable这里也要用

    //这下面就是接收pub轨迹点信息的代码了
    sub_center_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/ball/center_px", qos_profile,
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg){
        last_cx_ = msg->point.x;
        last_cy_ = msg->point.y;
        last_w_ = msg->point.z; // 直接用z存宽度
        last_stamp_ = msg->header.stamp;
        have_center_ = true;
        have_width_ = true;
        printIfReady();
      });
    // 不再需要sub_width_
    //这里新加一个pub,发卡尔曼滤波信息给visualize.py,用python实现可视化
    kf_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/ball/kf_pos", 10);
  }

  ~BallCoordSub() {
   
  }


private:
  KF kf_;
  std::vector<cv::Point3d> raw_points_;
  std::vector<cv::Point3d> kf_points_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_center_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr kf_pub_;
  rclcpp::Time last_stamp_;

  void printIfReady() {
    if (have_center_ && have_width_) {
      // solvePnP方法：用球心和球面上下左右五点做PnP解算，直接带参数就行，不用套公式了
      double cx = last_cx_;
      double cy = last_cy_;
      double w = last_w_;
      double h = last_w_; // 假设为正圆
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
        double now = this->now().seconds();
        kf_.update(obs, now);
        raw_points_.emplace_back(obs[0], obs[1], obs[2]);
        kf_points_.emplace_back(kf_.getPosition()[0], kf_.getPosition()[1], kf_.getPosition()[2]);
        for (int i = 1; i <= 2; ++i) {
          double t_interp = now + i * 0.01;
          kf_.predict(t_interp);
          Eigen::Vector3d pred = kf_.getPosition();
          kf_points_.emplace_back(pred[0], pred[1], pred[2]);
        }
        // 发布卡尔曼滤波结果，带时间戳
        geometry_msgs::msg::PointStamped kf_msg;
        kf_msg.header.stamp = last_stamp_;
        kf_msg.point.x = kf_.getPosition()[0];
        kf_msg.point.y = kf_.getPosition()[1];
        kf_msg.point.z = kf_.getPosition()[2];
        kf_pub_->publish(kf_msg);
        RCLCPP_INFO(this->get_logger(), "PnP Camera coords: (%.3f, %.3f, %.3f) [m]", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
      } else {
        RCLCPP_WARN(this->get_logger(), "solvePnP failed!");
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
