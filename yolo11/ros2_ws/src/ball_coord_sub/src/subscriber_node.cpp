#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "pnp_solver.h"
#include "kalman_manager.h"
#include "../../common/config.h"

//注意：卡尔曼滤波器里面update方法实际上只有一个参数，就是观测坐标，dt是因为之前（KF.cpp的那个版本，没有引入加加速度的）predict方法在update里面调用了所以传进去了

// 篮球坐标订阅节点类，负责接收2D检测结果，进行PnP解算和卡尔曼滤波跟踪
class BallCoordSub : public rclcpp::Node {
public:
  // 构造函数：初始化节点，读取相机参数，设置订阅者和发布者
  BallCoordSub() : Node("ball_coord_sub") {
    // 从json文件读取相机内参
    std::ifstream ifs(DEFAULT_CALIB_PATH);
    nlohmann::json j;
    ifs >> j;
    CameraMatrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        CameraMatrix.at<double>(i, k) = j["camera_matrix"][i][k];
    DistCoeffs = cv::Mat(1, j["distortion_coefficients"].size(), CV_64F);
    for (size_t i = 0; i < j["distortion_coefficients"].size(); ++i)
      DistCoeffs.at<double>(0, i) = j["distortion_coefficients"][i];

    PnPSolver_ = std::make_unique<PnPSolver>(CameraMatrix, DistCoeffs, BALL_D);

    this->declare_parameter("fps", DEFAULT_FPS);
    auto param_fps = this->get_parameter("fps");
    Fps = param_fps.as_double();
    RCLCPP_INFO(this->get_logger(), "默认 FPS: %.2f", Fps);

    SubFps = this->create_subscription<std_msgs::msg::Float32>(
      TOPIC_FPS, 10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        Fps = msg->data;
        RCLCPP_INFO(this->get_logger(), "更新 FPS: %.2f", Fps);
      });

    rclcpp::QoS QosProfile(QOS_DEPTH);
    QosProfile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    SubCenter = create_subscription<geometry_msgs::msg::PointStamped>(
      TOPIC_CENTER_PX, QosProfile,
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg){
        LastCx = msg->point.x;
        LastCy = msg->point.y;
        LastW = msg->point.z;
        // 时间戳不再需要，详见publisher_node.py那边的注释
        // LastStamp = msg->header.stamp;
        HaveCenter = true;
        HaveWidth = true;
        PrintIfReady(msg);
      });
    KfPub = this->create_publisher<geometry_msgs::msg::PointStamped>(TOPIC_KF_POS, 100);
  }

private:
  // 处理接收到的消息，当中心点和宽度都收到时进行PnP解算和卡尔曼滤波
  // 参数msg接收到的PointStamped消息
  // PrintIfReady这个名字是因为开始测试的时候主要是想打印出来看看结果对不对，后来逻辑扩展了也懒得改名字了：（
  void PrintIfReady(const geometry_msgs::msg::PointStamped::SharedPtr& msg) {
    std::stringstream Ss(msg->header.frame_id);
    int ObjId, FrameNum;
    char Delim;
    Ss >> ObjId >> Delim >> FrameNum;
    RCLCPP_INFO(this->get_logger(), "收到篮球id: %d, frame: %d", ObjId, FrameNum);

    if (HaveCenter && HaveWidth) {
      // 处理缺失帧的插值
      //publisher那边发送的frame_id是-1就代表空帧，空帧我这里套了两个循环，
      //最里面的循环就是正常的卡尔曼滤波预测一系列点插值的过程，和下面的一样
      //外层循环次数基于missingFrame也就是缺失的帧数，决定了补多少次
      //missingFrame就是去除空帧后相邻两帧，他们的帧索引之差减一
      int lastFrame = LastFrameMap.count(ObjId) ? LastFrameMap[ObjId] : (FrameNum - 1);
      if (FrameNum > lastFrame + 1) {
        for (int missingFrame = lastFrame + 1; missingFrame < FrameNum; ++missingFrame) {
          for (int i = 0; i < KM_RATE; ++i) {
            kalmanManager.Predict(ObjId, KM_FIX / Fps);
            Eigen::Vector3d kfPos = kalmanManager.GetPosition(ObjId);
            geometry_msgs::msg::PointStamped kfMsg;
            kfMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(missingFrame) + "_" + std::to_string(i + 1);
            kfMsg.point.x = kfPos[0];
            kfMsg.point.y = kfPos[1];
            kfMsg.point.z = kfPos[2];
            KfPub->publish(kfMsg);
          }
        }
      }
      LastFrameMap[ObjId] = FrameNum;

      Eigen::Vector3d Obs;
      if (LastW > 0) {
        std::vector<cv::Point2f> ImagePoints = {
            {static_cast<float>(LastCx), static_cast<float>(LastCy)},
            {static_cast<float>(LastCx), static_cast<float>(LastCy - LastW / 2)},
            {static_cast<float>(LastCx), static_cast<float>(LastCy + LastW / 2)},
            {static_cast<float>(LastCx - LastW / 2), static_cast<float>(LastCy)},
            {static_cast<float>(LastCx + LastW / 2), static_cast<float>(LastCy)}
        };
        
        // 进行PnP解算
        if (PnPSolver_->Solve(ImagePoints, Obs)) {
          RCLCPP_INFO(this->get_logger(), "PnP 解算: (%.3f, %.3f, %.3f) id=%d", Obs[0], Obs[1], Obs[2], ObjId);

          // 卡尔曼滤波逻辑
          kalmanManager.Update(ObjId, Obs, 1.0 / (Fps * KM_RATE));
          kalmanManager.Predict(ObjId, KM_FIX / Fps);

          // 发布原始点
          geometry_msgs::msg::PointStamped RawMsg;
          RawMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(FrameNum) + "_raw";
          RawMsg.point.x = Obs[0];
          RawMsg.point.y = Obs[1];
          RawMsg.point.z = Obs[2];
          KfPub->publish(RawMsg);
        } else {
          RCLCPP_WARN(this->get_logger(), "solvePnP 没成功");
        }
      }

      // 用卡尔曼滤波器的predict方法进行插值，并发布KF点
      for (int i = 0; i < KM_RATE; i++) {
        kalmanManager.Predict(ObjId, KM_FIX / Fps);
        Eigen::Vector3d KfPos = kalmanManager.GetPosition(ObjId);
        geometry_msgs::msg::PointStamped KfMsg;
        KfMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(FrameNum) + "_" + std::to_string(i + 1);
        KfMsg.point.x = KfPos[0];
        KfMsg.point.y = KfPos[1];
        KfMsg.point.z = KfPos[2];
        KfPub->publish(KfMsg);
      }
      HaveCenter = HaveWidth = false;
    }
  }

  // ROS2 订阅者和发布者
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr SubCenter;  // 订阅篮球中心点坐标
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr SubFps;  // 订阅FPS信息
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr KfPub;  // 发布卡尔曼滤波后的位置
  
  // 时间戳和状态变量（时间戳已移除，同步依赖帧索引）
  double Fps;  // 当前视频的FPS
  double LastCx, LastCy, LastW;  // 最后接收到的中心点坐标和宽度
  bool HaveCenter, HaveWidth;  // 标志位，表示是否收到中心点和宽度数据
  
  // 相机参数和解算器
  cv::Mat CameraMatrix, DistCoeffs;  // 相机内参矩阵和畸变系数
  std::unique_ptr<PnPSolver> PnPSolver_;  // PnP解算器
  KalmanManager kalmanManager;  // 卡尔曼滤波管理器
  std::unordered_map<int, int> LastFrameMap;  // 各目标最后处理的帧号
};

// 主函数：初始化ROS2，创建节点并开始spin
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto Node = std::make_shared<BallCoordSub>();
  rclcpp::spin(Node);
  rclcpp::shutdown();
  return 0;
}
