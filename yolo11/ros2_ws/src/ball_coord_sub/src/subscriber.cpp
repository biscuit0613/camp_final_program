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
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_center_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_fps_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr kf_pub_;
  rclcpp::Time last_stamp_;
  std::unordered_map<int, KF> kf_map_; // 整了一个字典，key是目标的id,值是对应的卡尔曼滤波实例，这样就可以不同目标不同滤波。
  std::unordered_map<int, int> last_frame_num_map_; // 各目标（键：球的id）最后发布的帧号(值)
  std::unordered_map<int, double> last_time_map_;    // 各目标最后时间（秒）
  double fps_; // 从订阅获取

  void printIfReady(const geometry_msgs::msg::PointStamped::SharedPtr& msg) {
    // 解析frame_id，获取obj_id和frame_num
    std::stringstream ss(msg->header.frame_id);//publisher那边发送的frame_id是frame_idx,是帧索引（累计帧数）
    int obj_id, frame_num;
    char delim;
    ss >> obj_id >> delim >> frame_num;
    //在publisher那边发送的球id和帧之间用_分隔，delim读取分隔符字符并跳过分隔符。
    RCLCPP_INFO(this->get_logger(), "收到篮球id: %d, frame: %d", obj_id, frame_num);

    // if (obj_id == -1) {
    //   // 空帧：对所有活跃目标在该帧时刻进行一次预测并发布
    //   double time = frame_num / fps_;
    //   for (auto& pair : kf_map_) {
    //     int id = pair.first;
    //     pair.second.predict(time);
    //     Eigen::Vector3d pred = pair.second.getPosition();
    //     geometry_msgs::msg::PointStamped kf_msg;
    //     kf_msg.header.stamp = msg->header.stamp;
    //     kf_msg.header.frame_id = std::to_string(id) + "_" + std::to_string(frame_num);
    //     kf_msg.point.x = pred[0];
    //     kf_msg.point.y = pred[1];
    //     kf_msg.point.z = pred[2];
    //     kf_pub_->publish(kf_msg);
    //     last_frame_num_map_[id] = frame_num; // 推进各自的最后帧号
    //     last_time_map_[id] = time;
    //     RCLCPP_INFO(this->get_logger(), "空帧预测: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)", id, frame_num, pred[0], pred[1], pred[2]);
    //   }
    //   return;
    // }

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
        RCLCPP_INFO(this->get_logger(), "PnP 解算后的相机坐标系下的球中心点(3d): (%.3f, %.3f, %.3f) [m] id=%d", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), obj_id);
        Eigen::Vector3d obs(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        double time_cur = frame_num / fps_;  // 使用帧号时间同步

        // 插值缺失帧（基于该目标上一次的帧号）
        int last_frame_id = last_frame_num_map_.count(obj_id) ? last_frame_num_map_[obj_id] : (frame_num - 1);
        int missing = frame_num - last_frame_id-1;
        int corrected_missing = 0;// 记录实际插值的帧数
        int KMrate = 15;//对于识别出球的帧之间，补帧的倍率（这里3倍补帧）
        if (missing >= 1 ) {// 说明中间有帧没识别出球，进行插值，需要嵌套两层循环
          for (size_t i = 0; i < missing; i++)// 外层循环，针对缺失的帧数
          {
            Eigen::Vector3d current_obs = obs;  // 初始化观测为 PnP解算的结果
            kf_map_[obj_id].update(current_obs, 1.0 / (fps_ * KMrate));  // 初始化KF
            // 内层循环，针对每个缺失帧进行 KMrate 次插值
            for (int j = 1; j <= KMrate; ++j) {  
              int frame_interp = last_frame_id + j;  // 计算插值帧号
              double t_interp = frame_interp / fps_;  // 计算插值时间
              kf_map_[obj_id].predict(1.0 / (fps_ * KMrate));  // 卡尔曼滤波器预测
              Eigen::Vector3d pred = kf_map_[obj_id].getPosition();  // 获取预测位置
              current_obs = pred;
              kf_map_[obj_id].update(current_obs, 1.0 / (fps_ * KMrate));  // 用当前估计更新滤波器
              geometry_msgs::msg::PointStamped kf_msg;  // 创建消息
              kf_msg.header.stamp = msg->header.stamp;  // 设置时间戳
              kf_msg.header.frame_id = std::to_string(obj_id) + "_" + std::to_string(frame_interp);  // 设置帧ID
              kf_msg.point.x = pred[0];  
              kf_msg.point.y = pred[1];  
              kf_msg.point.z = pred[2];  
              kf_pub_->publish(kf_msg);  
              last_frame_num_map_[obj_id] = frame_interp;  
              last_time_map_[obj_id] = t_interp + 1.0 / fps_/KMrate;  // 更新最后时间
              corrected_missing++;  // 增加插值计数，目前不知道有啥用，先留着吧
              RCLCPP_INFO(this->get_logger(), "插值点: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)",  obj_id, frame_interp, static_cast<float>(pred[0]), static_cast<float>(pred[1]), static_cast<float>(pred[2]));  // 日志输出
            }
            missing--; 
          }
        } 
        if (corrected_missing > 0) {
          RCLCPP_INFO(this->get_logger(), "插了%d帧", corrected_missing); 
        }
        kf_map_[obj_id].update(obs, 1.0 / (fps_ * KMrate));  // 初始化KF
        for (int i = 0; i < KMrate; i++)
        {
          kf_map_[obj_id].predict(1.0 / (fps_ * KMrate));  // 预测
          Eigen::Vector3d kf_pos = kf_map_[obj_id].getPosition();  // 获取预测估计位置
          kf_map_[obj_id].update(kf_pos, 1.0 / (fps_ * KMrate));  // 更新
          RCLCPP_INFO(this->get_logger(), "Raw point: (%.3f, %.3f, %.3f), KF point: (%.3f, %.3f, %.3f)", 
            kf_pos[0], kf_pos[1], kf_pos[2],
            kf_map_[obj_id].getPosition()[0], kf_map_[obj_id].getPosition()[1], kf_map_[obj_id].getPosition()[2]);
          // 发布当前迭代轮数的卡尔曼滤波结果
          geometry_msgs::msg::PointStamped kf_msg;
          kf_msg.header.stamp = last_stamp_;
          kf_msg.header.frame_id = std::to_string(obj_id) + "_" + std::to_string(frame_num);
          kf_msg.point.x = kf_map_[obj_id].getPosition()[0];
          kf_msg.point.y = kf_map_[obj_id].getPosition()[1];
          kf_msg.point.z = kf_map_[obj_id].getPosition()[2];
          kf_pub_->publish(kf_msg);
          // last_frame_num_map_[obj_id] = frame_num;
          // last_time_map_[obj_id] = current_time;
          // RCLCPP_INFO(this->get_logger(), "KF 迭代第%d次发布: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)", i + 1, obj_id, frame_num, kf_msg.point.x, kf_msg.point.y, kf_msg.point.z);
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "solvePnP 没成功");
      }
      have_center_ = have_width_ = false;
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
