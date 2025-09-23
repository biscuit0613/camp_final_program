#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../kmfilter/KalmanFilterJerk.hpp"
#include <Eigen/Dense>
#include <unordered_map>

//已弃用

//这个是没有拆分的最开始的一大坨subscriber,pnp解算和卡尔曼滤波都写在里面
//后来把pnp解算和卡尔曼滤波都拆分出去单独成了类


class BallCoordSub : public rclcpp::Node {
public:
  BallCoordSub() : Node("ball_coord_sub") {
    // 从camera_calibration.json读取相机内参，用于solvePnP
    std::ifstream ifs("../src/camera_calibration.json");
    nlohmann::json j;
    ifs >> j;
    CameraMatrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        CameraMatrix_.at<double>(i, k) = j["camera_matrix"][i][k];
    DistCoeffs_ = cv::Mat(1, j["distortion_coefficients"].size(), CV_64F);
    for (size_t i = 0; i < j["distortion_coefficients"].size(); ++i)
      DistCoeffs_.at<double>(0, i) = j["distortion_coefficients"][i];

    //如果接收不到fps参数，就用默认的fps
    this->declare_parameter("fps", 28.0);
    auto param_fps = this->get_parameter("fps");
    Fps_ = param_fps.as_double();
    RCLCPP_INFO(this->get_logger(), "默认 FPS: %.2f", Fps_);

    // 订阅 FPS,后续用在卡尔曼补帧中传给predict方法的dt参数，时间步长这一块
    SubFps_ = this->create_subscription<std_msgs::msg::Float32>(
      "/ball/fps",
      10,
      [this](const std_msgs::msg::Float32::SharedPtr msg) {
        Fps_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "更新 FPS: %.2f", Fps_);
      });


    //跟pub那边一样，也建立一个qos
    rclcpp::QoS QosProfile(100);
    //qos_profile（）里面的数是队列深度
    QosProfile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    //pub里面用了reliable这里也要用

    //这下面就是创建订阅，接收pub轨迹点信息的代码了
    SubCenter = create_subscription<geometry_msgs::msg::PointStamped>(
      "/ball/center_px", //topic的名称
      QosProfile,//qos,就是在pub里面定义的那个
      // 下面这一大坨是回调函数，就是收到消息了该怎么办。用了lambda表达式，比较简洁
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg){
        LastCx = msg->point.x;
        LastCy_ = msg->point.y;
        LastW_ = msg->point.z; // 直接用z存宽度
        LastStamp_ = msg->header.stamp;
        HaveCenter = true;
        HaveWidth = true;
        PrintIfReady(msg);
        //显然这里面没有获取id，因为id和上面获取的信息在 ROS 消息结构中的位置和用途不一样
        //坐标是放在point里面的，是浮点数，可以直接拿来用
      });
    //下面加一个pub,发卡尔曼滤波信息给visualize.py,用python实现可视化
    KfPub = this->create_publisher<geometry_msgs::msg::PointStamped>("/ball/kf_pos", 100);
  }

  ~BallCoordSub() {
   
  }


private:
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr SubCenter_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr SubFps_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr KfPub_;
  rclcpp::Time LastStamp_; // 最后接收到的消息时间戳
  std::unordered_map<int, KalmanFilterJerk> KfMap_; // 整了一个字典，key是目标的id,值是对应的卡尔曼滤波实例，这样就可以不同目标不同滤波。
  std::unordered_map<int, int> LastFrameNumMap_; // 各目标（键：球的id）最后发布的帧号(值)
  std::unordered_map<int, double> LastTimeMap_;    // 各目标最后时间（秒）
  double Fps_; // 从订阅获取

  void PrintIfReady(const geometry_msgs::msg::PointStamped::SharedPtr& msg) {
    // 解析消息头的字符串，获取obj_id和frame_num
    std::stringstream Ss(msg->header.frame_id);//publisher那边发送的frame_id是frame_idx,是帧索引（累计帧数）
    int ObjId, FrameNum;
    char Delim;
    Ss >> ObjId >> Delim >> FrameNum;
    //在publisher那边发送的球id和帧之间用_分隔，delim占用分隔符字符并跳过分隔符。
    RCLCPP_INFO(this->get_logger(), "收到篮球id: %d, frame: %d", ObjId, FrameNum);

    //失败的空帧处理逻辑
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

    if (HaveCenter && HaveWidth) {
      // solvePnP方法：用球心和球面上下左右五点做PnP解算，直接带参数就行，不用套公式了
      float cx = LastCx;
      float cy = LastCy;
      float w = LastW;
      float h = LastW; 
      float D = 0.246; // 球实际直径，单位：米
      float r = D / 2.0;
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
      bool pnp_ok = cv::solvePnP(objectPoints, imagePoints, CameraMatrix, DistCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
      if (pnp_ok) {
        RCLCPP_INFO(this->get_logger(), "PnP 解算后的相机坐标系下的球中心点(3d): (%.3f, %.3f, %.3f) [m] id=%d", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), ObjId);
        Eigen::Vector3d obs(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        //obs是观测值，也就是PnP解算出来的坐标
        double TimeCur = FrameNum / Fps_;  // 使用帧号时间同步

        // 插值缺失帧（基于该目标上一次的帧号）
        int LastFrameId = LastFrameNumMap_.count(ObjId) ? LastFrameNumMap_[ObjId] : (FrameNum - 1);
        int Missing = FrameNum - LastFrameId-1;
        int CorrectedMissing = 0;// 记录实际插值的帧数
        int KmRate = 6;//对于识别出球的帧之间，补帧的倍率(数值越大，补帧点迹越断裂)
        float KmFix = 1.0;//这是一个神奇的参数，可以调节插值的效果（应该是步长的问题），先留着(数值越大，插值点越方向越凌乱，点迹越离散)
        if (Missing >= 1 ) {// 说明中间有帧没识别出球，进行插值，需要嵌套两层循环
          for (size_t i = 0; i < Missing; i++)// 外层循环，针对缺失的帧数
          {
            Eigen::Vector3d CurrentObs = obs;  // 初始化观测为 PnP解算的结果
            KfMap_[ObjId].update(CurrentObs, 1.0 / (Fps_ * KmRate));  // 初始化KF
            // 内层循环，针对每个缺失帧进行 Kmrate 次插值
            for (int j = 1; j <= KmRate; ++j) {  
              int FrameInterp = LastFrameId + j;  // 计算插值帧号
              double TInterp = FrameInterp / (Fps_ * KmRate);  // 计算插值时间
              KfMap_[ObjId].predict(KmFix / (Fps_ * KmRate));  // 卡尔曼滤波器预测
              Eigen::Vector3d Pred = KfMap_[ObjId].getPosition();  // 获取预测位置
              CurrentObs = Pred;
              // kf_map_[obj_id].update(current_obs, KMfix / (fps_ * KMrate));  // 用当前估计更新滤波器
              geometry_msgs::msg::PointStamped KfMsg;  // 创建消息
              KfMsg.header.stamp = msg->header.stamp;  // 设置时间戳
              KfMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(FrameInterp);  // 设置帧ID
              KfMsg.point.x = Pred[0];  
              KfMsg.point.y = Pred[1];  
              KfMsg.point.z = Pred[2];  
              KfPub->publish(KfMsg);  
              LastFrameNumMap_[ObjId] = FrameInterp;  
              LastTimeMap_[ObjId] = TInterp + 1.0 / (Fps_ * KmRate);  // 更新最后时间
              CorrectedMissing++;  // 增加插值计数，目前不知道有啥用，先留着吧
              RCLCPP_INFO(this->get_logger(), "插值点: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)",  ObjId, FrameInterp, static_cast<float>(Pred[0]), static_cast<float>(Pred[1]), static_cast<float>(Pred[2]));  // 日志输出
            }
            Missing--; 
          }
        } 
        if (CorrectedMissing > 0) {
          RCLCPP_INFO(this->get_logger(), "插了%d帧", CorrectedMissing); 
        }
        KfMap_[ObjId].predict(KmFix / Fps_ );  // 预测
        KfMap_[ObjId].update(obs, 1.0 / (Fps_ * KmRate));  // 初始化KF
        
        // 发布原始观测点
        geometry_msgs::msg::PointStamped RawMsg;
        RawMsg.header.stamp = LastStamp_;
        RawMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(FrameNum) + "_raw";
        RawMsg.point.x = obs[0];
        RawMsg.point.y = obs[1];
        RawMsg.point.z = obs[2];
        KfPub->publish(RawMsg); 
        
        for (int i = 0; i < KmRate; i++)
        {
          KfMap_[ObjId].predict(KmFix / (Fps_ * KmRate));  // 预测
          Eigen::Vector3d KfPos = KfMap_[ObjId].getPosition();  // 获取预测估计位置
          // kf_map_[obj_id].update(kf_pos, (1.0*KMfix) / (fps_ * KMrate));  // 更新
          RCLCPP_INFO(this->get_logger(), "Raw point: (%.3f, %.3f, %.3f), KF point: (%.3f, %.3f, %.3f)", 
            obs[0], obs[1], obs[2],  // 使用 obs 作为 Raw point
            KfPos[0], KfPos[1], KfPos[2]);
          // 发布当前迭代轮数的卡尔曼滤波结果
          geometry_msgs::msg::PointStamped KfMsg;
          KfMsg.header.stamp = LastStamp_ += rclcpp::Duration::from_seconds( KmFix / (Fps_ * KmRate));
          KfMsg.header.frame_id = std::to_string(ObjId) + "_" + std::to_string(FrameNum);
          KfMsg.point.x = KfPos[0];
          KfMsg.point.y = KfPos[1];
          KfMsg.point.z = KfPos[2];
          KfPub->publish(KfMsg);
          // last_frame_num_map_[obj_id] = frame_num;
          // last_time_map_[obj_id] = current_time;
          // RCLCPP_INFO(this->get_logger(), "KF 迭代第%d次发布: id=%d, frame=%d, pos=(%.3f, %.3f, %.3f)", i + 1, obj_id, frame_num, kf_msg.point.x, kf_msg.point.y, kf_msg.point.z);
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "solvePnP 没成功");
      }
      HaveCenter = HaveWidth = false;
    }
  }
  double LastCx{0}, LastCy{0}, LastW{0};
  bool HaveCenter{false}, HaveWidth{false};

  cv::Mat CameraMatrix, DistCoeffs;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto Node = std::make_shared<BallCoordSub>();
  rclcpp::spin(Node);
  rclcpp::shutdown();
  return 0;
}
