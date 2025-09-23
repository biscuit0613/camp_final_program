#ifndef CONFIG_H
#define CONFIG_H

// 视频和模型配置
#define DEFAULT_VIDEO_PATH "test1/rgb.mp4"//yolo识别的视频路径，在publisher和可视化里面都要用到
#define DEFAULT_MODEL_PATH "v1.pt"//yolo模型路径
#define DEFAULT_CONF_THRESH 0.65//yolo检测的置信度阈值，低于这个数就不发了
#define DEFAULT_CALIB_PATH "../src/camera_calibration.json"//相机标定文件路径
#define DEFAULT_FPS 28.0//默认fps，因为每个视频帧率都不一样所以要单独获取，没啥意义。

// 篮球物理参数
#define BALL_D 0.246//篮球直径，单位米

// 卡尔曼滤波参数
#define KM_RATE 6 //对于识别出球的帧之间，补帧的倍率(数值越大，补帧点迹越断裂)
#define KM_FIX 2.0 //这个在分子上，用来调节插值的效果（应该是调时间步长），先留着(数值越大，插值点越方向越凌乱，点迹越离散)

// ROS2话题名称
#define TOPIC_CENTER_PX "/ball/center_px" //球心坐标和宽度的topic
#define TOPIC_WIDTH_PX "/ball/width_px" //球宽度的topic
#define TOPIC_FPS "/ball/fps" //帧率的topic
#define TOPIC_KF_POS "/ball/kf_pos" //卡尔曼滤波后的球坐标topic

// QoS配置
#define QOS_DEPTH 100 //QoS队列深度

#endif
