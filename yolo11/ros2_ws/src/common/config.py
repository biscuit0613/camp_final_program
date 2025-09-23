# common/config.py
# 共享配置和常量定义文件，包含整个系统的配置参数

# 视频和模型配置参数
DEFAULT_VIDEO_PATH = 'test1/rgb.mp4'  # YOLO识别的视频路径，在publisher和可视化模块中都会用到
DEFAULT_MODEL_PATH = 'v1.pt'  # YOLO模型文件路径
DEFAULT_CONF_THRESH = 0.65  # YOLO检测的置信度阈值，低于此值的不发送
DEFAULT_CALIB_PATH = 'ball_coord_sub/src/camera_calibration.json'  # 相机标定参数文件路径
DEFAULT_FPS = 28.0  # 默认帧率，因为每个视频帧率不同，需要单独获取，没啥用

BALL_DIAMETER = 0.246  # 篮球直径，单位：米，用于PnP解算

# 卡尔曼滤波相关参数
KM_RATE = 6  # 对于识别出球的帧之间，补帧的倍率（经测试发现数值越大，补帧点迹越断裂）
KM_FIX = 1.0  # 插值效果调节参数（经测试发现数值越大，插值点方向越凌乱，点迹越离散）

# ROS2话题名称定义
TOPIC_CENTER_PX = '/ball/center_px'  # 球心坐标和宽度的发布话题
TOPIC_WIDTH_PX = '/ball/width_px'  # 球宽度的发布话题
TOPIC_FPS = '/ball/fps'  # 帧率信息的发布话题
TOPIC_KF_POS = '/ball/kf_pos'  # 卡尔曼滤波后的球坐标发布话题

# QoS（服务质量）设置
QOS_DEPTH = 100  # QoS队列深度，影响消息缓冲大小
QOS_RELIABILITY = 'RELIABLE'  # QoS可靠性策略，确保消息可靠传递
