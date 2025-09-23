# publisher_node.py
# 从原来的publisher.py拆分出来的文件，只负责yolo识别和发布点迹
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
import cv2
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from yolo_detector import YoloDetector
from common.config import *

class YoloBallPublisher(Node):
    def __init__(self):
        super().__init__('yolo_ball_publisher')

        # 声明 ROS 参数
        self.declare_parameter('video_path', DEFAULT_VIDEO_PATH)
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('conf_thresh', DEFAULT_CONF_THRESH)

        # 创建 ROS 发布器
        qos = QoSProfile(depth=QOS_DEPTH, reliability=QoSReliabilityPolicy.RELIABLE)
        self.PubCenter = self.create_publisher(PointStamped, TOPIC_CENTER_PX, qos)
        self.PubWidth = self.create_publisher(Float32, TOPIC_WIDTH_PX, qos)
        self.PubFps = self.create_publisher(Float32, TOPIC_FPS, qos)

        # 获取参数值
        VideoPath = self.get_parameter('video_path').get_parameter_value().string_value
        VideoPath = os.path.abspath(VideoPath)
        print(f" 正在打开视频: {VideoPath}")
        ModelPath = self.get_parameter('model_path').get_parameter_value().string_value
        self.Conf = self.get_parameter('conf_thresh').get_parameter_value().double_value

        # 初始化YOLO检测器
        self.Detector = YoloDetector(ModelPath)

        # 打开视频文件
        self.Cap = cv2.VideoCapture(VideoPath, cv2.CAP_FFMPEG)
        if not self.Cap.isOpened():
            self.get_logger().error(f"打不开视频: {VideoPath}")
            self.destroy_node()
            return

        # 获取 FPS 并发布
        self.Fps = self.Cap.get(cv2.CAP_PROP_FPS)
        fpsMsg = Float32()
        fpsMsg.data = self.Fps
        self.PubFps.publish(fpsMsg)
        self.get_logger().info(f"视频 FPS: {self.Fps}")

        # 初始化轨迹字典
        self.Trajectories = {}
        self.PublishCount = 0
        self.FrameIdx = 0

        self.OutPath = 'yolo_detection_output.mp4'
        self.OutWriter = None

    def Loop(self):
        Ok, Frame = self.Cap.read()
        if not Ok:
            self.get_logger().info('视频放完了')
            self.get_logger().info(f'总共发布的坐标数: {self.PublishCount}')
            if self.OutWriter is not None:
                self.OutWriter.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            return False

        self.FrameIdx += 1

        # 使用YOLO检测
        Xywh, Ids = self.Detector.Detect(Frame, self.Conf)
        if Xywh is not None:
            for i, box in enumerate(Xywh):
                XCenter, YCenter, W, H = map(float, box)
                Cx = XCenter
                Cy = YCenter
                Width = max(1.0, W)
                #ObjId才是真正传给cpp那边的id
                # 这里简单处理一下，只用0和1两个id，避免id跳变
                # 实际上yolo的id并不稳定，尤其是篮球这种高速运动的物体，要是真用yolo的id卡尔曼滤波器会乱，而且轨迹可视化会花
                ObjId = min(i, 1)
                print(f'检测框{i}的ID是{ObjId}')
                if ObjId not in self.Trajectories:
                    self.Trajectories[ObjId] = []
                self.Trajectories[ObjId].append((Cx, Cy))

                MsgCenter = PointStamped()
                # stamp里面的时间戳之前是用来同步点迹的，但发现只靠帧索引（frameidx）更直接,就不用绝对时间戳了
                # MsgCenter.header.stamp = self.get_clock().now().to_msg()
                MsgCenter.header.frame_id = f"{ObjId}_{self.FrameIdx}"
                MsgCenter.point.x = Cx
                MsgCenter.point.y = Cy
                MsgCenter.point.z = Width
                self.PubCenter.publish(MsgCenter)
                self.PublishCount += 1
        else:
            # 空帧
            MsgCenter = PointStamped()
            MsgCenter.header.stamp = self.get_clock().now().to_msg()
            MsgCenter.header.frame_id = f"0_{self.FrameIdx}"
            MsgCenter.point.x = 0.0
            MsgCenter.point.y = 0.0
            MsgCenter.point.z = 0.0
            self.PubCenter.publish(MsgCenter)
            self.PublishCount += 1

        # 视频保存
        if self.OutWriter is None:
            Height, Width = Frame.shape[:2]
            Fps = self.Cap.get(cv2.CAP_PROP_FPS)
            self.OutWriter = cv2.VideoWriter(self.OutPath, cv2.VideoWriter_fourcc(*'mp4v'), Fps, (Width, Height))

        self.OutWriter.write(Frame)
        return True

def Main():
    rclpy.init()
    Node = YoloBallPublisher()
    while rclpy.ok():
        if not Node.Loop():
            break
        rclpy.spin_once(Node, timeout_sec=0.01)
        if cv2.waitKey(int(1000 / Node.Fps)) & 0xFF == 27:
            break
    Node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
