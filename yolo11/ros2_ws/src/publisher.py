import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, PointStamped
from std_msgs.msg import Float32
from ultralytics import YOLO
import cv2
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class YoloBallPublisher(Node):
    def __init__(self):
        super().__init__('yolo_ball_publisher')

        # 声明 ROS 参数（可在 launch 或命令行中覆盖）
        self.declare_parameter('video_path', 'test4/rgb.mp4')
        self.declare_parameter('model_path', 'v1.pt')
        self.declare_parameter('ball_class_id', 0)
        self.declare_parameter('conf_thresh', 0.7)

        # 创建 ROS 发布器：中心点和宽度
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        #qos就是质量服务配置文件，用于设置消息的传输质量，
        # pub端的qos得和sub端的qos一致
        # qos里面第一个参数是深度（就是能存多少），第二个参数是可靠性策略（这里reliable,确保消息送达）
        self.pub_center = self.create_publisher(PointStamped, '/ball/center_px', qos)
        self.pub_width  = self.create_publisher(Float32, '/ball/width_px', qos)

        # 获取参数值
        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        video_path = os.path.abspath(video_path)
        print(f"[DEBUG] Try to open video: {video_path}")
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.ball_cls = self.get_parameter('ball_class_id').get_parameter_value().integer_value
        self.conf = self.get_parameter('conf_thresh').get_parameter_value().double_value

        # 加载 YOLO 模型
        self.model = YOLO(model_path)

        # 打开视频文件
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video: {video_path}")
            self.destroy_node()
            return

        # 设置定时器周期（根据视频帧率）
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        period = 1.0 / fps if fps and fps > 0 else 0.03
        self.timer = self.create_timer(period, self.loop)

        # 初始化轨迹字典：每个 obj_id 对应一个点序列
        self.trajectories = {}
        # 统计发布次数
        self.publish_count = 0

        self.out_path = 'yolo_detection_output.mp4'
        self.out_writer = None

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().info('Video ended.')
            self.get_logger().info(f'Total published ball coordinates: {self.publish_count}')
            if self.out_writer is not None:
                self.out_writer.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            return

        # YOLO 跟踪推理
        results = self.model.track(frame, persist=True, conf=self.conf)

        if len(results) and results[0].boxes is not None:
            boxes = results[0].boxes
            xywh = boxes.xywh.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            # 遍历所有检测框
            for i, box in enumerate(xywh):
                if int(clss[i]) != self.ball_cls:
                    continue

                x_center, y_center, w, h = map(float, box)
                # 直接用xywh的中心点和宽度
                cx = x_center
                cy = y_center
                width = max(1.0, w)

                obj_id = i  #  YOLOv8 + tracker可以改为 boxes.id[i]

                if obj_id not in self.trajectories:
                    self.trajectories[obj_id] = []
                self.trajectories[obj_id].append((cx, cy))

                # 用PointStamped带时间戳发布中心点，z存宽度
                msg_center = PointStamped()
                msg_center.header.stamp = self.get_clock().now().to_msg()
                msg_center.point.x = cx
                msg_center.point.y = cy
                msg_center.point.z = width

                self.pub_center.publish(msg_center)
                self.publish_count += 1

                # 可视化检测框和中心点
                x1 = int(cx - width / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + width / 2)
                y2 = int(cy + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)

                # 可视化轨迹线
                pts = self.trajectories[obj_id]
                for j in range(1, len(pts)):
                    pt1 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                    pt2 = (int(pts[j][0]), int(pts[j][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                break  # 只处理一个目标，保持简单

        # 初始化视频保存器
        if self.out_writer is None:
            height, width = frame.shape[:2]
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.out_writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 保存视频帧
        self.out_writer.write(frame)

        # 显示图像窗口
        cv2.imshow("YOLO Tracking", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloBallPublisher()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroy_all_windows()

if __name__ == "__main__":
    main()