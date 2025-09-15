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

        # 声明 ROS 参数
        self.declare_parameter('video_path', 'test5/rgb.mp4')
        self.declare_parameter('model_path', 'v1.pt')
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
        print(f" 正在打开视频: {video_path}")
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = self.get_parameter('conf_thresh').get_parameter_value().double_value

        # 加载 YOLO 模型
        self.model = YOLO(model_path)

        # 打开视频文件
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.get_logger().error(f"打不开视频: {video_path}")
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
            self.get_logger().info('视频放完了')
            self.get_logger().info(f'总共发布的坐标数(和cpp那边对下帐): {self.publish_count}')
            if self.out_writer is not None:
                self.out_writer.release()
            cv2.destroyAllWindows()
            self.destroy_node()
            return

        # YOLO 跟踪推理，采用track方法，（detect只有框没有id）
        #这里挺纠结的，detect方法只能检测框，没有id，track方法能检测框还能给每个框分配一个id，但是检测率不如detect
        #尤其是在test2里面，篮球拍的速度太快要么换id了要么直接不检测了
        results = self.model.track(frame, persist=True, conf=self.conf)

        if len(results) and results[0].boxes is not None:
            boxes = results[0].boxes
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else [1.0]*len(xywh)
            if hasattr(boxes, 'id') and boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                print('当前帧的id字典是',ids)
            else:
                ids = list(range(len(xywh)))
            # 遍历所有检测框，根据id不同发信息。
            for i, box in enumerate(xywh):
                x_center, y_center, w, h = map(float, box)
                cx = x_center
                cy = y_center
                width = max(1.0, w)
                obj_id = int(ids[i])
                print(f'检测框{i}的ID是{obj_id}')
                if obj_id not in self.trajectories:
                    self.trajectories[obj_id] = []
                self.trajectories[obj_id].append((cx, cy))
                # 用PointStamped带时间戳发布中心点
                #PointStamped是geometry_msgs包里定义的一个消息类型，包含一个三维点和时间戳信息，由header和point两个部分组成
                #header包含时间戳和坐标系信息，point包含三维坐标
                msg_center = PointStamped()
                msg_center.header.stamp = self.get_clock().now().to_msg()
                msg_center.header.frame_id = str(obj_id)  # 传递id
                msg_center.point.x = cx
                msg_center.point.y = cy
                msg_center.point.z = width
                self.pub_center.publish(msg_center)
                self.publish_count += 1
                # 把框和中心点可视化
                x1 = int(cx - width / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + width / 2)
                y2 = int(cy + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
                # 显示框id和置信度
                label = f'id:{obj_id},conf:{confs[i]:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                pts = self.trajectories[obj_id] #画轨迹
                for j in range(1, len(pts)):
                    pt1 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                    pt2 = (int(pts[j][0]), int(pts[j][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # 视频保存
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