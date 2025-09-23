import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, PointStamped
from std_msgs.msg import Float32
from ultralytics import YOLO
import cv2
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

#已弃用

#最开始的publisher,yolo识别和发布点迹写在一起。后来把它拆分成了publisher_node.py和yolo_detector.py两个文件

class YoloBallPublisher(Node):
    def __init__(self):
        super().__init__('yolo_ball_publisher')

        # 声明 ROS 参数
        self.declare_parameter('video_path', 'test2/rgb.mp4')
        self.declare_parameter('model_path', 'v1.pt')
        self.declare_parameter('conf_thresh', 0.65)

        # 创建 ROS 发布器：中心点和宽度
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        #qos就是质量服务配置文件，用于设置消息的传输质量，
        # pub端的qos得和sub端的qos一致
        # qos里面第一个参数是深度（就是能存多少），第二个参数是可靠性策略（这里reliable,确保消息送达）
        self.PubCenter = self.create_publisher(PointStamped, '/ball/center_px', qos)
        self.PubWidth  = self.create_publisher(Float32, '/ball/width_px', qos)
        self.PubFps = self.create_publisher(Float32, '/ball/fps', qos)

        # 获取参数值
        VideoPath = self.get_parameter('video_path').get_parameter_value().string_value
        VideoPath = os.path.abspath(VideoPath)
        print(f" 正在打开视频: {VideoPath}")
        ModelPath = self.get_parameter('model_path').get_parameter_value().string_value
        self.Conf = self.get_parameter('conf_thresh').get_parameter_value().double_value

        # 加载 YOLO 模型
        self.Model = YOLO(ModelPath)

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

        # 初始化轨迹字典：每个 obj_id 对应一个点
        self.Trajectories = {}
        # 统计发布次数：包括发出的球的坐标数还有检测的帧数
        self.PublishCount = 0#发布的坐标数
        self.FrameIdx = 0#帧索引，也就是累计帧数。无论是否检测到球，是否发出坐标都会+1

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
            return False  # 视频结束，返回 False

        self.FrameIdx += 1#帧索引

        # YOLO 跟踪推理，采用track方法，（detect只有框没有id）
        #这里挺纠结的，detect方法只能检测框，没有id，track方法能检测框还能给每个框分配一个id，但是检测率不如detect
        #尤其是在test2里面，篮球拍的速度太快要么换id了要么直接不检测了
        Results = self.Model.track(Frame, persist=True, conf=self.Conf)
        #yolo的track方法返回一个结果列表，里面每个元素对应一帧图像，包含检测到的对象信息
        if len(Results) and Results[0].boxes is not None:#boxes就是检测到的框
            Boxes = Results[0].boxes
            # boxes的属性：
            # boxes.xywh 中心点xy和宽高wh
            # boxes.conf 置信度
            # boxes.id 跟踪id
            # boxes.cls 类别
            Xywh = Boxes.xywh.cpu().numpy()#把tensor转成numpy数组，2*2,第一行xy第二行wh
            # confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else [1.0]*len(xywh)
            if hasattr(Boxes, 'id') and Boxes.id is not None:
                Ids = Boxes.id.cpu().numpy()
                print('当前帧的id字典是',Ids)
                # 这里获取的球id只是为了打印出来看看的，真正发出去的id是下面限制在0和1之间
            else:
                Ids = list(range(len(Xywh)))
            # 遍历所有检测框，根据检测框id不同发信息。
            for i, box in enumerate(Xywh):
                XCenter, YCenter, W, H = map(float, box)
                Cx = XCenter
                Cy = YCenter
                Width = max(1.0, W)
                # obj_id才是传去cpp的id
                ObjId = min(i, 1)  # 限制 ID 为 0 或 1（最多两个球），不然滤波器那边会乱。
                print(f'检测框{i}的ID是{ObjId}')
                if ObjId not in self.Trajectories:
                    self.Trajectories[ObjId] = []
                self.Trajectories[ObjId].append((Cx, Cy))
                # 用PointStamped带时间戳发布中心点
                #PointStamped是geometry_msgs包里定义的一个消息类型，包含一个三维点和时间戳信息，由header和point两个部分组成
                #header包含时间戳和坐标系信息，point包含三维坐标
                MsgCenter = PointStamped()#创建一个PointStamped消息对象
                MsgCenter.header.stamp = self.get_clock().now().to_msg()#时间戳
                MsgCenter.header.frame_id = f"{ObjId}_{self.FrameIdx}"  # 传递球的id和帧号
                #消息头长这样：obj_id_frame_id，sub那边解析一下字符串就行
                #也许可以改为传数组过去，但是懒得重构了
                MsgCenter.point.x = Cx
                MsgCenter.point.y = Cy
                MsgCenter.point.z = Width
                self.PubCenter.publish(MsgCenter)
                self.PublishCount += 1#发布坐标的计数器
                # 把框和中心点可视化（由 visualize.py 处理）
                # x1 = int(cx - width / 2)
                # y1 = int(cy - h / 2)
                # x2 = int(cx + width / 2)
                # y2 = int(cy + h / 2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
                # 显示框id和置信度
                # label = f'id:{obj_id},conf:{confs[i]:.2f}'
                # cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                # pts = self.trajectories[obj_id] #画轨迹
                # for j in range(1, len(pts)):
                #     pt1 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                #     pt2 = (int(pts[j][0]), int(pts[j][1]))
                #     cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        else:
            # 空帧，发布空消息
            # 这个对应的是原来失败的空帧处理逻辑，除了时间戳和帧ID以外没有其他用途
            MsgCenter = PointStamped()
            MsgCenter.header.stamp = self.get_clock().now().to_msg()
            MsgCenter.header.frame_id = f"-1_{self.FrameIdx}"
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

        # 保存视频帧
        self.OutWriter.write(Frame)
        # 这里本来还有yolo识别的显示窗口，给删了，可视化交给 visualize.py 处理
        return True  # 继续循环

def Main():
    rclpy.init()
    Node = YoloBallPublisher()
    while rclpy.ok():
        if not Node.Loop():
            break  # 视频结束，退出循环
        rclpy.spin_once(Node, timeout_sec=0.01)
        if cv2.waitKey(int(1000 / Node.Fps)) & 0xFF == 27:
            break
    Node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()