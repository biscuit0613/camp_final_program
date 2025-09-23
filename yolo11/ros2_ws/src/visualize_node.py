# visualize_node.py
# 从原来visualize.py拆分出来的文件，只负责订阅卡尔曼滤波结果并在视频上显示轨迹
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import cv2
from trajectory_renderer import TrajectoryRenderer
from common.config import *

# 这里详细解释一下用帧索引同步的方法：
"""
    比如说yolo在第十五帧检测到了球,Publisher: frame_id = "0_15" (球ID=0, 帧=15)
    然后数据传给sub,进行插值,插值的帧格式:"{ball_id}_{frame_num}_{sub_frame}"
        Subscriber: frame_id = "0_15_1", "0_15_2", "0_15_3"...
    最后给可视化这一块:把这个格式转化成小数,t_rel = frame_num + sub_frame / 100.0
        Visualizer: 显示所有 frame_idx <= 当前播放帧 的点
"""

class KFVisualizer(Node):
    def __init__(self, VideoPath, CalibPath):
        super().__init__('kf_visualizer')
        self.Subscription = self.create_subscription(
            PointStamped, 
            TOPIC_KF_POS,  # 这个接收的就是subscriber.cpp里面发布的卡尔曼滤波后的结果
            self.ListenerCallback,  # 回调函数，这里处理时间戳同步的问题
            100)
        self.Renderer = TrajectoryRenderer(VideoPath, CalibPath)

    def ListenerCallback(self, msg):
        # 把原来回调函数的逻辑挪出去单独做成一个类TrajectoryRenderer
        # 这里仅仅是把收到的点传给这个类去处理
        self.Renderer.AddPoint(msg)

    def Run(self):
        print('等待卡尔曼滤波的结果传来中')
        while not self.Renderer.PointsDict and rclpy.ok():  # 这里检查点的字典是否为空以及订阅的消息来了没来。
            rclpy.spin_once(self, timeout_sec=0.1)
        print('收到数据开始播放卡尔曼滤波效果视频')

        cap = cv2.VideoCapture(self.Renderer.VideoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = 'kf_visualization_output.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0  # 帧索引
        while cap.isOpened():
            # 多 spin 几次以处理更多消息，减少滞后
            for i in range(KM_RATE):
                rclpy.spin_once(self, timeout_sec=0.001)
            ret, frame = cap.read()
            if not ret:
                break
            # t_frame = frame_idx / fps
            frame = self.Renderer.RenderFrame(frame, frame_idx)
            frame_idx += 1
            out.write(frame)
            cv2.imshow('KF Trajectory on Video', frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f'Visualization video saved to {out_path}')

def main(args=None):
    rclpy.init(args=args)
    video_path = DEFAULT_VIDEO_PATH
    calib_path = DEFAULT_CALIB_PATH
    node = KFVisualizer(video_path, calib_path)
    try:
        node.Run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
