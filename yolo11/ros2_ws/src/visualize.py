import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
import json
import bisect

class KFVisualizer(Node):
    def __init__(self, video_path, calib_path):
        super().__init__('kf_visualizer')
        self.t0 = 0.0
        self.subscription = self.create_subscription(
            PointStamped, 
            '/ball/kf_pos',#这个接收的就是subscriber.cpp里面发布的卡尔曼滤波后的结果
            self.listener_callback, #回调函数，这里处理时间戳同步的问题
            100)
        # 读取相机内参，后面用于把三维滤波结果投到二维视频上
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        self.camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(calib['distortion_coefficients'], dtype=np.float64)
        self.video_path = video_path
        self.points_dict = {}#点的字典, key是ball——id，value是时间戳+点的坐标列表

    #订阅里面回调函数的具体实现：（处理球的id还有时间戳的同步）
    def listener_callback(self, msg):
       
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9#收到python那边pub的时间戳
        if self.t0 == 0.0:
            self.t0 = t#第一次收到消息前t0一直是0,收到消息后，初始化t0
        t_rel = t - self.t0# 这里是相对时间，因为publisher那边获取的是绝对时间，两个启动时间如果不一样就会有误差，导致点划不完或者多了。
        ball_id = int(msg.header.frame_id) if msg.header.frame_id else 0#和publisher.cpp里面差不多的获取方法。
        print(f'收到球的id是 {ball_id}: 坐标是{msg.point.x}, {msg.point.y}, {msg.point.z}')
        if ball_id not in self.points_dict:
            self.points_dict[ball_id] = []
        self.points_dict[ball_id].append((t_rel, (msg.point.x, msg.point.y, msg.point.z)))#在点的字典里依次添加点

    def run(self):
        print('等待卡尔曼滤波的结果传来中')
        while not self.points_dict and rclpy.ok():#这里检查点的字典是否为空以及订阅的消息来了没来。
            rclpy.spin_once(self, timeout_sec=0.1)
        print('收到数据开始播放卡尔曼滤波效果视频')

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = 'kf_visualization_output.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0#帧索引
        base_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
        #整了几个不同的颜色，确保不同id都有不同颜色，免得糊成依托。
        while cap.isOpened():
            rclpy.spin_once(self, timeout_sec=0.01)
            ret, frame = cap.read()
            if not ret:
                break
            t_frame = frame_idx / fps
            if self.t0 is not None:
                # 固定颜色映射：ID 0 红，ID 1 绿
                id2color = {0: (255, 0, 0), 1: (0, 255, 0)}
                for ball_id, pts in self.points_dict.items():
                    ts = [p[0] for p in pts]
                    i_end = bisect.bisect_left(ts, t_frame)
                    color = id2color.get(ball_id, (0, 0, 255))  # 默认蓝
                    for i in range(i_end):
                        x, y, z = pts[i][1]
                        pt3d = np.array([[x, y, z]], dtype=np.float32)
                        rvec = np.zeros((3, 1), dtype=np.float64)
                        tvec = np.zeros((3, 1), dtype=np.float64)
                        pt2d, _ = cv2.projectPoints(pt3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                        px, py = int(pt2d[0][0][0]), int(pt2d[0][0][1])
                        cv2.circle(frame, (px, py), 4, color, -1)
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
    video_path = 'test5/rgb.mp4'
    calib_path = 'ball_coord_sub/src/camera_calibration.json'
    node = KFVisualizer(video_path, calib_path)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
