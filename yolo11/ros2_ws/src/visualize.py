#已弃用
# 最开始的可视化逻辑，后来把它拆分成了visualize_node.py和trajectory_renderer.py两个文件
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
import json
import bisect


class KFVisualizer(Node):
    def __init__(self, VideoPath, CalibPath):
        super().__init__('kf_visualizer')
        self.T0 = None
        self.Subscription = self.create_subscription(
            PointStamped, 
            '/ball/kf_pos',#这个接收的就是subscriber.cpp里面发布的卡尔曼滤波后的结果
            self.ListenerCallback, #回调函数，这里处理时间戳同步的问题
            100)
        # 读取相机内参，后面用于把三维滤波结果投到二维视频上
        with open(CalibPath, 'r') as f:
            calib = json.load(f)
        self.CameraMatrix = np.array(calib['camera_matrix'], dtype=np.float64)
        self.DistCoeffs = np.array(calib['distortion_coefficients'], dtype=np.float64)
        self.VideoPath = VideoPath
        self.PointsDict = {}#点的字典, key是ball——id，value是时间戳+点的坐标列表
        # 预读取视频 FPS
        cap = cv2.VideoCapture(VideoPath)
        self.Fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    #订阅里面回调函数的具体实现：（处理球的id还有时间戳的同步）
    def ListenerCallback(self, msg):
        parts = msg.header.frame_id.split('_')
        ball_id_str = parts[0]
        frame_num_str = parts[1]
        is_raw = len(parts) > 2 and parts[2] == 'raw'  # 检查是否为原始观测点
        ball_id = int(ball_id_str)
        frame_num = int(frame_num_str)
        t_rel = frame_num / self.Fps
        print(f'收到球的id是 {ball_id}: 坐标是{msg.point.x}, {msg.point.y}, {msg.point.z}, frame={frame_num}, raw={is_raw}')
        # print(f'收到球的id是 {ball_id}: 坐标是{msg.point.x}, {msg.point.y}, {msg.point.z}, frame={frame_num}')
        if ball_id not in self.PointsDict:
            self.PointsDict[ball_id] = []
        self.PointsDict[ball_id].append((t_rel, (msg.point.x, msg.point.y, msg.point.z)))#在点的字典里依次添加点

    def Run(self):
        print('等待卡尔曼滤波的结果传来中')
        while not self.PointsDict and rclpy.ok():#这里检查点的字典是否为空以及订阅的消息来了没来。
            rclpy.spin_once(self, timeout_sec=0.1)
        print('收到数据开始播放卡尔曼滤波效果视频')

        cap = cv2.VideoCapture(self.VideoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = 'kf_visualization_output.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0 # 帧索引
        while cap.isOpened():
            # 多 spin 几次以处理更多消息，减少滞后
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.001)
            ret, frame = cap.read()
            if not ret:
                break
            t_frame = frame_idx / fps
            if self.T0 != 0.0:
                for ball_id, pts in self.PointsDict.items():
                    ts = [p[0] for p in pts]
                    i_end = bisect.bisect_left(ts, t_frame)
                    color = (0, 255, 0) if ball_id == 0 else (0, 0, 255) if ball_id == 1 else (255, 0, 0)
                    for i in range(i_end):
                        x, y, z = pts[i][1]
                        pt3d = np.array([[x, y, z]], dtype=np.float32)
                        rvec = np.zeros((3, 1), dtype=np.float64)
                        tvec = np.zeros((3, 1), dtype=np.float64)
                        pt2d, _ = cv2.projectPoints(pt3d, rvec, tvec, self.CameraMatrix, self.DistCoeffs)
                        px = int(pt2d[0][0][0].item())
                        py = int(pt2d[0][0][1].item())
                        cv2.circle(frame, (px, py), 2, color, -1)
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
    video_path = 'test2/rgb.mp4'
    calib_path = 'ball_coord_sub/src/camera_calibration.json'
    node = KFVisualizer(video_path, calib_path)
    try:
        node.Run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
