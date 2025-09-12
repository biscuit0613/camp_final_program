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
        self.points = []  # [(timestamp, (x, y, z))]
        self.t0 = None
        self.subscription = self.create_subscription(
            PointStamped, '/ball/kf_pos', self.listener_callback, 10)
        # 读取相机内参
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        self.camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(calib['distortion_coefficients'], dtype=np.float64)
        self.video_path = video_path

    def listener_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        t_rel = t - self.t0
        self.points.append((t_rel, (msg.point.x, msg.point.y, msg.point.z)))

    def run(self):
        # 等待至少收到一个点
        print('Waiting for first KF point...')
        while not self.points and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        print('Received first KF point, start playing video.')

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = 'kf_visualization_output.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0
        while cap.isOpened():
            rclpy.spin_once(self, timeout_sec=0.01)
            ret, frame = cap.read()
            if not ret:
                break
            t_frame = frame_idx / fps
            if self.points and self.t0 is not None:
                t_frame_rel = t_frame
                ts = [p[0] for p in self.points]
                idx = bisect.bisect_left(ts, t_frame_rel)
                # 这个循环方法是保留历史点迹的
                for i in range (idx):
                    x,y,z = self.points[i][1]
                    pt3d = np.array([[x, y, z]], dtype=np.float32)
                    rvec = np.zeros((3, 1), dtype=np.float64)
                    tvec = np.zeros((3, 1), dtype=np.float64)
                    pt2d, _ = cv2.projectPoints(pt3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    px, py = int(pt2d[0][0][0]), int(pt2d[0][0][1])
                    cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)
                # 下面这个方法不保留历史点迹，看起来更舒服一点
                # if idx < len(self.points):
                #     x, y, z = self.points[idx][1]
                #     pt3d = np.array([[x, y, z]], dtype=np.float32)
                #     rvec = np.zeros((3, 1), dtype=np.float64)
                #     tvec = np.zeros((3, 1), dtype=np.float64)
                #     pt2d, _ = cv2.projectPoints(pt3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                #     px, py = int(pt2d[0][0][0]), int(pt2d[0][0][1])
                #     cv2.circle(frame, (px, py), 8, (255, 0, 0), -1)
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
    video_path = 'test4/rgb.mp4'
    calib_path = 'ball_coord_sub/src/camera_calibration.json'
    node = KFVisualizer(video_path, calib_path)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
