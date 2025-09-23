# trajectory_renderer.py
# 从原来的visualize.py拆分出来的文件，只负责渲染轨迹到视频上
import cv2
import numpy as np
import json
import bisect

class TrajectoryRenderer:
    def __init__(self, VideoPath, CalibPath):
        self.VideoPath = VideoPath
        self.PointsDict = {}
        # 读取相机内参
        with open(CalibPath, 'r') as f:
            calib = json.load(f)
        self.CameraMatrix = np.array(calib['camera_matrix'], dtype=np.float64)
        self.DistCoeffs = np.array(calib['distortion_coefficients'], dtype=np.float64)
        # 预读取视频 FPS
        cap = cv2.VideoCapture(VideoPath)
        self.Fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def AddPoint(self, msg):
        parts = msg.header.frame_id.split('_')
        ball_id_str = parts[0]
        frame_num_str = parts[1]
        sub_frame_str = parts[2] if len(parts) > 2 and parts[2].isdigit() else "0"
        is_raw = len(parts) > 3 and parts[3] == 'raw'
        ball_id = int(ball_id_str)
        frame_num = int(frame_num_str)
        sub_frame = int(sub_frame_str)
        # 用1/100作为权重，最多99个插值点。
        t_rel = frame_num + sub_frame / 100.0
        print(f'收到球的id是 {ball_id}: 坐标是{msg.point.x}, {msg.point.y}, {msg.point.z}, frame={frame_num}.{sub_frame}, raw={is_raw}')
        if ball_id not in self.PointsDict:
            self.PointsDict[ball_id] = []
        self.PointsDict[ball_id].append((t_rel, (msg.point.x, msg.point.y, msg.point.z)))

    def RenderFrame(self, frame, frame_idx):
        if self.PointsDict:
            for ball_id, pts in self.PointsDict.items():
                ts = [p[0] for p in pts]
                i_end = bisect.bisect_left(ts, frame_idx + 1)  # 显示所有 <= 当前帧索引的点
                # 比如说frameidx=15的插值点的是 15.01, 15.02, 15.03，根据这个比大小，就显示所有小于16的点
                color = (0, 255, 0) if ball_id == 0 else (0, 0, 255) if ball_id == 1 else (255, 0, 0)
                for i in range(i_end):
                    x, y, z = pts[i][1]
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                        continue  # 跳过无效点
                    pt3d = np.array([[x, y, z]], dtype=np.float32)
                    rvec = np.zeros((3, 1), dtype=np.float64)
                    tvec = np.zeros((3, 1), dtype=np.float64)
                    pt2d, _ = cv2.projectPoints(pt3d, rvec, tvec, self.CameraMatrix, self.DistCoeffs)
                    px = int(pt2d[0][0][0].item())
                    py = int(pt2d[0][0][1].item())
                    cv2.circle(frame, (px, py), 2, color, -1)
        return frame
