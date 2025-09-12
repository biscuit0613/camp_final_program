import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('v1.pt')  # 替换为你的权重文件

# 打开视频
video_path = '../test2/rgb.mp4'  
cap = cv2.VideoCapture(video_path)

# 视频保存设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_tracks.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 轨迹字典
trajectories = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model.track(frame, save=True, persist=True, conf=0.7)  # 使用track方法进行目标跟踪

    # 处理每个目标
    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        obj_id = i  # 如有ID可用ID，否则用i

        # 记录轨迹
        if obj_id not in trajectories:
            trajectories[obj_id] = []
        trajectories[obj_id].append((cx, cy))

        # 画检测框和中心点
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 画轨迹
        pts = trajectories[obj_id]
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j - 1], pts[j], (255, 0, 0), 2)

        # 分割掩码可视化（如模型支持分割）
        if results[0].masks is not None:
            mask = results[0].masks.data[i].cpu().numpy()
            mask = (mask > 0.5).astype('uint8') * 255
            colored_mask = cv2.merge([mask // 2, mask, mask // 2])
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    # 写入输出视频
    out.write(frame)
    cv2.imshow('YOLO11 Detect & Track', frame)
    if cv2.waitKey(1) == 27:  # 按ESC退出
        break

cap.release()
out.release()
cv2.destroyAllWindows()