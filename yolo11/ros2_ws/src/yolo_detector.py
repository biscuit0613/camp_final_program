# yolo_detector.py
# 从publisher.py拆分出来的YOLO检测器模块，封装YOLO的检测逻辑
from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, ModelPath):
        # 初始化YOLO模型
        self.Model = YOLO(ModelPath)

    def Detect(self, Frame, Conf):
        # 使用YOLO进行跟踪推理，返回检测框和ID
        Results = self.Model.track(Frame, persist=True, conf=Conf)
        # yolo的track方法返回一个结果列表results，里面每个元素对应一帧图像，包含检测到的对象信息
        # 元素的boxes就是框，属性如下：
        # boxes.xywh 中心点xy和宽高wh
        # boxes.conf 置信度
        # boxes.id 跟踪id
        # #这里的id是yolo的id,但并不是传给cpp那边的id.具体原因看publisher_node.py里面处理ObjId那一块的注释
        # boxes.cls 类别
        if len(Results) and Results[0].boxes is not None:
            Boxes = Results[0].boxes
            Xywh = Boxes.xywh.cpu().numpy()  # 中心点xy和宽高wh
            if hasattr(Boxes, 'id') and Boxes.id is not None:
                Ids = Boxes.id.cpu().numpy()  # 跟踪ID
                # print('当前帧的id是', Ids)
                # # 这里获取的球id只是为了打印出来看看的，测试用
            else:
                Ids = list(range(len(Xywh)))  # 如果没有ID，把索引作为默认ID
                # print('当前帧的默认的id是', Ids)
            return Xywh, Ids
        return None, None
