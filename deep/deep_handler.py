import torch
import numpy as np
from typing import List

from opts import opt
from deep.extractor import Extractor
from deep.detection import Detection


class Handler:
    """
    加载det模型和Reid的模型, 实现图像帧输入, 检测和Reid输出
    """

    def __init__(self):
        super(Handler, self).__init__()
        cfg = opt.cfg
        # 本地加载模型与权重
        self.model = torch.hub.load(cfg.DET.YOLO, 'custom', cfg.DET.WEIGHT, source='local')
        assert self.model is not None  # 判断是否加载成功
        self.model.classes = cfg.DET.CLASSES  # 类别
        self.model.conf = cfg.DET.CONF_THRES  # 置信度
        self.model.iou = cfg.DET.IOU_THRES  # NMS的IOU
        self.size = cfg.DET.IMG_SIZE
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 模型是否使用半精度
        if cfg.DET.HALF and device != 'cpu':
            self.model.half()
        self.height = cfg.DET.MIN_HEIGHT
        self.extractor = Extractor(cfg)  # 提取特征网络

    @staticmethod
    def _crop(det: np.ndarray, img: np.ndarray) -> list:
        im_crops = []
        for bbox in det:
            # 解析坐标
            x1, y1, x2, y2 = bbox[0:4]
            # 抠图
            im = img[int(y1):int(y2), int(x1):int(x2), :]
            im_crops.append(im)
        return im_crops

    def _det(self, img: np.ndarray) -> np.ndarray:
        """
        :param img: RGB图像
        :return: 检测结果
        """
        detections = self.model(img, size=self.size)
        det = detections.xyxy[0].cpu().numpy()  # 检测结果 n × 6(坐标×4, conf, cls)
        return det

    def _features(self, im_crops: list) -> np.ndarray:
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def sample_extract(self, im_crops: list) -> np.ndarray:
        """
        :param im_crops: 图像集合 List[cv.Mat]
        :return: 特征集合 n × 512
        """
        return self._features(im_crops)

    def __call__(self, img: np.ndarray) -> List[Detection]:
        """
        :param img: RGB图像
        :return: 检测 + 重识别
        """

        det = self._det(img)  # 检测 [n × 6]
        im_crops = Handler._crop(det, img)  # 切出的图 [n]
        features = self._features(im_crops)  # 提取出特征集合[n × 512]
        dets = []  # 集合对象
        # 返回检测结果和特征集合
        for idx, d in enumerate(det):
            flag = True if self.height == 0 or self.height < (d[3] - d[1]) else False
            if not flag:
                continue  # 跳过改检测
            else:
                dets.append(Detection(d, features[idx]))  # 加入检测和特征

        return dets


if __name__ == '__main__':
    import cv2

    handler = Handler()
    path = r"E:/MySpace/resource/video/test_01.mp4"
    cap = cv2.VideoCapture(path)

    while True:
        res, frame = cap.read()
        if not res:
            break
        i = frame[:, :, (2, 1, 0)]
        det = handler(i)
        for d in det:
            print(d)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
