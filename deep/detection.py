import numpy as np


class Detection(object):
    """
    封装检测类
    """

    def __init__(self, det: np.ndarray, feat: np.ndarray):
        self.tlbr = np.asarray(det[0:4], dtype=np.float)  # 左上右下坐标
        self.confidence = float(det[4])  # 置信度
        self.feature = np.asarray(feat, dtype=np.float32)  # 保存特征

    def to_xyah(self):
        """
        转中心坐标、宽高比、高, 用于KF初始化
        :return:
        """
        ret = self.to_tlwh()  # 拷贝一份
        ret[:2] += ret[2:] / 2  # x + w / 2, y + h / 2
        ret[2] /= ret[3]  # w = w / h, h
        return ret

    def to_tlwh(self) -> np.ndarray:
        """
        转左上坐标和wh
        :return: 新坐标
        """
        ret = self.tlbr.copy()  # 拷贝一份
        ret[2] = ret[2] - ret[0]  # w
        ret[3] = ret[3] - ret[1]  # h
        return ret

    def get_int_pos(self) -> list:
        """
        :return: 返回int坐标list
        """
        res = [int(e) for e in self.tlbr]
        return res

    def __str__(self) -> str:
        """
        打印方便查看
        :return: 检测对象字符串
        """
        return 'pos: ' + str(self.tlbr) + ' conf: ' + str(self.confidence)
