import cv2
import numpy as np
from typing import List
from deep.detection import Detection

# 颜色采样 [2047, 32767, 1048575]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def computer_color_by_id(label: int) -> tuple:
    """
    根据身份标签计算颜色
    :param label: 标签, id
    :return: 颜色元组
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_dets(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """
    绘制检测框
    Args:
        img: 帧图像
        dets: 检测列表

    Returns: 返回绘制矩阵框后的图像
    """
    if len(dets) == 0:
        return img  # 无检测

    for det in dets:
        color = (0, 0, 255)  # 红色
        x1, y1, x2, y2 = det.get_int_pos()  # 获取int类型坐标
        # 显示检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # 检测矩阵框

    return img


def draw_tracks(img: np.ndarray, info: dict) -> np.ndarray:
    """
    :param img: 帧图像
    :param info: 包含轨迹id、位置信息、置信度
    :return: 绘制后的img
    """

    if len(info) == 0:
        return img

    for tid, position in info.items():
        color = computer_color_by_id(tid)  # 根据id获取颜色
        x1, y1, x2, y2, _ = [int(e) for e in position]  # 坐标信息
        # 显示文字 tid
        label: str = '{}{:d}'.format("", tid)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)[0]  # 字体
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_8)  # 检测矩阵框
        cv2.rectangle(img, (x1, y1), (x1 + label_size[0] + 2, y1 + label_size[1] + 4), color, -1)  # label背景块
        # 白色的label
        cv2.putText(img, label, (x1, y1 + label_size[1] + 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


if __name__ == '__main__':
    print(computer_color_by_id(255))
