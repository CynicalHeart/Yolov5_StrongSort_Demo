import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray(
            [detections[i].to_tlwh() for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def d_iou(bbox, candidates):
    # bbox为单轨迹和检测集进行d_iou匹配, 左上,中心,右下
    bbox_tl, bbox_center, bbox_br = bbox[:2], bbox[:2] + \
                                    bbox[2:] / 2, bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]  # 检测左上集合
    candidates_center = candidates[:, :2] + candidates[:, 2:] / 2  # 检测中心集合
    candidates_br = candidates[:, :2] + candidates[:, 2:]  # 检测右下集合
    # 左上角全取最小
    tl = np.c_[np.minimum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.minimum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    # 右下角取最大
    br = np.c_[np.maximum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.maximum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    center_dist = np.power(candidates_center[:, 0] - bbox_center[0], 2) + np.power(
        candidates_center[:, 1] - bbox_center[1], 2)

    outer_dist = np.power(br[:, 0] - tl[:, 0], 2) + \
                 np.power(br[:, 1] - tl[:, 1], 2)
    res = center_dist / outer_dist
    np.clip(res, a_min=-1, a_max=1)

    return iou(bbox, candidates) - res


def d_iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    实现D-IOU算法, 即本帧检测中心点与轨迹中心点比较
    :return: 返回代价矩阵
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # 初始化0代价矩阵
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        bbox = np.asarray(tracks[track_idx].to_tlwh())  # 转左上、w、h
        candidates = np.asarray(
            [detections[i].tlwh for i in detection_indices])  # 提出坐标

        cost_matrix[row, :] = (1. - d_iou(bbox, candidates)) * 0.5  # 填充一行的代价矩阵
    return cost_matrix


""" 
if __name__ == '__main__':
    b = np.array([2, 2, 4, 6])
    d = np.array([[0, 6, 2, 6], [4, 5, 6, 5]])
    r = d_iou(b, d)
    print(1. - iou(b, d))
    print((1. - r) * 0.5) 
"""
