import numpy as np
import scipy.linalg
from opts import opt

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)  # 8 * 8矩阵F: 表示预测步骤

        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt  # 初始化单位矩阵 + dt(1), 默认匀速运动

        self._update_mat = np.eye(ndim, 2 * ndim)  # 4 * 8测量矩阵H

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20  # 位置方差因素
        self._std_weight_velocity = 1. / 160  # 速度方差因素

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement  # P: 位置信息初始化(cx, cy, a: [w/h], h)
        mean_vel = np.zeros_like(mean_pos)  # V: 上述观测值对应的速度变量
        # 拼接形成x_k, (cx, cy, a, h, 0, 0, 0, 0), 初始化速度都为0
        mean = np.r_[mean_pos, mean_vel]

        # 协方差矩阵P, 元素值越大, 表明不确定越大, shape = (1, 8)
        std = [
            2 * self._std_weight_position *
            measurement[3],  # 2 * 0.05 * h, 高度缩小十倍
            2 * self._std_weight_position * measurement[3],
            1e-2,  # 0.01
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * \
            measurement[3],  # 10 * (1 / 160) * h, h / 16
            10 * self._std_weight_velocity * measurement[3],
            1e-5,  # 0.00001
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))  # 对每个元素进行平方, 并形成8×8对角矩阵, 对角元素是平方值
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],  # 0.05 * ph = ph / 20
            self._std_weight_position * mean[3],
            1e-2,  # 0.01
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],  # (1 / 160) * ph
            self._std_weight_velocity * mean[3],
            1e-5,  # 0.00001
            self._std_weight_velocity * mean[3]]
        # 根据预测向量的高度生成协方差噪声: Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x_k = F·x_{k-1}, 无变速运动影响, 即没有控制矩阵和控制向量
        mean = np.dot(self._motion_mat, mean)
        # P' = F·P·F^T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=.0):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: (dyh) 检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],  # 0.05 * ph = ph / 20
            self._std_weight_position * mean[3],
            1e-1,  # 0.1
            self._std_weight_position * mean[3]]

        # NSA Kalman滤波
        if opt.NSA:
            std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))  # 4 * 4的检测器噪声矩阵R

        # x' = H·x_k, 转换为检测空间下(cx, cy, a, h)
        mean = np.dot(self._update_mat, mean)
        # P' = H·P_k·H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov  # P' + R

    def update(self, mean, covariance, measurement, confidence=.0):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: (dyh)检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(
            mean, covariance, confidence)  # 映射到检测空间, 得到Hx和S
        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean  # 计算差值

        new_mean = mean + np.dot(innovation, kalman_gain.T)  # 更新x'_k = x' + k(z-Hx')
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))  # 更新P'_k = P' - k·H·K^T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)  # H转变至检测空间

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        # 将协方差进行三角分解
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean  # 检测和预测的偏差
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha  # 计算马氏距离
