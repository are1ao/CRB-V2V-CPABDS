import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # 2D残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P

        residual = np.linalg.norm(y)  # 2D残差的范数
        return residual

class IMM:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.mu = np.array([0.5, 0.5])  # CV/CA模型概率
        self.PI = np.array([[0.9, 0.1], [0.1, 0.9]])  # 模型转移矩阵

        # 2D CV模型（x, vx, y, vy）
        F_cv = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        H_cv = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # 观测x/y坐标

        # 2D CA模型（x, vx, ax, y, vy, ay）
        F_ca = np.array([
            [1, dt, 0.5*dt*dt, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5*dt*dt],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        H_ca = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])  # 观测x/y坐标

        # 初始化两个模型的卡尔曼滤波器
        self.models = [
            KalmanFilter(F_cv, H_cv, np.eye(4)*0.1, np.eye(2)*1.0,
                         np.zeros(4), np.eye(4)),  # CV模型
            KalmanFilter(F_ca, H_ca, np.eye(6)*0.1, np.eye(2)*1.0,
                         np.zeros(6), np.eye(6))   # CA模型
        ]

    def step(self, z):
        """
        2D IMM步骤：输入z为[x, y]观测值，返回融合残差和模型概率
        """
        residuals = []
        likelihood = []

        # 每个模型独立预测+更新
        for i, model in enumerate(self.models):
            model.predict()
            residual = model.update(z)
            residuals.append(residual)
            likelihood.append(np.exp(-residual))  # 似然估计

        # 更新模型概率
        likelihood = np.array(likelihood)
        self.mu = self.mu * likelihood
        self.mu = self.mu / np.sum(self.mu)  # 归一化

        # 融合残差（加权平均）
        total_residual = np.dot(self.mu, residuals)
        return total_residual, self.mu