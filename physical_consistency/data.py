import numpy as np
import random


class Vehicle:
    def __init__(self, vid, mode="CV"):
        self.vid = vid
        self.mode = mode

        # 初始状态
        self.x = random.uniform(0, 10)
        self.y = random.uniform(0, 10)
        self.vx = random.uniform(1, 3)
        self.vy = random.uniform(1, 3)
        self.ax = 0
        self.ay = 0

    def step(self, dt):
        if self.mode == "CV":
            # 匀速
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.mode == "CA":
            # 匀加速
            self.ax += np.random.randn() * 0.1
            self.ay += np.random.randn() * 0.1

            self.vx += self.ax * dt
            self.vy += self.ay * dt

            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.mode == "TURN":
            # 简单转弯
            theta = 0.1  # 转角
            vx_new = self.vx * np.cos(theta) - self.vy * np.sin(theta)
            vy_new = self.vx * np.sin(theta) + self.vy * np.cos(theta)

            self.vx, self.vy = vx_new, vy_new

            self.x += self.vx * dt
            self.y += self.vy * dt

    def get_msg(self, t, noise_std=0.5, attack=False):
        pos = np.array([
            self.x + np.random.randn() * noise_std,
            self.y + np.random.randn() * noise_std
        ])

        vel = np.array([
            self.vx + np.random.randn() * noise_std,
            self.vy + np.random.randn() * noise_std
        ])

        # 模拟攻击（幽灵车 / 跳变）
        if attack:
            if random.random() < 0.2:
                pos += np.random.uniform(10, 20, size=2)

        return {
            "vehicle_id": self.vid,
            "timestamp": t,
            "pos": pos,
            "vel": vel
        }


class DataGenerator:
    def __init__(self, num_vehicles=10):

        self.vehicles = []
        modes = ["CV", "CA", "TURN"]

        for i in range(num_vehicles):
            mode = random.choice(modes)
            self.vehicles.append(Vehicle(f"car_{i}", mode))

        # 随机选一些恶意车
        self.attack_vehicles = set(random.sample(
            [v.vid for v in self.vehicles],
            k=max(1, num_vehicles // 5)
        ))

    def step(self, t, dt=0.1):
        msgs = []

        for v in self.vehicles:
            v.step(dt)

            attack = v.vid in self.attack_vehicles

            msg = v.get_msg(t, attack=attack)

            msgs.append(msg)

        return msgs