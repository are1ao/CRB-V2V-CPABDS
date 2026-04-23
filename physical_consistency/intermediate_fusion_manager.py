import numpy as np
from utils import physical_score, trajectory_score, rsu_score

class IntermediateFusionManager:
    def __init__(self):
        self.reputation = {}          # 车辆最终信誉值
        self.vel_history = {}         # 速度历史（用于轨迹分数）
        self.neighbor_votes = {}      # 邻居投票（用于RSU分数）
        self.beta = 0.5               # 信誉更新率
        self.weights = {              # 中间层融合权重
            "physical": 0.4,
            "trajectory": 0.3,
            "rsu": 0.3
        }

    def get_vel_history(self, vid):
        """获取车辆速度历史，无则初始化"""
        if vid not in self.vel_history:
            self.vel_history[vid] = []
        return self.vel_history[vid]

    def update_vel_history(self, vid, vel):
        """更新速度历史（保留最近5帧）"""
        history = self.get_vel_history(vid)
        history.append(vel)
        if len(history) > 5:
            history.pop(0)
        self.vel_history[vid] = history

    def get_neighbor_votes(self, vid):
        """获取车辆邻居投票，无则初始化"""
        if vid not in self.neighbor_votes:
            self.neighbor_votes[vid] = []
        return self.neighbor_votes[vid]

    def update_neighbor_votes(self, vid, vote):
        """更新邻居投票（保留最近10个）"""
        votes = self.get_neighbor_votes(vid)
        votes.append(vote)
        if len(votes) > 10:
            votes.pop(0)
        self.neighbor_votes[vid] = votes

    def fuse_scores(self, phy_score, traj_score, rsu_score):
        """中间层融合核心：加权融合多源分数"""
        fused_score = (
            self.weights["physical"] * phy_score +
            self.weights["trajectory"] * traj_score +
            self.weights["rsu"] * rsu_score
        )
        return np.clip(fused_score, 0.0, 1.0)

    def compute_all_scores(self, residual, vid, vel):
        """计算所有单源分数，并更新历史"""
        # 1. 物理分数（IMM残差）
        phy_score = physical_score(residual)
        # 2. 轨迹分数（速度平滑度）
        self.update_vel_history(vid, vel)
        traj_score = trajectory_score(self.get_vel_history(vid))
        # 3. RSU分数（邻居投票）
        rsu_score_val = rsu_score(self.get_neighbor_votes(vid))
        # 中间层融合
        fused_score = self.fuse_scores(phy_score, traj_score, rsu_score_val)
        return {
            "physical": phy_score,
            "trajectory": traj_score,
            "rsu": rsu_score_val,
            "fused": fused_score
        }

    def update_reputation(self, vid, fused_score):
        """更新车辆信誉值"""
        if vid not in self.reputation:
            self.reputation[vid] = 0.5  # 初始信誉
        # 信誉更新（带边界裁剪）
        self.reputation[vid] = self.reputation[vid] + self.beta * (fused_score - self.reputation[vid])
        self.reputation[vid] = np.clip(self.reputation[vid], 0.0, 1.0)
        return self.reputation[vid]

    def get_vote(self, fused_score):
        """基于融合分数的投票决策"""
        return -1 if fused_score < 0.6 else 1