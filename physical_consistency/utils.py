import numpy as np

def physical_score(residual, sigma=5.0):
    """物理一致性分数：基于IMM残差的高斯惩罚"""
    return np.exp(- residual**2 / sigma**2)

def trajectory_score(vel_history, window=5, sigma_v=1.0):
    """轨迹平滑度分数：基于速度序列的方差惩罚"""
    if len(vel_history) < window:
        return 0.5  # 初始默认分数
    vel_array = np.array(vel_history)
    vel_var = np.var(vel_array, axis=0)  # 计算x/y方向速度方差
    traj_score = np.exp(-np.sum(vel_var) / sigma_v**2)
    return np.clip(traj_score, 0.0, 1.0)

def rsu_score(neighbor_votes, sigma_r=0.5):
    """RSU邻居投票分数：基于周边车辆一致性"""
    if not neighbor_votes:
        return 0.5  # 无邻居时默认分数
    avg_vote = np.mean(neighbor_votes)
    rsu_score = np.exp(-(1 - avg_vote)**2 / sigma_r**2)
    return np.clip(rsu_score, 0.0, 1.0)

    # utils.py 末尾新增
def check_physical_consistency(data, physical_rules: dict) -> bool:
    """
    物理一致性校验通用函数
    :param data: 待校验数据（如融合后的特征/物理量）
    :param physical_rules: 物理规则配置（如量纲范围、守恒定律参数）
    :return: 校验通过返回True，否则False
    """
    # 1. 量纲一致性校验（示例）
    if "dimension" in physical_rules:
        target_dim = physical_rules["dimension"]
        if not hasattr(data, "dim") or data.dim != target_dim:
            raise ValueError(f"数据量纲不符，期望{target_dim}，实际{data.dim}")
    
    # 2. 数值范围校验（示例：如物理量不能为负）
    if "value_range" in physical_rules:
        min_val, max_val = physical_rules["value_range"]
        if (data < min_val).any() or (data > max_val).any():
            raise ValueError(f"数据超出物理范围[{min_val}, {max_val}]")
    
    # 3. 扩展：守恒定律、时空一致性等校验逻辑
    return True