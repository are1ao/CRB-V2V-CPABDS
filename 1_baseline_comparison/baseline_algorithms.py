"""
SOTA基线算法实现模块
包含以下对比方案：
1. DRAMBR (Dynamic Reputation-based Anomaly-aware Misbehavior detection)
2. PlexeMDS (Plexe Misbehavior Detection System)
3. 静态信誉方案 (Static Reputation)
4. 多数投票方案 (Majority Voting)
5. 无信任融合 (No Trust Fusion)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import json


class BaselineAlgorithm:
    """基线算法基类"""
    def __init__(self, name: str):
        self.name = name
        self.reputations = {}
        
    def initialize_reputations(self, vehicle_ids: List[str], initial_value: float = 0.5):
        """初始化声誉值"""
        self.reputations = {vid: initial_value for vid in vehicle_ids}
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        """更新声誉值 - 子类实现"""
        raise NotImplementedError
        
    def get_reputation(self, vehicle_id: str) -> float:
        """获取声誉值"""
        return self.reputations.get(vehicle_id, 0.5)
    
    def get_all_reputations(self) -> Dict[str, float]:
        """获取所有声誉值"""
        return self.reputations.copy()


class DRAMBR(BaselineAlgorithm):
    """
    DRAMBR算法实现
    基于论文: "DRAMBR: Dynamic Reputation-based Anomaly-aware Misbehavior detection"
    
    核心特点：
    - 动态声誉更新
    - 基于异常检测的行为评估
    - 使用贝叶斯更新机制
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, 
                 anomaly_threshold: float = 0.25, decay_factor: float = 0.95):
        super().__init__("DRAMBR")
        self.alpha = alpha
        self.beta = beta
        self.anomaly_threshold = anomaly_threshold
        self.decay_factor = decay_factor
        self.interaction_count = defaultdict(int)
        self.positive_count = defaultdict(int)
        self.negative_count = defaultdict(int)
        
    def detect_anomaly(self, observation: Dict) -> bool:
        """检测消息是否异常（调整为更敏感的阈值）"""
        anomaly_score = 0.0
        checks = 0
        
        if 'position_error' in observation:
            anomaly_score += min(observation['position_error'] / 4.0, 1.0)
            checks += 1
            
        if 'velocity_error' in observation:
            anomaly_score += min(observation['velocity_error'] / 3.0, 1.0)
            checks += 1
            
        if 'timestamp_error' in observation:
            anomaly_score += min(observation['timestamp_error'] / 1.0, 1.0)
            checks += 1
            
        if 'message_frequency' in observation:
            expected_freq = 10
            freq_error = abs(observation['message_frequency'] - expected_freq) / expected_freq
            anomaly_score += min(freq_error, 1.0)
            checks += 1
            
        if checks == 0:
            return False
            
        avg_anomaly = anomaly_score / checks
        return avg_anomaly > self.anomaly_threshold
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        """使用贝叶斯更新机制更新声誉"""
        self.interaction_count[vehicle_id] += 1
        is_anomaly = self.detect_anomaly(observation)
        
        if is_anomaly:
            self.negative_count[vehicle_id] += 1
            current_rep = self.reputations.get(vehicle_id, 0.5)
            new_rep = current_rep * (1 - self.beta)
            self.reputations[vehicle_id] = max(0.0, new_rep)
        else:
            self.positive_count[vehicle_id] += 1
            current_rep = self.reputations.get(vehicle_id, 0.5)
            new_rep = current_rep + self.alpha * (1 - current_rep)
            self.reputations[vehicle_id] = min(1.0, new_rep)
            
        if self.interaction_count[vehicle_id] % 10 == 0:
            self.reputations[vehicle_id] = (
                self.reputations[vehicle_id] * self.decay_factor + 
                0.5 * (1 - self.decay_factor)
            )
    
    def get_statistics(self) -> Dict:
        return {
            'total_interactions': sum(self.interaction_count.values()),
            'positive_interactions': sum(self.positive_count.values()),
            'negative_interactions': sum(self.negative_count.values()),
            'avg_reputation': np.mean(list(self.reputations.values())) if self.reputations else 0.5
        }


class PlexeMDS(BaselineAlgorithm):
    """
    PlexeMDS算法实现
    基于Plexe平台的简化误行为检测系统
    """
    def __init__(self, window_size: int = 20, trust_threshold: float = 0.7,
                 distrust_threshold: float = 0.3):
        super().__init__("PlexeMDS")
        self.window_size = window_size
        self.trust_threshold = trust_threshold
        self.distrust_threshold = distrust_threshold
        self.observation_history = defaultdict(list)
        
    def evaluate_message(self, observation: Dict) -> float:
        """评估单条消息的可信度（调整为更敏感的阈值）"""
        score = 1.0
        
        if 'position_error' in observation:
            if observation['position_error'] > 1.5:
                score *= 0.7
            if observation['position_error'] > 3.0:
                score *= 0.4
            if observation['position_error'] > 5.0:
                score *= 0.2
                
        if 'velocity_error' in observation:
            if observation['velocity_error'] > 1.0:
                score *= 0.7
            if observation['velocity_error'] > 2.5:
                score *= 0.4
            if observation['velocity_error'] > 4.0:
                score *= 0.2
                
        if 'timestamp_error' in observation:
            if observation['timestamp_error'] > 0.5:
                score *= 0.6
            if observation['timestamp_error'] > 1.0:
                score *= 0.3
                
        if 'message_frequency' in observation:
            freq_error = abs(observation['message_frequency'] - 10.0)
            if freq_error > 0.5:
                score *= 0.8
            if freq_error > 1.0:
                score *= 0.5
                
        return score
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        """基于滑动窗口更新声誉"""
        message_score = self.evaluate_message(observation)
        self.observation_history[vehicle_id].append(message_score)
        
        if len(self.observation_history[vehicle_id]) > self.window_size:
            self.observation_history[vehicle_id].pop(0)
            
        avg_score = np.mean(self.observation_history[vehicle_id])
        
        if avg_score >= self.trust_threshold:
            self.reputations[vehicle_id] = 1.0
        elif avg_score <= self.distrust_threshold:
            self.reputations[vehicle_id] = 0.0
        else:
            self.reputations[vehicle_id] = avg_score
            
    def get_statistics(self) -> Dict:
        trusted = sum(1 for r in self.reputations.values() if r >= self.trust_threshold)
        distrusted = sum(1 for r in self.reputations.values() if r <= self.distrust_threshold)
        return {
            'trusted_vehicles': trusted,
            'distrusted_vehicles': distrusted,
            'partial_trust_vehicles': len(self.reputations) - trusted - distrusted,
            'avg_reputation': np.mean(list(self.reputations.values())) if self.reputations else 0.5
        }


class StaticReputation(BaselineAlgorithm):
    """静态信誉方案 - 固定信誉值不更新"""
    def __init__(self, fixed_value: float = 0.5):
        super().__init__("StaticReputation")
        self.fixed_value = fixed_value
        
    def initialize_reputations(self, vehicle_ids: List[str], initial_value: float = None):
        value = initial_value if initial_value is not None else self.fixed_value
        self.reputations = {vid: value for vid in vehicle_ids}
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        if vehicle_id not in self.reputations:
            self.reputations[vehicle_id] = self.fixed_value
            
    def get_statistics(self) -> Dict:
        return {
            'fixed_value': self.fixed_value,
            'num_vehicles': len(self.reputations)
        }


class MajorityVoting(BaselineAlgorithm):
    """多数投票方案 - 基于观测数据的简单投票"""
    def __init__(self, voting_threshold: float = 0.6, anomaly_threshold: float = 2.0):
        super().__init__("MajorityVoting")
        self.voting_threshold = voting_threshold
        self.anomaly_threshold = anomaly_threshold
        self.vote_history = defaultdict(lambda: {'positive': 0, 'negative': 0})
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        anomaly_score = 0.0
        
        if 'position_error' in observation:
            anomaly_score += observation['position_error']
        if 'velocity_error' in observation:
            anomaly_score += observation['velocity_error']
        if 'timestamp_error' in observation:
            anomaly_score += observation['timestamp_error']
        
        is_good = anomaly_score < self.anomaly_threshold
        
        if is_good:
            self.vote_history[vehicle_id]['positive'] += 1
        else:
            self.vote_history[vehicle_id]['negative'] += 1
            
        total_votes = (self.vote_history[vehicle_id]['positive'] + 
                      self.vote_history[vehicle_id]['negative'])
        
        if total_votes > 0:
            positive_ratio = self.vote_history[vehicle_id]['positive'] / total_votes
            self.reputations[vehicle_id] = positive_ratio
        else:
            self.reputations[vehicle_id] = 0.5
            
    def get_statistics(self) -> Dict:
        trusted = sum(1 for r in self.reputations.values() if r >= 0.5)
        return {
            'trusted_vehicles': trusted,
            'distrusted_vehicles': len(self.reputations) - trusted,
            'total_votes': sum(v['positive'] + v['negative'] 
                             for v in self.vote_history.values())
        }


class NoTrustFusion(BaselineAlgorithm):
    """无信任融合 - 所有车辆平等对待"""
    def __init__(self):
        super().__init__("NoTrustFusion")
        
    def initialize_reputations(self, vehicle_ids: List[str], initial_value: float = 1.0):
        self.reputations = {vid: 1.0 for vid in vehicle_ids}
        
    def update_reputation(self, vehicle_id: str, observation: Dict):
        if vehicle_id not in self.reputations:
            self.reputations[vehicle_id] = 1.0
            
    def get_statistics(self) -> Dict:
        return {
            'strategy': 'equal_trust',
            'num_vehicles': len(self.reputations)
        }


class BaselineComparison:
    """基线算法对比工具"""
    def __init__(self):
        self.algorithms = {}
        self.results = defaultdict(dict)
        
    def add_algorithm(self, algorithm: BaselineAlgorithm):
        self.algorithms[algorithm.name] = algorithm
        
    def initialize_all(self, vehicle_ids: List[str], initial_value: float = 0.5):
        for algo in self.algorithms.values():
            algo.initialize_reputations(vehicle_ids, initial_value)
            
    def process_observation(self, vehicle_id: str, observation: Dict):
        for algo_name, algo in self.algorithms.items():
            algo.update_reputation(vehicle_id, observation)
            
    def get_comparison_results(self) -> pd.DataFrame:
        results = []
        for algo_name, algo in self.algorithms.items():
            stats = algo.get_statistics()
            row = {
                'Algorithm': algo_name,
                'Avg_Reputation': np.mean(list(algo.reputations.values())) if algo.reputations else 0.5,
                'Std_Reputation': np.std(list(algo.reputations.values())) if algo.reputations else 0.0,
                'Min_Reputation': min(algo.reputations.values()) if algo.reputations else 0.5,
                'Max_Reputation': max(algo.reputations.values()) if algo.reputations else 0.5,
            }
            row.update(stats)
            results.append(row)
        return pd.DataFrame(results)
    
    def save_results(self, output_path: str):
        df = self.get_comparison_results()
        df.to_csv(output_path, index=False)
        print(f"[OK] 对比结果已保存至: {output_path}")
        
        detailed_results = {}
        for algo_name, algo in self.algorithms.items():
            detailed_results[algo_name] = algo.get_all_reputations()
            
        json_path = output_path.replace('.csv', '_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"[OK] 详细声誉值已保存至: {json_path}")
        return df


def create_all_baselines(include_enhanced_drambr: bool = True) -> Dict[str, BaselineAlgorithm]:
    """创建所有基线算法实例"""
    baselines = {
        'DRAMBR': DRAMBR(alpha=0.3, beta=0.7, anomaly_threshold=0.3),
        'PlexeMDS': PlexeMDS(window_size=20, trust_threshold=0.7),
        'StaticReputation': StaticReputation(fixed_value=0.5),
        'MajorityVoting': MajorityVoting(voting_threshold=0.5),
        'NoTrustFusion': NoTrustFusion(),
    }
    if include_enhanced_drambr:
        try:
            from enhanced_drambr import EnhancedDRAMBR
            baselines['EnhancedDRAMBR'] = EnhancedDRAMBR()
        except ImportError:
            pass
    return baselines
