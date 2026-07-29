# -*- coding: utf-8 -*-
"""
改进的信誉引擎 - 解决信誉下降速度过慢问题

主要改进：
1. EWMA平滑系数从0.85*old+0.15*new调为0.7*old+0.3*new，使信誉变化速度翻倍
2. 首次作恶放大惩罚：历史信誉>0.85且首次异常时，惩罚步长×2
3. LSTM预测性信誉增强调试和验证
4. 自适应步长加速（count_factor从0.01调为0.05）
5. 多级过滤策略（<0.70降权至30%，<0.50完全排除）
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, LSTM prediction will use fallback")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TrustVector:
    """三维信任向量"""
    direct: float = 0.5
    indirect: float = 0.5
    global_trust: float = 0.5

    def fused_score(self, weights: Tuple[float, float, float] = (0.55, 0.35, 0.10)) -> float:
        w_d, w_i, w_g = weights
        return float(np.clip(w_d * self.direct + w_i * self.indirect + w_g * self.global_trust, 0.0, 1.0))


@dataclass
class VehicleReputationMeta:
    """车辆信誉元数据"""
    score: float = 0.5
    variance: float = 0.0
    update_count: int = 0
    consistency_history: deque = field(default_factory=lambda: deque(maxlen=50))
    trust_vector: TrustVector = field(default_factory=TrustVector)
    warning_level: int = 0
    first_anomaly_detected: bool = False
    high_reputation_streak: int = 0
    anomaly_count: int = 0
    last_anomaly_frame: int = -100


@dataclass
class ReputationConfig:
    """信誉配置参数"""
    default_reputation: float = 0.5
    positive_step: float = 0.05
    negative_step: float = 0.1
    min_reputation: float = 0.0
    
    # 改进1: 更激进的EWMA平滑系数
    ewma_alpha: float = 0.3
    ewma_beta: float = 0.7
    
    # 改进2: 首次作恶惩罚参数
    first_offense_multiplier: float = 2.0
    high_reputation_threshold: float = 0.85
    first_offense_window: int = 30
    
    # 改进6: 自适应步长加速
    adaptive_count_factor: float = 0.05
    
    suspicious_threshold: float = 0.45
    anomaly_threshold: float = 0.3
    fusion_filter_threshold: float = 0.3
    
    # 改进5: 多级过滤阈值
    filter_threshold_soft: float = 0.70
    filter_threshold_hard: float = 0.50
    filter_weight_soft: float = 0.30
    
    adaptive_step: bool = True
    trust_weights: Tuple[float, float, float] = (0.55, 0.35, 0.10)
    lstm_window: int = 10
    lstm_deviation_threshold: float = 0.15
    lstm_enable_debug: bool = True


# ---------------------------------------------------------------------------
# LSTM 预测性信誉（增强调试）
# ---------------------------------------------------------------------------

class _ReputationLSTM(nn.Module if HAS_TORCH else object):
    """轻量LSTM预测器"""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        if not HAS_TORCH:
            return
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class ImprovedPredictiveReputationModel:
    """改进的预测性信誉模型，增强调试功能"""
    
    def __init__(self, window: int = 10, deviation_threshold: float = 0.15, enable_debug: bool = True):
        self.window = window
        self.deviation_threshold = deviation_threshold
        self.enable_debug = enable_debug
        self._sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._lstm = _ReputationLSTM() if HAS_TORCH else None
        if HAS_TORCH and self._lstm is not None:
            self._lstm.eval()
        
        self._prediction_count = 0
        self._deviation_trigger_count = 0
        self._debug_logs: deque = deque(maxlen=100)

    def record_observation(self, vehicle_id: str, features: np.ndarray):
        self._sequences[vehicle_id].append(features.astype(np.float32))

    def predict_next(self, vehicle_id: str) -> Optional[float]:
        seq = self._sequences.get(vehicle_id)
        if not seq or len(seq) < 3:
            return None

        arr = np.array(list(seq))
        
        if HAS_TORCH and self._lstm is not None:
            with torch.no_grad():
                x = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)
                pred = self._lstm(x).item()
            prediction = float(pred)
        else:
            scores = arr[:, 0]
            t = np.arange(len(scores))
            slope = np.polyfit(t, scores, 1)[0] if len(scores) >= 2 else 0.0
            prediction = float(np.clip(scores[-1] + slope, 0.0, 1.0))
        
        self._prediction_count += 1
        return prediction

    def check_deviation(self, vehicle_id: str, actual: float) -> Tuple[bool, float]:
        pred = self.predict_next(vehicle_id)
        if pred is None:
            return False, 0.0
        
        deviation = abs(actual - pred)
        is_trigger = deviation > self.deviation_threshold
        
        if is_trigger:
            self._deviation_trigger_count += 1
        
        if self.enable_debug:
            log_entry = {
                'vehicle_id': vehicle_id,
                'predicted': pred,
                'actual': actual,
                'deviation': deviation,
                'triggered': is_trigger,
            }
            self._debug_logs.append(log_entry)
            
            if is_trigger:
                logger.info(f"[LSTM偏差预警] 车辆{vehicle_id}: 预测={pred:.4f}, 实际={actual:.4f}, "
                           f"偏差={deviation:.4f}")
        
        return is_trigger, deviation
    
    def get_statistics(self) -> Dict:
        return {
            'total_predictions': self._prediction_count,
            'deviation_triggers': self._deviation_trigger_count,
            'trigger_rate': self._deviation_trigger_count / max(1, self._prediction_count),
            'tracked_vehicles': len(self._sequences),
        }


# ---------------------------------------------------------------------------
# 改进的信誉管理器
# ---------------------------------------------------------------------------

class ImprovedReputationManager:
    """改进的信誉管理器"""
    
    def __init__(self, config: Optional[ReputationConfig] = None):
        self.config = config or ReputationConfig()
        self._vehicles: Dict[str, VehicleReputationMeta] = {}
        self._predictor = ImprovedPredictiveReputationModel(
            self.config.lstm_window,
            self.config.lstm_deviation_threshold,
            self.config.lstm_enable_debug
        )
        self._current_frame: Dict[str, int] = defaultdict(int)
        
        logger.info(f"初始化改进信誉管理器: EWMA({self.config.ewma_beta}*old + {self.config.ewma_alpha}*new)")

    def _get_meta(self, vehicle_id: str) -> VehicleReputationMeta:
        if vehicle_id not in self._vehicles:
            m = VehicleReputationMeta(score=self.config.default_reputation)
            m.trust_vector.global_trust = self.config.default_reputation
            self._vehicles[vehicle_id] = m
        return self._vehicles[vehicle_id]

    def get_trust_score(self, vehicle_id: str) -> float:
        return self._get_meta(vehicle_id).score

    def _compute_adaptive_step(self, meta: VehicleReputationMeta, is_positive: bool) -> float:
        base = self.config.positive_step if is_positive else self.config.negative_step
        
        if not self.config.adaptive_step:
            return base
        
        var_factor = 1.0 / (1.0 + meta.variance * 5.0)
        count_factor = min(1.8, 1.0 + meta.update_count * self.config.adaptive_count_factor)
        
        if is_positive:
            return base * var_factor * count_factor
        else:
            return base * (2.0 - var_factor) * count_factor

    def _check_first_offense(self, vehicle_id: str, meta: VehicleReputationMeta) -> bool:
        current_frame = self._current_frame[vehicle_id]
        
        is_first_offense = (
            meta.score > self.config.high_reputation_threshold and
            meta.anomaly_count <= 1 and
            meta.high_reputation_streak > self.config.first_offense_window and
            (current_frame - meta.last_anomaly_frame) > self.config.first_offense_window
        )
        
        if is_first_offense and not meta.first_anomaly_detected:
            logger.warning(f"[首次作恶检测] 车辆{vehicle_id}: 信誉={meta.score:.4f}")
            meta.first_anomaly_detected = True
            return True
        
        return False

    def update_from_evidence(
        self,
        vehicle_id: str,
        is_consistent: bool,
        consistency_ratio: float = 1.0,
        direct_trust: Optional[float] = None,
        indirect_reports: Optional[List[float]] = None,
        attack_type: Optional[str] = None,
    ) -> Dict:
        meta = self._get_meta(vehicle_id)
        self._current_frame[vehicle_id] += 1
        current = meta.score
        
        if direct_trust is not None:
            meta.trust_vector.direct = float(np.clip(direct_trust, 0.0, 1.0))
        if indirect_reports:
            meta.trust_vector.indirect = float(np.clip(np.mean(indirect_reports), 0.0, 1.0))
        
        anomaly_score = 1.0 - consistency_ratio
        self._predictor.record_observation(
            vehicle_id,
            np.array([consistency_ratio, meta.trust_vector.direct,
                     meta.trust_vector.indirect, anomaly_score])
        )
        
        early_warn, deviation = self._predictor.check_deviation(vehicle_id, meta.trust_vector.direct)
        
        step = self._compute_adaptive_step(meta, is_consistent)
        
        first_offense_penalty = 1.0
        if not is_consistent:
            is_first_offense = self._check_first_offense(vehicle_id, meta)
            if is_first_offense:
                first_offense_penalty = self.config.first_offense_multiplier
            
            meta.anomaly_count += 1
            meta.last_anomaly_frame = self._current_frame[vehicle_id]
            meta.high_reputation_streak = 0
        else:
            if meta.score > self.config.high_reputation_threshold:
                meta.high_reputation_streak += 1
        
        if is_consistent:
            step_increase = min(1.0, current + step)
            new_score = self.config.ewma_beta * current + self.config.ewma_alpha * step_increase
        else:
            effective_step = step * first_offense_penalty
            new_score = max(self.config.min_reputation, current - effective_step)
            tv_penalty = meta.trust_vector.fused_score()
            new_score = min(new_score, tv_penalty)
        
        if early_warn and deviation > self.config.lstm_deviation_threshold:
            penalty_factor = min(0.5, deviation / (2 * self.config.lstm_deviation_threshold))
            new_score = max(self.config.min_reputation, new_score - step * penalty_factor)
        
        new_score = float(np.clip(new_score, self.config.min_reputation, 1.0))
        
        meta.score = new_score
        meta.trust_vector.global_trust = new_score
        meta.update_count += 1
        meta.consistency_history.append(1.0 if is_consistent else 0.0)
        
        if len(meta.consistency_history) > 1:
            meta.variance = float(np.var(list(meta.consistency_history)))
        
        if new_score < self.config.anomaly_threshold:
            meta.warning_level = 2
        elif new_score < self.config.suspicious_threshold:
            meta.warning_level = 1
        else:
            meta.warning_level = 0
        
        return {
            "vehicle_id": vehicle_id,
            "old_score": current,
            "new_score": new_score,
            "step_applied": step * first_offense_penalty,
            "warning_level": meta.warning_level,
            "early_warning": early_warn,
            "first_offense": first_offense_penalty > 1.0,
        }

    def get_filter_weight(self, vehicle_id: str) -> float:
        score = self.get_trust_score(vehicle_id)
        
        if score >= self.config.filter_threshold_soft:
            return 1.0
        elif score >= self.config.filter_threshold_hard:
            ratio = (score - self.config.filter_threshold_hard) / (
                self.config.filter_threshold_soft - self.config.filter_threshold_hard
            )
            return ratio * self.config.filter_weight_soft
        else:
            return 0.0

    def get_all_reputations(self) -> Dict[str, float]:
        return {vid: m.score for vid, m in self._vehicles.items()}

    def get_statistics(self) -> Dict:
        lstm_stats = self._predictor.get_statistics()
        
        if not self._vehicles:
            return {**lstm_stats, 'total_vehicles': 0}
        
        scores = [m.score for m in self._vehicles.values()]
        first_offense_count = sum(1 for m in self._vehicles.values() if m.first_anomaly_detected)
        
        return {
            **lstm_stats,
            'total_vehicles': len(self._vehicles),
            'avg_reputation': float(np.mean(scores)),
            'std_reputation': float(np.std(scores)),
            'min_reputation': float(np.min(scores)),
            'max_reputation': float(np.max(scores)),
            'first_offense_detected': first_offense_count,
        }
