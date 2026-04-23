# -*- coding: utf-8 -*-
"""
重叠视场投票模块 - 基于WBF的信誉加权融合
"""

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion


# ==================== 类1：投票器 ====================
class OverlapFieldVoter:
    """执行加权融合的核心类"""
    
    def __init__(self, iou_thr=0.5, skip_box_thr=0.0001, conf_type='avg', device='cuda'):
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type
        self.device = device
        
    def vote_detection_level(self, boxes_list, scores_list, labels_list, trust_scores=None):
        """检测框级投票融合，信誉值作为权重"""
        n_vehicles = len(boxes_list)
        if n_vehicles == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
        
        if trust_scores is None:
            weights = np.ones(n_vehicles)
        else:
            weights = np.array(trust_scores)
            # 信誉值低于0.3的车辆直接排除
            valid_mask = weights >= 0.3
            if not np.all(valid_mask):
                boxes_list = [boxes_list[i] for i in range(n_vehicles) if valid_mask[i]]
                scores_list = [scores_list[i] for i in range(n_vehicles) if valid_mask[i]]
                labels_list = [labels_list[i] for i in range(n_vehicles) if valid_mask[i]]
                weights = weights[valid_mask]
                n_vehicles = len(boxes_list)
            
            if n_vehicles == 0:
                return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
        
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights.tolist(),
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
            conf_type=self.conf_type
        )
        return fused_boxes, fused_scores, fused_labels
    
    def vote_feature_level(self, feature_list, trust_scores, fusion_method='weighted_sum'):
        """特征级投票融合（备用）"""
        if not feature_list:
            return None
        n_vehicles = len(feature_list)
        if n_vehicles == 1:
            return feature_list[0]
        
        weights = torch.tensor(trust_scores, device=self.device, dtype=torch.float32)
        weights = torch.clamp(weights, min=0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(n_vehicles, device=self.device) / n_vehicles
        
        fused_feature = torch.zeros_like(feature_list[0])
        for feat, w in zip(feature_list, weights):
            fused_feature = fused_feature + feat * w
        return fused_feature


# ==================== 类2：信誉管理器 ====================
class ReputationManager:
    """存储和更新车辆信誉值"""
    
    def __init__(self, default_reputation=0.5, update_rate=0.1, min_reputation=0.3):
        self.reputation_cache = {}
        self.default_reputation = default_reputation
        self.update_rate = update_rate
        self.min_reputation = min_reputation  # 信誉值下限
        
    def get_trust_score(self, vehicle_id):
        if vehicle_id in self.reputation_cache:
            return self.reputation_cache[vehicle_id]
        return self.default_reputation
    
    def set_trust_score(self, vehicle_id, score):
        score = max(self.min_reputation, min(1.0, score))
        self.reputation_cache[vehicle_id] = score
    
    def update_from_voting_consistency(self, vehicle_id, is_consistent):
        current_score = self.get_trust_score(vehicle_id)
        if is_consistent:
            new_score = min(1.0, current_score + self.update_rate)
        else:
            new_score = max(self.min_reputation, current_score - self.update_rate)
        self.reputation_cache[vehicle_id] = new_score
    
    def batch_update_from_voting(self, fused_result, original_detections, vehicle_ids, iou_thr=0.5):
        """批量更新信誉值"""
        fused_boxes, fused_scores, fused_labels = fused_result
        if len(fused_boxes) == 0:
            return {vid: False for vid in vehicle_ids}
        
        consistency_dict = {}
        for vehicle_id in vehicle_ids:
            if vehicle_id not in original_detections:
                consistency_dict[vehicle_id] = False
                continue
            
            detections = original_detections[vehicle_id]
            vehicle_boxes = detections.get('boxes', [])
            vehicle_labels = detections.get('labels', [])
            
            if len(vehicle_boxes) == 0:
                consistency_dict[vehicle_id] = False
                continue
            
            consistent_count = 0
            total_matchable = 0
            
            for i, vbox in enumerate(vehicle_boxes):
                vlabel = vehicle_labels[i] if i < len(vehicle_labels) else None
                for j, fbox in enumerate(fused_boxes):
                    iou = self._calculate_iou(vbox, fbox)
                    if iou > iou_thr:
                        total_matchable += 1
                        if vlabel == fused_labels[j]:
                            consistent_count += 1
                        break
            
            if total_matchable > 0:
                consistency_ratio = consistent_count / total_matchable
                is_consistent = consistency_ratio > 0.7
            else:
                is_consistent = False
            
            consistency_dict[vehicle_id] = is_consistent
            self.update_from_voting_consistency(vehicle_id, is_consistent)
        
        return consistency_dict
    
    @staticmethod
    def _calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0


# ==================== 类3：主入口系统 ====================
class OverlapFieldVotingSystem:
    """整合投票器和信誉管理器的主入口"""
    
    def __init__(self, iou_thr=0.5, update_rate=0.1, default_reputation=0.5, min_reputation=0.3):
        self.voter = OverlapFieldVoter(iou_thr=iou_thr)
        self.reputation_manager = ReputationManager(
            default_reputation=default_reputation,
            update_rate=update_rate,
            min_reputation=min_reputation
        )
    
    def set_reputation(self, vehicle_id, score):
        self.reputation_manager.set_trust_score(vehicle_id, score)
    
    def get_reputation(self, vehicle_id):
        return self.reputation_manager.get_trust_score(vehicle_id)
    
    def get_all_reputations(self):
        return self.reputation_manager.reputation_cache.copy()
    
    def load_reputations_from_cache(self, cache_dict):
        for vid, score in cache_dict.items():
            self.set_reputation(vid, score)
    
    def fuse(self, detections_dict):
        """执行融合"""
        vehicle_ids = list(detections_dict.keys())
        trust_scores = [self.get_reputation(vid) for vid in vehicle_ids]
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for vid in vehicle_ids:
            data = detections_dict[vid]
            boxes_list.append(data['boxes'])
            scores_list.append(data['scores'])
            labels_list.append(data['labels'])
        
        return self.voter.vote_detection_level(boxes_list, scores_list, labels_list, trust_scores)
    
    def update_reputations(self, fused_result, detections_dict):
        """更新信誉值"""
        vehicle_ids = list(detections_dict.keys())
        return self.reputation_manager.batch_update_from_voting(
            fused_result, detections_dict, vehicle_ids
        )