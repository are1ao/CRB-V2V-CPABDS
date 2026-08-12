# -*- coding: utf-8 -*-
"""Overlap-field voting for detection-level trust updates."""

import numpy as np
import sys
from typing import List, Dict, Tuple, Optional

# ✅ 新增：导入空间冲突检测模块
from physical_consistency import (
    detect_spatial_collusion,
    compute_visibility_weights,
    get_suspicious_vehicles,
    get_visibility_weights
)


# ========== 核心投票器 ==========
class OverlapFieldVoter:
    """Cluster 2D detections with score-reputation weighted voting."""

    def __init__(self, iou_thr=0.5, skip_box_thr=1e-4):
        self.iou_thr = float(iou_thr)
        self.skip_box_thr = float(skip_box_thr)

    @staticmethod
    def empty_output():
        """Return the standard empty voting output tuple."""
        return np.zeros((0, 4), dtype=np.float32), \
            np.zeros((0,), dtype=np.float32), \
            np.zeros((0,), dtype=np.int32)

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two 2D boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0.0 else 0.0

    @staticmethod
    def apply_gradient_weighting(reputation):
        """
        应用梯度降权逻辑（共谋防御：低信誉车辆被排除）
        
        Args:
            reputation: 原始信誉值 (0-1)
            
        Returns:
            float: 应用梯度降权后的权重系数
        """
        rep_weight = max(reputation, 0.0) ** 2
        
        if reputation < 0.5:
            return 0.0
        elif reputation < 0.7:
            t = (reputation - 0.5) / (0.7 - 0.5)
            gradient_factor = 0.2 + t * 0.3
            return rep_weight * gradient_factor
        elif reputation < 0.85:
            t = (reputation - 0.7) / (0.85 - 0.7)
            gradient_factor = 0.5 + t * 0.5
            return rep_weight * gradient_factor
        else:
            return rep_weight

    def vote_detection_level(self, boxes_list, scores_list, labels_list,
                             reputation_scores=None):
        """核心投票融合函数（生成共识参考）"""
        if not boxes_list:
            return self.empty_output()
        
        if not (len(boxes_list) == len(scores_list) == len(labels_list)):
            raise ValueError(
                f"boxes_list, scores_list, labels_list must have same length. "
                f"Got {len(boxes_list)}, {len(scores_list)}, {len(labels_list)}"
            )
        
        if reputation_scores is not None and len(reputation_scores) != len(boxes_list):
            raise ValueError(
                f"reputation_scores length ({len(reputation_scores)}) "
                f"must match boxes_list length ({len(boxes_list)})"
            )

        flattened = []
        for agent_idx, boxes in enumerate(boxes_list):
            reputation = 1.0 if reputation_scores is None else \
                float(reputation_scores[agent_idx])
            
            rep_weight = self.apply_gradient_weighting(reputation)
            
            if rep_weight <= 0.0:
                continue

            for box_idx, box in enumerate(boxes):
                score = float(scores_list[agent_idx][box_idx])
                label = int(labels_list[agent_idx][box_idx])
                weight = score * rep_weight
                if weight <= self.skip_box_thr + sys.float_info.epsilon:
                    continue
                flattened.append({
                    'box': np.asarray(box, dtype=np.float32),
                    'score': score,
                    'label': label,
                    'weight': weight,
                })

        if not flattened:
            return self.empty_output()

        flattened.sort(key=lambda item: item['weight'], reverse=True)
        clusters = []
        for det in flattened:
            assigned = False
            for cluster in clusters:
                if det['label'] != cluster['label']:
                    continue
                if self.calculate_iou(det['box'], cluster['mean_box']) < \
                        self.iou_thr:
                    continue
                cluster['box_sum'] += det['box'] * det['weight']
                cluster['score_sum'] += det['score'] * det['weight']
                cluster['sum_weight'] += det['weight']
                cluster['mean_box'] = cluster['box_sum'] / \
                    cluster['sum_weight']
                assigned = True
                break
            if not assigned:
                clusters.append({
                    'label': det['label'],
                    'mean_box': det['box'].copy(),
                    'box_sum': det['box'] * det['weight'],
                    'score_sum': det['score'] * det['weight'],
                    'sum_weight': det['weight'],
                })

        fused_boxes = []
        fused_scores = []
        fused_labels = []
        for cluster in clusters:
            if cluster['sum_weight'] <= self.skip_box_thr + sys.float_info.epsilon:
                continue
            fused_boxes.append(cluster['mean_box'])
            fused_scores.append(cluster['score_sum'] / cluster['sum_weight'])
            fused_labels.append(cluster['label'])

        if not fused_boxes:
            return self.empty_output()

        return np.asarray(fused_boxes, dtype=np.float32), \
            np.asarray(fused_scores, dtype=np.float32), \
            np.asarray(fused_labels, dtype=np.int32)


# ========== 自适应阈值计算工具函数 ==========
def calculate_adaptive_threshold(num_detections: int, reputation: float) -> float:
    """
    根据检测框数量和信誉值计算自适应匹配率阈值
    
    参数：
        num_detections: 这辆车检测到了多少个目标
        reputation: 这辆车的当前信誉值 (0-1)
    
    返回：
        float: 调整后的阈值 (范围控制在0.35-0.85之间)
    
    逻辑说明：
        1. 检测框越少 → 阈值越低（考虑领头车视野受限）
        2. 信誉越高 → 阈值越低（历史表现好，值得信任）
        3. 信誉越低 → 阈值越高（历史有问题，严格审查）
    """
    # 第一步：根据检测数量确定基础阈值
    if num_detections <= 3:
        base_threshold = 0.5
    elif num_detections <= 5:
        base_threshold = 0.6
    else:
        base_threshold = 0.7
    
    # 第二步：根据信誉值计算调节因子
    if reputation >= 0.7:
        factor = 1.0 - (reputation - 0.7) / 0.3 * 0.15
    elif reputation >= 0.4:
        factor = 1.0
    else:
        factor = 1.0 + (0.4 - reputation) / 0.4 * 0.3
    
    # 第三步：计算最终阈值
    final_threshold = base_threshold * factor
    
    # 第四步：限制阈值范围
    if final_threshold < 0.35:
        final_threshold = 0.35
    if final_threshold > 0.85:
        final_threshold = 0.85
    
    return final_threshold


# ========== 辅助函数：物理视野过滤器 ==========
def filter_detections_by_visibility(target_position: Tuple,
                                     agent_det: Dict,
                                     max_visible_distance: float = 200.0) -> Dict:
    """
    根据目标车辆的位置，过滤参考车辆中在物理上不可见的检测框。
    
    参数：
        target_position: 被评估车辆的位置 (x, y, z)
        agent_det: 参考车辆的检测数据，包含 boxes, scores, labels, position
        max_visible_distance: 最大可视距离（米），默认200m
    
    返回：
        Dict: 过滤后的检测数据，只包含目标车辆物理上能看到的检测框
    """
    agent_boxes = agent_det.get('boxes', np.array([]))
    
    # ✅ 修复：使用 len() 判断而非直接对数组做布尔判断
    if len(agent_boxes) == 0:
        return {
            'boxes': np.array([]),
            'scores': np.array([]),
            'labels': np.array([]),
            'position': agent_det.get('position', (0, 0, 0)),
            'reputation': agent_det.get('reputation', 1.0)
        }
    
    agent_scores = agent_det.get('scores', np.array([]))
    agent_labels = agent_det.get('labels', np.array([]))
    
    valid_boxes = []
    valid_scores = []
    valid_labels = []
    
    target_pos = np.array(target_position[:2])
    
    for i, box in enumerate(agent_boxes):
        # 计算检测框中心点（世界坐标）
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        center = np.array([center_x, center_y])
        
        # 计算距离
        distance = np.linalg.norm(target_pos - center)
        
        # 如果距离超过最大可视距离，过滤掉这个检测框
        if distance > max_visible_distance:
            continue
        
        # 通过检查，保留该检测框
        valid_boxes.append(box)
        if i < len(agent_scores):
            valid_scores.append(agent_scores[i])
        else:
            valid_scores.append(1.0)
        if i < len(agent_labels):
            valid_labels.append(agent_labels[i])
        else:
            valid_labels.append(0)
    
    return {
        'boxes': np.array(valid_boxes) if valid_boxes else np.array([]),
        'scores': np.array(valid_scores) if valid_scores else np.array([]),
        'labels': np.array(valid_labels) if valid_labels else np.array([]),
        'position': agent_det.get('position', (0, 0, 0)),
        'reputation': agent_det.get('reputation', 1.0)
    }


# ========== 主系统类 ==========
class OverlapFieldVotingSystem:
    """Build voting consensus and check leave-one-out consistency."""

    def __init__(self, iou_thr=0.5, skip_box_thr=1e-4,
                 min_reference_agents=1, min_matched_boxes=1,
                 enable_collusion_detection=True,
                 max_visible_distance=200.0):
        self.voter = OverlapFieldVoter(iou_thr=iou_thr,
                                       skip_box_thr=skip_box_thr)
        self.min_reference_agents = int(min_reference_agents)
        self.min_matched_boxes = int(min_matched_boxes)
        self.enable_collusion_detection = enable_collusion_detection
        self.max_visible_distance = float(max_visible_distance)

    def fuse(self, detections_dict):
        """数据准备与融合"""
        agent_ids = list(detections_dict.keys())
        reputations = [detections_dict[agent_id].get('reputation', 1.0)
                       for agent_id in agent_ids]
        boxes_list = [detections_dict[agent_id]['boxes']
                       for agent_id in agent_ids]
        scores_list = [detections_dict[agent_id]['scores']
                       for agent_id in agent_ids]
        labels_list = [detections_dict[agent_id]['labels']
                       for agent_id in agent_ids]
        return self.voter.vote_detection_level(boxes_list, scores_list,
                                               labels_list, reputations)

    def compute_consistency_leave_one_out(self, detections_dict, iou_thr=0.5,
                                          ego_position=None):
        """简化接口：返回每辆车的一致性判断"""
        details = self.compute_consistency_details(detections_dict,
                                                   iou_thr=iou_thr,
                                                   ego_position=ego_position)
        return {
            agent_id: item['consistent']
            for agent_id, item in details.items()
            if agent_id != '_collusion'
        }

    def compute_consistency_details(self, detections_dict, iou_thr=0.5,
                                     ego_position=None):
        """
        Return leave-one-out consistency and debug metadata.
        
        新增参数：
            ego_position: 自车位置 (x, y, z)，用于空间冲突检测
        """
        # 空间冲突检测
        collusion_result = None
        if self.enable_collusion_detection and ego_position is not None:
            collusion_result = detect_spatial_collusion(
                detections_dict, 
                ego_position
            )
        
        details = {}
        
        for target_id, target_det in detections_dict.items():
            # 获取被评估车辆的位置
            target_position = target_det.get('position', (0, 0, 0))
            
            # ============================================================
            # 核心修改：基于物理视野约束构建参考集
            # 对于每一辆参考车辆，只保留目标车辆物理上能看到的检测框
            # ============================================================
            filtered_reference_detections = {}
            
            for agent_id, agent_det in detections_dict.items():
                if agent_id == target_id:
                    continue
                
                # 过滤掉目标车辆看不到的检测框
                filtered_agent_det = filter_detections_by_visibility(
                    target_position=target_position,
                    agent_det=agent_det,
                    max_visible_distance=self.max_visible_distance
                )
                
                # 如果该参考车辆过滤后仍有有效检测框，则加入参考集
                if len(filtered_agent_det.get('boxes', [])) > 0:
                    filtered_reference_detections[agent_id] = filtered_agent_det
            
            reference_count = len(filtered_reference_detections)
            
            if reference_count < self.min_reference_agents:
                details[target_id] = {
                    'consistent': None,
                    'reason': 'insufficient_reference_agents_after_visibility_filter',
                    'reference_agent_count': reference_count,
                    'matched_boxes': 0,
                    'unmatched_boxes': len(target_det.get('boxes', [])),
                    'consistency_ratio': None,
                    'filtered_reference_agents': list(filtered_reference_detections.keys()),
                }
                continue

            # 使用过滤后的参考集进行融合
            fused_reference = self.fuse(filtered_reference_detections)
            
            target_reputation = target_det.get('reputation', 1.0)
            
            # 注意：visibility_weight 在这里不再用于惩罚目标车辆
            # 因为已经在构建参考集时过滤了物理上不可见的检测框
            details[target_id] = self.compare_to_fused_details(
                target_det, fused_reference, iou_thr=iou_thr,
                target_reputation=target_reputation,
                visibility_weight=1.0,
                reference_count=reference_count
            )
            details[target_id]['reference_agent_count'] = reference_count
            details[target_id]['filtered_reference_agents'] = list(
                filtered_reference_detections.keys()
            )
        
        if collusion_result:
            details['_collusion'] = collusion_result
        
        return details

    def compare_to_fused(self, detections, fused_output, iou_thr=0.5):
        """Return whether one agent's detections agree with fused boxes."""
        return self.compare_to_fused_details(
            detections,
            fused_output,
            iou_thr=iou_thr)['consistent']

    def compare_to_fused_details(self, detections, fused_output, iou_thr=0.5,
                              target_reputation=1.0,
                              visibility_weight=1.0,
                              reference_count=0):
        """
        Return consistency plus matched/unmatched debug counts.
        
        参数：
            target_reputation: 被评估车辆的当前信誉值，用于自适应阈值计算
            visibility_weight: 视野权重（当前已不再用于惩罚，保留为1.0）
            reference_count: 参考车辆数量
        """
        fused_boxes, _, fused_labels = fused_output
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        
        if len(fused_boxes) == 0 or len(boxes) == 0:
            return {
                'consistent': False,
                'reason': 'empty_reference_or_target',
                'matched_boxes': 0,
                'unmatched_boxes': len(boxes),
                'consistency_ratio': 0.0,
                'adaptive_threshold': 0.0,
                'target_reputation': target_reputation,
                'visibility_weight': visibility_weight,
                'num_detections': len(boxes),
            }

        matched = 0
        label_matched = 0
        for box_idx, box in enumerate(boxes):
            label = labels[box_idx] if box_idx < len(labels) else None
            best_idx = None
            best_iou = 0.0
            for fused_idx, fused_box in enumerate(fused_boxes):
                iou = self.voter.calculate_iou(box, fused_box)
                if iou > iou_thr and iou > best_iou:
                    best_idx = fused_idx
                    best_iou = iou
            if best_idx is None:
                continue
            matched += 1
            if label == fused_labels[best_idx]:
                label_matched += 1

        num_detections = len(boxes)
        match_ratio = float(matched) / float(num_detections) if num_detections > 0 else 0.0
        label_ratio = float(label_matched) / float(matched) if matched > 0 else 0.0

        # 自适应阈值（基于检测数和信誉）
        adaptive_threshold = calculate_adaptive_threshold(
            num_detections=num_detections,
            reputation=target_reputation
        )
        
        # 参考车辆数量微调
        if reference_count <= 2 and adaptive_threshold < 0.6:
            adaptive_threshold = min(0.6, adaptive_threshold * 1.05)
        
        # 判断一致性：直接用原始匹配率比较阈值
        if matched < self.min_matched_boxes:
            consistent = False
            reason = 'insufficient_matched_boxes'
        else:
            consistent = (match_ratio >= adaptive_threshold)
            reason = 'matched' if consistent else 'threshold_not_met'

        return {
            'consistent': consistent,
            'reason': reason,
            'matched_boxes': matched,
            'unmatched_boxes': num_detections - matched,
            'consistency_ratio': match_ratio,
            'label_consistency': label_ratio,
            'adaptive_threshold': adaptive_threshold,
            'target_reputation': target_reputation,
            'visibility_weight': visibility_weight,
            'num_detections': num_detections,
            'reference_count': reference_count,
        }

    # ========== 对外接口（供jy调用） ==========
    
    def get_suspicious_vehicles(self, detections_dict, ego_position) -> List:
        """接口：hx → jy（疑似共谋车辆）"""
        return get_suspicious_vehicles(detections_dict, ego_position)
    
    def get_visibility_weights(self, detections_dict, target_positions) -> Dict:
        """接口：hx → jy（视野权重）"""
        return get_visibility_weights(detections_dict, target_positions)