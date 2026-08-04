# -*- coding: utf-8 -*-
"""
物理一致性验证 - 空间冲突检测 + 视野权重
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


def cluster_by_world_coordinate(all_vehicle_detections: Dict, precision: float = 0.1) -> Dict:
    """按世界坐标聚类检测框"""
    clusters = defaultdict(list)
    
    for vehicle_id, det_data in all_vehicle_detections.items():
        for box in det_data.get('boxes', []):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            grid_x = round(center_x / precision) * precision
            grid_y = round(center_y / precision) * precision
            clusters[(grid_x, grid_y)].append(vehicle_id)
    
    return clusters


def check_mutual_visibility(vehicles: List, coord: Tuple, all_positions: Dict, 
                            max_distance: float = 200) -> bool:
    """检查多辆车是否都能看到目标位置"""
    for vid in vehicles:
        if vid not in all_positions:
            return False
        pos = all_positions[vid]
        distance = np.linalg.norm(np.array(pos[:2]) - np.array(coord))
        if distance > max_distance:
            return False
    return True


def detect_spatial_collusion(all_vehicle_detections: Dict, 
                              ego_position: Tuple,
                              max_distance: float = 200, 
                              min_vehicles: int = 2) -> Dict:
    """
    检测共谋攻击：多车在同一世界坐标报告同一幽灵
    
    Returns:
        dict: {
            'suspicious_vehicles': [149, 152],
            'conflict_coords': [(x1, y1), ...],
            'confidence': 0.95,
            'has_collusion': True/False
        }
    """
    all_positions = {}
    for vid, det_data in all_vehicle_detections.items():
        all_positions[vid] = det_data.get('position', (0, 0, 0))
    
    clusters = cluster_by_world_coordinate(all_vehicle_detections, precision=0.1)
    
    suspicious_vehicles = set()
    conflict_coords = []
    
    for coord, vehicles in clusters.items():
        if len(vehicles) >= min_vehicles:
            visible = check_mutual_visibility(vehicles, coord, all_positions, max_distance)
            if not visible:
                for vid in vehicles:
                    suspicious_vehicles.add(vid)
                conflict_coords.append(coord)
    
    confidence = min(0.95, 0.5 + len(conflict_coords) * 0.1 + len(suspicious_vehicles) * 0.02) if conflict_coords else 0.0
    
    return {
        'suspicious_vehicles': list(suspicious_vehicles),
        'conflict_coords': conflict_coords,
        'confidence': confidence,
        'has_collusion': len(suspicious_vehicles) > 0
    }


def calculate_visibility_weight(vehicle_position: Tuple, 
                                 target_position: Tuple,
                                 max_distance: float = 200) -> float:
    """计算车辆是否真的能看到目标位置"""
    pos = np.array(vehicle_position[:2])
    target = np.array(target_position)
    
    distance = np.linalg.norm(pos - target)
    if distance > max_distance:
        return 0.1
    
    distance_weight = max(0.3, 1.0 - distance / 300)
    return distance_weight


def compute_visibility_weights(all_vehicle_detections: Dict, 
                                target_positions: List) -> Dict:
    """计算所有车辆的视野权重"""
    weights = {}
    for vid, det_data in all_vehicle_detections.items():
        pos = det_data.get('position', (0, 0, 0))
        min_weight = 1.0
        for target in target_positions:
            w = calculate_visibility_weight(pos, target)
            min_weight = min(min_weight, w)
        weights[vid] = min_weight
    return weights


# ========== 对外接口 ==========

def get_suspicious_vehicles(all_vehicle_detections: Dict, 
                             ego_position: Tuple) -> List:
    """接口：hx → jy（疑似共谋车辆）"""
    result = detect_spatial_collusion(all_vehicle_detections, ego_position)
    return result['suspicious_vehicles']


def get_visibility_weights(all_vehicle_detections: Dict, 
                            target_positions: List) -> Dict:
    """接口：hx → jy（视野权重）"""
    return compute_visibility_weights(all_vehicle_detections, target_positions)