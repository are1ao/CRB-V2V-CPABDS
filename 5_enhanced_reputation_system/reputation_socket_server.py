# -*- coding: utf-8 -*-
"""
Reputation Socket Server for VEINS Integration
监听8888端口，接收VEINS发来的验证上报JSON，调用现有信誉算法处理，返回信誉更新结果
"""

import socket
import json
import logging
import threading
from typing import Dict, Optional
from datetime import datetime

from improved_reputation_engine import ImprovedReputationManager, ReputationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReputationSocketServer:
    """信誉系统Socket服务器"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
        # 初始化信誉管理器
        config = ReputationConfig()
        self.reputation_manager = ImprovedReputationManager(config)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'connected_vehicles': set(),
        }
        
        logger.info(f"初始化信誉服务器: {host}:{port}")
    
    def start(self):
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"✓ 信誉服务器启动成功: {self.host}:{self.port}")
            logger.info("等待VEINS客户端连接...")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    logger.info(f"新连接: {client_address}")
                    
                    # 为每个客户端创建新线程
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"接受连接错误: {e}")
        
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise
        finally:
            self.stop()
    
    def handle_client(self, client_socket: socket.socket, client_address):
        """处理客户端连接"""
        try:
            while self.running:
                # 接收数据（假设每条消息以换行符结尾）
                data = b''
                while not data.endswith(b'\n'):
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                
                if not data:
                    logger.info(f"客户端断开: {client_address}")
                    break
                
                # 解析JSON请求
                try:
                    request = json.loads(data.decode('utf-8').strip())
                    response = self.process_request(request)
                    
                    # 发送响应
                    response_data = json.dumps(response).encode('utf-8') + b'\n'
                    client_socket.sendall(response_data)
                    
                    self.stats['total_requests'] += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                    error_response = {
                        'status': 'error',
                        'message': f'Invalid JSON: {str(e)}'
                    }
                    client_socket.sendall(json.dumps(error_response).encode('utf-8') + b'\n')
        
        except Exception as e:
            logger.error(f"客户端处理错误 {client_address}: {e}")
        finally:
            client_socket.close()
    
    def process_request(self, request: Dict) -> Dict:
        """
        处理信誉更新请求
        
        请求格式:
        {
            "type": "update_reputation",
            "vehicle_id": "147",
            "observation": {
                "position_error": 5.2,
                "velocity_error": 2.1,
                "timestamp_error": 0.1,
                "message_frequency": 10.0,
                "frame_idx": 75
            },
            "timestamp": 1234567890.123
        }
        
        响应格式:
        {
            "status": "success",
            "vehicle_id": "147",
            "old_score": 0.85,
            "new_score": 0.72,
            "filter_weight": 1.0,
            "warning_level": 0,
            "first_offense": false,
            "timestamp": 1234567890.124
        }
        """
        try:
            request_type = request.get('type')
            
            if request_type == 'update_reputation':
                return self._handle_update_reputation(request)
            
            elif request_type == 'get_reputation':
                return self._handle_get_reputation(request)
            
            elif request_type == 'get_filter_weight':
                return self._handle_get_filter_weight(request)
            
            elif request_type == 'get_statistics':
                return self._handle_get_statistics(request)
            
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown request type: {request_type}'
                }
        
        except Exception as e:
            logger.error(f"请求处理错误: {e}")
            self.stats['failed_updates'] += 1
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _handle_update_reputation(self, request: Dict) -> Dict:
        """处理信誉更新请求"""
        vehicle_id = str(request.get('vehicle_id'))
        observation = request.get('observation', {})
        
        # 记录连接的车辆
        self.stats['connected_vehicles'].add(vehicle_id)
        
        # 计算一致性
        pos_err = observation.get('position_error', 0.0)
        vel_err = observation.get('velocity_error', 0.0)
        
        is_consistent = pos_err < 2.0 and vel_err < 1.5
        consistency_ratio = 1.0 - min(1.0, (pos_err + vel_err) / 10.0)
        
        # 更新信誉
        update_result = self.reputation_manager.update_from_evidence(
            vehicle_id=vehicle_id,
            is_consistent=is_consistent,
            consistency_ratio=consistency_ratio,
            direct_trust=consistency_ratio,
        )
        
        # 获取过滤权重
        filter_weight = self.reputation_manager.get_filter_weight(vehicle_id)
        
        self.stats['successful_updates'] += 1
        
        # 构建响应
        response = {
            'status': 'success',
            'vehicle_id': vehicle_id,
            'old_score': update_result['old_score'],
            'new_score': update_result['new_score'],
            'filter_weight': filter_weight,
            'warning_level': update_result['warning_level'],
            'first_offense': update_result.get('first_offense', False),
            'early_warning': update_result.get('early_warning', False),
            'timestamp': datetime.now().timestamp()
        }
        
        return response
    
    def _handle_get_reputation(self, request: Dict) -> Dict:
        """获取车辆信誉"""
        vehicle_id = str(request.get('vehicle_id'))
        reputation = self.reputation_manager.get_trust_score(vehicle_id)
        
        return {
            'status': 'success',
            'vehicle_id': vehicle_id,
            'reputation': reputation,
            'timestamp': datetime.now().timestamp()
        }
    
    def _handle_get_filter_weight(self, request: Dict) -> Dict:
        """获取过滤权重"""
        vehicle_id = str(request.get('vehicle_id'))
        filter_weight = self.reputation_manager.get_filter_weight(vehicle_id)
        
        return {
            'status': 'success',
            'vehicle_id': vehicle_id,
            'filter_weight': filter_weight,
            'timestamp': datetime.now().timestamp()
        }
    
    def _handle_get_statistics(self, request: Dict) -> Dict:
        """获取统计信息"""
        rep_stats = self.reputation_manager.get_statistics()
        
        return {
            'status': 'success',
            'statistics': {
                **rep_stats,
                'total_requests': self.stats['total_requests'],
                'successful_updates': self.stats['successful_updates'],
                'failed_updates': self.stats['failed_updates'],
                'connected_vehicles': len(self.stats['connected_vehicles']),
            },
            'timestamp': datetime.now().timestamp()
        }
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("服务器已停止")


def main():
    """主函数"""
    server = ReputationSocketServer(host='0.0.0.0', port=8888)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("\n收到停止信号...")
        server.stop()
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        raise


if __name__ == "__main__":
    main()
