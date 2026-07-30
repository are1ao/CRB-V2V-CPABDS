# -*- coding: utf-8 -*-
"""
完整对比实验脚本 - 改进信誉系统 vs 基线算法

攻击场景：
1. ghost_teleport  (瞬移攻击)
2. ghost_drift     (漂移攻击)
3. ghost_rev       (逆向幽灵攻击)
4. brake_burst     (刹车欺诈)
5. static_obstacle (静态假障碍注入，独立 episode_0000)

观测提取改为「数据驱动」：
优先用帧内 attack / vehicles 证据计算四维输入，避免仅按标签合成导致
DRAMBR / PlexeMDS 信誉曲线失真。本 CARLA 导出不含消息到达时间序列，
timestamp_error / message_frequency 在无时序攻击证据时保持近正常。

对比算法：
ImprovedDRAMBR / DRAMBR / PlexeMDS / StaticReputation / MajorityVoting / NoTrustFusion
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from improved_reputation_engine import ImprovedReputationManager, ReputationConfig
from baseline_algorithms import DRAMBR, PlexeMDS, StaticReputation, MajorityVoting, NoTrustFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 场景短名映射（用于结果展示）
SCENARIO_SHORT_NAMES = {
    "carla_export_n10_v200x50_ghost_teleport_pcd": "teleport",
    "carla_export_10k_n10_v200x50_ghost_drift": "drift",
    "carla_export_n10_v200x50_ghost_rev_pcd": "reverse",
    "carla_export_n10_v200x50_brake_burst": "brake",
    "static_obstacle": "obstacle",
}

# 独立 episode（不在 DataSet/<scenario>/episode_* 结构下）
STANDALONE_EPISODES = {
    "static_obstacle": ROOT / "episode_0000",
}

ALL_SCENARIOS = list(SCENARIO_SHORT_NAMES.keys())

# 算法期望的 CAM 频率中心（surrogate 内部写死 10Hz）
EXPECTED_MSG_FREQ = 10.0


def _merge_attack_params(meta: Dict) -> Dict:
    """兼容 DataSet 的 attack_params 与独立 episode 的 attack_config。"""
    params = {}
    params.update(meta.get("attack_config") or {})
    params.update(meta.get("attack_params") or {})
    return params


def resolve_attack_window(attack_params: Dict, attack_label: str, num_frames: int) -> Tuple[int, int]:
    """从 meta 参数解析攻击窗口（无帧扫描时的回退）。"""
    if "frame_start" in attack_params and "frame_end" in attack_params:
        return int(attack_params["frame_start"]), int(attack_params["frame_end"])

    if "burst_start" in attack_params:
        start = int(attack_params["burst_start"])
        duration = int(attack_params.get("burst_frames", 20))
        return start, min(start + duration, num_frames)

    mode = str(attack_params.get("violation_mode", "")).lower()
    label = str(attack_label or "").lower()

    if any(k in mode or k in label for k in ("static_obstacle", "obstacle")):
        return 0, num_frames

    if any(k in mode or k in label for k in ("teleport", "reverse", "ghost")):
        return 50, num_frames

    return 50, min(100, num_frames)


def _frame_injected_ids(frame: Dict) -> List[int]:
    """从单帧提取注入的假目标/障碍物 ID。"""
    att = frame.get("attack") or {}
    ids: List[int] = []
    for x in att.get("injected_ado_ids") or []:
        ids.append(int(x))
    oid = att.get("obstacle_id")
    if oid is not None:
        ids.append(int(oid))

    for k, v in (frame.get("vehicles") or {}).items():
        try:
            kid = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        if v.get("is_obstacle") or v.get("static"):
            ids.append(kid)
        elif kid < 0 or kid >= 90000:
            ids.append(kid)
    return sorted(set(ids))


def _frame_is_attack_active(frame: Dict, adversary: bool) -> bool:
    """判断该帧攻击车是否正在作恶（优先帧内证据）。"""
    if not adversary:
        return False
    att = frame.get("attack") or {}
    if "is_active_this_frame" in att:
        return bool(att.get("is_active_this_frame"))
    if att.get("obstacle_injected"):
        return True
    if _frame_injected_ids(frame):
        return True
    label = str(att.get("attack_label") or "").lower()
    if label in ("static_obstacle", "ghost_vehicle") and att.get("is_adversary"):
        # ghost 可能偶发空注入列表：仍视作攻击身份，但强度由误差决定
        return bool(att.get("pcd_injected") or att.get("kinematic_violation") or label == "static_obstacle")
    return False


def infer_attack_window_from_data(
    vehicle_data: Dict[Any, List[Dict]],
    adversary_ids: List[int],
    attack_params: Dict,
    attack_label: str,
    num_frames: int,
) -> Tuple[int, int]:
    """扫描攻击车帧，得到真实活跃窗口；扫不到则回退 meta。"""
    active = []
    adv_set = {int(a) for a in adversary_ids}
    for cav_id, frames in vehicle_data.items():
        if int(cav_id) not in adv_set:
            continue
        for i, fr in enumerate(frames):
            if _frame_is_attack_active(fr, True):
                active.append(i)
    if active:
        return int(min(active)), int(max(active)) + 1
    return resolve_attack_window(attack_params, attack_label, num_frames)


def _neighbor_see_count(
    vehicle_data: Dict[Any, List[Dict]],
    frame_idx: int,
    cav_id: int,
    injected: List[int],
) -> Tuple[int, int]:
    """有多少邻居也看到了同一批注入目标。"""
    if not injected:
        return 0, 0
    inj = set(injected)
    see, total = 0, 0
    for other_id, frames in vehicle_data.items():
        if int(other_id) == int(cav_id):
            continue
        if frame_idx >= len(frames):
            continue
        total += 1
        other_ids = set(_frame_injected_ids(frames[frame_idx]))
        # 邻居正常车辆列表也可能不含负 ID；用 vehicles key 判断可见性更稳
        other_vehs = set()
        for k in (frames[frame_idx].get("vehicles") or {}):
            try:
                other_vehs.add(int(k))
            except (TypeError, ValueError):
                pass
        if inj & (other_ids | other_vehs):
            see += 1
    return see, total


class DatasetLoader:
    """数据集加载器：支持 DataSet/<scenario>/episode_* 与独立 episode。"""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

    def resolve_episode_path(self, scenario: str, episode_idx: int) -> Path:
        if scenario in STANDALONE_EPISODES:
            path = STANDALONE_EPISODES[scenario]
            if not path.exists():
                raise FileNotFoundError(f"Standalone episode not found: {path}")
            return path
        episode_path = self.dataset_root / scenario / f"episode_{episode_idx:04d}"
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode not found: {episode_path}")
        return episode_path

    def load_episode(self, scenario: str, episode_idx: int) -> Dict:
        episode_path = self.resolve_episode_path(scenario, episode_idx)

        with open(episode_path / "meta.yaml", "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        vehicle_data = {}
        for cav_id in meta["cav_ids"]:
            vehicle_dir = episode_path / str(cav_id)
            frames = []
            for frame_file in sorted(vehicle_dir.glob("*.yaml")):
                with open(frame_file, "r", encoding="utf-8") as f:
                    frames.append(yaml.safe_load(f))
            vehicle_data[cav_id] = frames

        attack_params = _merge_attack_params(meta)
        return {
            "meta": meta,
            "vehicle_data": vehicle_data,
            "num_frames": meta["num_frames"],
            "adversary_ids": meta.get("adversary_cav_ids", []),
            "attack_params": attack_params,
            "attack_label": meta.get("attack_label", ""),
            "episode_path": str(episode_path),
        }

    def list_episodes(self, scenario: str) -> List[int]:
        if scenario in STANDALONE_EPISODES:
            return [0] if STANDALONE_EPISODES[scenario].exists() else []
        scenario_path = self.dataset_root / scenario
        if not scenario_path.exists():
            return []
        episodes = [int(p.name.split("_")[1]) for p in scenario_path.glob("episode_*")]
        return sorted(episodes)


class ObservationExtractor:
    """
    完全数据驱动观测提取器：优先使用帧内真实攻击证据。

    四维输出（DRAMBR / PlexeMDS / ImprovedDRAMBR 共用）：
      position_error, velocity_error, timestamp_error, message_frequency

    核心逻辑：
    1. 扫描当前帧 attack 字段、注入目标列表、邻居共识
    2. 基于真实位姿偏差、速度不一致、独占目标计算误差
    3. 仅在无帧内证据时回退 meta 标签（弱降级）
    """

    @staticmethod
    def compute_observation(
        vehicle_data: Dict[Any, List[Dict]],
        frame_idx: int,
        cav_id: int,
        adversary_ids: List[int],
        attack_params: Dict,
        attack_label: str,
        num_frames: int,
        attack_window: Tuple[int, int],
        rng: np.random.Generator,
    ) -> Dict:
        frames = vehicle_data.get(cav_id) or vehicle_data.get(int(cav_id))
        if not frames or frame_idx >= len(frames):
            return {}

        frame = frames[frame_idx]
        att = frame.get("attack") or {}
        ego_speed = float(frame.get("ego_speed", 0.0) or 0.0)
        ego_pos = frame.get("true_ego_pos") or frame.get("lidar_pose") or [0, 0, 0]

        frame_start, frame_end = attack_window
        is_adversary = int(cav_id) in {int(a) for a in adversary_ids}
        
        # ---- 帧内证据：优先判断当前帧是否实际攻击 ----
        is_attacking = _frame_is_attack_active(frame, is_adversary)
        injected = _frame_injected_ids(frame)
        
        # 回退：窗口内但缺帧内标记（ghost 偶发）
        if not is_attacking and is_adversary and frame_start <= frame_idx < frame_end:
            label_lower = str(attack_label or att.get("attack_label") or "").lower()
            # brake 依赖 burst_start 精确窗口；ghost 可宽松标记
            if "brake" not in label_lower and attack_params.get("mode") != "burst":
                is_attacking = True

        # ---- 基线噪声（正常车辆） ----
        position_error = abs(float(rng.normal(0.25, 0.10)))
        velocity_error = abs(float(rng.normal(0.15, 0.08)))
        timestamp_error = abs(float(rng.normal(0.04, 0.015)))
        message_frequency = float(EXPECTED_MSG_FREQ + rng.normal(0, 0.25))
        evidence = "benign_noise"

        # ---- 攻击分支 1：注入假目标（ghost / obstacle） ----
        if is_attacking and injected:
            see_count, total_neighbors = _neighbor_see_count(vehicle_data, frame_idx, cav_id, injected)
            solo_ratio = 1.0 - (see_count / max(1, total_neighbors))

            # 假目标位置
            fake_loc = att.get("fake_world_location") or att.get("world_location")
            if fake_loc is not None:
                dist = float(np.linalg.norm(np.asarray(fake_loc[:2], float) - np.asarray(ego_pos[:2], float)))
            else:
                dist = float(
                    att.get("insertion_gap_m") or att.get("obstacle_distance_m")
                    or attack_params.get("offset_m") or attack_params.get("base_offset_m") or 12.0
                )

            # 假目标速度
            fake_speed = att.get("fake_speed_reported")
            if fake_speed is None:
                oid = injected[0]
                veh = (frame.get("vehicles") or {}).get(oid) or (frame.get("vehicles") or {}).get(str(oid)) or {}
                fake_speed = float(veh.get("speed", 0.0) or 0.0)
            fake_speed = float(fake_speed)

            # 基线：独占假目标 → 位置/速度不一致强
            position_error = solo_ratio * (3.5 + min(14.0, 0.35 * dist))
            velocity_error = abs(ego_speed - fake_speed) * 0.30 + 1.2 * solo_ratio

            # 细分攻击模式增强
            mode = str(attack_params.get("violation_mode") or att.get("kinematic_violation") or "").lower()
            label_lower = str(attack_label or att.get("attack_label") or "").lower()

            if "teleport" in mode or "teleport" in label_lower:
                position_error = max(position_error, solo_ratio * (8.0 + 0.4 * dist))
                velocity_error = max(velocity_error, solo_ratio * (2.5 + abs(fake_speed) * 0.2))
            elif "drift" in mode or "drift" in label_lower:
                offset = float(attack_params.get("offset_m") or att.get("dist_ahead_m") or dist)
                reported = float(attack_params.get("reported_speed_mps") or fake_speed)
                position_error = max(position_error, solo_ratio * offset * 0.55)
                velocity_error = max(velocity_error, solo_ratio * (abs(ego_speed - reported) + 0.8))
            elif "reverse" in mode or "reverse" in label_lower or "rev" in label_lower:
                position_error = max(position_error, solo_ratio * (0.4 * dist + 4.0))
                velocity_error = max(velocity_error, solo_ratio * (abs(ego_speed) + abs(fake_speed) + 3.0))
            elif "obstacle" in label_lower or att.get("obstacle_injected"):
                position_error = max(position_error, solo_ratio * (4.0 + min(12.0, dist * 0.35)))
                velocity_error = max(velocity_error, abs(ego_speed - fake_speed) * 0.25 + 1.5 * solo_ratio)

            # 渐进：前几帧略缓
            if frame_idx >= frame_start:
                ramp = min(1.0, (frame_idx - frame_start + 1) / 6.0)
                position_error *= 0.55 + 0.45 * ramp
                velocity_error *= 0.55 + 0.45 * ramp

            evidence = "injected_object_solo" if solo_ratio > 0.7 else "injected_object_shared"

        # ---- 攻击分支 2：刹车欺诈（无假目标注入） ----
        elif is_attacking and (att.get("severity_reported") is not None or "brake" in str(attack_label).lower()):
            severity_reported = float(att.get("severity_reported") or attack_params.get("severity_mps2") or 8.0)
            actual_decel = float(att.get("target_actual_decel_mps2") or 0.0)
            gap = abs(severity_reported - actual_decel)

            position_error = 1.5 + 0.15 * gap
            velocity_error = 0.7 * severity_reported + 0.5 * gap
            evidence = "brake_fraud_report"

            if frame_idx >= frame_start:
                ramp = min(1.0, (frame_idx - frame_start + 1) / 6.0)
                position_error *= 0.55 + 0.45 * ramp
                velocity_error *= 0.55 + 0.45 * ramp

        # ---- 攻击分支 3：窗口内但缺帧证据（弱回退） ----
        elif is_attacking:
            progress = (frame_idx - frame_start) / max(1, frame_end - frame_start)
            ramp = min(1.0, (frame_idx - frame_start + 1) / 8.0)
            position_error = ramp * (5.0 + 3.0 * progress) + abs(rng.normal(0, 0.4))
            velocity_error = ramp * (2.0 + 1.5 * progress) + abs(rng.normal(0, 0.3))
            evidence = "label_fallback_weak"

        return {
            "position_error": float(max(0.0, position_error)),
            "velocity_error": float(max(0.0, velocity_error)),
            "timestamp_error": float(max(0.0, timestamp_error)),
            "message_frequency": float(max(0.0, message_frequency)),
            "is_attacking": bool(is_attacking),
            "is_adversary": bool(is_adversary),
            "frame_idx": frame_idx,
            "attack_window": [frame_start, frame_end],
            "evidence": evidence,
            "injected_ids": injected,
        }


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, dataset_root: str, output_dir: str, seed: int = 42):
        self.loader = DatasetLoader(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.seed = seed
        self.algorithms = {
            "ImprovedDRAMBR": None,
            "DRAMBR": DRAMBR(alpha=0.15, beta=0.10, anomaly_threshold=0.35, decay_factor=0.98),
            "PlexeMDS": PlexeMDS(window_size=20, trust_threshold=0.7),
            "StaticReputation": StaticReputation(fixed_value=0.5),
            "MajorityVoting": MajorityVoting(voting_threshold=0.6),
            "NoTrustFusion": NoTrustFusion(),
        }

    def run_single_episode(self, scenario: str, episode_idx: int) -> Dict:
        logger.info("运行 %s episode_%04d", scenario, episode_idx)
        rng = np.random.default_rng(self.seed + episode_idx)

        episode_data = self.loader.load_episode(scenario, episode_idx)
        meta = episode_data["meta"]
        vehicle_data = episode_data["vehicle_data"]
        adversary_ids = [int(v) for v in episode_data["adversary_ids"]]
        attack_params = dict(episode_data["attack_params"] or {})
        attack_label = episode_data["attack_label"]
        num_frames = int(episode_data["num_frames"])

        frame_start, frame_end = infer_attack_window_from_data(
            vehicle_data, adversary_ids, attack_params, attack_label, num_frames
        )
        attack_params["frame_start"] = frame_start
        attack_params["frame_end"] = frame_end
        meta = dict(meta)
        meta["attack_params"] = attack_params
        meta["attack_short_name"] = SCENARIO_SHORT_NAMES.get(scenario, scenario)

        vehicle_ids = [str(v) for v in meta["cav_ids"]]

        config = ReputationConfig()
        improved_manager = ImprovedReputationManager(config)
        for vid in vehicle_ids:
            improved_manager._get_meta(vid)
        self.algorithms["ImprovedDRAMBR"] = improved_manager

        for algo in self.algorithms.values():
            if hasattr(algo, "initialize_reputations"):
                algo.initialize_reputations(vehicle_ids, initial_value=1.0)  # ✅ 修复：初始信任所有车辆

        reputation_history = defaultdict(lambda: defaultdict(list))
        filter_weight_history = defaultdict(list)
        detection_events = defaultdict(dict)

        # ✅ 修复：在第一帧之前记录初始信誉值
        for cav_id in meta["cav_ids"]:
            vehicle_id = str(cav_id)
            for algo_name, algo in self.algorithms.items():
                if algo_name == "ImprovedDRAMBR":
                    initial_rep = improved_manager.get_trust_score(vehicle_id)
                elif hasattr(algo, "get_reputation"):
                    initial_rep = algo.get_reputation(vehicle_id)
                else:
                    initial_rep = 1.0
                reputation_history[algo_name][vehicle_id].append(float(initial_rep))

        for frame_idx in range(num_frames):
            for cav_id in meta["cav_ids"]:
                vehicle_id = str(cav_id)
                obs = ObservationExtractor.compute_observation(
                    vehicle_data,
                    frame_idx,
                    cav_id,
                    adversary_ids,
                    attack_params,
                    attack_label,
                    num_frames,
                    (frame_start, frame_end),
                    rng,
                )
                if not obs:
                    continue

                for algo_name, algo in self.algorithms.items():
                    if algo_name == "ImprovedDRAMBR":
                        is_consistent = obs["position_error"] < 2.0 and obs["velocity_error"] < 1.5
                        consistency_ratio = 1.0 - min(
                            1.0, (obs["position_error"] + obs["velocity_error"]) / 10.0
                        )
                        improved_manager.update_from_evidence(
                            vehicle_id,
                            is_consistent=is_consistent,
                            consistency_ratio=consistency_ratio,
                            direct_trust=consistency_ratio,
                        )
                        reputation = improved_manager.get_trust_score(vehicle_id)
                        filter_weight = improved_manager.get_filter_weight(vehicle_id)
                        filter_weight_history[vehicle_id].append(filter_weight)
                    else:
                        if hasattr(algo, "update_reputation"):
                            algo.update_reputation(vehicle_id, obs)
                        reputation = (
                            algo.get_reputation(vehicle_id)
                            if hasattr(algo, "get_reputation")
                            else 0.5
                        )

                    reputation_history[algo_name][vehicle_id].append(float(reputation))
                    if (
                        vehicle_id not in detection_events[algo_name]
                        and reputation < 0.5
                    ):
                        detection_events[algo_name][vehicle_id] = frame_idx

        return {
            "scenario": scenario,
            "attack_short_name": SCENARIO_SHORT_NAMES.get(scenario, scenario),
            "episode_idx": episode_idx,
            "meta": meta,
            "adversary_ids": [str(v) for v in adversary_ids],
            "reputation_history": {k: dict(v) for k, v in reputation_history.items()},
            "filter_weight_history": dict(filter_weight_history),
            "detection_events": {k: dict(v) for k, v in detection_events.items()},
            "num_frames": num_frames,
            "attack_window": [frame_start, frame_end],
            "obs_mode": "data_driven",
        }

    def run_scenario(self, scenario: str, num_episodes: int = 3) -> List[Dict]:
        episodes = self.loader.list_episodes(scenario)[:num_episodes]
        if not episodes:
            logger.warning("场景无可用 episode: %s", scenario)
            return []

        results = []
        for ep_idx in tqdm(episodes, desc=f"Running {SCENARIO_SHORT_NAMES.get(scenario, scenario)}"):
            try:
                results.append(self.run_single_episode(scenario, ep_idx))
            except Exception as e:
                logger.error("Error in %s episode %s: %s", scenario, ep_idx, e)
        return results

    def run_all_scenarios(self, scenarios: Optional[List[str]] = None, num_episodes: int = 3):
        scenarios = scenarios or ALL_SCENARIOS
        all_results = {}
        for scenario in scenarios:
            logger.info(
                "\n%s\n运行场景: %s (%s)\n%s",
                "=" * 60,
                scenario,
                SCENARIO_SHORT_NAMES.get(scenario, "?"),
                "=" * 60,
            )
            all_results[scenario] = self.run_scenario(scenario, num_episodes)

        output_file = self.output_dir / "experiment_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\n实验结果已保存至: %s", output_file)
        return all_results


def main():
    dataset_root = str(ROOT / "DataSet")
    output_dir = str(Path(__file__).resolve().parent / "results")

    num_episodes = int(os.environ.get("EPISODES", "3"))
    # 可选：SCENARIOS=teleport,obstacle 或完整目录名
    raw = os.environ.get("SCENARIOS", "").strip()
    if raw:
        wanted = [s.strip() for s in raw.split(",") if s.strip()]
        scenarios = []
        for w in wanted:
            if w in SCENARIO_SHORT_NAMES:
                scenarios.append(w)
            else:
                matched = [k for k, v in SCENARIO_SHORT_NAMES.items() if v == w or w in k]
                scenarios.extend(matched or [w])
    else:
        scenarios = ALL_SCENARIOS

    runner = ExperimentRunner(dataset_root, output_dir)
    runner.run_all_scenarios(scenarios, num_episodes=num_episodes)
    logger.info("\n实验完成！场景: %s", [SCENARIO_SHORT_NAMES.get(s, s) for s in scenarios])


if __name__ == "__main__":
    main()
