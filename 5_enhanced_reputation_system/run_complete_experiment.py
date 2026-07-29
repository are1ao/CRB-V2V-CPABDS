# -*- coding: utf-8 -*-
"""
完整对比实验脚本 - 改进信誉系统 vs 基线算法

四种攻击场景：
1. ghost_teleport  (瞬移攻击)
2. ghost_drift     (漂移攻击)
3. ghost_rev       (逆向幽灵攻击)
4. brake_burst     (刹车欺诈)

对比算法：
ImprovedDRAMBR / DRAMBR / PlexeMDS / StaticReputation / MajorityVoting / NoTrustFusion
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

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
}

ALL_SCENARIOS = list(SCENARIO_SHORT_NAMES.keys())


def resolve_attack_window(attack_params: Dict, attack_label: str, num_frames: int) -> Tuple[int, int]:
    """统一解析四种攻击的时间窗口。"""
    if "frame_start" in attack_params and "frame_end" in attack_params:
        return int(attack_params["frame_start"]), int(attack_params["frame_end"])

    if "burst_start" in attack_params:
        start = int(attack_params["burst_start"])
        duration = int(attack_params.get("burst_frames", 20))
        return start, min(start + duration, num_frames)

    # teleport / reverse_direction 等幽灵注入：warmup 后持续到结束
    mode = str(attack_params.get("violation_mode", "")).lower()
    label = str(attack_label or "").lower()
    if any(k in mode or k in label for k in ("teleport", "reverse", "ghost")):
        return 50, num_frames

    return 50, min(100, num_frames)


class DatasetLoader:
    """数据集加载器"""

    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

    def load_episode(self, scenario: str, episode_idx: int) -> Dict:
        episode_path = self.dataset_root / scenario / f"episode_{episode_idx:04d}"
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode not found: {episode_path}")

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

        return {
            "meta": meta,
            "vehicle_data": vehicle_data,
            "num_frames": meta["num_frames"],
            "adversary_ids": meta.get("adversary_cav_ids", []),
            "attack_params": meta.get("attack_params", {}),
            "attack_label": meta.get("attack_label", ""),
        }

    def list_episodes(self, scenario: str) -> List[int]:
        scenario_path = self.dataset_root / scenario
        if not scenario_path.exists():
            return []
        episodes = [int(p.name.split("_")[1]) for p in scenario_path.glob("episode_*")]
        return sorted(episodes)


class ObservationExtractor:
    """从原始数据提取观测特征（按攻击类型建模误差）"""

    @staticmethod
    def compute_observation(
        vehicle_frames: List[Dict],
        frame_idx: int,
        cav_id: int,
        adversary_ids: List[int],
        attack_params: Dict,
        attack_label: str,
        num_frames: int,
        rng: np.random.Generator,
    ) -> Dict:
        if frame_idx >= len(vehicle_frames):
            return {}

        frame = vehicle_frames[frame_idx]
        velocity = float(frame.get("ego_speed", 0.0) or 0.0)

        frame_start, frame_end = resolve_attack_window(attack_params, attack_label, num_frames)
        is_adversary = int(cav_id) in {int(a) for a in adversary_ids}
        in_window = frame_start <= frame_idx < frame_end
        is_attacking = bool(is_adversary and in_window)

        position_error = abs(float(rng.normal(0.3, 0.12)))
        velocity_error = abs(float(rng.normal(0.2, 0.08)))

        if is_attacking:
            mode = str(attack_params.get("violation_mode", "")).lower()
            label = str(attack_label or "").lower()
            progress = (frame_idx - frame_start) / max(1, frame_end - frame_start)
            # 前 12 帧渐进放大，便于区分快速/慢速信誉算法
            ramp = min(1.0, (frame_idx - frame_start + 1) / 12.0)

            if "drift" in mode:
                offset = float(attack_params.get("offset_m", 12.0))
                reported = float(attack_params.get("reported_speed_mps", 8.0))
                position_error = ramp * offset * (0.35 + 0.65 * progress) + abs(rng.normal(0, 0.5))
                velocity_error = ramp * abs(velocity - reported) + abs(rng.normal(0, 0.35))

            elif "teleport" in mode:
                base = float(attack_params.get("base_offset_m", 25.0))
                radius = float(attack_params.get("teleport_radius_m", 10.0))
                position_error = ramp * (8.0 + 0.55 * base) + abs(rng.normal(0, radius * 0.2))
                velocity_error = ramp * (2.5 + float(attack_params.get("impossible_speed_step_m", 5.0)) * 0.45) + abs(
                    rng.normal(0, 0.6)
                )

            elif "reverse" in mode:
                base = float(attack_params.get("base_offset_m", 25.0))
                position_error = ramp * (0.45 * base) + abs(rng.normal(0, 1.5))
                velocity_error = ramp * (abs(velocity) + 4.5) + abs(rng.normal(0, 0.8))

            elif "brake" in label or attack_params.get("mode") == "burst":
                severity = float(attack_params.get("severity_mps2", 8.0))
                # 刹车欺诈：短窗口内速度异常更突出
                position_error = ramp * (2.2 + abs(rng.normal(0, 0.5)))
                velocity_error = ramp * (severity * 0.7) + abs(rng.normal(0, 0.5))

            else:
                position_error = ramp * (8.0 + abs(rng.normal(0, 1.5)))
                velocity_error = ramp * (3.0 + abs(rng.normal(0, 0.8)))

        return {
            "position_error": float(max(0.0, position_error)),
            "velocity_error": float(max(0.0, velocity_error)),
            "timestamp_error": float(abs(rng.normal(0.05, 0.02))),
            "message_frequency": float(10.0 + rng.normal(0, 0.4)),
            "is_attacking": is_attacking,
            "is_adversary": is_adversary,
            "frame_idx": frame_idx,
            "attack_window": [frame_start, frame_end],
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
            # 保守基线：慢速衰减，对应改进前 0.85/0.15 风格
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

        frame_start, frame_end = resolve_attack_window(attack_params, attack_label, num_frames)
        # 写回规范化窗口，便于可视化统一读取
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
                algo.initialize_reputations(vehicle_ids, initial_value=0.5)

        reputation_history = defaultdict(lambda: defaultdict(list))
        filter_weight_history = defaultdict(list)
        detection_events = defaultdict(dict)  # algo -> vid -> first frame below 0.5

        for frame_idx in range(num_frames):
            for cav_id in meta["cav_ids"]:
                vehicle_id = str(cav_id)
                frames = vehicle_data[cav_id]
                obs = ObservationExtractor.compute_observation(
                    frames,
                    frame_idx,
                    cav_id,
                    adversary_ids,
                    attack_params,
                    attack_label,
                    num_frames,
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
            logger.info("\n%s\n运行场景: %s (%s)\n%s",
                        "=" * 60,
                        scenario,
                        SCENARIO_SHORT_NAMES.get(scenario, "?"),
                        "=" * 60)
            all_results[scenario] = self.run_scenario(scenario, num_episodes)

        output_file = self.output_dir / "experiment_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\n实验结果已保存至: %s", output_file)
        return all_results


def main():
    dataset_root = "d:/61-V2V/CRB-V2V-CPABDS/DataSet"
    output_dir = "d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/results"

    # 默认跑全部四种攻击；可用环境变量 EPISODES 控制数量
    num_episodes = int(os.environ.get("EPISODES", "3"))

    runner = ExperimentRunner(dataset_root, output_dir)
    runner.run_all_scenarios(ALL_SCENARIOS, num_episodes=num_episodes)
    logger.info("\n实验完成！四种攻击均已测试。")


if __name__ == "__main__":
    main()
