# -*- coding: utf-8 -*-
"""薄封装：用改造后的主实验管线试跑独立 episode_0000 (static_obstacle)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_complete_experiment import ExperimentRunner, ROOT, ObservationExtractor
from run_complete_experiment import infer_attack_window_from_data

OUT = Path(__file__).resolve().parent / "results" / "try_episode_0000"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(str(ROOT / "DataSet"), str(OUT))
    result = runner.run_single_episode("static_obstacle", 0)

    # 抽样打印攻击车观测，确认 data-driven 生效
    ep = runner.loader.load_episode("static_obstacle", 0)
    adv = int(ep["adversary_ids"][0])
    window = tuple(result["attack_window"])
    rng = np.random.default_rng(0)
    print(f"attack={ep['attack_label']} adv={adv} window={window} obs_mode={result['obs_mode']}")
    for fi in (0, 50, 100, 150, 199):
        obs = ObservationExtractor.compute_observation(
            ep["vehicle_data"], fi, adv, ep["adversary_ids"],
            ep["attack_params"], ep["attack_label"], ep["num_frames"], window, rng,
        )
        print(
            f"  f={fi:3d} pos={obs['position_error']:.2f} vel={obs['velocity_error']:.2f} "
            f"ts={obs['timestamp_error']:.3f} freq={obs['message_frequency']:.2f} "
            f"attacking={obs['is_attacking']} evidence={obs['evidence']}"
        )

    adv_s = str(adv)
    print("算法摘要 (攻击车):")
    for algo, hist in result["reputation_history"].items():
        if algo not in ("ImprovedDRAMBR", "DRAMBR", "PlexeMDS"):
            continue
        curve = hist[adv_s]
        det = result["detection_events"].get(algo, {}).get(adv_s)
        print(f"  {algo:16s} final={curve[-1]:.3f} min={min(curve):.3f} detect@{det}")

    slim = {
        "scenario": result["scenario"],
        "attack_short_name": result["attack_short_name"],
        "attack_window": result["attack_window"],
        "obs_mode": result["obs_mode"],
        "adversary_ids": result["adversary_ids"],
        "detection_events": {
            k: v for k, v in result["detection_events"].items()
            if k in ("ImprovedDRAMBR", "DRAMBR", "PlexeMDS")
        },
        "adv_curves": {
            k: result["reputation_history"][k][adv_s]
            for k in ("ImprovedDRAMBR", "DRAMBR", "PlexeMDS")
        },
    }
    out_json = OUT / "comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    print(f"[SAVE] {out_json}")

    # 写入标准 results 结构，便于 advanced_visualization 直接读
    std = {"static_obstacle": [result]}
    with open(OUT / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(std, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = {"ImprovedDRAMBR": "#C0392B", "DRAMBR": "#2471A3", "PlexeMDS": "#1E8449"}
        for name, curve in slim["adv_curves"].items():
            ax.plot(curve, label=name, color=colors[name], lw=2)
        ax.axhline(0.5, color="#888", ls="--", lw=1)
        ax.set_title("static_obstacle / episode_0000 (data-driven)")
        ax.set_xlabel("frame")
        ax.set_ylabel("reputation")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        png = OUT / "reputation_curves.png"
        fig.tight_layout()
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"[SAVE] {png}")
    except Exception as e:  # noqa: BLE001
        print(f"绑图跳过: {e}")


if __name__ == "__main__":
    main()
