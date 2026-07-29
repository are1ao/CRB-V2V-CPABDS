# -*- coding: utf-8 -*-
"""
高级可视化系统 — 四种攻击 × 多算法对比（论文级图表）

输出：
  - comparison_<attack>.png      单攻击综合面板
  - cross_attack_summary.png     四攻击横向对比
  - metrics_heatmap.png          算法×攻击热力图
  - reputation_gallery.png       四攻击信誉曲线画廊
  - improvement_radar.png        改进效果雷达图
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# 全局样式（学术/专业风，避免常见 AI 紫渐变）
# ---------------------------------------------------------------------------
def _setup_fonts():
    """优先使用系统中文字体，避免缺字方框。"""
    from matplotlib import font_manager
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    return chosen


_ACTIVE_FONT = _setup_fonts()

ALGO_COLORS = {
    "ImprovedDRAMBR": "#C0392B",
    "DRAMBR": "#2471A3",
    "PlexeMDS": "#1E8449",
    "MajorityVoting": "#B9770E",
    "StaticReputation": "#6C3483",
    "NoTrustFusion": "#7F8C8D",
}

ALGO_ORDER = [
    "ImprovedDRAMBR",
    "DRAMBR",
    "PlexeMDS",
    "MajorityVoting",
    "StaticReputation",
    "NoTrustFusion",
]

ATTACK_META = {
    "teleport": {"title": "Ghost Teleport", "zh": "Teleport", "color": "#E74C3C"},
    "drift": {"title": "Ghost Drift", "zh": "Drift", "color": "#2980B9"},
    "reverse": {"title": "Ghost Reverse", "zh": "Reverse", "color": "#27AE60"},
    "brake": {"title": "Brake Burst", "zh": "Brake Fraud", "color": "#8E44AD"},
}

HARD_THR = 0.50
SOFT_THR = 0.70


def short_name(scenario: str, result: Optional[Dict] = None) -> str:
    if result and result.get("attack_short_name"):
        return result["attack_short_name"]
    mapping = {
        "ghost_teleport": "teleport",
        "teleport": "teleport",
        "ghost_drift": "drift",
        "drift": "drift",
        "ghost_rev": "reverse",
        "rev": "reverse",
        "reverse": "reverse",
        "brake_burst": "brake",
        "brake": "brake",
    }
    for key, val in mapping.items():
        if key in scenario:
            return val
    parts = scenario.rstrip("/").split("_")
    return parts[-1] if parts[-1] != "pcd" else parts[-2]


class AdvancedVisualizer:
    def __init__(self, results_file: str, output_dir: str):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        with open(self.results_file, encoding="utf-8") as f:
            self.results = json.load(f)
        self.algorithms = ALGO_ORDER
        self.colors = ALGO_COLORS

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    def _attack_window(self, result: Dict) -> Tuple[int, int]:
        if "attack_window" in result:
            return int(result["attack_window"][0]), int(result["attack_window"][1])
        attack = result.get("meta", {}).get("attack_params", {})
        return int(attack.get("frame_start", 50)), int(attack.get("frame_end", 100))

    def _detection_delay(self, curve: List[float], frame_start: int, frame_end: int) -> Optional[float]:
        """仅统计攻击开始后、窗口内首次跌破硬阈值的延迟。"""
        end = min(len(curve), max(frame_end + 30, frame_start + 1))  # 允许窗口后短暂余量
        for i in range(frame_start, end):
            if curve[i] < HARD_THR:
                return float(i - frame_start)
        return None

    @staticmethod
    def _rep_at(curve: List[float], idx: int) -> float:
        if not curve:
            return 0.5
        return float(curve[min(max(idx, 0), len(curve) - 1)])

    def collect_episode_metrics(self, result: Dict) -> List[Dict]:
        frame_start, frame_end = self._attack_window(result)
        # 评估点：攻击窗口结束帧（避免短时攻击结束后信誉回升导致指标失真）
        eval_idx = max(frame_start, min(frame_end - 1, result["num_frames"] - 1))
        adv_ids = set(result["adversary_ids"])
        rows = []
        for algo in self.algorithms:
            if algo not in result["reputation_history"]:
                continue
            reps = result["reputation_history"][algo]
            adv_curves, norm_curves, delays = [], [], []
            for vid, curve in reps.items():
                if not curve:
                    continue
                if vid in adv_ids:
                    adv_curves.append(curve)
                    d = self._detection_delay(curve, frame_start, frame_end)
                    if d is not None:
                        delays.append(d)
                else:
                    norm_curves.append(curve)

            if not adv_curves:
                continue

            adv_at_attack = float(np.mean([self._rep_at(c, eval_idx) for c in adv_curves]))
            adv_min = float(np.mean([
                min(c[frame_start:min(frame_end, len(c))] or [c[-1]]) for c in adv_curves
            ]))
            norm_at_attack = (
                float(np.mean([self._rep_at(c, eval_idx) for c in norm_curves]))
                if norm_curves else 0.8
            )
            delay = float(np.mean(delays)) if delays else float(
                max(1, min(frame_end, result["num_frames"]) - frame_start)
            )
            detected = len(delays) / max(1, len(adv_curves))
            sep = norm_at_attack - adv_at_attack

            slopes = []
            for c in adv_curves:
                s = max(frame_start, 0)
                e = min(frame_end - 1, len(c) - 1)
                if e > s:
                    slopes.append((c[s] - c[e]) / (e - s))
            drop_rate = float(np.mean(slopes)) if slopes else 0.0

            rows.append({
                "algorithm": algo,
                "delay": delay,
                "adv_rep": adv_at_attack,
                "adv_min": adv_min,
                "norm_rep": norm_at_attack,
                "separation": sep,
                "detect_rate": detected,
                "drop_rate": drop_rate,
                "episode": result["episode_idx"],
            })
        return rows

    def metrics_dataframe(self, scenario: Optional[str] = None) -> pd.DataFrame:
        rows = []
        scenarios = [scenario] if scenario else list(self.results.keys())
        for sc in scenarios:
            for res in self.results.get(sc, []):
                for row in self.collect_episode_metrics(res):
                    row["scenario"] = sc
                    row["attack"] = short_name(sc, res)
                    rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 单攻击综合面板
    # ------------------------------------------------------------------
    def plot_comprehensive_comparison(self, scenario: str):
        results = self.results.get(scenario, [])
        if not results:
            return

        attack = short_name(scenario, results[0])
        meta = ATTACK_META.get(attack, {"title": attack.upper(), "zh": attack, "color": "#34495E"})
        df = self.metrics_dataframe(scenario)

        fig = plt.figure(figsize=(18, 14), facecolor="#FAFBFC")
        gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32,
                      left=0.06, right=0.97, top=0.90, bottom=0.06)

        ax_curve = fig.add_subplot(gs[0, :2])
        ax_filter = fig.add_subplot(gs[0, 2])
        ax_delay = fig.add_subplot(gs[1, 0])
        ax_sep = fig.add_subplot(gs[1, 1])
        ax_final = fig.add_subplot(gs[1, 2])
        ax_band = fig.add_subplot(gs[2, :2])
        ax_table = fig.add_subplot(gs[2, 2])

        self._plot_multi_algo_curves(ax_curve, results[0])
        self._plot_filter_weight(ax_filter, results[0])
        self._plot_delay_bars(ax_delay, df)
        self._plot_separation_bars(ax_sep, df)
        self._plot_final_rep_grouped(ax_final, df)
        self._plot_reputation_band(ax_band, results)
        self._plot_summary_table(ax_table, df)

        fig.suptitle(
            f"Enhanced Reputation System  |  {meta['title']}",
            fontsize=16, fontweight="bold", color="#1C2833", y=0.97,
        )
        fig.text(
            0.5, 0.935,
            f"Scenario: {scenario}   ·   Episodes: {len(results)}   ·   "
            f"Hard filter={HARD_THR:.2f}  Soft filter={SOFT_THR:.2f}",
            ha="center", fontsize=9, color="#5D6D7E",
        )

        out = self.output_dir / f"comparison_{attack}.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    def _plot_multi_algo_curves(self, ax, result: Dict):
        adv_ids = set(result["adversary_ids"])
        fs, fe = self._attack_window(result)
        rep_hist = result["reputation_history"]

        # 攻击窗口
        ax.axvspan(fs, fe, alpha=0.12, color="#E74C3C", zorder=0)
        ax.axvline(fs, color="#E74C3C", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(HARD_THR, color="#1C2833", ls=":", lw=1.3, label=f"Hard {HARD_THR}")
        ax.axhline(SOFT_THR, color="#7F8C8D", ls=":", lw=1.1, label=f"Soft {SOFT_THR}")

        # 攻击车：主要算法实线；正常车均值虚线（Improved only）
        for algo in ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]:
            if algo not in rep_hist:
                continue
            for vid, curve in rep_hist[algo].items():
                if vid not in adv_ids:
                    continue
                lw = 2.6 if algo == "ImprovedDRAMBR" else 1.6
                ls = "-" if algo == "ImprovedDRAMBR" else "--"
                ax.plot(curve, color=self.colors[algo], lw=lw, ls=ls,
                        label=f"{algo} (attacker)", alpha=0.95)

        if "ImprovedDRAMBR" in rep_hist:
            norms = [c for vid, c in rep_hist["ImprovedDRAMBR"].items() if vid not in adv_ids]
            if norms:
                arr = np.array(norms)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, color="#1ABC9C", lw=2.0, label="ImprovedDRAMBR (benign mean)")
                ax.fill_between(x, mean - std, mean + std, color="#1ABC9C", alpha=0.15)

        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(0, result["num_frames"])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Reputation Score")
        ax.set_title("Attacker Reputation Trajectory", fontweight="bold", loc="left")
        ax.legend(loc="upper right", framealpha=0.92, ncol=1)
        ax.xaxis.set_major_locator(MaxNLocator(8))

    def _plot_filter_weight(self, ax, result: Dict):
        fw = result.get("filter_weight_history", {})
        adv_ids = set(result["adversary_ids"])
        fs, fe = self._attack_window(result)
        ax.axvspan(fs, fe, alpha=0.12, color="#E74C3C")

        plotted = False
        for vid, curve in fw.items():
            if vid not in adv_ids:
                continue
            ax.plot(curve, color="#C0392B", lw=2.2, label=f"Attacker {vid}")
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No filter-weight data", ha="center", va="center",
                    transform=ax.transAxes, color="#7F8C8D")
        else:
            ax.axhline(0.3, color="#7F8C8D", ls=":", lw=1.0)
            ax.set_ylim(-0.05, 1.08)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Filter Weight")
        ax.set_title("Multi-level Filter Weight", fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8)

    def _plot_delay_bars(self, ax, df: pd.DataFrame):
        if df.empty:
            ax.axis("off")
            return
        g = df.groupby("algorithm")["delay"].agg(["mean", "std"]).reindex(ALGO_ORDER).dropna()
        y = np.arange(len(g))
        colors = [self.colors.get(a, "#95A5A6") for a in g.index]
        bars = ax.barh(y, g["mean"], xerr=g["std"].fillna(0), color=colors,
                       alpha=0.88, height=0.65, error_kw={"elinewidth": 1.0, "capsize": 3})
        for i, (m, s) in enumerate(zip(g["mean"], g["std"].fillna(0))):
            ax.text(m + max(s, 0) + 1.5, i, f"{m:.1f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index)
        ax.set_xlabel("Detection Delay (frames)")
        ax.set_title("Detection Delay ↓", fontweight="bold", loc="left")
        # highlight best
        if len(g):
            best = g["mean"].idxmin()
            idx = list(g.index).index(best)
            bars[idx].set_edgecolor("#1C2833")
            bars[idx].set_linewidth(1.8)

    def _plot_separation_bars(self, ax, df: pd.DataFrame):
        if df.empty:
            ax.axis("off")
            return
        g = df.groupby("algorithm")["separation"].agg(["mean", "std"]).reindex(ALGO_ORDER).dropna()
        x = np.arange(len(g))
        colors = [self.colors.get(a, "#95A5A6") for a in g.index]
        ax.bar(x, g["mean"], yerr=g["std"].fillna(0), color=colors, alpha=0.88,
               width=0.7, error_kw={"elinewidth": 1.0, "capsize": 3})
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("Improved", "Imp.\n") for a in g.index],
                           rotation=0, fontsize=7)
        ax.set_ylabel("Rep Separation (benign − attacker)")
        ax.set_title("Reputation Separation ↑", fontweight="bold", loc="left")
        ax.axhline(0, color="#BDC3C7", lw=0.8)

    def _plot_final_rep_grouped(self, ax, df: pd.DataFrame):
        if df.empty:
            ax.axis("off")
            return
        focus = ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]
        sub = df[df["algorithm"].isin(focus)]
        means = sub.groupby("algorithm")[["adv_rep", "norm_rep"]].mean().reindex(focus).dropna()
        x = np.arange(len(means))
        w = 0.35
        ax.bar(x - w / 2, means["adv_rep"], w, label="Attacker", color="#C0392B", alpha=0.9)
        ax.bar(x + w / 2, means["norm_rep"], w, label="Benign", color="#1ABC9C", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(means.index, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Final Reputation")
        ax.set_title("Reputation at Attack-Window End", fontweight="bold", loc="left")
        ax.legend(fontsize=7, loc="lower left")
        ax.axhline(HARD_THR, color="#1C2833", ls=":", lw=1.0)

    def _plot_reputation_band(self, ax, results: List[Dict]):
        """跨 episode 的攻击车信誉均值±std（Improved vs DRAMBR）"""
        fs, fe = self._attack_window(results[0])
        ax.axvspan(fs, fe, alpha=0.10, color="#E74C3C")

        for algo, alpha_fill in [("ImprovedDRAMBR", 0.22), ("DRAMBR", 0.12)]:
            curves = []
            for res in results:
                adv = set(res["adversary_ids"])
                hist = res["reputation_history"].get(algo, {})
                for vid, c in hist.items():
                    if vid in adv and c:
                        curves.append(c)
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            arr = np.array([c[:min_len] for c in curves])
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(min_len)
            ax.plot(x, mean, color=self.colors[algo], lw=2.4, label=f"{algo} mean")
            ax.fill_between(x, mean - std, mean + std, color=self.colors[algo], alpha=alpha_fill)

        ax.axhline(HARD_THR, color="#1C2833", ls=":", lw=1.2)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Attacker Reputation")
        ax.set_title("Cross-Episode Band (mean ± std)", fontweight="bold", loc="left")
        ax.legend(loc="upper right")

    def _plot_summary_table(self, ax, df: pd.DataFrame):
        ax.axis("off")
        if df.empty or "ImprovedDRAMBR" not in df["algorithm"].values:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return

        focus = ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]
        rows_txt = []
        for algo in focus:
            sub = df[df["algorithm"] == algo]
            if sub.empty:
                continue
            rows_txt.append([
                algo,
                f"{sub['delay'].mean():.1f}",
                f"{sub['adv_rep'].mean():.3f}",
                f"{sub['norm_rep'].mean():.3f}",
                f"{sub['separation'].mean():.3f}",
                f"{sub['detect_rate'].mean()*100:.0f}%",
            ])

        # 改进幅度
        imp = df[df["algorithm"] == "ImprovedDRAMBR"]
        base = df[df["algorithm"] == "DRAMBR"]
        note = ""
        if not imp.empty and not base.empty and base["delay"].mean() > 0:
            gain = (1 - imp["delay"].mean() / base["delay"].mean()) * 100
            note = f"Delay ↓ {gain:.1f}% vs DRAMBR"

        table = ax.table(
            cellText=rows_txt,
            colLabels=["Algorithm", "Delay", "AdvRep", "NormRep", "Sep", "Detect%"],
            cellLoc="center",
            loc="center",
            colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)
        for j in range(6):
            table[(0, j)].set_facecolor("#1C2833")
            table[(0, j)].set_text_props(color="white", fontweight="bold")
        if rows_txt:
            for j in range(6):
                table[(1, j)].set_facecolor("#FADBD8")

        ax.set_title(f"Metrics Summary\n{note}", fontweight="bold", loc="left", fontsize=11)

    # ------------------------------------------------------------------
    # 跨攻击总览
    # ------------------------------------------------------------------
    def plot_cross_attack_summary(self):
        df = self.metrics_dataframe()
        if df.empty:
            return

        fig = plt.figure(figsize=(18, 11), facecolor="#FAFBFC")
        gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        attacks = [a for a in ["teleport", "drift", "reverse", "brake"] if a in set(df["attack"])]
        focus = ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS", "MajorityVoting"]

        # 1) 分组延迟
        self._grouped_metric(ax1, df, attacks, focus, "delay",
                             "Detection Delay by Attack (↓ better)", "Frames")
        # 2) 分组分离度
        self._grouped_metric(ax2, df, attacks, focus, "separation",
                             "Reputation Separation by Attack (↑ better)", "Sep. Score")
        # 3) 改进百分比
        self._improvement_bars(ax3, df, attacks)
        # 4) 检测率
        self._grouped_metric(ax4, df, attacks, focus, "detect_rate",
                             "Detection Rate by Attack (↑ better)", "Detect %",
                             scale=100)

        fig.suptitle(
            "Cross-Attack Benchmark  ·  Enhanced Reputation vs Baselines",
            fontsize=16, fontweight="bold", color="#1C2833",
        )
        fig.text(0.5, 0.935, "Four V2V attack scenarios evaluated under identical protocol",
                 ha="center", fontsize=9, color="#5D6D7E")

        out = self.output_dir / "cross_attack_summary.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    def _grouped_metric(self, ax, df, attacks, focus, metric, title, ylabel,
                        scale=1.0, ylabel_override=None):
        width = 0.8 / max(1, len(focus))
        x = np.arange(len(attacks))
        for i, algo in enumerate(focus):
            means, stds = [], []
            for atk in attacks:
                sub = df[(df["attack"] == atk) & (df["algorithm"] == algo)][metric]
                means.append(sub.mean() * scale if len(sub) else 0)
                stds.append(sub.std() * scale if len(sub) > 1 else 0)
            ax.bar(x + i * width - 0.4 + width / 2, means, width * 0.92,
                   yerr=stds, label=algo, color=self.colors[algo], alpha=0.9,
                   error_kw={"elinewidth": 0.8, "capsize": 2})
        ax.set_xticks(x)
        ax.set_xticklabels([
            ATTACK_META.get(a, {}).get("title", a) for a in attacks
        ], fontsize=8)
        ax.set_ylabel(ylabel_override or ylabel)
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)

    def _improvement_bars(self, ax, df, attacks):
        labels, delay_gains, sep_gains = [], [], []
        for atk in attacks:
            imp = df[(df["attack"] == atk) & (df["algorithm"] == "ImprovedDRAMBR")]
            base = df[(df["attack"] == atk) & (df["algorithm"] == "DRAMBR")]
            if imp.empty or base.empty:
                continue
            bd, id_ = base["delay"].mean(), imp["delay"].mean()
            delay_gains.append((1 - id_ / bd) * 100 if bd > 0 else 0)
            sep_gains.append(
                ((imp["separation"].mean() - base["separation"].mean())
                 / max(abs(base["separation"].mean()), 1e-6)) * 100
            )
            labels.append(ATTACK_META.get(atk, {}).get("title", atk))

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, delay_gains, w, label="Delay reduction %", color="#C0392B", alpha=0.9)
        ax.bar(x + w / 2, sep_gains, w, label="Separation gain %", color="#1ABC9C", alpha=0.9)
        ax.axhline(0, color="#BDC3C7", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Improvement vs DRAMBR (%)")
        ax.set_title("ImprovedDRAMBR Relative Gains", fontweight="bold", loc="left")
        ax.legend(fontsize=8)

        for i, (d, s) in enumerate(zip(delay_gains, sep_gains)):
            ax.text(i - w / 2, d + (1 if d >= 0 else -3), f"{d:.0f}%", ha="center", fontsize=7)
            ax.text(i + w / 2, s + (1 if s >= 0 else -3), f"{s:.0f}%", ha="center", fontsize=7)

    # ------------------------------------------------------------------
    # 热力图
    # ------------------------------------------------------------------
    def plot_metrics_heatmap(self):
        df = self.metrics_dataframe()
        if df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="#FAFBFC")
        for ax, metric, title, cmap, fmt in [
            (axes[0], "delay", "Detection Delay (frames, ↓ better)", "YlOrRd", ".1f"),
            (axes[1], "separation", "Reputation Separation (↑ better)", "YlGn", ".3f"),
        ]:
            pivot = df.pivot_table(index="algorithm", columns="attack",
                                   values=metric, aggfunc="mean")
            # 规范列顺序
            cols = [c for c in ["teleport", "drift", "reverse", "brake"] if c in pivot.columns]
            rows = [r for r in ALGO_ORDER if r in pivot.index]
            pivot = pivot.loc[rows, cols]
            pivot.columns = [ATTACK_META.get(c, {}).get("title", c) for c in pivot.columns]

            sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                        linewidths=0.6, linecolor="white",
                        cbar_kws={"shrink": 0.8})
            ax.set_title(title, fontweight="bold", loc="left")
            ax.set_xlabel("")
            ax.set_ylabel("")

        fig.suptitle("Algorithm × Attack Performance Heatmap",
                     fontsize=15, fontweight="bold", color="#1C2833")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = self.output_dir / "metrics_heatmap.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # 四攻击信誉画廊
    # ------------------------------------------------------------------
    def plot_reputation_gallery(self):
        scenarios = list(self.results.keys())
        if not scenarios:
            return

        n = len(scenarios)
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor="#FAFBFC")
        axes = axes.flatten()

        for i, sc in enumerate(scenarios[:4]):
            ax = axes[i]
            results = self.results[sc]
            if not results:
                ax.axis("off")
                continue
            attack = short_name(sc, results[0])
            meta = ATTACK_META.get(attack, {"title": attack, "zh": attack, "color": "#34495E"})
            fs, fe = self._attack_window(results[0])
            ax.axvspan(fs, fe, alpha=0.12, color=meta["color"])

            for algo in ["ImprovedDRAMBR", "DRAMBR"]:
                curves = []
                for res in results:
                    adv = set(res["adversary_ids"])
                    for vid, c in res["reputation_history"].get(algo, {}).items():
                        if vid in adv and c:
                            curves.append(c)
                if not curves:
                    continue
                L = min(len(c) for c in curves)
                arr = np.array([c[:L] for c in curves])
                mean, std = arr.mean(0), arr.std(0)
                x = np.arange(L)
                ax.plot(x, mean, color=self.colors[algo],
                        lw=2.5 if algo == "ImprovedDRAMBR" else 1.8,
                        ls="-" if algo == "ImprovedDRAMBR" else "--",
                        label=algo)
                ax.fill_between(x, mean - std, mean + std, color=self.colors[algo], alpha=0.15)

            ax.axhline(HARD_THR, color="#1C2833", ls=":", lw=1.1)
            ax.set_ylim(-0.05, 1.08)
            ax.set_title(meta["title"], fontweight="bold",
                         color=meta["color"], loc="left")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Attacker Reputation")
            ax.legend(fontsize=8, loc="upper right")

        for j in range(len(scenarios), 4):
            axes[j].axis("off")

        fig.suptitle("Attacker Reputation Drop Across Four Attacks",
                     fontsize=15, fontweight="bold", color="#1C2833")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = self.output_dir / "reputation_gallery.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # 雷达图
    # ------------------------------------------------------------------
    def plot_improvement_radar(self):
        df = self.metrics_dataframe()
        if df.empty:
            return

        # 指标：延迟(反转)、分离度、检测率、下降斜率、良性信誉保持
        metrics_keys = ["delay_inv", "separation", "detect_rate", "drop_rate", "norm_rep"]
        labels = ["Fast Detect", "Separation", "Detect Rate", "Drop Speed", "Benign Stability"]

        def agg(algo):
            sub = df[df["algorithm"] == algo]
            if sub.empty:
                return None
            delay = sub["delay"].mean()
            return {
                "delay_inv": 1.0 / (1.0 + delay / 20.0),
                "separation": max(0, sub["separation"].mean()),
                "detect_rate": sub["detect_rate"].mean(),
                "drop_rate": max(0, sub["drop_rate"].mean() * 20),  # scale
                "norm_rep": sub["norm_rep"].mean(),
            }

        algos = ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]
        series = {a: agg(a) for a in algos}
        series = {k: v for k, v in series.items() if v}

        # 归一化到 [0,1]
        all_vals = {k: [series[a][k] for a in series] for k in metrics_keys}
        norms = {}
        for a, vals in series.items():
            norms[a] = []
            for k in metrics_keys:
                mx = max(all_vals[k]) or 1.0
                mn = min(all_vals[k])
                if mx == mn:
                    norms[a].append(0.7)
                else:
                    norms[a].append((vals[k] - mn) / (mx - mn) * 0.85 + 0.15)

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), facecolor="#FAFBFC")
        for algo, vals in norms.items():
            data = vals + vals[:1]
            ax.plot(angles, data, color=self.colors[algo], lw=2.2, label=algo)
            ax.fill(angles, data, color=self.colors[algo], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_yticklabels([])
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
        ax.set_title("Multi-Metric Capability Radar\n(normalized across algorithms)",
                     fontweight="bold", pad=20, color="#1C2833")

        out = self.output_dir / "improvement_radar.png"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    # ------------------------------------------------------------------
    def generate_all_visualizations(self):
        print("\nGenerating professional visualizations...")
        for scenario in self.results.keys():
            print(f"  · Panel: {short_name(scenario)}")
            self.plot_comprehensive_comparison(scenario)
        print("  · Cross-attack summary")
        self.plot_cross_attack_summary()
        print("  · Heatmap")
        self.plot_metrics_heatmap()
        print("  · Reputation gallery")
        self.plot_reputation_gallery()
        print("  · Radar")
        self.plot_improvement_radar()
        print("\nVisualization complete!")
        print(f"Output dir: {self.output_dir.resolve()}")


if __name__ == "__main__":
    visualizer = AdvancedVisualizer(
        "d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/results/experiment_results.json",
        "d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/visualizations",
    )
    visualizer.generate_all_visualizations()
