"""
Plot experiment results from results.tsv (multi-panel progress chart).

Usage:
    python plot_results.py              # show plot
    python plot_results.py --save       # save to progress.png
    python plot_results.py --watch      # auto-refresh every 60s (for overnight runs)
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path="results.tsv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    return df


def plot(df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_bpb = axes[0, 0]
    ax_mfu = axes[0, 1]
    ax_vram = axes[1, 0]
    ax_tput = axes[1, 1]

    kept = df[df["status"] == "keep"].copy()
    discarded = df[df["status"].isin(["discard", "crash"])].copy()
    valid = df[df["val_bpb"] > 0].copy()  # non-crash experiments

    fig.suptitle(
        f"Autoresearch Progress: {len(df)} Experiments, {len(kept)} Kept Improvements",
        fontsize=14, fontweight="bold",
    )

    # --- Panel 1: val_bpb ---
    if not discarded.empty:
        valid_disc = discarded[discarded["val_bpb"] > 0]
        if not valid_disc.empty:
            ax_bpb.scatter(
                valid_disc.index, valid_disc["val_bpb"],
                c="#cccccc", s=30, zorder=2, alpha=0.6,
                edgecolors="white", linewidth=0.3, label="Discarded",
            )
    if not kept.empty:
        kept["best_so_far"] = kept["val_bpb"].cummin()
        ax_bpb.scatter(
            kept.index, kept["val_bpb"],
            c="#2ecc71", s=60, zorder=4, edgecolors="white", linewidth=0.5,
            label="Kept",
        )
        ax_bpb.step(
            kept.index, kept["best_so_far"],
            where="post", color="#2ecc71", linewidth=2, zorder=3,
            label="Running best",
        )
        for idx, row in kept.iterrows():
            desc = str(row.get("description", ""))
            if len(desc) > 40:
                desc = desc[:37] + "..."
            ax_bpb.annotate(
                desc,
                xy=(idx, row["val_bpb"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6, color="#333333", rotation=25,
                ha="left", va="bottom",
            )
    # Auto-zoom y-axis to useful range (exclude extreme outliers)
    if not valid.empty:
        valid_bpb = valid["val_bpb"]
        bpb_median = valid_bpb.median()
        # Cap y-axis at median + 0.05 to exclude catastrophic outliers
        y_max = min(valid_bpb.max(), bpb_median + 0.05)
        y_min = valid_bpb.min() - 0.005
        ax_bpb.set_ylim(y_min, y_max)
    ax_bpb.set_xlabel("Experiment #", fontsize=10)
    ax_bpb.set_ylabel("val_bpb (lower is better)", fontsize=10)
    ax_bpb.legend(fontsize=8, loc="upper right")
    ax_bpb.grid(True, alpha=0.2)

    # --- Panel 2: Wall-Clock Time to 200M Tokens (minutes) ---
    ax_time = ax_mfu
    if "tok_per_sec" in valid.columns:
        time_data = valid[pd.notna(valid["tok_per_sec"]) & (valid["tok_per_sec"] > 0)].copy()
        if not time_data.empty:
            token_budget = 200_000_000
            time_data["wall_minutes"] = (token_budget / time_data["tok_per_sec"]) / 60
            colors = ["#2ecc71" if s == "keep" else "#cccccc" for s in time_data["status"]]
            ax_time.bar(time_data.index, time_data["wall_minutes"], color=colors, edgecolor="white", linewidth=0.3)
            ax_time.axhline(y=time_data["wall_minutes"].mean(), color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label=f"Mean: {time_data['wall_minutes'].mean():.1f} min")
            ax_time.legend(fontsize=8)
    ax_time.set_xlabel("Experiment #", fontsize=10)
    ax_time.set_ylabel("Minutes to 200M tokens", fontsize=10)
    ax_time.set_title("Time to Completion (lower is better)", fontsize=11)
    ax_time.grid(True, alpha=0.2, axis="y")

    # --- Panel 3: Training VRAM (GB) ---
    if "memory_gb" in valid.columns:
        vram_data = valid[pd.notna(valid["memory_gb"]) & (valid["memory_gb"] > 0)]
        if not vram_data.empty:
            colors = ["#2ecc71" if s == "keep" else "#cccccc" for s in vram_data["status"]]
            ax_vram.bar(vram_data.index, vram_data["memory_gb"], color=colors, edgecolor="white", linewidth=0.3)
            ax_vram.axhline(y=12.0, color="#e74c3c", linestyle="-", linewidth=1.5, alpha=0.8, label="GPU limit (12 GB)")
            ax_vram.set_ylim(0, 13)
            ax_vram.legend(fontsize=8)
    ax_vram.set_xlabel("Experiment #", fontsize=10)
    ax_vram.set_ylabel("Training VRAM (GB)", fontsize=10)
    ax_vram.set_title("Memory Usage", fontsize=11)
    ax_vram.grid(True, alpha=0.2, axis="y")

    # --- Panel 4: Throughput (tok/sec) ---
    if "tok_per_sec" in valid.columns:
        tput_data = valid[pd.notna(valid["tok_per_sec"]) & (valid["tok_per_sec"] > 0)]
        if not tput_data.empty:
            colors = ["#2ecc71" if s == "keep" else "#cccccc" for s in tput_data["status"]]
            ax_tput.bar(tput_data.index, tput_data["tok_per_sec"], color=colors, edgecolor="white", linewidth=0.3)
            ax_tput.axhline(y=tput_data["tok_per_sec"].mean(), color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6, label=f"Mean: {tput_data['tok_per_sec'].mean():,.0f} tok/s")
            ax_tput.legend(fontsize=8)
    ax_tput.set_xlabel("Experiment #", fontsize=10)
    ax_tput.set_ylabel("Throughput (tok/sec)", fontsize=10)
    ax_tput.set_title("Training Throughput", fontsize=11)
    ax_tput.grid(True, alpha=0.2, axis="y")

    # Summary stats text box on val_bpb panel
    if not kept.empty:
        best_bpb = kept["val_bpb"].min()
        best_row = kept.loc[kept["val_bpb"].idxmin()]
        stats_lines = [f"Best: {best_bpb:.6f}"]
        if "mfu" in df.columns and pd.notna(best_row.get("mfu")):
            stats_lines.append(f"MFU: {best_row['mfu']:.1f}%")
        if "tok_per_sec" in df.columns and pd.notna(best_row.get("tok_per_sec")):
            stats_lines.append(f"Throughput: {best_row['tok_per_sec']:.0f} tok/s")
        if "memory_gb" in df.columns and pd.notna(best_row.get("memory_gb")):
            stats_lines.append(f"Train VRAM: {best_row['memory_gb']:.1f}/12.0 GB")
        stats_text = "\n".join(stats_lines)
        ax_bpb.text(
            0.02, 0.02, stats_text,
            transform=ax_bpb.transAxes, fontsize=9,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch results")
    parser.add_argument("--save", action="store_true", help="Save to progress.png")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--file", default="results.tsv", help="Path to results TSV")
    args = parser.parse_args()

    if args.watch:
        plt.ion()
        print("Watching results.tsv — Ctrl+C to stop")
        while True:
            df = load_results(args.file)
            if df is not None:
                plt.clf()
                plot(df, save_path="progress.png" if args.save else None)
                plt.pause(0.1)
                kept = df[df["status"] == "keep"]
                if not kept.empty:
                    print(
                        f"\r[{time.strftime('%H:%M:%S')}] {len(df)} experiments, "
                        f"best={kept['val_bpb'].min():.6f}",
                        end="", flush=True,
                    )
            time.sleep(60)
    else:
        df = load_results(args.file)
        if df is None:
            print("No results.tsv found or it's empty. Run some experiments first.")
            return
        print(f"Loaded {len(df)} experiments from {args.file}")
        kept = df[df["status"] == "keep"]
        if not kept.empty:
            best = kept.loc[kept["val_bpb"].idxmin()]
            desc = str(best['description']).encode('ascii', errors='replace').decode()
            print(f"Best val_bpb: {best['val_bpb']:.6f} (experiment #{best.name}, {desc})")
        plot(df, save_path="progress.png" if args.save else None)
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
