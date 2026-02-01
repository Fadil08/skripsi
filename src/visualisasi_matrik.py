import pandas as pd
import  os
import numpy as np
import matplotlib.pyplot as plt

def visualize_split_comparison():
    """
    Visualisasi perbandingan performa antar split (90/10, 80/20, 70/30)
    dari runs/results.csv
    """
    csv_path = "runs/results.csv"
    if not os.path.exists(csv_path):
        print("[WARN] runs/results.csv tidak ditemukan.")
        return

    df = pd.read_csv(csv_path)

    # Pastikan kolom ada
    required_cols = {"split", "aug", "acc", "f1_macro", "miou"}
    if not required_cols.issubset(df.columns):
        print("[WARN] Kolom CSV tidak lengkap:", df.columns)
        return

    out_dir = "runs/figures/split_comparison"
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("acc", "Accuracy"),
        ("f1_macro", "Macro F1-score"),
        ("miou", "Mean IoU (mIoU)")
    ]

    splits_order = ["90_10", "80_20", "70_30"]

    for metric_key, metric_name in metrics:
        plt.figure(figsize=(8, 5))

        for aug in sorted(df["aug"].unique()):
            sub = df[df["aug"] == aug].copy()
            sub["split"] = pd.Categorical(sub["split"], categories=splits_order, ordered=True)
            sub = sub.sort_values("split")

            plt.plot(
                sub["split"],
                sub[metric_key],
                marker="o",
                linewidth=2,
                label=f"Aug: {aug}"
            )

            # tambahkan nilai di titik
            for x, y in zip(sub["split"], sub[metric_key]):
                plt.text(x, y + 0.005, f"{y:.2f}", ha="center", fontsize=9)

        plt.ylim(0, 1.0)
        plt.xlabel("Dataset Split")
        plt.ylabel(metric_name)
        plt.title(f"Perbandingan {metric_name} antar Split Dataset")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"compare_{metric_key}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"[OK] Saved: {out_path}")
