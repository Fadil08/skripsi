import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

from visualisasi_matrik import visualize_split_comparison

def load_label_map(path="runs/label_map.json"):
    if not os.path.exists(path):
        raise RuntimeError(f"Label map tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx2cls = {v: k for k, v in label_map.items()}
    classes = [idx2cls[i] for i in range(len(idx2cls))]
    return classes

def plot_confusion_matrix(cm, class_names, title, out_path, normalize=True):
    cm = cm.astype(np.float64)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_iou_bar(iou_per_class, class_names, title, out_path):
    iou_per_class = np.array(iou_per_class, dtype=np.float64)
    miou = float(np.mean(iou_per_class)) if len(iou_per_class) else 0.0

    plt.figure(figsize=(11, 5))
    x = np.arange(len(class_names))
    plt.bar(x, iou_per_class)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("IoU")
    plt.title(f"{title} | mIoU={miou:.3f}")
    plt.grid(True, axis="y", alpha=0.3)

    # label nilai di atas bar
    for i, v in enumerate(iou_per_class):
        plt.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def visualize_runs():
    """
    Visualisasi untuk hasil eksperimen training (runs/cm_*.npy dan runs/metrics_*.json)
    """
    classes = load_label_map("runs/label_map.json")
    out_dir = "runs/figures"
    os.makedirs(out_dir, exist_ok=True)

    cm_files = sorted(glob.glob("runs/cm_*.npy"))
    if not cm_files:
        print("[WARN] Tidak ada runs/cm_*.npy. Jalankan train.py dulu.")
        return

    for cm_path in cm_files:
        tag = os.path.basename(cm_path).replace("cm_", "").replace(".npy", "")
        cm = np.load(cm_path)

        metrics_path = f"runs/metrics_{tag}.json"
        title = tag
        iou_per_class = None
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            title = f"{m['split']} | {m['aug']} | acc={m['acc']:.3f} f1m={m['f1_macro']:.3f} mIoU={m['miou']:.3f}"
            iou_per_class = m.get("iou_per_class", None)

        # confusion matrix
        plot_confusion_matrix(cm, classes, title, os.path.join(out_dir, f"cm_norm_{tag}.png"), normalize=True)
        plot_confusion_matrix(cm, classes, title, os.path.join(out_dir, f"cm_count_{tag}.png"), normalize=False)

        # IoU bar
        if iou_per_class is not None:
            plot_iou_bar(iou_per_class, classes, title, os.path.join(out_dir, f"iou_{tag}.png"))

        print(f"[OK] Saved figures for {tag}")

    print(f"Selesai. Lihat folder: {out_dir}")

def visualize_alas_purwo():
    """
    Visualisasi untuk external test Alas Purwo:
    - runs/alas_purwo/cm_alas_purwo.npy
    - runs/alas_purwo/metrics_alas_purwo.json
    """
    classes = load_label_map("runs/label_map.json")
    out_dir = "runs/alas_purwo/figures"
    os.makedirs(out_dir, exist_ok=True)

    cm_path = "runs/alas_purwo/cm_alas_purwo.npy"
    metrics_path = "runs/alas_purwo/metrics_alas_purwo.json"

    if not os.path.exists(cm_path) or not os.path.exists(metrics_path):
        print("[WARN] File Alas Purwo belum ada. Jalankan test_alas_purwo.py dulu.")
        return

    cm = np.load(cm_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    title = f"ALAS PURWO | model={m.get('model_tag','?')} | acc={m['acc']:.3f} f1m={m['f1_macro']:.3f} mIoU={m['miou']:.3f}"

    plot_confusion_matrix(cm, classes, title, os.path.join(out_dir, "cm_norm_alas_purwo.png"), normalize=True)
    plot_confusion_matrix(cm, classes, title, os.path.join(out_dir, "cm_count_alas_purwo.png"), normalize=False)
    plot_iou_bar(m["iou_per_class"], classes, title, os.path.join(out_dir, "iou_alas_purwo.png"))

    print(f"[OK] Saved Alas Purwo figures to: {out_dir}")

if __name__ == "__main__":
    # visualisasi untuk semua eksperimen train + external test
    # visualize_runs()
    # visualize_alas_purwo()
    visualize_split_comparison()
