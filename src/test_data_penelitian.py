import os, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from  dataset import *
from splits import stratified_split
from model import EffNetLSTM

def iou_from_confusion(cm: np.ndarray, eps: float = 1e-9):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + eps)
    return iou, float(np.mean(iou))

def main():
    alas_root = "dataset/alas_purwo/features_5s_mel64"
    label_map_path = "runs/label_map.json"

    # pilih model tag terbaikmu (contoh: "80_20__bg_noise")
    best_tag = "80_20__bg_noise"
    model_path = f"runs/models/best_{best_tag}.pt"

    out_dir = "runs/alas_purwo"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(label_map_path):
        raise RuntimeError("runs/label_map.json tidak ada. Jalankan training dulu.")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    ds = BirdSegDataset(root=alas_root, label_map=label_map, aug="none")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = EffNetLSTM(num_classes=len(label_map), n_steps=3).to(device)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Checkpoint tidak ditemukan: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds += p
            trues += y.cpu().numpy().tolist()

    cm = confusion_matrix(trues, preds)
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, output_dict=True, zero_division=0)
    f1w = report["weighted avg"]["f1-score"]
    f1m = report["macro avg"]["f1-score"]
    iou_vec, miou = iou_from_confusion(cm)

    metrics = {
        "test_set": "Alas Purwo (external test)",
        "model_tag": best_tag,
        "n_samples": int(len(ds)),
        "acc": float(acc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "miou": float(miou),
        "iou_per_class": [float(v) for v in iou_vec],
    }

    np.save(os.path.join(out_dir, "cm_alas_purwo.npy"), cm)
    with open(os.path.join(out_dir, "metrics_alas_purwo.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(out_dir, "alas_purwo_results.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model_tag","n_samples","acc","f1_weighted","f1_macro","miou"])
        w.writerow([best_tag, metrics["n_samples"], acc, f1w, f1m, miou])

    print("\n=== EXTERNAL TEST: ALAS PURWO ===")
    print(f"Model: {best_tag}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"ACC:  {acc:.4f}")
    print(f"F1w:  {f1w:.4f}")
    print(f"F1m:  {f1m:.4f}")
    print(f"mIoU: {miou:.4f}")
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
