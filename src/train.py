# # src/train.py
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix, classification_report
# from splits import stratified_split
# from tqdm import tqdm

# from dataset import BirdSegDataset
# from model import EffNetLSTM
# import numpy as np

# def iou_from_confusion(cm: np.ndarray, eps: float = 1e-9):
#     """
#     cm: confusion matrix shape (K, K)
#     returns:
#       iou_per_class: (K,)
#       miou: float
#     """
#     cm = cm.astype(np.float64)
#     tp = np.diag(cm)
#     fp = cm.sum(axis=0) - tp
#     fn = cm.sum(axis=1) - tp
#     denom = tp + fp + fn + eps
#     iou = tp / denom
#     miou = float(np.mean(iou))
#     return iou, miou

# def run_experiment(data_root, aug, split_name, test_size, num_epochs=3, batch_size=32, lr=1e-4, seed=42, num_workers=None, patience=2):
#     # build full list
#     full_ds = BirdSegDataset(root=data_root, aug="none")
#     label_map = full_ds.label_map
#     items = full_ds.items

#     train_items, val_items = stratified_split(items, test_size=test_size, seed=seed)

#     train_ds = BirdSegDataset(root=data_root, aug=aug, files=train_items, label_map=label_map)
#     val_ds   = BirdSegDataset(root=data_root, aug="none", files=val_items, label_map=label_map)

#     # DataLoader performance tweaks
#     if num_workers is None:
#         try:
#             cpu_count = os.cpu_count() or 1
#         except Exception:
#             cpu_count = 1
#         num_workers = min(4, cpu_count)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     pin_memory = True if device == "cuda" else False
#     print("Device:", device)
#     if device == "cuda":
#         print("GPU:", torch.cuda.get_device_name(0))


#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
#     val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

#     model = EffNetLSTM(num_classes=len(label_map)).to(device)

#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     crit = nn.CrossEntropyLoss()

   

#     best_f1 = -1
#     best_state = None
#     epochs_no_improve = 0

#     for epoch in range(1, num_epochs+1):
#         model.train()
#         train_iter = train_loader
#         for x, y in (tqdm(train_iter, desc=f"Train {split_name} {aug} epoch {epoch}") if len(train_loader) > 0 else train_iter):
#             x = x.to(device)
#             y = y.to(device)
#             opt.zero_grad()
#             logits = model(x)
#             loss = crit(logits, y)
#             loss.backward()
#             opt.step()

#         # eval
#         model.eval()
#         preds, trues = [], []
#         with torch.no_grad():
#             val_iter = val_loader
#             for x, y in (tqdm(val_iter, desc=f"Val   {split_name} {aug} epoch {epoch}") if len(val_loader) > 0 else val_iter):
#                 x = x.to(device)
#                 logits = model(x)
#                 p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
#                 preds += p
#                 trues += list(y)

#         cm = confusion_matrix(trues, preds)
#         report = classification_report(trues, preds, output_dict=True, zero_division=0)
#         f1 = report["weighted avg"]["f1-score"]

#         print(f"[{split_name} | {aug}] epoch {epoch}  val_f1={f1:.4f}")
#         if f1 > best_f1:
#             best_f1 = f1
#             best_state = {k:v.cpu() for k,v in model.state_dict().items()}
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"Early stopping: no improvement for {patience} epochs. Stopping.")
#             break

#     return best_f1, cm, label_map, best_state

# if __name__ == "__main__":
#     data_root = "dataset/processed"

#     splits = {
#         "90_10": 0.10,
#     }
#     augs = ["none","gain"]
#     #  augs = ["none", "gain", "time_stretch", "pitch_shift", "bg_noise"]
#     results = []
#     # sensible defaults — tune as needed
#     default_batch = 32
#     default_num_workers = None
#     for split_name, test_size in splits.items():
#         for aug in augs:
#             best_f1, cm, label_map, state = run_experiment(
#                 data_root=data_root,
#                 aug=aug,
#                 split_name=split_name,
#                 test_size=test_size,
#                 batch_size=default_batch,
#                 num_workers=default_num_workers,
#                 patience=2,
#             )
#             results.append((split_name, aug, best_f1))
#     print(results)
# src/train.py
import os
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from splits import stratified_split
from tqdm import tqdm

from dataset import BirdSegDataset
from model import EffNetLSTM


def iou_from_confusion(cm: np.ndarray, eps: float = 1e-9):
    """
    cm: confusion matrix shape (K, K)
    returns:
      iou_per_class: (K,)
      miou: float
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn + eps
    iou = tp / denom
    miou = float(np.mean(iou))
    return iou, miou


def save_run_artifacts(split_name, aug, cm, metrics, label_map):
    """
    Simpan confusion matrix + metrics agar bisa divisualisasikan.
    """
    os.makedirs("runs", exist_ok=True)

    tag = f"{split_name}__{aug}"

    # simpan confusion matrix
    np.save(f"runs/cm_{tag}.npy", cm)

    # simpan metrics json
    with open(f"runs/metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # simpan label_map (sekali saja / overwrite tidak masalah)
    with open("runs/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # append ringkasan ke CSV
    csv_path = "runs/results.csv"
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["split", "aug", "acc", "f1_weighted", "f1_macro", "miou"])
        w.writerow([
            metrics["split"], metrics["aug"],
            metrics["acc"], metrics["f1_weighted"], metrics["f1_macro"], metrics["miou"]
        ])


def run_experiment(
    data_root, aug, split_name, test_size,
    num_epochs=3, batch_size=32, lr=1e-4, seed=42,
    num_workers=None, patience=2
):
    # build full list
    full_ds = BirdSegDataset(root=data_root, aug="none")
    label_map = full_ds.label_map
    items = full_ds.items

    if len(items) == 0:
        raise RuntimeError(f"Dataset kosong. Cek folder: {data_root} (harus berisi subfolder kelas dengan file .wav)")

    train_items, val_items = stratified_split(items, test_size=test_size, seed=seed)

    train_ds = BirdSegDataset(root=data_root, aug=aug, files=train_items, label_map=label_map)
    val_ds   = BirdSegDataset(root=data_root, aug="none", files=val_items, label_map=label_map)

    # DataLoader performance tweaks
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model = EffNetLSTM(num_classes=len(label_map)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None
    best_cm = None
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ===== TRAIN =====
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train {split_name} {aug} epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # ===== EVAL =====
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val   {split_name} {aug} epoch {epoch}"):
                x = x.to(device)
                logits = model(x)
                p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds += p
                trues += y.cpu().numpy().tolist()

        cm = confusion_matrix(trues, preds)
        acc = accuracy_score(trues, preds)

        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        f1_weighted = report["weighted avg"]["f1-score"]
        f1_macro = report["macro avg"]["f1-score"]

        iou_vec, miou = iou_from_confusion(cm)

        print(f"[{split_name} | {aug}] epoch {epoch}  acc={acc:.4f}  f1w={f1_weighted:.4f}  f1m={f1_macro:.4f}  mIoU={miou:.4f}")

        # === pilih metrik acuan best ===
        # pakai f1_macro (lebih adil untuk imbalance), bisa ganti ke f1_weighted kalau mau
        score = f1_macro

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_cm = cm.copy()
            best_metrics = {
                "split": split_name,
                "aug": aug,
                "epoch_best": epoch,
                "acc": float(acc),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
                "miou": float(miou),
                "iou_per_class": [float(v) for v in iou_vec],
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement for {patience} epochs. Stopping.")
            break

    # simpan artifacts best untuk visualisasi
    save_run_artifacts(split_name, aug, best_cm, best_metrics, label_map)

    return best_metrics, best_cm, label_map, best_state


if __name__ == "__main__":
    data_root = "dataset/processed"

    splits = {
        # "90_10": 0.10,
        "80_20": 0.20,
        # "70_30": 0.30,
    }
    augs = ["none", "gain"]
    # augs = ["none", "gain", "time_stretch", "pitch_shift", "bg_noise"]

    results = []
    default_batch = 32
    default_num_workers = None

    for split_name, test_size in splits.items():
        for aug in augs:
            metrics, cm, label_map, state = run_experiment(
                data_root=data_root,
                aug=aug,
                split_name=split_name,
                test_size=test_size,
                batch_size=default_batch,
                num_workers=default_num_workers,
                patience=2,
                num_epochs=50,
            )
            results.append((split_name, aug, metrics["acc"], metrics["f1_macro"], metrics["miou"]))

    print("Summary:", results)
    print("Saved to runs/: results.csv, cm_*.npy, metrics_*.json, label_map.json")
