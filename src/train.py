import os, json, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from splits import stratified_split
from dataset import BirdSegDataset
from model import EffNetLSTM

def iou_from_confusion(cm: np.ndarray, eps: float = 1e-9):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + eps)
    return iou, float(np.mean(iou))

def save_artifacts(tag, cm, metrics, label_map, best_state):
    os.makedirs("runs", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    np.save(f"runs/cm_{tag}.npy", cm)

    with open(f"runs/metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("runs/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    torch.save(best_state, f"runs/models/best_{tag}.pt")

    csv_path = "runs/results.csv"
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tag","split","aug","acc","f1_weighted","f1_macro","miou","epoch_best"])
        w.writerow([tag, metrics["split"], metrics["aug"], metrics["acc"], metrics["f1_weighted"], metrics["f1_macro"], metrics["miou"], metrics["epoch_best"]])

def run_experiment(
    data_root, split_name, test_size, aug,
    num_epochs=10, batch_size=32, lr=1e-4, seed=42,
    num_workers=0, patience=3, freeze_backbone_epochs=2
):
    # build full list
    full = BirdSegDataset(root=data_root, aug="none")
    label_map = full.label_map
    items = full.items

    train_items, val_items = stratified_split(items, test_size=test_size, seed=seed)

    train_ds = BirdSegDataset(root=data_root, files=train_items, label_map=label_map, aug=aug)
    val_ds   = BirdSegDataset(root=data_root, files=val_items,   label_map=label_map, aug="none")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EffNetLSTM(num_classes=len(label_map), n_steps=3).to(device)

    def set_backbone_trainable(trainable: bool):
        for p in model.backbone.parameters():
            p.requires_grad = trainable

    if freeze_backbone_epochs > 0:
        set_backbone_trainable(False)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None
    best_cm = None
    best_metrics = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(True)
            opt = torch.optim.Adam(model.parameters(), lr=lr * 0.3)

        # TRAIN
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train {split_name} {aug} ep{epoch}"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # VAL
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val   {split_name} {aug} ep{epoch}"):
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

        print(f"[{split_name}|{aug}] ep{epoch} acc={acc:.4f} f1w={f1w:.4f} f1m={f1m:.4f} mIoU={miou:.4f}")

        # pakai macro-F1 sebagai best criterion (lebih adil untuk imbalance)
        score = f1m
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_cm = cm.copy()
            best_metrics = {
                "split": split_name,
                "aug": aug,
                "epoch_best": epoch,
                "acc": float(acc),
                "f1_weighted": float(f1w),
                "f1_macro": float(f1m),
                "miou": float(miou),
                "iou_per_class": [float(v) for v in iou_vec],
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping: no improvement {patience} epochs.")
            break

    tag = f"{split_name}__{aug}"
    save_artifacts(tag, best_cm, best_metrics, label_map, best_state)
    return best_metrics

if __name__ == "__main__":
    data_root = "dataset/processed"

    splits = {"90_10": 0.10, "80_20": 0.20, "70_30": 0.30}
    augs = ["none", "gain", "bg_noise"]  # tambah "time_mask" kalau mau

    results = []
    for split_name, test_size in splits.items():
        for aug in augs:
            m = run_experiment(
                data_root=data_root,
                split_name=split_name,
                test_size=test_size,
                aug=aug,
                num_epochs=10,
                batch_size=32,
                num_workers=0,          # Windows CPU: stabil
                patience=3,
                freeze_backbone_epochs=2
            )
            results.append((split_name, aug, m["acc"], m["f1_macro"], m["miou"]))

    print("Summary:", results)
    print("Saved: runs/results.csv + runs/cm_*.npy + runs/models/best_*.pt")
