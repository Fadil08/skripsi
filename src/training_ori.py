# src/train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from splits import stratified_split

from dataset import BirdSegDataset
from model import EffNetLSTM
import numpy as np

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

def run_experiment(data_root, aug, split_name, test_size, num_epochs=3, batch_size=16, lr=1e-4, seed=42):
    # build full list
    full_ds = BirdSegDataset(root=data_root, aug="none")
    label_map = full_ds.label_map
    items = full_ds.items

    train_items, val_items = stratified_split(items, test_size=test_size, seed=seed)

    train_ds = BirdSegDataset(root=data_root, aug=aug, files=train_items, label_map=label_map)
    val_ds   = BirdSegDataset(root=data_root, aug="none", files=val_items, label_map=label_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffNetLSTM(num_classes=len(label_map)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1
    best_state = None

    for epoch in range(1, num_epochs+1):
        model.train()
        for x, y in train_loader:
            # x, y = x.to(device), torch.tensor(y).to(device)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                p = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds += p
                trues += list(y)

        cm = confusion_matrix(trues, preds)
        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        f1 = report["weighted avg"]["f1-score"]

        print(f"[{split_name} | {aug}] epoch {epoch}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    return best_f1, cm, label_map, best_state

if __name__ == "__main__":
    data_root = "dataset/processed"

    splits = {
        "90_10": 0.10,
        # "80_20": 0.20,
        # "70_30": 0.30
    }
    augs = ["none", "gain", "time_stretch", "pitch_shift", "bg_noise"]

    results = []
    for split_name, test_size in splits.items():
        for aug in augs:
            best_f1, cm, label_map, state = run_experiment(
                data_root=data_root, aug=aug, split_name=split_name, test_size=test_size
            )
            results.append((split_name, aug, best_f1))
    print(results)
