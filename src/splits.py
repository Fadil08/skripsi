# src/splits.py
from sklearn.model_selection import train_test_split

def stratified_split(items, test_size, seed=42):
    """
    items: list of (filepath, label_idx)
    test_size: float, e.g. 0.2 for 80:20
    """
    X = [fp for fp, y in items]
    y = [y for fp, y in items]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    train_items = list(zip(X_train, y_train))
    val_items = list(zip(X_val, y_val))
    return train_items, val_items
