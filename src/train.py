import copy
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models import KidneyNet
from src.preprocess import prepare_splits


def train_and_export(
    csv_path: str = "data/kidney_dataset.csv", artifact_dir: str = "artifacts"
) -> dict:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor, raw_cols = prepare_splits(csv_path)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr.reshape(-1, 1))),
        batch_size=64,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), device=device)
    X_test_t = torch.tensor(X_test, device=device)

    model = KidneyNet(in_features=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(100):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 10:
            print(f"Early stopping at epoch {epoch+1} | Best Val Loss: {best_val_loss:.4f}")
            break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds, digits=4))

    joblib.dump(preprocessor, out_dir / "kidney_preprocessor.joblib")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_features": X_train.shape[1],
            "expected_columns": raw_cols,
        },
        out_dir / "kidney_net_checkpoint.pt",
    )
    return {"best_val_loss": best_val_loss}
