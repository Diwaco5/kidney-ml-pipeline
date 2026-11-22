from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from src.models import KidneyNet


class KidneyPredictor:
    def __init__(self, artifact_dir: str | Path = "artifacts"):
        base = Path(artifact_dir)
        self.preprocessor = joblib.load(base / "kidney_preprocessor.joblib")
        checkpoint = torch.load(base / "kidney_net_checkpoint.pt", map_location="cpu")

        self.expected_columns = checkpoint["expected_columns"]
        self.model = KidneyNet(in_features=checkpoint["in_features"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_one(self, patient_data: dict, threshold: float = 0.5) -> dict:
        df = pd.DataFrame([patient_data])[self.expected_columns]
        X_proc = self.preprocessor.transform(df).astype(np.float32)

        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(torch.tensor(X_proc))).item())

        return {
            "ckd_probability": round(prob, 4),
            "prediction": int(prob >= threshold),
            "label": "CKD" if prob >= threshold else "No CKD",
        }
