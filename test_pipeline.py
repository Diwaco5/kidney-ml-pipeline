import numpy as np
import pandas as pd
import torch
from src.models import KidneyNet, ScratchLogisticRegression
from src.predict import KidneyPredictor
from src.preprocess import load_and_clean_data
from src.train import train_and_export


def test_gfr_leakage_removed(tmp_path):
    csv = tmp_path / "sample.csv"
    pd.DataFrame({
        "Age": list((40, 65)),
        "Creatinine": list((0.9, 2.8)),
        "GFR": list((95.0, 22.0)),
        "Medication": list(("None", "ACE Inhibitor")),
        "CKD_Status": list((0, 1)),
    }).to_csv(csv, index=False)

    X, y = load_and_clean_data(csv)
    assert "GFR" not in X.columns
    assert list(y) == list((0, 1))


def test_scratch_and_pytorch_shapes():
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=20).astype(np.float32)

    scratch = ScratchLogisticRegression(n_iters=10).fit(X, y)
    assert len(scratch.loss_history) == 10
    assert scratch.predict_proba(X).shape == (20,)

    net = KidneyNet(in_features=8)
    out = net(torch.tensor(X))
    assert out.shape == (20, 1)


def test_end_to_end_train_and_predict(tmp_path):
    rng = np.random.default_rng(42)
    csv = tmp_path / "kidney.csv"
    pd.DataFrame({
        "Age": rng.integers(25, 80, size=60),
        "Creatinine": rng.uniform(0.6, 4.5, size=60),
        "BUN": rng.uniform(8, 60, size=60),
        "Medication": rng.choice(list(("None", "ACE Inhibitor", "ARB")), size=60),
        "CKD_Status": rng.choice(list((0, 1)), size=60),
    }).to_csv(csv, index=False)

    artifacts = tmp_path / "artifacts"
    train_and_export(csv_path=str(csv), artifact_dir=str(artifacts))

    predictor = KidneyPredictor(artifact_dir=artifacts)
    res = predictor.predict_one({
        "Age": 65,
        "Creatinine": 3.2,
        "BUN": 45.0,
        "Medication": "None",
    })
    assert 0.0 <= res["ckd_probability"] <= 1.0
    assert res["prediction"] in (0, 1)


def test_api_health_and_predict():
    from fastapi.testclient import TestClient
    from src.api import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        resp = client.post(
            "/predict",
            json={
                "Creatinine": 2.9,
                "BUN": 42.0,
                "Urine_Output": 850.0,
                "Diabetes": 1,
                "Hypertension": 1,
                "Age": 62.0,
                "Protein_in_Urine": 1.8,
                "Water_Intake": 1.5,
                "Medication": "None",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["ckd_probability"] <= 1.0
        assert body["prediction"] in (0, 1)
