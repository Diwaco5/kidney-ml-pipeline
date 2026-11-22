# Kidney Disease Risk Prediction Pipeline (`kidney_ml`)

![Pytest CI](https://github.com<Diwaco5>/kidney-ml-pipeline/actions/workflows/tests.yml/badge.svg)

An end-to-end, containerized machine learning pipeline and FastAPI inference service comparing a **from-scratch NumPy Logistic Regression** implementation against a **PyTorch Multi-Layer Perceptron (`KidneyNet`)** on clinical biomarker data.

## System Architecture
```text
  kidney_dataset.csv
         │
         ▼
 ┌───────────────────┐
 │  src/preprocess   │  ──► Drops GFR (leakage guard), median/mode imputation,
 └─────────┬─────────┘      StandardScaler + OneHotEncoder (ColumnTransformer)
           │
           ▼
 ┌───────────────────┐
 │    src/models     │  ──► ScratchLogisticRegression (NumPy + L2)
 │   & src/train     │  ──► KidneyNet (PyTorch MLP + ReduceLROnPlateau + Early Stopping)
 └─────────┬─────────┘
           │  Saves kidney_preprocessor.joblib & kidney_net_checkpoint.pt
           ▼
 ┌───────────────────┐
 │    src/predict    │  ──► FastAPI (/health, /predict) + Pydantic v2 validation
 │    & src/api      │  ──► Containerized via Dockerfile + GitHub Actions CI (pytest)
 └───────────────────┘
```

## Key Engineering Highlights
- **Leakage-Free Preprocessing (`src/preprocess.py`):** Encapsulates imputation, standard scaling, and one-hot encoding inside a Scikit-Learn `ColumnTransformer` fit strictly on training splits, while stripping post-diagnosis leakage features (`GFR`).
- **From-Scratch vs. Deep Learning (`src/models.py`):** Implements vectorized binary cross-entropy with L2 regularization in pure NumPy alongside a regularized PyTorch MLP (`BCEWithLogitsLoss`, Dropout, Adam, `ReduceLROnPlateau`, and early stopping).
- **REST API & Containerization (`src/api.py`, `Dockerfile`):** Exposes real-time `/predict` and `/health` endpoints validated with Pydantic v2 and packaged for Docker deployment.
- **Automated CI Testing (`tests/test_pipeline.py`):** GitHub Actions runs `pytest` on every push to verify leakage guards, model tensor shapes, training/inference parity, and FastAPI endpoint responses.

## Benchmark Results (1,000 Unseen Test Samples)

| Model | Accuracy | Precision | Recall | F1-Score | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ScratchLogisticRegression` (NumPy) | 98.60% | 94.95% | 100.00% | 97.41% | 14 | 0 |
| `KidneyNet` (PyTorch MLP) | **99.30%** | **97.41%** | **100.00%** | **98.69%** | **7** | **0** |

## Quickstart

### 1. Local / Virtual Environment
```bash
pip install -r requirements.txt
python -m src.train
PYTHONPATH=. pytest -v
```

### 2. Run the FastAPI Server
```bash
uvicorn src.api:app --reload
# Interactive Swagger UI available at http://127.0.0
```

### 3. Run with Docker
```bash
docker build -t kidney-ml-api .
docker run -p 8000:8000 kidney-ml-api
```
