# Kidney Disease Risk Prediction Pipeline (`kidney_ml`)

An end-to-end, modular machine learning pipeline comparing a **from-scratch NumPy Logistic Regression** implementation against a **PyTorch Multi-Layer Perceptron (`KidneyNet`)** on clinical biomarker data.

## Key Engineering Highlights
- **Leakage-Free Preprocessing (`src/preprocess.py`):** Encapsulates median/mode imputation, standard scaling, and one-hot encoding inside a Scikit-Learn `ColumnTransformer` fit strictly on training splits, while removing post-diagnosis leakage features (`GFR`).
- **From-Scratch vs. Deep Learning (`src/models.py`):** Implements vectorized binary cross-entropy with L2 regularization in pure NumPy alongside a regularized PyTorch MLP (`BCEWithLogitsLoss`, Dropout, Adam, `ReduceLROnPlateau`, and early stopping).
- **Production-Ready Inference (`src/predict.py`):** Persists fitted transformers and model checkpoints for single-record scoring.
- **Automated Testing (`tests/test_pipeline.py`):** Includes `pytest` unit tests verifying leakage guards, tensor shapes, and end-to-end training/inference.

## Benchmark Results (1,000 Unseen Test Samples)

| Model | Accuracy | Precision | Recall | F1-Score | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ScratchLogisticRegression` (NumPy) | 98.60% | 94.95% | 100.00% | 97.41% | 14 | 0 |
| `KidneyNet` (PyTorch MLP) | **99.30%** | **97.41%** | **100.00%** | **98.69%** | **7** | **0** |

## Quickstart
```bash
pip install -r requirements.txt
python -m src.train
PYTHONPATH=. pytest -v
```
