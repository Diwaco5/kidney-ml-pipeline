import numpy as np
import torch
import torch.nn as nn


class ScratchLogisticRegression:
    def __init__(self, lr: float = 0.05, n_iters: int = 1500, l2: float = 0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        bce = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        l2_pen = (self.l2 / (2.0 * len(y_true))) * np.sum(self.weights ** 2)
        return float(bce + l2_pen)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchLogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []
        y_1d = y.ravel().astype(np.float64)
        X_64 = X.astype(np.float64)

        for _ in range(self.n_iters):
            y_pred = self._sigmoid(X_64 @ self.weights + self.bias)
            self.loss_history.append(self._compute_loss(y_1d, y_pred))

            error = y_pred - y_1d
            dw = (1.0 / n_samples) * (X_64.T @ error) + (self.l2 / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X.astype(np.float64) @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class KidneyNet(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
