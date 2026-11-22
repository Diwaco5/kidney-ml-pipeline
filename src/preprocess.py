from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_clean_data(
    csv_path: str | Path, target_col: str | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, na_values=["?", "\t?", "nan", "NaN", ""])
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    target = target_col or df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    if target == "CKD_Status" and "GFR" in X.columns:
        X = X.drop(columns=["GFR"])

    if y.dtype == "object":
        y = y.str.lower().map({"ckd": 1, "notckd": 0, "yes": 1, "no": 0}).fillna(y)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ]
    )


def prepare_splits(
    csv_path: str | Path, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    X, y = load_and_clean_data(csv_path)
    X_train_df, X_test_df, y_train_ser, y_test_ser = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df).astype(np.float32)
    X_test = preprocessor.transform(X_test_df).astype(np.float32)

    return (
        X_train,
        X_test,
        y_train_ser.to_numpy(dtype=np.float32),
        y_test_ser.to_numpy(dtype=np.float32),
        preprocessor,
        X_train_df.columns.tolist(),
    )
