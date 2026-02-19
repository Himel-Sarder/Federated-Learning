import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_csv(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    if y.dtype == object:
        _, y = np.unique(y, return_inverse=True)

    return X, y

def preprocess_split(X, y, seed=42, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def split_iid(X_train, y_train, num_clients: int, seed=42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    splits = np.array_split(idx, num_clients)
    return [(X_train[s], y_train[s]) for s in splits]

def make_loader(Xc, yc, batch_size=32, shuffle=True) -> DataLoader:
    Xt = torch.tensor(Xc, dtype=torch.float32)
    yt = torch.tensor(yc, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)
