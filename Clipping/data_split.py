import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def make_iid_client_loaders(num_clients=2, batch_size=32, seed=42):
    np.random.seed(seed)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2,
        random_state=seed,
        stratify=data.target
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    idx = np.random.permutation(len(X_train))
    splits = np.array_split(idx, num_clients)

    train_loaders = []
    for cid in range(num_clients):
        client_idx = splits[cid]
        train_loaders.append(
            DataLoader(
                TensorDataset(X_train[client_idx], y_train[client_idx]),
                batch_size=batch_size,
                shuffle=True
            )
        )

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]
    return train_loaders, test_loader, input_dim
