## 1) Create Client-wise DataLoaders (IID)

```python
# data_split.py
import numpy as np
import torch
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def make_iid_loaders(num_clients=2, batch_size=32, seed=42):
    np.random.seed(seed)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    indices = np.random.permutation(len(X_train))
    splits = np.array_split(indices, num_clients)

    train_loaders = []
    class_dist = []

    for cid in range(num_clients):
        idx = splits[cid]
        y_client = y_train[idx].numpy()
        class_dist.append(Counter(y_client))

        train_loaders.append(
            DataLoader(
                TensorDataset(X_train[idx], y_train[idx]),
                batch_size=batch_size,
                shuffle=True
            )
        )

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]

    return train_loaders, test_loader, input_dim, class_dist
```

### Check class balance

```python
train_loaders, test_loader, input_dim, class_dist = make_iid_loaders(num_clients=2)
for i, d in enumerate(class_dist):
    print(f"Client {i}:", dict(d))
```

---

## 2) Create Client-wise DataLoaders (Non-IID by label skew)

```python
# data_split.py
import numpy as np
import torch
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def make_noniid_loaders(num_clients=2, batch_size=32, seed=42, skew=0.9):
    """
    skew=0.9 means each client gets ~90% from one class and ~10% from the other
    """
    np.random.seed(seed)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    splits0 = np.array_split(idx0, num_clients)
    splits1 = np.array_split(idx1, num_clients)

    train_loaders = []
    class_dist = []

    for cid in range(num_clients):
        if cid % 2 == 0:
            major = splits0[cid]
            minor = splits1[cid][: max(1, int(len(splits1[cid]) * (1 - skew)))]
        else:
            major = splits1[cid]
            minor = splits0[cid][: max(1, int(len(splits0[cid]) * (1 - skew)))]

        idx = np.concatenate([major, minor])
        np.random.shuffle(idx)

        y_client = y_train[idx]
        class_dist.append(Counter(y_client))

        Xc = torch.tensor(X_train[idx], dtype=torch.float32)
        yc = torch.tensor(y_client, dtype=torch.long)

        train_loaders.append(
            DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=True)
        )

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    return train_loaders, test_loader, input_dim, class_dist
```

### Check Non-IID distribution

```python
train_loaders, test_loader, input_dim, class_dist = make_noniid_loaders(num_clients=2, skew=0.9)
for i, d in enumerate(class_dist):
    print(f"Client {i}:", dict(d))
```

---

## 3) Use Client-wise DataLoader inside Flower client.py

```python
# client.py (important parts)
import sys
import flwr as fl
import torch
from utils import Net
from data_split import make_iid_loaders  # or make_noniid_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cid = int(sys.argv[1])          # client id: 0, 1, 2...
num_clients = 2

train_loaders, test_loader, input_dim, _ = make_iid_loaders(num_clients=num_clients)
train_loader = train_loaders[cid]

model = Net(input_dim).to(DEVICE)
```

Run two clients:

```bash
python client.py 0
python client.py 1
```


If you tell me how many clients you want (2, 3, 5), I can give you the exact `server.py` settings and the run commands for that number.
