## Dataset Split into Multiple Clients (Federated Learning)

### Goal

In Federated Learning, each client must have its **own local dataset**.
So we split one dataset into multiple parts, where each part represents one client.

This simulates multiple hospitals/devices.

---

## 1. Why Split the Dataset?

* To simulate multiple clients in FL
* Each client trains on its own private data
* Enables IID and Non-IID experiments
* Makes FL experiments realistic

---

## 2. General Approach

1. Load the full dataset
2. Decide the number of clients (e.g., 2, 3, 5)
3. Split the training set into multiple subsets
4. Create a DataLoader for each client

---

## 3. IID Split (Random Split) – Example Code

This creates **similar distributions** across clients.

```python
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data_iid(num_clients=2, batch_size=32):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Shuffle indices and split equally
    num_samples = len(X_train)
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)

    train_loaders = []
    for cid in range(num_clients):
        idx = splits[cid]
        train_loaders.append(
            DataLoader(TensorDataset(X_train[idx], y_train[idx]), batch_size=batch_size, shuffle=True)
        )

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]

    return train_loaders, test_loader, input_dim
```

**What this does:**

* Randomly splits training data into `num_clients` equal parts
* Each client sees a similar class distribution (IID)

---

## 4. How to Use in Flower Client

Modify your `client.py` to accept a `cid` (client id):

```python
import sys
cid = int(sys.argv[1])  # client id from command line

train_loaders, test_loader, input_dim = load_data_iid(num_clients=2)
train_loader = train_loaders[cid]
```

Run clients like this:

```bash
python client.py 0
python client.py 1
```

Now each client uses a different subset of the dataset.
