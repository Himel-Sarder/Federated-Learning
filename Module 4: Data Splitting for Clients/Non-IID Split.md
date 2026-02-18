## Non-IID Split by Label/Class (Federated Learning)

### Goal

Create a realistic federated setup where each client has **skewed class distributions** (heterogeneous data).
This is the most common Non-IID scenario and a major challenge for FedAvg.

---

## 1. What is Non-IID (by Label/Class)?

* Each client receives data from **only a subset of classes** or **highly skewed class proportions**
* Clients’ local objectives differ
* Causes client drift and slower FedAvg convergence
* Reflects real-world settings (e.g., hospitals specializing in certain cases)

---

## 2. Two Common Non-IID Patterns

### Pattern A: Label Partition (Extreme Non-IID)

Each client gets data from **only one class** (or a small subset of classes).
This is very challenging and often hurts FedAvg performance badly.

### Pattern B: Skewed Label Distribution (Mild Non-IID)

Each client has **both classes**, but with very different ratios (e.g., 90:10 vs 10:90).

For medical datasets, Pattern B is more realistic and recommended.

---

## 3. Non-IID Split by Label (Skewed Distribution) – Code

This ensures each client has **both classes but heavily skewed**:

```python
import numpy as np
import torch
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data_noniid_by_label(num_clients=2, batch_size=32, seed=42, skew=0.9):
    """
    skew=0.9 means each client gets 90% of one class and 10% of the other
    """
    np.random.seed(seed)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Split by class
    idx_class0 = np.where(y_train == 0)[0]
    idx_class1 = np.where(y_train == 1)[0]

    np.random.shuffle(idx_class0)
    np.random.shuffle(idx_class1)

    # Split class indices across clients
    splits_0 = np.array_split(idx_class0, num_clients)
    splits_1 = np.array_split(idx_class1, num_clients)

    client_indices = []

    for cid in range(num_clients):
        # Each client is biased toward one class
        if cid % 2 == 0:
            major = splits_0[cid]
            minor = splits_1[cid][: int(len(splits_1[cid]) * (1 - skew))]
        else:
            major = splits_1[cid]
            minor = splits_0[cid][: int(len(splits_0[cid]) * (1 - skew))]

        idx = np.concatenate([major, minor])
        np.random.shuffle(idx)
        client_indices.append(idx)

    train_loaders = []
    class_distributions = []

    for cid, idx in enumerate(client_indices):
        y_client = y_train[idx]
        class_distributions.append(Counter(y_client))

        X_client = torch.tensor(X_train[idx], dtype=torch.float32)
        y_client = torch.tensor(y_client, dtype=torch.long)

        train_loaders.append(
            DataLoader(TensorDataset(X_client, y_client), batch_size=batch_size, shuffle=True)
        )

    # Global test loader (can be shared)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    return train_loaders, test_loader, input_dim, class_distributions
```

---

## 4. Verify Non-IID Split (Class Distribution Check)

```python
train_loaders, test_loader, input_dim, class_dist = load_data_noniid_by_label(num_clients=2, skew=0.9)

for i, dist in enumerate(class_dist):
    print(f"Client {i} class distribution:", dict(dist))
```

**Expected output (example):**

```
Client 0 class distribution: {0: 160, 1: 18}
Client 1 class distribution: {1: 300, 0: 15}
```

This confirms **strong label skew** (Non-IID).

---

## 5. How to Plug into Flower Clients

```python
# client.py
import sys
cid = int(sys.argv[1])

train_loaders, test_loader, input_dim, _ = load_data_noniid_by_label(num_clients=2, skew=0.9)
train_loader = train_loaders[cid]
```

Run:

```bash
python client.py 0
python client.py 1
```

---

## 6. What You Should Observe in Results

Compared to IID:

* Slower convergence
* Lower final accuracy
* Higher variance across rounds
* Minority class recall drops
* More rounds needed

This is expected and should be reported.
