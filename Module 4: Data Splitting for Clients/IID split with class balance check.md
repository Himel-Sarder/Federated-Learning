## 1. What is IID Split?

IID (Independent and Identically Distributed) split means:

* Each client gets a **random subset** of the training data
* Each client has a **similar class distribution**
* Feature distributions are approximately the same
* This simulates the ideal case for Federated Learning (FedAvg works best here)

---

## 2. IID Split with Stratification (Recommended)

To ensure class balance, we:

* Use `stratify=y` in train/test split
* Then randomly distribute samples across clients

### Code: IID Split with Class Balance Check

```python
import numpy as np
import torch
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data_iid_with_balance(num_clients=2, batch_size=32, seed=42):
    np.random.seed(seed)

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=seed, stratify=data.target
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Shuffle indices
    num_samples = len(X_train)
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)

    train_loaders = []
    class_distributions = []

    for cid in range(num_clients):
        idx = splits[cid]
        y_client = y_train[idx].numpy()
        dist = Counter(y_client)
        class_distributions.append(dist)

        train_loaders.append(
            DataLoader(
                TensorDataset(X_train[idx], y_train[idx]),
                batch_size=batch_size,
                shuffle=True
            )
        )

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]

    return train_loaders, test_loader, input_dim, class_distributions
```

---

## 3. Check Class Balance per Client

After calling the function:

```python
train_loaders, test_loader, input_dim, class_dist = load_data_iid_with_balance(num_clients=2)

print("Class distribution per client:")
for i, dist in enumerate(class_dist):
    print(f"Client {i}: {dict(dist)}")
```

### Expected Output (Example)

```
Client 0: {0: 170, 1: 283}
Client 1: {0: 167, 1: 289}
```

You should see **similar ratios** of class 0 and class 1 across clients.
This confirms your split is IID.

---

## 4. How to Plug into Flower Clients

Use client IDs:

```python
# client.py
import sys

cid = int(sys.argv[1])

train_loaders, test_loader, input_dim, _ = load_data_iid_with_balance(num_clients=2)
train_loader = train_loaders[cid]
```

Run:

```bash
python client.py 0
python client.py 1
```

---

## 5. Why Class Balance Check Matters

* Ensures your IID assumption is valid
* Makes your FedAvg results comparable to centralized training
* Prevents misleading conclusions due to skewed splits
* Important for thesis methodology section
