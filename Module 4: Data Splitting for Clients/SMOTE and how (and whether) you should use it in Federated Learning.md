## 1. SMOTE in Federated Learning (Important Caveats)

Using SMOTE in FL is **tricky**:

### Pros

* Improves minority class learning locally
* Can improve recall and F1-score
* Helps clients with few minority samples

### Cons

* Each client generates synthetic data independently
* This introduces distribution shifts across clients
* Can increase client drift in Non-IID settings
* Makes theoretical FL assumptions weaker

**Best practice in FL:**
Prefer **class-weighted loss** first.
Use SMOTE only as an ablation or secondary experiment.

---

## 2. How to Apply SMOTE (Client-Side) – Example

Apply SMOTE **inside each client before creating the DataLoader**:

```python
from imblearn.over_sampling import SMOTE
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# Example usage inside client data loading
X_client, y_client = X_client.numpy(), y_client.numpy()
X_res, y_res = apply_smote(X_client, y_client)

X_res = torch.tensor(X_res, dtype=torch.float32)
y_res = torch.tensor(y_res, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_res, y_res), batch_size=32, shuffle=True)
```

Install dependency:

```bash
pip install imbalanced-learn
```
