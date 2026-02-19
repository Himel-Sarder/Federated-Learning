## Minimal Code Changes

### 1) Add watermark config (server side)

In `src/fl_server.py`, send watermark parameters to clients:

```python
import numpy as np
import flwr as fl

# Create a fixed watermark for this experiment/run
WATERMARK_BITS = 64
np.random.seed(42)
P = np.random.randn(WATERMARK_BITS, 32)  # 32 = last hidden size in your MLP
b = np.random.choice([-1.0, 1.0], size=(WATERMARK_BITS,))

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    acc = sum(n * m["accuracy"] for n, m in metrics) / max(total, 1)
    return {"accuracy": acc}

def fit_config(server_round: int):
    return {
        "wm_strength": 1e-4,        # small regularization strength
        "wm_P": P.tolist(),        # send projection
        "wm_b": b.tolist(),        # send target bits
    }

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    fraction_evaluate=1.0,
    min_evaluate_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
)
```

---

### 2) Modify client training to include watermark loss

In `src/train_eval.py`, add a watermark regularizer:

```python
import torch
import torch.nn as nn

def watermark_loss(last_layer_weights, P, b):
    # last_layer_weights: shape [out_dim, hidden_dim]
    w_vec = last_layer_weights.view(-1)
    P = torch.tensor(P, dtype=w_vec.dtype, device=w_vec.device)
    b = torch.tensor(b, dtype=w_vec.dtype, device=w_vec.device)

    proj = torch.matmul(P, w_vec[:P.shape[1]])  # simple projection
    return torch.mean((proj - b) ** 2)
```

Update `train_one_epoch`:

```python
def train_one_epoch(model, loader, device, lr=1e-3, wm_cfg=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)

        # Add watermark regularization (light)
        if wm_cfg is not None:
            P = wm_cfg["wm_P"]
            b = wm_cfg["wm_b"]
            strength = wm_cfg["wm_strength"]
            last_layer = model.net[-1].weight  # final linear layer
            loss = loss + strength * watermark_loss(last_layer, P, b)

        loss.backward()
        opt.step()
```

---

### 3) Pass watermark config from Flower client

In `src/fl_client.py`:

```python
def fit(self, parameters, config):
    set_params(self.model, parameters)
    local_epochs = int(config.get("local_epochs", 1))
    lr = float(config.get("lr", 5e-4))

    wm_cfg = {
        "wm_P": config.get("wm_P", None),
        "wm_b": config.get("wm_b", None),
        "wm_strength": float(config.get("wm_strength", 0.0)),
    }
    if wm_cfg["wm_P"] is None:
        wm_cfg = None

    for _ in range(local_epochs):
        train_one_epoch(self.model, self.train_loader, DEVICE, lr=lr, wm_cfg=wm_cfg)

    return get_params(self.model), len(self.train_loader.dataset), {}
```

---

## Watermark Verification (After Training)

Add a small script:

```python
import torch
import numpy as np
from src.model import MLP

def verify_watermark(model, P, b):
    with torch.no_grad():
        W = model.net[-1].weight.view(-1).cpu().numpy()
    proj = np.dot(P, W[:P.shape[1]])
    recovered = np.sign(proj)
    match = (recovered == np.sign(b)).mean()
    return match

# match close to 1.0 = watermark detected
```

You can report **watermark detection accuracy** (e.g., 95% bits matched) in your thesis.
