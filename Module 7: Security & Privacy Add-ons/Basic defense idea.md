## Basic Defense Ideas in Federated Learning

### Goal

Reduce the impact of **malicious or abnormal client updates** on the global model while keeping the system simple.

---

## 1) Update Clipping (Most Practical Basic Defense)

### What it is

Limit the magnitude (norm) of each client’s update so no single client can dominate aggregation.

### Why it works

* Malicious clients often send **scaled or extreme updates**
* Clipping bounds their influence
* Stabilizes FedAvg under Non-IID and noise

### Light Implementation (Client-side)

```python
import numpy as np

def clip_updates(params, clip_norm=1.0):
    clipped = []
    for p in params:
        norm = np.linalg.norm(p)
        if norm > clip_norm:
            p = p * (clip_norm / (norm + 1e-8))
        clipped.append(p)
    return clipped
```

Use in `fit()`:

```python
def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(model, train_loader, epochs=1)

    params = self.get_parameters({})
    params = clip_updates(params, clip_norm=1.0)

    return params, len(train_loader.dataset), {}
```

**When to use:**
Default basic defense in experiments and demos.

---

## 2) Norm-based Filtering (Concept)

### What it is

Server (or client) monitors update norms and flags or down-weights abnormal clients.

### Why it helps

* Poisoned updates often have unusually large norms
* Simple anomaly detection reduces attack impact

**Conceptual server rule:**

* If update norm > threshold → down-weight or ignore

(Implementation is more complex server-side in Flower; mention conceptually.)

---

## 3) Robust Aggregation (Concept)

### What it is

Replace mean (FedAvg) with more robust aggregators:

* Median
* Trimmed mean
* Krum

### Why it helps

* Reduces influence of outliers
* More resilient to poisoning

**Trade-off:**

* More computation
* Can hurt convergence when no attackers

Mention as future work.

---

## 4) Client Reputation / Auditing (Concept)

### What it is

Track historical behavior of clients:

* Persistent anomalies → down-weight or blacklist

**Trade-off:**

* Requires client identity and history
* Adds system complexity

---

## 5) Combine with DP (Optional)

* Clip updates (for robustness)
* Add small DP noise (for privacy)
* This combo is commonly used in practice
