<img width="797" height="407" alt="image" src="https://github.com/user-attachments/assets/f0816ecf-0ce6-4b12-8fbb-c52c226375b4" />

## Model Poisoning Attack (Federated Learning)

### What is Model Poisoning?

**Model poisoning** is an attack where a malicious client intentionally sends **manipulated model updates** to the server to:

* degrade the global model’s performance, or
* introduce targeted misbehavior (backdoors), or
* bias predictions toward an attacker’s goal.

This is a major threat in open federated settings where clients are not fully trusted.

---

## How Model Poisoning Works (Concept)

A malicious client can:

* Train on **corrupted labels** (label flipping)
* Send **scaled or crafted gradients**
* Inject a **backdoor pattern** so that specific inputs are misclassified
* Skip training and send adversarial updates

Because the server aggregates updates (e.g., FedAvg), the malicious update can influence the global model.

---

## Types of Model Poisoning

1. **Untargeted poisoning**
   Goal: degrade overall accuracy (e.g., random noise in updates)

2. **Targeted poisoning (Backdoor attacks)**
   Goal: cause specific inputs to be misclassified while keeping overall accuracy high

---

## Why Model Poisoning is Effective in FL

* No access to raw data for server to verify updates
* FedAvg trusts client updates
* Small number of clients → one malicious client can have large impact
* Non-IID data amplifies attack impact

---

## Impact on FedAvg

* Reduced convergence speed
* Lower final accuracy
* Potential targeted misclassification
* Unstable training across rounds

---

## Basic Defenses (Lightweight)

### 1) Update Clipping (Client-side or Server-side)

Limit the magnitude of updates so one client cannot dominate aggregation.

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

---

### 2) Robust Aggregation (Concept)

Instead of mean (FedAvg), use:

* Median
* Trimmed mean
* Krum

These reduce the influence of outliers/malicious clients (conceptual for thesis).

---

### 3) Client Auditing (Concept)

* Monitor update norms
* Flag anomalous clients
* Temporarily exclude suspicious clients
