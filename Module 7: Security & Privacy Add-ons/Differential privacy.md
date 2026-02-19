## Differential Privacy (DP) in Federated Learning

### What is Differential Privacy?

**Differential Privacy (DP)** is a formal privacy guarantee that ensures the output of a computation (e.g., model updates) does not significantly change whether any **single data point** is included or removed from the training set.

In FL, DP limits how much information about an individual sample can be inferred from model updates or the final model.

---

## Why Differential Privacy is Needed in FL

Even in Federated Learning:

* Model updates can leak sensitive information
* Gradient inversion and membership inference attacks exist
* Sharing updates repeatedly increases leakage risk

DP reduces this risk by **adding calibrated noise** to updates.

---

## How DP Works (High-Level)

Typical DP pipeline (client-side DP):

1. Compute gradients locally
2. **Clip gradients** to bound sensitivity
3. **Add Gaussian noise** to clipped gradients
4. Send noisy updates to server

This provides an ((\varepsilon, \delta))-DP guarantee (privacy budget).

---

## Light Implementation (Educational, Not Formal DP)

This is a simple way to approximate DP in your Flower client for experiments:

```python
import numpy as np

def clip_and_add_noise(params, clip_norm=1.0, sigma=0.01):
    noisy = []
    for p in params:
        norm = np.linalg.norm(p)
        if norm > clip_norm:
            p = p * (clip_norm / (norm + 1e-8))
        noise = sigma * np.random.randn(*p.shape)
        noisy.append(p + noise)
    return noisy
```

Use in `fit()`:

```python
def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(model, train_loader, epochs=1)

    params = self.get_parameters({})
    params = clip_and_add_noise(params, clip_norm=1.0, sigma=0.01)

    return params, len(train_loader.dataset), {}
```

**Note:** This is for demonstration only. Proper DP requires calibrated noise and privacy accounting.

---

## What DP Protects Against

* Gradient leakage
* Gradient inversion attacks
* Membership inference attacks
* Sample-level privacy leakage

---

## What DP Does Not Protect Against

* Model poisoning attacks
* Backdoor attacks
* Inference from final model behavior (partially mitigated only)

DP is about **privacy**, not **robustness**.

---

## Trade-off: Privacy vs Utility

* More noise (lower ε) → stronger privacy, lower accuracy
* Less noise (higher ε) → weaker privacy, higher accuracy

This trade-off should be reported in experiments.

---

## DP vs Secure Aggregation (Quick Comparison)

| Aspect                          | Differential Privacy | Secure Aggregation           |
| ------------------------------- | -------------------- | ---------------------------- |
| Adds noise                      | Yes                  | No                           |
| Protects individual samples     | Yes                  | No (protects client updates) |
| Protects against curious server | Yes (partially)      | Yes                          |
| Accuracy impact                 | Yes (trade-off)      | No (ideally)                 |
| Complexity                      | Medium               | High                         |
