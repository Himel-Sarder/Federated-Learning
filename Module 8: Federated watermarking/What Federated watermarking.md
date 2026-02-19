## What is Federated Watermarking?

A way to embed an **ownership/traceable signature** into a model that is trained across multiple clients, so later you can **verify**:

* “This model was produced by our FL system”
* “This model is derived from our model”
* “Which client group/version produced it” (fingerprinting)

## Why it’s harder in FL

* No centralized training loop
* Model updates are averaged, so the watermark must be **robust to aggregation**
* Clients may be Non-IID, and updates are noisy

---

## Safer watermark types for FL

### A) Weight-based watermark (white-box, safest to describe/implement)

Embed a bitstring into weights by adding a tiny **regularization term** during local training.

**Idea**

* Pick a subset of parameters (e.g., last layer weight matrix)
* Define a secret projection matrix ( P ) and target bit vector ( b )
* Add a loss term that nudges ( P \cdot \text{vec}(W) ) toward ( b )

**Verification**

* Later, compute ( \hat{b} = \text{sign}(P \cdot \text{vec}(W)) )
* Compare ( \hat{b} ) with original ( b ) → watermark match score

**Why this fits FL**

* Each client applies the same small regularizer
* FedAvg preserves the effect (if the watermark strength is small but consistent)

### B) Model signing + hash commitments (black-box friendly, very safe)

Not a “hidden in weights” watermark, but strong for provenance:

* Server signs model checkpoints (hash + digital signature)
* You can prove the model was produced by your training pipeline
* Great for auditability, less about “survives fine-tuning”

### C) Fingerprinting (trace which FL run/client group)

Embed a run-specific identifier (version id) so different training runs produce different signatures.

---

## How to “use” it in your Flower project (high-level)

If you choose **weight-based watermarking**, the clean FL integration is:

1. **Server generates watermark key** (secret (P) and bits (b))
2. Server sends watermark config to clients via `on_fit_config_fn`
3. Clients add watermark regularization term to their training loss
4. Server still uses normal FedAvg aggregation
5. After training, you verify watermark from the final global model

Important: keep the watermark strength small so accuracy doesn’t drop.
