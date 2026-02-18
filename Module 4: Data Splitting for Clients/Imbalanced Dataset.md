## 1. IID Split with an Imbalanced Dataset

Even with IID splitting:

* Each client will also have ~92:8 distribution
* Minority class samples per client become **very small**
* Some clients may have very few minority samples
* Local training becomes noisy and unstable for the minority class
* FedAvg can bias the global model toward the majority class

**Conclusion:** IID does not fix imbalance; it only preserves the global imbalance at each client.

---

## 2. Non-IID Split with an Imbalanced Dataset (Worse Case)

If you split by label/class:

* Some clients may get almost only majority class
* Some clients may get very few or no minority samples
* FedAvg convergence degrades significantly
* The global model can become highly biased
* Minority class recall drops sharply

---

## 3. What You Should Do (Practical Solutions)

### A) Use Class-Weighted Loss (Recommended)

This helps the model care more about minority samples.

```python
import torch
from collections import Counter

# compute class weights from training labels
counts = Counter(y_train.numpy())
total = sum(counts.values())
weights = [total / counts[c] for c in sorted(counts.keys())]
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
```

Use this `loss_fn` in your `train()` function on each client.

---

### B) Use Evaluation Metrics Beyond Accuracy

Report:

* Precision
* Recall
* F1-score
* ROC-AUC

Accuracy alone is misleading for 92:8 imbalance.

---

### C) Client-Side Oversampling (Optional, Careful)

You can oversample minority class **locally on each client**:

* Random oversampling
* SMOTE (only if features are continuous and you do it carefully)

In FL, SMOTE can introduce distribution shifts across clients, so mention this as a limitation if used.

---

### D) Stratified Client Split (When Possible)

When creating clients, ensure each client has at least some minority samples:

* Enforce minimum minority samples per client
* Avoid clients with zero minority class

---

### E) Tune FedAvg Parameters

* Use smaller local epochs to reduce client drift
* Increase number of rounds
* Consider advanced strategies (FedProx, SCAFFOLD) if Non-IID + imbalance is severe

