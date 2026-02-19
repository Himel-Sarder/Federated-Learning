## 1) IID experiments (stable, fast convergence)

### When to use

Debugging, baseline results, comparing with centralized training.

### Server defaults (FedAvg)

* `num_rounds`: 10–20
* `fraction_fit`: 1.0 (use all clients)
* `min_fit_clients`: = number of clients you run
* `min_available_clients`: = number of clients you run
* `fraction_evaluate`: 1.0
* `min_evaluate_clients`: = number of clients you run

Example (2 clients):

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)
num_rounds = 15
```

### Client defaults

* local epochs: 1–3
* learning rate: 1e-3 (or 5e-4)
* batch size: 32/64

Recommended:

* `epochs=2`, `lr=0.001`, `batch_size=32`

---

## 2) Non-IID experiments (more rounds, reduce client drift)

### When to use

Thesis realism, label-skew clients, performance under heterogeneity.

### Why different

FedAvg struggles under Non-IID because client updates conflict (client drift). You want:

* more rounds
* fewer local epochs
* more clients per round (if possible)

### Server defaults (FedAvg)

* `num_rounds`: 30–60
* `fraction_fit`: 1.0 (small client count) or 0.6–0.9 (large client count)
* `min_fit_clients`: at least 60–100% of clients (prefer higher)
* `min_available_clients`: = number of clients you run
* evaluate on all clients

Example (5 clients):

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.8,
    min_fit_clients=4,
    min_available_clients=5,
    fraction_evaluate=1.0,
    min_evaluate_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average,
)
num_rounds = 40
```

### Client defaults

* local epochs: 1 (or 2 max)
* learning rate: 5e-4 (more stable)
* batch size: 32
* optional: early stopping locally is usually not used in FL rounds

Recommended:

* `epochs=1`, `lr=0.0005`, `batch_size=32`

If Non-IID is strong (clients almost single-class), consider switching strategy:

* FedProx (better default than FedAvg in Non-IID)

---

## 3) Imbalanced datasets (e.g., 92:8) (focus on recall/F1, not accuracy)

### When to use

Medical/rare event datasets.

### Key idea

Accuracy can look high even if the model ignores the minority class. So:

* use class-weighted loss
* report recall/F1/ROC-AUC
* keep training stable (not too aggressive epochs)

### Server defaults (FedAvg)

* `num_rounds`: 20–50
* `fraction_fit`: 1.0 (if few clients) or 0.7–1.0
* evaluate on all clients
* aggregate metrics (accuracy + add F1/recall if you compute them)

Example (2 clients):

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)
num_rounds = 30
```

### Client defaults (most important changes)

* local epochs: 1–2
* learning rate: 5e-4
* batch size: 32/64
* use class-weighted loss (recommended over SMOTE for FL baseline)

Recommended:

* `epochs=1`, `lr=0.0005`, `batch_size=32`
* `CrossEntropyLoss(weight=class_weights)`

And evaluate using:

* recall (minority class)
* F1-score
* ROC-AUC

---

## Quick presets (copy/paste values)

### IID preset (safe baseline)

* rounds: 15
* local epochs: 2
* lr: 1e-3
* fraction_fit: 1.0

### Non-IID preset (FedAvg)

* rounds: 40
* local epochs: 1
* lr: 5e-4
* fraction_fit: 0.8–1.0

### Imbalanced preset

* rounds: 30
* local epochs: 1
* lr: 5e-4
* loss: class-weighted CE
* report: F1/recall/ROC-AUC
