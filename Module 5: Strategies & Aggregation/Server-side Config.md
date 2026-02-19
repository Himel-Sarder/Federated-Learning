## Server-side Config (Flower)

Server-side config controls **how federated training runs**:

* how many rounds
* how many clients participate per round
* when rounds start
* how evaluation is done

These parameters directly affect **convergence, stability, and runtime**.

---

## Key Parameters You Must Know

### 1) `num_rounds`

**What it does:**
Number of federated communication rounds.

```python
config=fl.server.ServerConfig(num_rounds=15)
```

**Effect:**

* More rounds → higher accuracy (up to a point)
* More rounds → more communication cost
* Too few rounds → underfitting

**Typical values:** 10–30 (local simulation)

---

### 2) `fraction_fit`

**What it does:**
Fraction of available clients selected for training each round.

```python
fraction_fit=1.0   # use all clients
fraction_fit=0.5   # use 50% of clients per round
```

**Effect:**

* Higher fraction → more stable training
* Lower fraction → faster rounds, noisier updates
* With 2 clients, keep it `1.0`

---

### 3) `min_fit_clients`

**What it does:**
Minimum number of clients that must participate in training per round.

```python
min_fit_clients=2
```

**Effect:**

* If fewer clients connect than this number, the round will not start
* Prevents training with too few clients

---

### 4) `min_available_clients`

**What it does:**
Minimum number of clients that must be connected to the server to start a round.

```python
min_available_clients=2
```

**Effect:**

* Ensures enough clients are online
* Important for stability

---

### 5) `fraction_evaluate` and `min_evaluate_clients`

**What they do:**
Control how many clients evaluate the global model.

```python
fraction_evaluate=1.0
min_evaluate_clients=2
```

**Effect:**

* More evaluators → more reliable metrics
* Fewer evaluators → faster evaluation

---

### 6) `evaluate_metrics_aggregation_fn`

**What it does:**
Aggregates custom metrics (e.g., accuracy) across clients.

```python
def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    ns = [n for n, _ in metrics]
    return {"accuracy": sum(accs) / sum(ns)}
```

Without this, Flower will warn and won’t show aggregated accuracy.

---

## Full Example: Clean `server.py`

```python
import flwr as fl

def weighted_average(metrics):
    accs = [num_examples * m["accuracy"] for num_examples, m in metrics]
    nums = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accs) / sum(nums)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )
```

---

## How These Parameters Affect Results

| Parameter             | Increase Value → Effect               |
| --------------------- | ------------------------------------- |
| num_rounds            | Higher final accuracy, longer runtime |
| fraction_fit          | More stable training                  |
| min_fit_clients       | More reliable aggregation             |
| fraction_evaluate     | More stable metrics                   |
| min_available_clients | Prevents under-participation          |

---

## Common Mistakes

* `min_fit_clients` > actual running clients
  → training never starts
* `fraction_fit < 1.0` with very few clients
  → unstable results
* Not defining `evaluate_metrics_aggregation_fn`
  → accuracy not aggregated
