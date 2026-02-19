## Server-side Evaluation (Flower)

### What is Server-side Evaluation?

In Flower, **server-side evaluation** means the server collects evaluation results from clients and aggregates them to report **global performance per round** (e.g., global accuracy and loss).

The server itself does not usually hold data. Instead:

* Clients evaluate the global model locally using `evaluate()`
* The server aggregates those metrics

This is called “server-side evaluation” because the **final metric is computed and logged on the server**.

---

## How Server-side Evaluation Works (Flow)

1. Server finishes aggregation for a round
2. Server selects clients for evaluation (`fraction_evaluate`, `min_evaluate_clients`)
3. Server sends the global model to selected clients
4. Each client runs `evaluate()` on local test data
5. Clients return `(loss, num_examples, metrics)`
6. Server aggregates:

   * loss (weighted by `num_examples`)
   * metrics (using `evaluate_metrics_aggregation_fn`)
7. Server logs global metrics for that round

---

## Enable Proper Server-side Evaluation (Flower Code)

### 1) Client: Return metrics from `evaluate()`

```python
def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    acc = test(model, test_loader)
    loss = 1.0 - acc  # or compute real loss if you want
    return float(loss), len(test_loader.dataset), {"accuracy": float(acc)}
```

---

### 2) Server: Aggregate metrics

```python
import flwr as fl

def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    nums = [n for n, _ in metrics]
    return {"accuracy": sum(accs) / sum(nums)}

strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)
```

---

## Where You See Server-side Evaluation Results

In server logs, you will see something like:

```
[SUMMARY]
History (loss, distributed):
  round 1: 0.21
  round 2: 0.15
History (metrics, distributed):
  accuracy:
    round 1: 0.64
    round 2: 0.92
```

This “distributed” history is the **server-side aggregated evaluation**.

---

## Why Server-side Evaluation Matters

* Gives a **global view of model performance**
* Comparable to centralized test accuracy
* Enables plotting learning curves
* Needed for experiment comparison (IID vs Non-IID, FedAvg vs FedProx)

---

## Common Pitfalls

* Not providing `evaluate_metrics_aggregation_fn`
  → accuracy will not be shown on the server

* Returning wrong tuple from `evaluate()`
  → Flower will error or ignore metrics

* Using training data in `evaluate()`
  → inflated metrics (data leakage)
