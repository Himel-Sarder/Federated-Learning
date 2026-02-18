## 1. What is `evaluate()` in Flower?

### Definition

In Flower, `evaluate()` is the method that performs **local evaluation (testing)** of the global model on each client’s private test dataset during a federated learning round.

Flower calls `evaluate()` automatically after aggregation (or when configured) to measure model performance.

---

## 2. What Happens Inside `evaluate()`?

### Typical Implementation (Your PyTorch Example)

```python
def evaluate(self, parameters, config):
    self.set_parameters(parameters)     # 1) Load global model
    acc = test(model, test_loader)      # 2) Evaluate on local test data
    return float(1 - acc), len(test_loader.dataset), {"accuracy": float(acc)}
```

### Step-by-step

1. **Load global parameters**
   The client updates its local model with the latest global model received from the server.

2. **Local evaluation**
   The client evaluates the model on its private test dataset (no training here).

3. **Return metrics to server**
   The client returns:

   * loss (Flower expects a loss value)
   * number of test examples (used for weighted aggregation)
   * metrics dictionary (e.g., accuracy)

---

## 3. How Flower Aggregates Evaluation Metrics

By default, Flower aggregates **loss** across clients using a weighted average based on the number of test examples.

To aggregate **custom metrics (e.g., accuracy)**, you must define a server-side aggregation function.

### Example (Server-side Metric Aggregation)

```python
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)
```

### What this does

* Receives metrics from all clients:
  `(num_examples, {"accuracy": acc})`
* Computes a **weighted average accuracy**
* Reports a single global accuracy per round on the server

---

## 4. Evaluation Scheduling in Flower

Evaluation is typically scheduled:

* After each aggregation round
* According to the server strategy configuration

Example:

```python
strategy = fl.server.strategy.FedAvg(
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
)
```

This means all available clients evaluate the global model each round.

---

## 5. Why Weighted Aggregation Matters

Clients may have different test set sizes.
Weighted aggregation ensures that clients with more test samples contribute proportionally more to the global accuracy.

This approximates centralized evaluation on the combined test set.

---

## 6. Common Pitfalls

* Not providing `evaluate_metrics_aggregation_fn`
  This results in warnings and no aggregated accuracy on the server.

* Returning wrong values from `evaluate()`
  The return signature must be:
  `(loss: float, num_examples: int, metrics: dict)`

* Mixing train and test loaders
  Always evaluate on `test_loader`, not `train_loader`.
