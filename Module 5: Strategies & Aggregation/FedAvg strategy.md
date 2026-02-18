## FedAvg Strategy (Federated Averaging)

### What is FedAvg?

**FedAvg (Federated Averaging)** is the standard aggregation algorithm in Federated Learning.
It updates the global model by computing a **weighted average of client model updates** after local training.

It is the default strategy used in Flower.

---

## How FedAvg Works (Step-by-Step)

1. **Server initializes** a global model
2. **Server samples clients** for the round
3. **Server sends** global parameters to selected clients
4. **Each client trains locally** for a few epochs
5. **Clients send updated parameters** back to the server
6. **Server aggregates updates using FedAvg**
7. **Server updates** the global model
8. Repeat for multiple rounds

---

## Mathematical Formulation

[
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_t^k
]

Where:

* ( w_t^k ): model parameters from client (k) after local training
* ( n_k ): number of training samples on client (k)
* ( N = \sum_{k=1}^{K} n_k ): total number of samples

This ensures clients with more data have more influence.

---

## FedAvg in Flower (Server Code Example)

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

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

---

## Why FedAvg Works Well (and When It Doesn’t)

### Works well when:

* Data is IID or mildly Non-IID
* Clients have similar data sizes
* Local epochs are small to moderate

### Struggles when:

* Data is highly Non-IID
* Clients have very different data distributions
* Severe class imbalance exists
* Many local epochs cause client drift

---

## Common Improvements over FedAvg (Mention in Thesis)

* **FedProx**: adds a proximal term to reduce client drift
* **SCAFFOLD**: uses control variates to correct client drift
* **Personalized FL**: client-specific models
