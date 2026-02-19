Great — here’s the next part of **Module 6: Evaluation & Logging** you asked for.
## 1) Client Metrics Logging (clean logging per round)

Add proper logging inside your client `evaluate()` so you can see per-client metrics in the console and return them to the server.

```python
def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    acc = test(model, test_loader)

    # Optional: compute real loss if you have it
    loss = 1.0 - acc

    print(f"[Client {self.cid}] Eval -> accuracy: {acc:.4f}, loss: {loss:.4f}")

    return float(loss), len(test_loader.dataset), {"accuracy": float(acc)}
```

If you want client id:

```python
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
```

Run clients like:

```bash
python client.py 0
python client.py 1
```

---

## 2) Capture Server-side History (for plotting)

Flower keeps a `history` object on the server. Modify `server.py` to store it:

```python
import flwr as fl

def weighted_average(metrics):
    accs = [n * m["accuracy"] for n, m in metrics]
    nums = [n for n, _ in metrics]
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
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )
```

---

## 3) Plot Accuracy & Loss with Matplotlib

Create a new file: `plot_metrics.py`

```python
import matplotlib.pyplot as plt

def plot_history(history):
    rounds = [r for r, _ in history.losses_distributed]
    losses = [v for _, v in history.losses_distributed]

    acc_rounds = [r for r, _ in history.metrics_distributed["accuracy"]]
    accuracies = [v for _, v in history.metrics_distributed["accuracy"]]

    plt.figure()
    plt.plot(rounds, losses)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated Loss per Round")
    plt.show()

    plt.figure()
    plt.plot(acc_rounds, accuracies)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Accuracy per Round")
    plt.show()
```

Then call this after training finishes in `server.py`:

```python
import plot_metrics

history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=15),
    strategy=strategy,
)

plot_metrics.plot_history(history)
```

---

## 4) What You Get (for your report/thesis)

* Per-round **global accuracy curve**
* Per-round **global loss curve**
* Client-wise logs in terminal
* Ready figures for report

---

## 5) Common Issues

* If accuracy does not appear:

  * Ensure `evaluate_metrics_aggregation_fn` is set
* If history is None:

  * Ensure you capture the return value of `start_server`
* If matplotlib not installed:

  ```bash
  pip install matplotlib
  ```
