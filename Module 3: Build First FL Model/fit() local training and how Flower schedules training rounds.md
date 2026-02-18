## 1. What is `fit()` in Flower?

### Definition

In Flower, `fit()` is the method that performs **local training on each client** during one federated learning round.

Flower calls `fit()` automatically when the server schedules a training round.

---

## 2. What Happens Inside `fit()`?

### Typical Implementation (Your PyTorch Example)

```python
def fit(self, parameters, config):
    self.set_parameters(parameters)          # 1) Load global model
    train(model, train_loader, epochs=1)     # 2) Train locally on private data
    return self.get_parameters({}), len(train_loader.dataset), {}
```

### Step-by-step

1. **Load global parameters**
   The client updates its local model with the global model sent by the server using `set_parameters()`.

2. **Local training**
   The client trains the model on its private dataset for a fixed number of local epochs.

3. **Return updates to server**
   The client sends updated parameters back to the server along with:

   * the number of local samples (used by FedAvg for weighting)
   * optional training metrics

---

## 3. Why `fit()` is Important

* It defines **how much local learning** happens per round
* It controls:

  * local epochs
  * batch size
  * optimizer and learning rate
* It affects:

  * convergence speed
  * communication efficiency
  * model accuracy

More local epochs per `fit()` call usually reduce communication cost but can increase client drift in Non-IID settings.

---

## 4. How Flower Schedules Training Rounds

### Server-side Scheduling

Flower schedules training rounds based on the server configuration and strategy.

Example (`server.py`):

```python
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=15),
    strategy=strategy,
)
```

* `num_rounds=15` means 15 federated rounds will run
* Each round consists of:

  * client sampling
  * local training (`fit()`)
  * aggregation (FedAvg)
  * evaluation (`evaluate()`)

---

## 5. Client Sampling per Round

Flower’s strategy (e.g., FedAvg) decides how many clients participate in each round:

```python
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,         # fraction of available clients used for training
    min_fit_clients=2,        # minimum number of clients per round
    min_available_clients=2  # minimum connected clients to start a round
)
```

### Meaning

* `fraction_fit=1.0`
  Use all connected clients in each round

* `min_fit_clients=2`
  At least 2 clients must participate

* `min_available_clients=2`
  The round starts only if at least 2 clients are connected

---

## 6. Full Round Workflow (Textual Flow)

1. Server selects clients according to strategy
2. Server sends global model to selected clients
3. Flower triggers `fit()` on each selected client
4. Clients perform local training
5. Clients return updated parameters
6. Server aggregates updates (FedAvg)
7. Server optionally triggers `evaluate()`
8. Next round starts

---

## 7. Impact of `fit()` Design on Performance

* More local epochs:

  * Faster initial convergence
  * Less communication
  * Risk of client drift in Non-IID data

* Fewer local epochs:

  * More communication
  * More stable convergence across clients
