
## 1. What is `NumPyClient` in Flower?

`NumPyClient` is a **client interface (base class)** provided by Flower for Federated Learning.
You create your own client by **inheriting** from `fl.client.NumPyClient` and implementing a few methods.

Flower uses these methods to:

* send the global model to the client
* let the client train locally
* collect updated parameters and evaluation metrics

It is called **NumPyClient** because Flower exchanges model parameters as **NumPy arrays**.

---

## 2. Why use `NumPyClient`?

* Easy to implement
* Framework-agnostic (works with PyTorch, TensorFlow, plain NumPy models)
* You control training/evaluation code
* Flower handles communication and orchestration

---

## 3. Required/Typical Methods in `NumPyClient`

### 3.1 `get_parameters(self, config)`

**Purpose:**
Return the client’s current model parameters so the server can aggregate them.

**In your PyTorch code:**

```python
def get_parameters(self, config):
    return [val.cpu().numpy() for val in model.state_dict().values()]
```

Meaning:

* take model weights
* convert to NumPy
* return as a list

---

### 3.2 `set_parameters(self, parameters)`

**Purpose:**
Receive parameters from the server and load them into the local model.

**In your code:**

```python
def set_parameters(self, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
```

Meaning:

* match incoming arrays with parameter names
* convert NumPy arrays back to tensors
* load them into the model

---

### 3.3 `fit(self, parameters, config)`

**Purpose:**
This runs **local training** on the client in one FL round.

**Return values:**

* updated parameters
* number of local training examples (for FedAvg weighting)
* optional training metrics

Example:

```python
def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(model, train_loader, epochs=1)
    return self.get_parameters({}), len(train_loader.dataset), {}
```

Meaning:

* set global model
* train locally
* return updated weights and dataset size

---

### 3.4 `evaluate(self, parameters, config)`

**Purpose:**
Evaluate the global model on the client’s local test data.

**Return values:**

* loss
* number of test examples
* metrics (like accuracy)

Example:

```python
def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    acc = test(model, test_loader)
    return float(1 - acc), len(test_loader.dataset), {"accuracy": float(acc)}
```

Meaning:

* load global model
* test locally
* send loss/accuracy to server

---

## 4. The Full Life Cycle (What Flower Does Internally)

For each round, Flower typically does:

1. Server sends global parameters
2. Client calls `fit()`
3. Client trains locally
4. Client returns updated parameters
5. Server aggregates (FedAvg)
6. Server sends updated global parameters
7. Client calls `evaluate()`
8. Client returns accuracy/loss
9. Repeat for next round
