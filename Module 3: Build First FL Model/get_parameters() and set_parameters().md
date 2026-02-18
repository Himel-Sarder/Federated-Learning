
## What are `get_parameters()` and `set_parameters()`?

In Flower, the server and clients exchange model parameters every round.
These two methods define **how parameters are sent and received** on the client side.

* `get_parameters()` → send local model weights to the server
* `set_parameters()` → receive global model weights from the server and load them into the local model

Flower calls these methods automatically during each federated round.

---

## `get_parameters(self, config)`

### Purpose

Returns the client’s current model parameters so the server can aggregate them (e.g., using FedAvg).

### Example (PyTorch + Flower)

```python
def get_parameters(self, config):
    return [val.cpu().numpy() for val in model.state_dict().values()]
```

### What happens here

* `model.state_dict().values()` gets all model weights
* `.cpu()` ensures tensors are on CPU
* `.numpy()` converts tensors to NumPy arrays
* The list of NumPy arrays is sent to the server

### Why NumPy?

Flower uses NumPy arrays to stay framework-agnostic (works with PyTorch, TensorFlow, etc.).

---

## `set_parameters(self, parameters)`

### Purpose

Loads parameters received from the server into the local model before training or evaluation.

### Example

```python
def set_parameters(self, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
```

### What happens here

* `parameters` is a list of NumPy arrays from the server
* `zip(keys, parameters)` matches each array to the correct layer name
* `torch.tensor(v)` converts NumPy arrays back to tensors
* `model.load_state_dict(...)` loads the global model into the local model

---

## How Flower Uses These in One Round (Flow)

1. Server sends global model → client
2. Flower calls `set_parameters(parameters)`
3. Client trains locally (`fit()`)
4. Flower calls `get_parameters()`
5. Client sends updated parameters to server
6. Server aggregates (FedAvg)
7. Repeat for next round

---

## Common Mistakes to Avoid

* Not converting tensors to NumPy in `get_parameters()`
* Not matching parameter order in `set_parameters()`
* Forgetting `.cpu()` when running on GPU
* Returning parameters in a different order than the model’s `state_dict()`

