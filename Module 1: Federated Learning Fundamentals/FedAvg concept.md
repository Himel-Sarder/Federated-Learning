## FedAvg (Federated Averaging) Concept in Federated Learning

---

## 1. What is FedAvg?

**FedAvg (Federated Averaging)** is the standard aggregation algorithm used in Federated Learning.
It updates the global model by taking a **weighted average of the model parameters** trained by multiple clients, where each client’s contribution is proportional to the size of its local dataset.

---

## 2. Why FedAvg is Needed

In Federated Learning:

* Data is distributed across clients
* Each client trains the model on different local data
* The server needs a way to combine these different local models into one global model

FedAvg provides a simple and efficient way to combine client updates.

---

## 3. How FedAvg Works (Step-by-Step)

1. The server initializes a global model.
2. In each round, the server selects a subset of clients.
3. Each selected client receives the global model and trains it locally for a few epochs.
4. Each client sends its updated model parameters to the server.
5. The server aggregates these parameters using weighted averaging.
6. The aggregated model becomes the new global model and is sent back to clients for the next round.

This process is repeated until the model converges.

---

## 4. Mathematical Formulation
<img width="946" height="331" alt="image" src="https://github.com/user-attachments/assets/8edd6662-63c4-46d5-ba63-280d3c138827" />


This means that clients with more data have a larger influence on the global model update.

---

## 5. Simple Numerical Example

Assume three clients participate in one round:

* Client A has 100 samples
* Client B has 300 samples
* Client C has 600 samples

Total samples ( N = 1000 )

Weights:

* A: ( 100/1000 = 0.1 )
* B: ( 300/1000 = 0.3 )
* C: ( 600/1000 = 0.6 )

<img width="995" height="104" alt="image" src="https://github.com/user-attachments/assets/9ee01488-5e4c-40b7-b840-fafb6cd248dc" />


---

## 6. Pseudocode for FedAvg

```
Initialize global model w0
For each round t = 1 to T:
    Server selects a set of clients
    For each client k:
        Receive global model wt
        Train locally on client data
        Send updated model wt^k to server
    Server aggregates:
        wt+1 = Σ (nk / N) * wt^k
Return final global model
```

---

## 7. Advantages of FedAvg

* Simple and efficient
* Communication-efficient compared to sending raw data
* Scales to many clients
* Widely adopted baseline algorithm

---

## 8. Limitations of FedAvg

* Sensitive to Non-IID data
* Vulnerable to malicious or poisoned updates
* Slower convergence compared to centralized training
* Assumes honest clients
ant, I can continue with the next topic:
IID vs Non-IID data in Federated Learning and why it matters for FedAvg performance.
