## How Aggregation Works in Federated Learning (FedAvg)

### What is Aggregation?

**Aggregation** is the server-side process of combining model updates received from multiple clients into a single **global model** after each federated round.

In Flower, aggregation is controlled by the **strategy** (most commonly **FedAvg**).

---

## Step-by-Step Aggregation Workflow

1. **Server selects clients**
   Based on `fraction_fit` and `min_fit_clients`.

2. **Server sends global model**
   The current global parameters are broadcast to selected clients.

3. **Clients train locally (`fit()`)**
   Each client updates the model using its private data.

4. **Clients send updates**
   Each client returns:

   * updated parameters
   * number of local samples (`num_examples`)

5. **Server aggregates (FedAvg)**
   The server computes a **weighted average** of parameters.

6. **Global model update**
   The aggregated parameters become the new global model.

7. **Evaluation (optional)**
   Clients evaluate the global model and server aggregates metrics.

---

## FedAvg Aggregation Formula
<img width="964" height="307" alt="image" src="https://github.com/user-attachments/assets/cafe8e70-591a-4bb6-bb56-50c6dd71f38b" />

This gives more weight to clients with more data.

---

## How Flower Implements Aggregation (Conceptual)

* Flower receives a list of `(parameters, num_examples)` from clients
* For each layer:

  * Multiply client parameters by `num_examples`
  * Sum across clients
  * Divide by total samples
* The result becomes the new global parameter for that layer

---

## Why Weighting Matters

* Prevents small clients from dominating the update
* Approximates centralized training behavior
* Improves stability when client data sizes differ

---

## What Aggregation Looks Like in Logs

Typical Flower logs:

* `aggregate_fit: received X results and 0 failures`
* This means X client updates were aggregated successfully

---

## Limitations of FedAvg Aggregation

* Sensitive to Non-IID data (client drift)
* Can converge slowly under label skew
* Can bias toward dominant clients
* Does not provide privacy by itself

---

## Common Improvements to Aggregation

* FedProx: adds regularization to reduce client drift
* SCAFFOLD: corrects client drift with control variates
* Robust aggregation: median, trimmed mean (against malicious clients)
