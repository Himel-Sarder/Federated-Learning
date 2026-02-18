<img width="474" height="389" alt="image" src="https://github.com/user-attachments/assets/ee4e75b7-39eb-44d5-a10a-b5e1a64f93c3" />

## Client, Server, Round, and Aggregation in Federated Learning

## 1. Client

### Definition

A **client** is a participating node in Federated Learning that holds private local data and performs local model training.

### Role

* Stores private data locally (data never leaves the client)
* Receives the global model from the server
* Trains the model on local data
* Sends model updates (weights or gradients) back to the server

### Examples

* A hospital in a medical FL system
* A mobile phone in keyboard prediction
* An IoT device in sensor analytics

### Key Properties

* Data privacy is preserved
* Clients may have different data distributions
* Clients may be intermittently available

---

## 2. Server

### Definition

The **server** is the central coordinator that manages the federated training process.

### Role

* Initializes the global model
* Selects a subset of clients in each round
* Sends the current global model to selected clients
* Aggregates client updates
* Updates and redistributes the global model

### Key Properties

* Does not store raw client data
* Acts as the orchestrator of training
* Can apply security checks or aggregation strategies

---

## 3. Round (Communication Round)

### Definition

A **round** is one complete cycle of communication and training between the server and participating clients.

### What happens in one round

1. Server sends the current global model to selected clients
2. Clients train locally on their own data
3. Clients send model updates to the server
4. Server aggregates updates and updates the global model

### Importance

* Training progresses over multiple rounds
* More rounds usually lead to better convergence
* Communication cost increases with more rounds

---

## 4. Aggregation

### Definition

**Aggregation** is the process by which the server combines model updates received from multiple clients to form a new global model.

### Common Method

* Federated Averaging (FedAvg)
  The server computes a weighted average of client model parameters based on local dataset sizes.

### Purpose

* Combine distributed learning into one global model
* Ensure contributions from multiple clients are reflected
* Improve model generalization across clients

### Challenges

* Non-IID data can bias aggregation
* Malicious client updates can corrupt the global model
* Communication efficiency must be considered
If you want, the next topic can be:
IID vs Non-IID data in Federated Learning, with examples and impact on training.
