## 1. Centralized Learning

### Definition

In centralized learning, all training data is collected from different sources and stored in a single central server. The machine learning model is trained using this combined dataset.

### How it works

Data from multiple clients (such as hospitals, mobile devices, or organizations) is sent to a central server. The server trains the model using all the data and produces a global model.

### Architecture

Clients → Central Server (Data Storage + Training) → Trained Model

### Advantages

* Simple architecture
* Faster training because data is in one place
* Easy to debug and manage
* Mature and widely used approach

### Disadvantages

* High privacy risk because raw data is shared
* High data transfer cost and bandwidth usage
* Single point of failure
* Legal and compliance issues for sensitive data (healthcare, finance)

### Example

Hospitals upload patient medical records to a cloud server to train a cancer prediction model.

---

## 2. Federated Learning

### Definition

Federated Learning is a distributed machine learning approach where the model is trained across multiple clients without transferring raw data to a central server. Only model updates are shared.

### How it works

The central server sends an initial model to multiple clients. Each client trains the model locally using its own data. The clients send model updates (weights or gradients) back to the server. The server aggregates these updates to create a new global model. This process is repeated over multiple rounds.

### Architecture

Central Server → Sends Model to Clients
Clients → Local Training → Send Model Updates
Central Server → Aggregation (e.g., FedAvg) → Updated Global Model

### Advantages

* Preserves data privacy because raw data never leaves clients
* Reduces data transfer of sensitive information
* Better compliance with data protection regulations
* Suitable for distributed and edge environments

### Disadvantages

* Communication overhead due to multiple training rounds
* Non-IID data across clients can slow convergence
* Vulnerable to model poisoning and adversarial updates
* More complex system design and debugging

### Example

Hospitals keep patient data locally and only share trained model updates to collaboratively build a cancer diagnosis model.

---

## 3. Centralized vs Federated Learning (Comparison)

| Aspect            | Centralized Learning                 | Federated Learning                  |
| ----------------- | ------------------------------------ | ----------------------------------- |
| Data location     | Central server                       | Stays on client devices             |
| Privacy           | Low                                  | High                                |
| Data transfer     | Raw data transferred                 | Only model updates transferred      |
| Legal compliance  | Difficult for sensitive data         | Easier to comply with regulations   |
| Scalability       | Limited by data movement             | Better for distributed environments |
| Security risk     | High impact if server is compromised | Lower risk to raw data              |
| System complexity | Low                                  | Higher                              |
| Failure point     | Single point of failure              | More resilient                      |
