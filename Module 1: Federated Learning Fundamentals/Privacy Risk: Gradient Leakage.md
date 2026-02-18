<img width="725" height="344" alt="image" src="https://github.com/user-attachments/assets/fdb727f8-802e-43b1-b72d-4e17255a1ab8" />

## 1. What is Gradient Leakage?

### Definition

**Gradient leakage** is a privacy risk in Federated Learning where an attacker can **reconstruct or infer sensitive information about a client’s local training data** from the model updates (gradients or weights) sent by the client to the server.

Even though raw data is not shared in Federated Learning, **model updates can still leak information** about the underlying data.

---

## 2. Why Gradient Leakage Happens

In Federated Learning, clients send:

* Gradients
* Model weight updates

These updates are mathematically derived from local data.
Under certain conditions, an adversary can reverse-engineer:

* Input features
* Labels
* Sensitive attributes

This breaks the privacy guarantee if not protected.

---

## 3. How Gradient Leakage Attacks Work (High-Level)

An attacker (malicious server or eavesdropper) can:

1. Observe gradients sent by a client
2. Use optimization techniques to find an input that would produce the same gradients
3. Reconstruct approximate training samples
4. Infer sensitive attributes (e.g., disease label, image content)

This class of attacks is known as **gradient inversion** or **reconstruction attacks**.

---

## 4. Example (Healthcare Scenario)

In a federated medical system:

* A hospital trains a cancer prediction model locally
* The hospital sends model gradients to the server
* A malicious server analyzes gradients
* The server can potentially infer whether rare disease cases exist in that hospital’s data
* In extreme cases, parts of patient images or features can be reconstructed

This violates patient privacy.

---

## 5. Impact on FedAvg Performance and Privacy

* Gradient leakage does not directly reduce model accuracy
* It introduces serious **privacy risks**
* FedAvg alone does not provide privacy guarantees
* Additional privacy mechanisms are required

---

## 6. Common Defenses Against Gradient Leakage

1. Secure Aggregation
   The server only sees aggregated updates from many clients, not individual client updates.

2. Differential Privacy
   Noise is added to gradients before sending, limiting information leakage.

3. Encryption
   Updates are encrypted during transmission.

4. Gradient Clipping
   Limits the magnitude of gradients to reduce information leakage.

5. Federated Dropout
   Reduces information content per update by sending partial model updates.
