## Secure Aggregation (Federated Learning)

### What is Secure Aggregation?

**Secure aggregation** is a privacy-preserving protocol that ensures the server can only see the **aggregated (sum/average) of client updates**, not any **individual client’s model updates**.

Even if the server is curious or partially malicious, it cannot inspect a single client’s gradients or weights.

---

## Why Secure Aggregation is Needed

Federated Learning does **not** share raw data, but:

* Model updates can leak private information (gradient leakage)
* A curious server can analyze individual updates
* Sensitive attributes can be inferred

Secure aggregation mitigates this by **hiding individual updates** from the server.

---

## How Secure Aggregation Works (High-Level)

1. Each client masks (encrypts) its model update
2. Masks are designed to cancel out when summed
3. Server aggregates masked updates
4. Individual updates remain hidden
5. Only the final aggregate is revealed

This can be implemented using cryptographic masking or homomorphic encryption (protocol-level).

---

## What Secure Aggregation Protects Against

* Curious server inspecting client updates
* Gradient inversion attacks at server side
* Leakage of sensitive client-level information

---

## What Secure Aggregation Does NOT Protect Against

* Model poisoning by malicious clients
* Inference attacks using the final global model
* Privacy leakage from the trained model itself

Secure aggregation is **orthogonal** to robustness and DP.

---

## Secure Aggregation in Flower (Practical Note)

Flower itself focuses on orchestration.
Full secure aggregation requires protocol-level support or external libraries.

In practice (for thesis/experiments):

* You describe secure aggregation conceptually
* Optionally integrate a secure aggregation protocol
* Or use Flower-compatible setups with secure aggregation in deployment environments

## Comparison: Secure Aggregation vs Differential Privacy

| Aspect                                  | Secure Aggregation     | Differential Privacy    |
| --------------------------------------- | ---------------------- | ----------------------- |
| Protects individual updates from server | Yes                    | Partially               |
| Adds noise                              | No                     | Yes                     |
| Utility loss                            | None (ideally)         | Yes (accuracy drop)     |
| Protects against poisoning              | No                     | No                      |
| Complexity                              | High (cryptography)    | Medium                  |
| Typical use                             | Protocol-level privacy | Algorithm-level privacy |


## Light “Conceptual Demo” (Not Real Secure Aggregation)

For teaching purposes only (not cryptographically secure):

```python
# Conceptual masking idea (do NOT use in production)
masked_params = [p + np.random.randn(*p.shape) for p in params]
# server aggregates masked_params from all clients
# masks cancel out in sum if coordinated
```

This shows the idea of masking, but real secure aggregation needs proper cryptographic protocols.
