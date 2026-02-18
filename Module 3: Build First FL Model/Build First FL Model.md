## Topic 1: Simple Classification Model (Logistic Regression / MLP)

Before applying Federated Learning with Flower, you should understand a simple classification model. The two most common options are **Logistic Regression** and **MLP (Multilayer Perceptron)**. Both work well for tabular datasets such as the Breast Cancer dataset.

## 1) Logistic Regression (Baseline Model)

### Concept

Logistic Regression is a simple linear classifier with **no hidden layers**.

* Input features go directly to the output layer
* Suitable for binary classification
* Fast to train and less prone to overfitting
* Good baseline for comparison

### PyTorch Implementation

```python
import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)  # 2 classes

    def forward(self, x):
        return self.linear(x)
```

Use `CrossEntropyLoss` for training (no sigmoid/softmax needed in the model).
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

---

## 2) MLP (Multilayer Perceptron)

### Concept

MLP is a neural network with one or more hidden layers.

* Can learn non-linear relationships
* Higher model capacity than Logistic Regression
* Usually achieves better accuracy on complex datasets
* Slightly higher risk of overfitting

### PyTorch Implementation (1 Hidden Layer)

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```
