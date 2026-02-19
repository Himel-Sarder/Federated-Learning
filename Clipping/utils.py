import torch
import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

def train_one_epoch(model, loader, device, lr=1e-3, poison=False):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Poisoning (label flipping) for binary class {0,1}
        if poison:
            y = 1 - y

        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

def evaluate_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)
