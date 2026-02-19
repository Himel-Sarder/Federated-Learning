import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_one_epoch(model, loader, device, lr=1e-5):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    preds_all, y_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            preds = torch.argmax(logits, dim=1)

            total_loss += float(loss.item()) * yb.size(0)
            total_n += yb.size(0)

            preds_all.append(preds.cpu().numpy())
            y_all.append(yb.cpu().numpy())

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(y_all)

    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, acc
