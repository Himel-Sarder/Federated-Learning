import flwr as fl
import torch
from .model import MLP
from .train_eval import train_one_epoch, evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_params(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_params(model, params):
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)

class ThyroidClient(fl.client.NumPyClient):
    def __init__(self, cid: int, train_loader, test_loader, in_dim: int, n_classes: int):
        self.cid = cid
        self.model = MLP(in_dim, n_classes).to(DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-3))

        for _ in range(local_epochs):
            train_one_epoch(self.model, self.train_loader, DEVICE, lr=lr)

        return get_params(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, DEVICE)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}
