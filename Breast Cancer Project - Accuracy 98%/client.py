import flwr as fl
import torch
from utils import Net, load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, input_dim = load_data()
model = Net(input_dim).to(DEVICE)

def train(model, loader, epochs=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

def test(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, train_loader, epochs=1)
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test(model, test_loader)
        return float(1 - acc), len(test_loader.dataset), {"accuracy": float(acc)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
