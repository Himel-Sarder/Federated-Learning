import sys
import argparse
import numpy as np
import flwr as fl
import torch

from utils import LogisticModel, train_one_epoch, evaluate_acc
from data_split import make_iid_client_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_parameters(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]

def set_model_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)

def clip_params(params, clip_norm=1.0):
    clipped = []
    for p in params:
        norm = np.linalg.norm(p)
        if norm > clip_norm:
            p = p * (clip_norm / (norm + 1e-8))
        clipped.append(p)
    return clipped

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader, malicious=False, use_clip=False, clip_norm=1.0):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.malicious = malicious
        self.use_clip = use_clip
        self.clip_norm = clip_norm

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        # local training (poison if malicious)
        train_one_epoch(
            self.model,
            self.train_loader,
            DEVICE,
            lr=1e-3,
            poison=self.malicious
        )

        new_params = get_model_parameters(self.model)

        # defense: clip update magnitude (simple version: clip parameters)
        if self.use_clip:
            new_params = clip_params(new_params, clip_norm=self.clip_norm)

        num_examples = len(self.train_loader.dataset)
        return new_params, num_examples, {
            "malicious": float(self.malicious),
            "clip": float(self.use_clip),
        }

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        acc = evaluate_acc(self.model, self.test_loader, DEVICE)

        # log per client
        print(f"[Client {self.cid}] evaluate -> acc={acc:.4f} | malicious={self.malicious} | clip={self.use_clip}")

        # Flower expects: (loss, num_examples, metrics)
        loss = float(1.0 - acc)
        return loss, len(self.test_loader.dataset), {"accuracy": float(acc)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=int, help="client id (0 or 1)")
    parser.add_argument("--clip", type=int, default=0, help="1=enable clipping defense, 0=disable")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="clipping norm")
    args = parser.parse_args()

    cid = args.cid
    use_clip = bool(args.clip)

    num_clients = 2
    train_loaders, test_loader, input_dim = make_iid_client_loaders(num_clients=num_clients, batch_size=32)

    model = LogisticModel(input_dim).to(DEVICE)

    malicious = (cid == 1)  # client 1 is attacker

    client = FlowerClient(
        cid=cid,
        model=model,
        train_loader=train_loaders[cid],
        test_loader=test_loader,
        malicious=malicious,
        use_clip=use_clip,
        clip_norm=args.clip_norm
    )

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
