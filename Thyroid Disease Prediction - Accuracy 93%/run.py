import numpy as np
import flwr as fl

from src.config import (
    NUM_CLIENTS, NUM_ROUNDS, BATCH_SIZE, SEED,
    CSV_PATH, TARGET_COL, LR, LOCAL_EPOCHS
)
from src.data import load_csv, preprocess_split, split_iid, make_loader
from src.fl_client import ThyroidClient
from src.fl_server import build_strategy

def main():
    np.random.seed(SEED)

    X, y = load_csv(CSV_PATH, TARGET_COL)
    X_train, X_test, y_train, y_test, _ = preprocess_split(X, y, seed=SEED)

    in_dim = X_train.shape[1]
    n_classes = int(len(np.unique(y_train)))

    client_splits = split_iid(X_train, y_train, NUM_CLIENTS, seed=SEED)

    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    def client_fn(cid: str):
        cid_int = int(cid)
        Xc, yc = client_splits[cid_int]
        train_loader = make_loader(Xc, yc, batch_size=BATCH_SIZE, shuffle=True)
        return ThyroidClient(cid_int, train_loader, test_loader, in_dim, n_classes)

    strategy = build_strategy(NUM_CLIENTS, lr=LR, local_epochs=LOCAL_EPOCHS)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("Accuracy history:", history.metrics_distributed.get("accuracy"))

if __name__ == "__main__":
    main()
