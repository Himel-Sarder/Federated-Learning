import flwr as fl

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    acc = sum(n * m["accuracy"] for n, m in metrics) / max(total, 1)
    return {"accuracy": acc}

def build_strategy(num_clients: int, lr: float, local_epochs: int):
    def fit_config(server_round: int):
        return {"lr": lr, "local_epochs": local_epochs}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=1.0,
        min_evaluate_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
    )
    return strategy
