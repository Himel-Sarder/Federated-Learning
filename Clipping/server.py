import flwr as fl

def weighted_average(metrics):
    # metrics is: List[Tuple[num_examples, Dict[str, scalar]]]
    accs = [n * m["accuracy"] for n, m in metrics]
    ns = [n for n, _ in metrics]
    return {"accuracy": sum(accs) / sum(ns)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    fraction_evaluate=1.0,
    min_evaluate_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )
