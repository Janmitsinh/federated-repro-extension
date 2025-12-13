
import pickle
from pathlib import Path
import sys
import traceback
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedavgm.client import generate_client_fn
from fedavgm.dataset import partition
from fedavgm.server import get_evaluate_fn

from flwr.common import (
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    ReconnectIns,
    DisconnectRes,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History

class SequentialClientProxy(ClientProxy):
    def __init__(self, cid: str, client_fn):
        super().__init__(cid)
        self.client_fn = client_fn
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                self._client = self.client_fn(self.cid)
                if hasattr(self._client, "to_client"):
                    self._client = self._client.to_client()
            except TypeError:
                 self._client = self.client_fn(self.cid)
        return self._client

    def get_properties(self, ins: GetPropertiesIns, timeout: Optional[float]) -> GetPropertiesRes:
        return self.client.get_properties(ins)

    def get_parameters(self, ins: GetParametersIns, timeout: Optional[float]) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        return self.client.evaluate(ins)

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        return self.client.reconnect(ins)

def run_sequential_simulation(
    client_fn,
    num_clients,
    num_rounds,
    strategy,
) -> History:
    print("Starting Sequential Simulation (No Ray)...")
    history = History()

    parameters: Optional[Parameters] = strategy.initialize_parameters(client_manager=SimpleClientManager())

    client_manager = SimpleClientManager()
    for i in range(num_clients):
        cid = str(i)
        proxy = SequentialClientProxy(cid, client_fn)
        client_manager.register(proxy)

    for server_round in range(1, num_rounds + 1):

        client_instructions = strategy.configure_fit(
            server_round=server_round, 
            parameters=parameters, 
            client_manager=client_manager
        )
        
        if not client_instructions:
            print(f"Round {server_round}: No clients sampled, stopping.")
            break
            
        print(f"Round {server_round}: Training on {len(client_instructions)} clients...")

        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[BaseException] = []
        
        for client_proxy, fit_ins in client_instructions:
            try:
                fit_res = client_proxy.fit(fit_ins, timeout=None)
                results.append((client_proxy, fit_res))
            except Exception as e:
                print(f"Client {client_proxy.cid} failure: {e}")
                failures.append(e)

        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = strategy.aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures
        )
        
        aggregated_parameters, aggregated_metrics = aggregated_result
        
        if aggregated_parameters is not None:
            parameters = aggregated_parameters
   
        if strategy.evaluate_fn:
            
            eval_res = strategy.evaluate(server_round, parameters=parameters)
            if eval_res:
                loss, metrics = eval_res
                print(f"Round {server_round} global evaluation: loss={loss}, acc={metrics.get('accuracy', 'N/A')}")
                history.add_loss_centralized(server_round, loss)
                history.add_metrics_centralized(server_round, metrics)

    return history


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(2020)
    print(OmegaConf.to_yaml(cfg))

    x_train, y_train, x_test, y_test, input_shape, num_classes = instantiate(
        cfg.dataset
    )

    partitions = partition(x_train, y_train, cfg.num_clients, cfg.noniid.concentration)
    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    client_fn = generate_client_fn(partitions, cfg.model, num_classes)

    evaluate_fn = get_evaluate_fn(
        instantiate(cfg.model), x_test, y_test, cfg.num_rounds, num_classes
    )

    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)

    try:
        history = run_sequential_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            strategy=strategy,
        )
    except Exception as exc:
        print("Error during sequential simulation:")
        traceback.print_exc()
        raise

    # Save results
    try:
        _, final_acc = history.metrics_centralized["accuracy"][-1]
    except Exception:
        final_acc = None

    save_path = HydraConfig.get().runtime.output_dir
    strategy_name = strategy.__class__.__name__
    if cfg.dataset.input_shape == [32, 32, 3]:
        dataset_type = "cifar10"
    elif cfg.dataset.input_shape == [28, 28, 1]:
        dataset_type = "fmnist"
    else:
        dataset_type = "imagenette"

    def format_variable(x):
        return f"{x!r}" if isinstance(x, bytes) else x

    file_suffix: str = (
        f"_{format_variable(strategy_name)}"
        f"_{format_variable(dataset_type)}"
        f"_clients={format_variable(cfg.num_clients)}"
        f"_rounds={format_variable(cfg.num_rounds)}"
        f"_C={format_variable(cfg.server.reporting_fraction)}"
        f"_E={format_variable(cfg.client.local_epochs)}"
        f"_alpha={format_variable(cfg.noniid.concentration)}"
        f"_server-momentum={format_variable(cfg.server.momentum)}"
        f"_client-lr={format_variable(cfg.client.lr)}"
        f"_acc={format_variable(final_acc) if final_acc is not None else 'None'}"
    )

    filename = "results" + file_suffix + ".pkl"
    print(f">>> Saving {filename}...")
    results_path = Path(save_path) / filename
    results = {"history": history}

    with open(str(results_path), "wb") as hist_file:
        pickle.dump(results, hist_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
