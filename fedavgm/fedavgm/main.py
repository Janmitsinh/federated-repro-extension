"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import pickle
from pathlib import Path
import sys
import traceback

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Monkey patch to bypass Windows Job Object error (OSError: [Errno 0] AssignProcessToJobObject() failed)
try:
    import ray._private.utils
    import ray._private.services
    def noop(*args, **kwargs):
        pass
    ray._private.utils.set_kill_child_on_death_win32 = noop
    ray._private.services.set_kill_child_on_death_win32 = noop
except ImportError:
    pass # Ray might not be installed or used yet

from fedavgm.client import generate_client_fn
from fedavgm.dataset import partition
from fedavgm.server import get_evaluate_fn


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    np.random.seed(2020)

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = instantiate(
        cfg.dataset
    )

    partitions = partition(x_train, y_train, cfg.num_clients, cfg.noniid.concentration)

    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    # 3. Define your clients
    client_fn = generate_client_fn(partitions, cfg.model, num_classes)

    # 4. Define your strategy
    evaluate_fn = get_evaluate_fn(
        instantiate(cfg.model), x_test, y_test, cfg.num_rounds, num_classes
    )

    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)

    # 5. Start Simulation
    # 5. Start Simulation
    # Force local_mode=True for Windows to avoid 'CreateFileMapping() failed' (paging file too small / resource exhaustion)
    # The previous attempts with local_mode=False caused OOM/Handle leaks on this specific environment.
    ray_local_mode = True
    
    ray_init_args = {
        "local_mode": ray_local_mode,
        "include_dashboard": False,
        # "num_cpus": cfg.num_cpus, # Let Ray detect or use default in local mode
        # "num_gpus": cfg.num_gpus 
    }

    try:
        print(f"Starting simulation with ray_init_args: {ray_init_args}")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
            ray_init_args=ray_init_args,
        )
    except Exception as exc:
        print("Error starting simulation (Ray/TensorFlow). Attempting fallback.")
        print(f"Original error: {exc}")
        # traceback.print_exc(file=sys.stdout)
        
        # Fallback: Try the opposite mode
        fallback_mode = not ray_local_mode
        print(f"Retrying with local_mode={fallback_mode} ...")
        
        ray_init_args["local_mode"] = fallback_mode
        
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=cfg.num_clients,
                config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                strategy=strategy,
                client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
                ray_init_args=ray_init_args,
            )
        except Exception as exc2:
            print(f"Fallback failed: {exc2}")
            print("Aborting run.")
            raise

    # If we get here, history should be available
    try:
        _, final_acc = history.metrics_centralized["accuracy"][-1]
    except Exception:
        final_acc = None

    # 6. Save your results
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
