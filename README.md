# FedAvgM – Setup & Execution Guide

This README provides instructions for setting up the environment, installing dependencies, and running the `fedavgm` project locally.

---

## 1. Create & Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Upgrade pip and Install Dependencies

```
python -m pip install --upgrade pip
pip install "flwr[simulation]==1.5.0" hydra-core==1.3.2 matplotlib cython
```

Alternative
```
python -m pip install --upgrade "flwr[simulation]"
```

## 3. Install TensorFlow (CPU)
```
pip install tensorflow==2.12.1
```

If this fails
```
pip install tensorflow-cpu==2.12.0
```

Or:
```
python -m pip install --upgrade pip
python -m pip install tensorflow
```

## 4. Install Hydra
```
pip install hydra-core==1.3.2
```

## 5. Make the Project Importable
```
pip install -e .
```

## 6. Run the Project
```
python -m fedavgm.main
```

Notes

Always activate the virtual environment before running commands.

If TensorFlow installation fails, use the fallback options above.

pip install -e . ensures the project is importable as a module.

## 7. ModuleNotFoundError: No module named 'fedavgm'

# add parent folder to PYTHONPATH for this command only and run main.py
```
$env:PYTHONPATH = (Resolve-Path ..).Path
python ..\main.py dataset=imagenette model=mobilenetv2 num_clients=4 num_rounds=2 client.local_epochs=1
```
