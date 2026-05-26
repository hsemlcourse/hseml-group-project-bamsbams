from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling import MLP, PINN, load_model_weights
from src.preprocessing import denormalize, normalize

META_PATH = ROOT_DIR / "data" / "processed" / "pinn_dataset_meta.json"
PINN_WEIGHTS = ROOT_DIR / "models" / "pinn_main.pt"
MLP_WEIGHTS = ROOT_DIR / "models" / "mlp_baseline.pt"

app = FastAPI(title="PINN/MLP Temperature API", version="0.1.0")


class ModelStore:
    def __init__(self) -> None:
        self.pinn: torch.nn.Module | None = None
        self.mlp: torch.nn.Module | None = None


store = ModelStore()


def load_meta() -> dict:
    default = {
        "normalization": {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "t": [0.0, 1.0],
            "temperature": [20.0, 60.0],
        }
    }
    if not META_PATH.exists():
        return default
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def infer_arch_from_state_dict(weights_path: Path) -> tuple[int, int]:
    state = torch.load(weights_path, map_location="cpu")
    linear_weight_keys = sorted(
        [k for k in state if k.startswith("net.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if len(linear_weight_keys) < 2:
        raise ValueError("Cannot infer architecture from state_dict")

    first_w = state[linear_weight_keys[0]]
    hidden_dim = int(first_w.shape[0])
    num_hidden_layers = len(linear_weight_keys) - 1
    return hidden_dim, num_hidden_layers


def build_model(model_name: str, hidden_dim: int, num_hidden_layers: int) -> torch.nn.Module:
    model_cls = PINN if model_name == "PINN" else MLP
    return model_cls(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)


def load_model(model_name: str) -> torch.nn.Module:
    if model_name == "PINN":
        if store.pinn is None:
            if not PINN_WEIGHTS.exists():
                raise FileNotFoundError(f"Weights not found: {PINN_WEIGHTS}")
            h, n = infer_arch_from_state_dict(PINN_WEIGHTS)
            store.pinn = load_model_weights(
                build_model("PINN", h, n),
                PINN_WEIGHTS,
                map_location="cpu",
            )
        return store.pinn

    if store.mlp is None:
        if not MLP_WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found: {MLP_WEIGHTS}")
        h, n = infer_arch_from_state_dict(MLP_WEIGHTS)
        store.mlp = load_model_weights(
            build_model("MLP", h, n),
            MLP_WEIGHTS,
            map_location="cpu",
        )
    return store.mlp


def predict_single(
    model: torch.nn.Module,
    x_phys: float,
    y_phys: float,
    t_phys: float,
    mins_maxs: dict,
) -> float:
    x_norm = normalize(np.array([x_phys]), *mins_maxs["x"])[0]
    y_norm = normalize(np.array([y_phys]), *mins_maxs["y"])[0]
    t_norm = normalize(np.array([t_phys]), *mins_maxs["t"])[0]
    xyt = torch.tensor([[x_norm, y_norm, t_norm]], dtype=torch.float32)
    with torch.no_grad():
        temp_norm = model(xyt).cpu().numpy().squeeze()
    return float(denormalize(np.array([temp_norm]), *mins_maxs["temperature"])[0])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict:
    return load_meta()


@app.get("/predict/pinn")
def predict_pinn(
    x: float = Query(..., description="Physical x coordinate"),
    y: float = Query(..., description="Physical y coordinate"),
    t: float = Query(..., description="Physical time coordinate"),
) -> dict:
    meta_obj = load_meta()
    mins_maxs = meta_obj["normalization"]

    if not mins_maxs["x"][0] <= x <= mins_maxs["x"][1]:
        raise HTTPException(status_code=400, detail="x is out of range")
    if not mins_maxs["y"][0] <= y <= mins_maxs["y"][1]:
        raise HTTPException(status_code=400, detail="y is out of range")
    if not mins_maxs["t"][0] <= t <= mins_maxs["t"][1]:
        raise HTTPException(status_code=400, detail="t is out of range")

    try:
        model = load_model("PINN")
        temperature = predict_single(model, x, y, t, mins_maxs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model": "PINN",
        "x": x,
        "y": y,
        "t": t,
        "temperature": temperature,
    }


@app.get("/predict/mlp")
def predict_mlp(
    x: float = Query(..., description="Physical x coordinate"),
    y: float = Query(..., description="Physical y coordinate"),
) -> dict:
    meta_obj = load_meta()
    mins_maxs = meta_obj["normalization"]
    t = 1.0

    if not mins_maxs["x"][0] <= x <= mins_maxs["x"][1]:
        raise HTTPException(status_code=400, detail="x is out of range")
    if not mins_maxs["y"][0] <= y <= mins_maxs["y"][1]:
        raise HTTPException(status_code=400, detail="y is out of range")

    try:
        model = load_model("MLP")
        temperature = predict_single(model, x, y, t, mins_maxs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model": "MLP",
        "x": x,
        "y": y,
        "t": t,
        "temperature": temperature,
    }
