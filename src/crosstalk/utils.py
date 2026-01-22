import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch

from .fake_flux_crosstalk import ZToZCrosstalkPlant, CrosstalkConfig
from .train_calibrator import build_model, TrainConfig


# ----------------------------
# Plant I/O
# ----------------------------

def load_plant_from_npz(
    npz_path: Union[str, Path],
    *,
    control_key: str = "Z",
    target_key: str = "domega_meas",
    cfg_key: str = "plant_cfg_json",
    override_truth_from_npz: bool = True,
) -> Tuple[ZToZCrosstalkPlant, Dict[str, Any], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load Crosstalk plant + config from an npz file created by your generator.

    Returns:
        plant:      ZToZCrosstalkPlant
        cfg_dict:   dict (json-loaded config)
        control:    (N,n) control array (default key: "Z")
        target:     (N,n) measured target array (default key: "domega_meas")
        truth:      dict with optional keys "C_true","D_true"
    """
    npz = np.load(str(npz_path), allow_pickle=True)

    if cfg_key not in npz:
        raise KeyError(f"npz missing '{cfg_key}'. Found keys: {list(npz.keys())}")

    cfg_json_raw = npz[cfg_key]
    if isinstance(cfg_json_raw, np.ndarray):
        cfg_json_raw = cfg_json_raw.item()
    if not isinstance(cfg_json_raw, str):
        raise TypeError(f"{cfg_key} must be a JSON string; got {type(cfg_json_raw)}")

    cfg_dict = json.loads(cfg_json_raw)
    xt_cfg = CrosstalkConfig(**cfg_dict)
    plant = ZToZCrosstalkPlant(xt_cfg)

    # data arrays
    if control_key not in npz or target_key not in npz:
        raise KeyError(
            f"npz must contain '{control_key}' and '{target_key}'. Found keys: {list(npz.keys())}"
        )
    control = np.asarray(npz[control_key], dtype=np.float64)
    target = np.asarray(npz[target_key], dtype=np.float64)

    # truth override (highly recommended)
    truth: Dict[str, np.ndarray] = {}
    if override_truth_from_npz:
        if "C_true" in npz:
            plant.C_true = np.asarray(npz["C_true"], dtype=np.float64)
            truth["C_true"] = plant.C_true.copy()
        if "D_true" in npz:
            plant.D_true = np.asarray(npz["D_true"], dtype=np.float64)
            truth["D_true"] = plant.D_true.copy()

    return plant, cfg_dict, control, target, truth


def plant_forward(
    plant: ZToZCrosstalkPlant,
    control: np.ndarray,
    *,
    add_noise: bool = False,
) -> np.ndarray:
    """
    Forward simulate target from control using plant.
    """
    control = np.asarray(control, dtype=np.float64)
    return plant.forward(control, add_noise=bool(add_noise))


# ----------------------------
# Model I/O
# ----------------------------

def load_calibrator_pt(
    pt_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load inverse calibrator checkpoint (.pt) that stores:
        ckpt["state_dict"]
        ckpt["meta"]["cfg"] (TrainConfig kwargs)
        ckpt["meta"]["n"]
    """
    dev = torch.device(device)
    ckpt = torch.load(str(pt_path), map_location=dev)

    meta = ckpt.get("meta", {})
    cfg_dict = meta.get("cfg", {})
    n = meta.get("n", None)
    if n is None:
        raise ValueError(f"Checkpoint meta missing 'n': {pt_path}")

    cfg = TrainConfig(**cfg_dict)
    model = build_model(cfg, n=int(n), device=dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def get_model_pt_from_report(
    report: Dict[str, Any],
    model_name: str,
    *,
    key_path: Tuple[str, ...] = ("per_model",),
    pt_field: str = "model_pt",
) -> str:
    """
    Extract model_pt path from your sweep report.

    Expected structure:
        report["per_model"][model_name]["row"]["model_pt"]  (default)
    """
    try:
        return report["per_model"][model_name]["row"][pt_field]
    except Exception as e:
        raise KeyError(
            f"Cannot find pt path for model '{model_name}'. "
            f"Expected report['per_model'][model]['row']['{pt_field}']"
        ) from e


# ----------------------------
# One-hot evaluation (blank + calibrated)
# ----------------------------

def onehot_target(n: int, amp: float = 1.0) -> np.ndarray:
    return float(amp) * np.eye(int(n), dtype=np.float64)


def eval_onehot_blank(
    plant: ZToZCrosstalkPlant,
    *,
    amp: float = 1.0,
    add_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blank baseline:
        requested target T = amp*I
        NO calibration: control = T
        achieved target: plant.forward(control)
    Returns:
        T (n,n), control_blank (n,n), T_ach (n,n)
    """
    n = int(plant.cfg.n)
    T = onehot_target(n, amp=amp)
    control_blank = T.copy()
    T_ach = plant_forward(plant, control_blank, add_noise=add_noise)
    return T, control_blank, T_ach


def eval_onehot_calibrated(
    plant: ZToZCrosstalkPlant,
    model: Any,  # CalibrationModelBase-like (must have .predict and .n)
    *,
    amp: float = 1.0,
    add_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrated:
        requested target T = amp*I
        control_hat = model.predict(T)
        achieved target = plant.forward(control_hat)
    Returns:
        T (n,n), control_hat (n,n), T_ach (n,n)
    """
    T = onehot_target(int(model.n), amp=amp)

    control_hat = model.predict(T)
    if isinstance(control_hat, torch.Tensor):
        control_hat = control_hat.detach().cpu().numpy()
    control_hat = np.asarray(control_hat, dtype=np.float64)

    T_ach = plant_forward(plant, control_hat, add_noise=add_noise)
    return T, control_hat, T_ach



# ----------------------------
# Plot crosstalk matrix
# ----------------------------

def plot_matrix_two_colorbars(
    M: np.ndarray,
    *,
    title: str = "",
    diag_cmap: str = "vanimo",
    off_cmap: str = "seismic",
    diag_range=None,   # (vmin, vmax) or None
    off_range=None,    # (vmin, vmax) or None
    show_abs_off: bool = True,  # off-diag often better in abs
    aspect="auto",
):
    """
    Render a square matrix M with:
      - diagonal entries shown using diag_cmap + its own colorbar (top-right)
      - off-diagonal entries shown using off_cmap + its own colorbar (bottom-right)
    Two colorbars are both on the right, stacked vertically.

    show_abs_off: if True, off-diagonal uses abs(M_ij) for visibility (common for leakage).
    """

    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square (n,n), got {M.shape}")

    # Masks
    diag_mask = np.eye(n, dtype=bool)
    off_mask = ~diag_mask

    # Two masked arrays (only draw what belongs to each layer)
    M_diag = np.ma.array(M, mask=off_mask)
    M_off_raw = np.abs(M) if show_abs_off else M
    M_off = np.ma.array(M_off_raw, mask=diag_mask)

    # Default ranges (robust-ish)
    if diag_range is None:
        d = np.diag(M)
        vmin_d = float(np.quantile(d, 0.05))
        vmax_d = float(np.quantile(d, 0.95))
        if np.isclose(vmin_d, vmax_d):
            vmin_d, vmax_d = float(d.min()), float(d.max() + 1e-12)
        diag_range = (vmin_d, vmax_d)

    if off_range is None:
        off_vals = M_off_raw[off_mask]
        if off_vals.size == 0:
            off_range = (0.0, 1.0)
        else:
            # visualize small leakage: use 0..p99 (or a tiny epsilon)
            vmax_o = float(np.quantile(off_vals, 0.99))
            vmax_o = max(vmax_o, 1e-12)
            vmin_o = 0.0 if show_abs_off else float(np.quantile(off_vals, 0.01))
            off_range = (vmin_o, vmax_o)

    norm_diag = colors.Normalize(vmin=diag_range[0], vmax=diag_range[1])
    norm_off = colors.Normalize(vmin=off_range[0], vmax=off_range[1])

    # Layout: main axis + a right column split into two rows for colorbars
    fig = plt.figure()
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 0.06],   # right col for colorbars
        height_ratios=[1.0, 1.0],
        wspace=0.15, hspace=0.15
    )
    ax = fig.add_subplot(gs[:, 0])      # big left axis spans both rows
    cax_diag = fig.add_subplot(gs[0, 1]) # top-right colorbar axis
    cax_off = fig.add_subplot(gs[1, 1])  # bottom-right colorbar axis

    # Draw off-diagonal first (background)
    im_off = ax.imshow(M_off, cmap=off_cmap, norm=norm_off, aspect=aspect, interpolation="nearest")
    # Draw diagonal on top (foreground)
    im_diag = ax.imshow(M_diag, cmap=diag_cmap, norm=norm_diag, aspect=aspect, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("achieved channel")
    ax.set_ylabel("requested one-hot index")

    # Colorbars on the right (stacked)
    cb1 = fig.colorbar(im_diag, cax=cax_diag)
    cb1.set_label("diag scale")

    cb2 = fig.colorbar(im_off, cax=cax_off)
    cb2.set_label("off-diag scale (abs)" if show_abs_off else "off-diag scale")

    return fig, ax