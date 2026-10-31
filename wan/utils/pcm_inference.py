"""Four-phase inference schedule for distilled autoregressive models."""

import json
import math
from pathlib import Path


def validate_phase_sigmas(values, boundary: float) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("phase_sigmas must be a list of five numeric values")
    try:
        sigmas = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase_sigmas must contain five numeric values") from exc
    if len(sigmas) != 5 or not all(math.isfinite(value) for value in sigmas):
        raise ValueError("phase_sigmas must contain five finite values")
    if not (sigmas[0] <= 1.0 and sigmas[-1] == 0.0):
        raise ValueError("phase_sigmas must start at or below 1 and end at 0")
    if not all(sigmas[i] > sigmas[i + 1] for i in range(4)):
        raise ValueError("phase_sigmas must be strictly descending")
    if not sigmas[0] >= boundary > sigmas[1] > sigmas[2] > sigmas[3] > sigmas[4]:
        raise ValueError("phase_sigmas must route one high and three low steps")
    return sigmas


def load_phase_sigmas(checkpoint_dir: Path, boundary: float) -> tuple[float, ...]:
    path = checkpoint_dir / "inference_config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "phase_sigmas" not in config:
        raise ValueError(f"{path} must contain phase_sigmas")
    return validate_phase_sigmas(config["phase_sigmas"], boundary)


def pcm_step(state, velocity, sigma, sigma_next):
    """Deterministic flow-map transition between checkpoint phase endpoints."""
    return (state.float() + (float(sigma_next) - float(sigma)) * velocity.float()).to(state.dtype)
