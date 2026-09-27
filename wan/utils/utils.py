"""Output helpers used by the public inference entry point."""

import logging
import tempfile
from pathlib import Path

import imageio
import torch
import torchvision

__all__ = ["save_video"]


def save_video(
    tensor,
    save_file=None,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
):
    """Save a ``[B, C, T, H, W]`` tensor as an H.264 video grid."""
    if save_file is None:
        temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temporary.close()
        output_path = Path(temporary.name)
    else:
        output_path = Path(save_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frames = tensor.clamp(min(value_range), max(value_range))
        frames = torch.stack(
            [
                torchvision.utils.make_grid(
                    frame,
                    nrow=nrow,
                    normalize=normalize,
                    value_range=value_range,
                )
                for frame in frames.unbind(2)
            ],
            dim=1,
        ).permute(1, 2, 3, 0)
        frames = (frames * 255).to(torch.uint8).cpu().numpy()

        with imageio.get_writer(
            str(output_path), fps=fps, codec="libx264", quality=8
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception:
        logging.exception("Failed to save video to %s", output_path)
        raise

    return output_path
