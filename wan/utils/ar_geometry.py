"""Single source of truth for autoregressive chunk geometry.

Used by the inference pipeline. The latent stream
is a whole clip of ``N * chunk_size`` frames (N = the video's own chunk count),
split into contiguous chunks

    [ chunk_0 | chunk_1 | ... | chunk_{N-1} ]     (chunk c = frames [c*chunk : (c+1)*chunk])

There is NO separate seed frame. Frame 0 is simply the first frame of chunk 0
and is conditioned as the i2v anchor purely through ``cond_y`` (mask=1 at frame
0 + ``VAE([img, black...])``), never placed as a clean latent in the front-16
latent stream. Chunk c attends causally to chunks <= c (block-causal).
"""

from dataclasses import dataclass, field


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def build_i2v_mask(frame_num, lat_h, lat_w, device, dtype, vae_temporal_stride: int = 4):
    """i2v cond_y mask: 1 only at the first frame (the i2v anchor), 0 elsewhere,
    temporal-folded from ``[1, frame_num, lat_h, lat_w]`` to
    ``[vae_temporal_stride, latent_t, lat_h, lat_w]``.

    Follows the image-to-video conditioning layout. ``frame_num``
    must satisfy ``(frame_num - 1) % vae_temporal_stride == 0``.
    """
    import torch

    if (frame_num - 1) % vae_temporal_stride != 0:
        raise ValueError(
            f"frame_num={frame_num} must satisfy (frame_num - 1) % {vae_temporal_stride} == 0"
        )
    mask = torch.ones(1, frame_num, lat_h, lat_w, device=device, dtype=dtype)
    mask[:, 1:] = 0
    mask = torch.cat(
        [torch.repeat_interleave(mask[:, 0:1], repeats=vae_temporal_stride, dim=1), mask[:, 1:]],
        dim=1,
    )
    mask = mask.view(1, mask.shape[1] // vae_temporal_stride, vae_temporal_stride, lat_h, lat_w)
    mask = mask.transpose(1, 2)[0]
    return mask


@dataclass(frozen=True)
class ARGeometry:
    """Seed-less chunk-by-chunk AR geometry.

    The number of chunks N is a per-video property (variable), so this object
    only fixes the per-chunk geometry ``(chunk_size, vae_temporal_stride,
    patch_size)`` and derives N from a clip's latent-frame count via
    ``num_chunks``.
    """

    chunk_size: int
    vae_temporal_stride: int
    patch_size: tuple = field(default=(1, 2, 2))

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.vae_temporal_stride <= 0:
            raise ValueError(
                f"vae_temporal_stride must be positive, got {self.vae_temporal_stride}"
            )
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 entries, got {self.patch_size}")

    @property
    def latent_window_size(self) -> int:
        # One chunk in latent frames.
        return int(self.chunk_size)

    @property
    def frame_window_size(self) -> int:
        """Number of RGB frames per AR chunk (Wan VAE causal 4x temporal)."""
        return self.chunk_size * self.vae_temporal_stride

    def num_chunks(self, lat_f: int) -> int:
        """How many whole chunks fit in ``lat_f`` latent frames (floor)."""
        return int(lat_f) // int(self.chunk_size)

    def usable_latent_frames(self, lat_f: int) -> int:
        """Largest multiple of ``chunk_size`` <= ``lat_f`` (matches
        the inference pipeline: ``lat_f -= lat_f % chunk_size``)."""
        return self.num_chunks(lat_f) * int(self.chunk_size)

    def compute_seq_len(self, lat_f: int, lat_h: int, lat_w: int) -> int:
        """Single-sequence token count over ``lat_f`` latent frames."""
        p_t, p_h, p_w = self.patch_size
        return int(
            _ceil_div(lat_f, p_t) * _ceil_div(lat_h, p_h) * _ceil_div(lat_w, p_w)
        )
