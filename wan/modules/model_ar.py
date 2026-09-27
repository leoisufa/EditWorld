import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from wan.commons.communications import all_gather, all_to_all_4D
from wan.commons.parallel_states import get_parallel_state
from wan.modules.attention import flash_attention
GCA_PROMPT_WINDOW = 2

__all__ = ["GCAKVCache", "GCA_PROMPT_WINDOW", "WanModelAR"]


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def pad_for_3d_conv(x, kernel_size):
    _, _, frames, height, width = x.shape
    patch_t, patch_h, patch_w = kernel_size
    pad_t = (patch_t - (frames % patch_t)) % patch_t
    pad_h = (patch_h - (height % patch_h)) % patch_h
    pad_w = (patch_w - (width % patch_w)) % patch_w
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_token_freqs(x, token_freqs):
    batch_size, seq_len, num_heads, head_dim = x.shape
    if token_freqs.ndim != 4:
        raise ValueError(
            f"token_freqs must have shape [B, L, 1, D/2], got {tuple(token_freqs.shape)}"
        )
    expected_shape = (batch_size, seq_len, 1, head_dim // 2)
    if tuple(token_freqs.shape) != expected_shape:
        raise ValueError(
            "RoPE token frequency shape mismatch. "
            f"Expected {expected_shape}, got {tuple(token_freqs.shape)}"
        )

    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
    )
    freqs = token_freqs.to(device=x.device, dtype=x_complex.dtype)
    x_complex = x_complex * freqs
    return torch.view_as_real(x_complex).flatten(3).float()


# Reference images live OUTSIDE the video timeline at wide NEGATIVE temporal RoPE
# positions: ref i (0-indexed) sits at frame position -(i+1)*REF_TEMPORAL_STEP, so
# ref #1 is nearest the video (-256) and later refs are farther back. 256 latent
# frames is far beyond any real clip / the local attention window, so the model
# reads these as a distinct "out-of-timeline context" band rather than as frames
# immediately preceding the video. The spatial (h/w) RoPE axes are unchanged.
REF_TEMPORAL_STEP = 256


def pack_ctx_window(items, text_len):
    """Concatenate VARIABLE-length prompt embeddings into ONE fixed text_len block.

    ``items``: list of [len_i, dim] T5-space embeddings in importance-DESCENDING
    order (e.g. [a_B, a_i, a_{i-1}, a_{i-2}]) so that when the concatenation
    exceeds ``text_len`` the TAIL (= the least-important prompt's end) is
    truncated first. Shorter concatenations are zero-padded (the zeros become
    the usual MLP(0) constant sink after text_embedding). Returns [text_len, dim].
    """
    cat = items[0] if len(items) == 1 else torch.cat(items, dim=0)
    if cat.size(0) > text_len:
        cat = cat[:text_len]
    elif cat.size(0) < text_len:
        cat = torch.cat([cat, cat.new_zeros(text_len - cat.size(0), cat.size(1))], dim=0)
    return cat


@torch.amp.autocast("cuda", enabled=False)
def build_ref_token_freqs(freqs_table, positions, grid_h, grid_w, device):
    """Per-token RoPE freqs for reference frames: temporal axis computed
    analytically at (possibly negative) `positions`, spatial h/w axes taken
    from the shared `freqs_table` (identical to the video path). Returns a
    complex tensor shaped [1, len(positions)*grid_h*grid_w, 1, half_dim]
    consumable by rope_apply_token_freqs.
    """
    rope_dtype = torch.float64
    half_dim = freqs_table.size(1)
    f_cols = half_dim - 2 * (half_dim // 3)          # temporal band width
    freq_splits = freqs_table.split([f_cols, half_dim // 3, half_dim // 3], dim=1)
    # Temporal inv-freq identical to rope_params(_, 2*f_cols): 10000^(-j/f_cols).
    inv_freq = 1.0 / torch.pow(
        torch.tensor(10000.0, dtype=rope_dtype, device=device),
        torch.arange(f_cols, device=device, dtype=rope_dtype) / f_cols,
    )
    pos = torch.as_tensor(positions, dtype=rope_dtype, device=device)
    t_freqs = torch.polar(torch.ones(pos.numel(), f_cols, device=device, dtype=rope_dtype),
                          torch.outer(pos, inv_freq))   # [num_ref, f_cols] complex
    h_freqs = freq_splits[1][:grid_h].to(device)
    w_freqs = freq_splits[2][:grid_w].to(device)
    num_ref = pos.numel()
    grid = torch.cat(
        [
            t_freqs[:, None, None, :].expand(num_ref, grid_h, grid_w, -1),
            h_freqs[None, :, None, :].expand(num_ref, grid_h, grid_w, -1),
            w_freqs[None, None, :, :].expand(num_ref, grid_h, grid_w, -1),
        ],
        dim=-1,
    )
    return grid.reshape(1, num_ref * grid_h * grid_w, 1, half_dim)


def _sparse_kv_select(n_chunks, sink_chunks, recent_chunks, topk, sims):
    """Active memory set for generating the next chunk given ``n_chunks``
    committed blocks: sink ∪ recent ∪ top-k middle-history by
    pooled query-key similarity. ``sims``: [n_chunks] scores (only middle
    entries consulted) or None (no retrieval). Returns sorted chunk indices,
    preserving temporal order for KV concatenation."""
    required = set(range(min(sink_chunks, n_chunks)))
    required |= set(range(max(0, n_chunks - recent_chunks), n_chunks))
    mid = list(range(sink_chunks, max(sink_chunks, n_chunks - recent_chunks)))
    if topk > 0 and mid and sims is not None:
        order = sorted(mid, key=lambda j: -float(sims[j]))
        required |= set(order[: min(topk, len(mid))])
    return sorted(required)


def _pin_cpu(t):
    """Copy a device tensor into pinned host memory so later per-forward
    H2D fetches with non_blocking=True are truly asynchronous."""
    out = torch.empty_like(t, device="cpu", pin_memory=True)
    out.copy_(t)
    return out


class GCAKVCache:
    """Per-layer full-fidelity cache for Gated Causal Attention (GCA).

    Deliberately a plain object, NOT a dict: FSDP's root pre-forward rebuilds
    dict/list containers inside forward kwargs (tree_map for input device
    move / cast), so dict-level mutations made during a forward land in a
    throwaway copy and are silently dropped. An unknown class is a pytree
    LEAF and passes through BY REFERENCE, so commits (chunks_k/v appends,
    ``n`` increment, pk writes) survive across forwards. The rolling cache
    can stay a dict because all its state lives in preallocated tensors
    mutated in-place (tensor objects survive container rebuilds)."""

    __slots__ = ("topk", "sink_chunks", "recent_chunks", "offload",
                 "pk", "chunks_k", "chunks_v", "n")

    def __init__(self, topk, sink_chunks, recent_chunks, offload, pk):
        self.topk = int(topk)
        self.sink_chunks = int(sink_chunks)
        self.recent_chunks = int(recent_chunks)
        self.offload = bool(offload)
        self.pk = pk
        self.chunks_k = []
        self.chunks_v = []
        self.n = 0


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """KV-cache inference RoPE.

    Applies rotary embeddings to a single chunk whose frames begin at absolute
    ``start_frame``. Used by the sequential GCA KV-cache generation path.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class GatedCausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        local_attn_size=-1,
        sink_size=0,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        ref_kv=None,
        commit_ref=False,
        attend_ref=False,
        rope_offset_frames=0,
        ref_temporal_pos=None,
        commit_chunk=False,
    ):
        batch_size, seq_len, num_heads, head_dim = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(hidden_states):
            q = self.norm_q(self.q(hidden_states)).view(batch_size, seq_len, num_heads, head_dim)
            k = self.norm_k(self.k(hidden_states)).view(batch_size, seq_len, num_heads, head_dim)
            v = self.v(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
            return q, k, v

        # KV-cache inference path (sequential chunk-by-chunk). Single code path for
        # both single-GPU and multi-GPU Sequence-Parallel (Ulysses): when SP is
        # enabled the sequence arrives sharded across ranks, all_to_all gathers the
        # full sequence and scatters heads (so the KV cache holds only num_heads//sp
        # local heads per rank -> non-redundant), then all_to_all back. At sp==1 all
        # SP ops are no-ops and this is bit-identical to the plain path.
        # Mirrors the reference streaming self-attention: causal RoPE at the chunk's
        # absolute frame offset, then a rolling KV cache with a `local_attn_size`
        # sliding window and a `sink_size` attention sink (the first `sink_size`
        # frames are never evicted). `max_attention_size` bounds how many cached
        # tokens each query attends. `local_attn_size == -1` => unbounded global cache.
        if kv_cache is None and not commit_ref:
            raise ValueError("GCA requires an AR KV cache or a reference commit")
        q, k, v = qkv_fn(x)
        if frame_seqlen is None:
            frame_seqlen = int(grid_sizes[0][1].item() * grid_sizes[0][2].item())
        parallel_dims = get_parallel_state()
        sp_on = parallel_dims.sp_enabled
        if sp_on:
            sp_group = parallel_dims.sp_group
            local_seq = q.shape[1]
            # gather sequence, scatter heads -> [B, padded_seq, num_heads/sp, head_dim]
            q = all_to_all_4D(q, sp_group, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, sp_group, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, sp_group, scatter_dim=2, gather_dim=1)
        # Pre-RoPE pooled q/k for sparse retrieval scoring: positions must
        # not leak into the similarity (post-RoPE pooling carries a relative-
        # position phase that favors temporally near chunks). The full-res
        # cached blocks below keep RoPE — only the compact table is pre-RoPE.
        pool_q = pool_k = None
        if isinstance(kv_cache, GCAKVCache):
            _real = int(seq_lens[0]) if sp_on else k.shape[1]
            pool_q = q[:, :_real].float().mean(dim=1)   # [1,H,D]
            pool_k = k[:, :_real].float().mean(dim=1)   # [1,H,D]
        # RoPE position is offset by rope_offset_frames (ref frames occupy
        # t=0..n-1, video shifts to t=n) while the CACHE index stays video-
        # relative (current_start), so the rolling/global cache is unaffected.
        current_start_frame = int(current_start) // frame_seqlen + int(rope_offset_frames)
        if commit_ref and ref_temporal_pos is not None:
            # Ref frame: rope at its wide-negative temporal position (spatial
            # h/w from the table), instead of the video's table-sliced rope.
            gh, gw = int(grid_sizes[0][1]), int(grid_sizes[0][2])
            ref_tf = build_ref_token_freqs(freqs, [int(ref_temporal_pos)], gh, gw, q.device)
            ref_len = ref_tf.shape[1]
            q = torch.cat([rope_apply_token_freqs(q[:, :ref_len], ref_tf), q[:, ref_len:]], dim=1).type_as(v)
            k = torch.cat([rope_apply_token_freqs(k[:, :ref_len], ref_tf), k[:, ref_len:]], dim=1).type_as(v)
        else:
            q = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            k = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        if sp_on:
            # discard SP padding: only the real seq_lens tokens enter cache + attention
            seq_lens_int = int(seq_lens[0])
            q = q[:, :seq_lens_int]
            k = k[:, :seq_lens_int]
            v = v[:, :seq_lens_int]

        # ---- Reference-image bank ----
        # commit_ref: this call's tokens ARE ref frame(s); store their K/V into
        # the fixed ref bank (no rolling) and self-attend within the frame only.
        if commit_ref and ref_kv is not None:
            r0 = int(current_start)
            r1 = r0 + k.shape[1]
            ref_kv["k"][:, r0:r1] = k
            ref_kv["v"][:, r0:r1] = v
            x = flash_attention(q, k, v)
            if sp_on:
                local_heads = x.shape[2]
                padded_seq = local_seq * parallel_dims.sp
                if padded_seq > x.shape[1]:
                    x = torch.cat(
                        [x, x.new_zeros(x.shape[0], padded_seq - x.shape[1], local_heads, x.shape[3])],
                        dim=1,
                    )
                x = all_to_all_4D(x, sp_group, scatter_dim=1, gather_dim=2)
            return self.o(x.flatten(2))

        if isinstance(kv_cache, GCAKVCache):
            # ---- Sparse full-fidelity GCA memory ----
            # Per-chunk KV blocks (middle history lives on pinned CPU), plus a
            # resident fp32 PRE-RoPE pooled-key table. EVERY self-attention
            # call (each layer, each denoise step, each CFG stream / expert)
            # independently selects A = sink ∪ recent ∪ top-k(mid) with the
            # pre-RoPE pooled q of THIS call scored against THIS layer's
            # table. Selected CPU
            # blocks are fetched, used and dropped within this forward.
            n_c = int(kv_cache.n)
            sims = None
            mid_lo = kv_cache.sink_chunks
            mid_hi = max(mid_lo, n_c - kv_cache.recent_chunks)
            if kv_cache.topk > 0 and mid_hi > mid_lo:
                # Cosine over pre-RoPE pooled q/k: no relative-position
                # phase in the scores, and normalising by the pooled norms
                # removes the bias toward homogeneous chunks whose token
                # means don't self-cancel.
                pk = kv_cache.pk[:, :n_c]                               # [1,n,H,D] fp32
                dot = (pk * pool_q.unsqueeze(1)).sum(dim=(-1, -2)).squeeze(0)  # [n]
                k_sq = pk.pow(2).sum(dim=(-1, -2)).squeeze(0)                  # [n]
                q_sq = pool_q.pow(2).sum().reshape(1)                          # [1]
                if sp_on:
                    # q/pk hold only num_heads//sp local heads per rank:
                    # all_reduce the dot and norm pieces together so every
                    # rank computes identical cosine scores and retrieves
                    # the SAME chunks across all head groups.
                    packed = torch.cat([dot, k_sq, q_sq])
                    dist.all_reduce(packed, group=sp_group)
                    dot, k_sq, q_sq = packed[:n_c], packed[n_c:2 * n_c], packed[2 * n_c:]
                sims = dot / (q_sq.clamp_min(1e-12).sqrt() * k_sq.clamp_min(1e-12).sqrt())
            sel = _sparse_kv_select(
                n_c, kv_cache.sink_chunks, kv_cache.recent_chunks,
                kv_cache.topk, sims,
            )
            k_hist, v_hist = [], []
            for j in sel:
                kj, vj = kv_cache.chunks_k[j], kv_cache.chunks_v[j]
                if kj.device != q.device:
                    kj = kj.to(q.device, non_blocking=True)
                    vj = vj.to(q.device, non_blocking=True)
                k_hist.append(kj)
                v_hist.append(vj)
            k_all = torch.cat(k_hist + [k], dim=1) if k_hist else k
            v_all = torch.cat(v_hist + [v], dim=1) if v_hist else v
            if attend_ref and ref_kv is not None:
                k_all = torch.cat([ref_kv["k"], k_all], dim=1)
                v_all = torch.cat([ref_kv["v"], v_all], dim=1)
            x = flash_attention(q, k_all, v_all)
            if commit_chunk:
                # Final clean pass of this chunk: append the full-res block
                # (post-RoPE, as attention consumes it), write its PRE-RoPE
                # pooled key, demote the block that just left the recent
                # window to pinned CPU (offload).
                kv_cache.chunks_k.append(k.contiguous())
                kv_cache.chunks_v.append(v.contiguous())
                kv_cache.pk[:, n_c] = pool_k
                kv_cache.n = n_c + 1
                if kv_cache.offload:
                    out_idx = n_c - kv_cache.recent_chunks
                    if out_idx >= kv_cache.sink_chunks:
                        kv_cache.chunks_k[out_idx] = _pin_cpu(kv_cache.chunks_k[out_idx])
                        kv_cache.chunks_v[out_idx] = _pin_cpu(kv_cache.chunks_v[out_idx])
            if sp_on:
                local_heads = x.shape[2]
                padded_seq = local_seq * parallel_dims.sp
                if padded_seq > x.shape[1]:
                    x = torch.cat(
                        [x, x.new_zeros(x.shape[0], padded_seq - x.shape[1], local_heads, x.shape[3])],
                        dim=1,
                    )
                x = all_to_all_4D(x, sp_group, scatter_dim=1, gather_dim=2)
            return self.o(x.flatten(2))

        current_end = int(current_start) + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = q.shape[1]
        if self.local_attn_size == -1:
            # Global cache: indices advance identically, no eviction.
            local_end_index = current_end
            local_start_index = int(current_start)
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        elif (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
        ):
            # Cache full: evict oldest non-sink tokens, roll the window left,
            # keep the first `sink_tokens`, then append the new tokens.
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # Room left: append directly.
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v

        k_all = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
        v_all = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
        # Gated ref bank: during-chunks (attend_ref) prepend the fixed ref K/V.
        if attend_ref and ref_kv is not None:
            k_all = torch.cat([ref_kv["k"], k_all], dim=1)
            v_all = torch.cat([ref_kv["v"], v_all], dim=1)
        x = flash_attention(q, k_all, v_all)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        if sp_on:
            # pad back, then gather heads / scatter sequence -> [B, local_seq, num_heads, head_dim]
            local_heads = x.shape[2]
            padded_seq = local_seq * parallel_dims.sp
            if padded_seq > x.shape[1]:
                x = torch.cat(
                    [x, x.new_zeros(x.shape[0], padded_seq - x.shape[1], local_heads, x.shape[3])],
                    dim=1,
                )
            x = all_to_all_4D(x, sp_group, scatter_dim=1, gather_dim=2)
        x = x.flatten(2)
        x = self.o(x)
        return x



class WanCrossAttention(GatedCausalAttention):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        batch_size, num_heads, head_dim = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(batch_size, -1, num_heads, head_dim)
        # Cache the text K,V once (the prompt is fixed within a chunk's denoise
        # loop) — shared by the low- and high-noise DiT experts.
        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                crossattn_cache["is_init"] = True
                crossattn_cache["k"] = self.norm_k(self.k(context)).view(batch_size, -1, num_heads, head_dim)
                crossattn_cache["v"] = self.v(context).view(batch_size, -1, num_heads, head_dim)
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(batch_size, -1, num_heads, head_dim)
            v = self.v(context).view(batch_size, -1, num_heads, head_dim)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        local_attn_size=-1,
        sink_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = GatedCausalAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        ref_kv=None,
        commit_ref=False,
        attend_ref=False,
        rope_offset_frames=0,
        ref_temporal_pos=None,
        commit_chunk=False,
    ):
        if e.dtype != torch.float32:
            e = e.float()
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            kv_cache=kv_cache,
            current_start=current_start,
            max_attention_size=max_attention_size,
            frame_seqlen=frame_seqlen,
            ref_kv=ref_kv,
            commit_ref=commit_ref,
            attend_ref=attend_ref,
            rope_offset_frames=rope_offset_frames,
            ref_temporal_pos=ref_temporal_pos,
            commit_chunk=commit_chunk,
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            if c2ws_plucker_emb.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "control token length must strictly match hidden states. "
                    f"control={tuple(c2ws_plucker_emb.shape)}, hidden={tuple(x.shape)}"
                )
            # Camera injection runs in ambient dtype (bf16). Only the residual
            # add above is an fp32 island.
            c2ws_hidden_states = self.cam_injector_layer2(
                torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb))
            )
            c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            x = (1.0 + cam_scale) * x + cam_shift

        def cross_attn_ffn(hidden_states, encoder_hidden_states, modulation):
            normed = self.norm3(hidden_states)
            attn_out = self.cross_attn(
                normed,
                encoder_hidden_states,
                None,
                crossattn_cache=crossattn_cache,
            )
            hidden_states = hidden_states + attn_out
            y = self.ffn(
                self.norm2(hidden_states).float() * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                hidden_states = hidden_states + y * modulation[5].squeeze(2)
            return hidden_states

        return cross_attn_ffn(x, context, e)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        if e.dtype != torch.float32:
            e = e.float()
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModelAR(ModelMixin, ConfigMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        control_type="cam",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        local_attn_size=-1,
        sink_size=0,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        # i2v checkpoints concatenate latent + i2v mask + cond_y into 36 input
        # channels. The model instantiates with that shape so `from_pretrained`
        # can load the base weights without mismatch; AR inference keeps the full
        # 36-channel layout end-to-end (the AR-window cond_y stream is fed via
        # the `y=` kwarg at forward time).
        if model_type == "i2v" and in_dim == 16:
            in_dim = 36

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if control_type == "cam":
            control_dim = 6
        elif control_type == "act":
            control_dim = 7
        else:
            raise ValueError(f"Unsupported control_type: {control_type}")

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2],
            dim,
        )
        self.control_dim = control_dim
        self.control_channels = control_dim * 64
        if tuple(patch_size) != (1, 2, 2):
            raise ValueError(
                "AR uniform layout requires Wan patch_size=(1, 2, 2), "
                f"got {patch_size}"
            )
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

        self.init_weights()

    def _control_to_tensor(self, value, batch_size, name, expected_raw_shape=None):
        if isinstance(value, (list, tuple)):
            value = torch.cat(list(value), dim=0)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor or a list/tuple of Tensor chunks")
        if value.ndim != 5 or value.shape[0] != batch_size or value.shape[1] != self.control_channels:
            raise ValueError(
                f"{name} must have shape [B, {self.control_channels}, T, H, W], "
                f"got {tuple(value.shape)}"
            )
        if expected_raw_shape is not None and tuple(value.shape[2:]) != tuple(expected_raw_shape):
            raise ValueError(
                f"{name} raw T/H/W must strictly match its latent branch. "
                f"Expected {tuple(expected_raw_shape)}, got {tuple(value.shape[2:])}"
            )
        return value

    def _patch_control_segment(self, control, patch_kernel, linear, expected_tokens, device, name):
        control = pad_for_3d_conv(control.to(device=device, dtype=self.patch_embedding.weight.dtype), patch_kernel)
        tokens = rearrange(
            control,
            "b c (f c1) (h c2) (w c3) -> b (f h w) (c c1 c2 c3)",
            c1=patch_kernel[0],
            c2=patch_kernel[1],
            c3=patch_kernel[2],
        )
        if tokens.size(1) != expected_tokens:
            raise ValueError(
                f"{name} control token length mismatch: expected {expected_tokens}, got {tokens.size(1)}"
            )
        tokens = linear(tokens)
        hidden_states = self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(tokens)))
        return tokens + hidden_states

    def _build_control_tokens(
        self,
        dit_cond_dict,
        batch_size,
        target_tokens,
        target_raw_shape,
        device,
    ):
        if dit_cond_dict is None or "c2ws_plucker_emb" not in dit_cond_dict:
            return None

        control_target = self._control_to_tensor(
            dit_cond_dict["c2ws_plucker_emb"],
            batch_size,
            "c2ws_plucker_emb",
            expected_raw_shape=target_raw_shape,
        )
        control_tokens = self._patch_control_segment(
            control_target,
            self.patch_size,
            self.patch_embedding_wancamctrl,
            target_tokens.size(1),
            device,
            "c2ws_plucker_emb",
        )
        return {"c2ws_plucker_emb": control_tokens}

    def _build_time_embeddings(self, t, batch_size, target_seq_len, device):
        if t.dim() == 1:
            target_t = t.to(device=device)[:, None].expand(batch_size, target_seq_len)
        elif t.dim() == 2 and tuple(t.shape) == (batch_size, target_seq_len):
            target_t = t.to(device=device)
        else:
            raise ValueError(
                f"t must have shape [B] or [B, target_seq_len={target_seq_len}], got {tuple(t.shape)}"
            )

        with torch.amp.autocast("cuda", dtype=torch.float32):
            flat_t = target_t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, flat_t)
                .unflatten(0, (batch_size, target_t.size(1)))
                .float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        return e.float(), e0.float()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        ref_kv=None,
        commit_ref=False,
        attend_ref=False,
        rope_offset_frames=0,
        ref_temporal_pos=None,
        commit_chunk=False,
    ):
        # ---- KV-cache inference path (sequential chunk-by-chunk generation) ----
        # History lives in the persistent per-layer kv_cache (committed clean KVs),
        # so there is NO history-token concatenation, no doubled sequence, and no
        # mask — causality is structural in the cache. Single code path for both
        # single-GPU and multi-GPU Sequence-Parallel: when SP is enabled the sequence
        # is padded to a multiple of sp and sharded across ranks (context parallel);
        # each block's self-attn does the Ulysses all_to_all, then `head` runs on the
        # local shard and the sequence is all-gathered before unpatchify. At sp==1 no
        # padding/chunk/gather happens and this is bit-identical to the plain path.
        if kv_cache is None and not commit_ref:
            raise ValueError("WanModelAR requires an AR KV cache or a reference commit")
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None and self.in_dim == x[0].shape[0] + y[0].shape[0]:
            x = [torch.cat([latent, cond], dim=0) for latent, cond in zip(x, y)]
        batch_size = len(x)
        x5d = [self.patch_embedding(latent.unsqueeze(0)) for latent in x]
        grid_sizes = torch.stack(
            [torch.tensor(latent.shape[2:], dtype=torch.long) for latent in x5d]
        )
        x = torch.cat([latent.flatten(2).transpose(1, 2) for latent in x5d], dim=0)
        real_seq_len = x.size(1)
        # Camera control tokens built at the REAL length, before any SP padding.
        dit_cond_dict = self._build_control_tokens(
            dit_cond_dict, batch_size, x, None, device
        )
        parallel_dims = get_parallel_state()
        sp_on = parallel_dims.sp_enabled
        _saved_sp_meta = None
        if sp_on:
            sp = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            sp_group = parallel_dims.sp_group
            # This path shards the current chunk's short sequence. Clear global SP
            # sequence metadata so every all_to_all/all_gather here discovers the
            # real per-rank shard from the tensor shapes; restore it before returning.
            _saved_sp_meta = parallel_dims.sp_split_sizes
            parallel_dims.clear_sp_sequence_info()
            padded_seq = ((real_seq_len + sp - 1) // sp) * sp
            if padded_seq > real_seq_len:
                x = torch.cat([x, x.new_zeros(x.shape[0], padded_seq - real_seq_len, x.shape[2])], dim=1)
        else:
            padded_seq = real_seq_len
        e, e0 = self._build_time_embeddings(t, batch_size, padded_seq, device)
        # VARIABLE-length prompt window: the caller passes the items in
        # importance-descending order (e.g. [a_B, a_p, a_{p-1}, a_{p-2}]);
        # concatenate them into ONE text_len block (trunc tail / zero-pad).
        context = self.text_embedding(
            pack_ctx_window([item for item in context], self.text_len).unsqueeze(0)
        )
        if sp_on:
            if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
                cam = dit_cond_dict["c2ws_plucker_emb"]
                if cam.size(1) < padded_seq:
                    cam = torch.cat([cam, cam.new_zeros(cam.size(0), padded_seq - cam.size(1), cam.size(2))], dim=1)
                elif cam.size(1) > padded_seq:
                    cam = cam[:, :padded_seq]
                dit_cond_dict = dict(dit_cond_dict)
                dit_cond_dict["c2ws_plucker_emb"] = cam.chunk(sp, dim=1)[sp_rank]
            # Context-parallel shard along the sequence dimension.
            x = x.chunk(sp, dim=1)[sp_rank]
            e = e.chunk(sp, dim=1)[sp_rank]
            e0 = e0.chunk(sp, dim=1)[sp_rank]
        # seq_lens is the FULL real length (self-attn slices SP padding off).
        seq_lens = torch.full((batch_size,), real_seq_len, dtype=torch.long, device=device)
        if frame_seqlen is None:
            frame_seqlen = int(grid_sizes[0][1].item() * grid_sizes[0][2].item())
        for block_index, block in enumerate(self.blocks):
            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=None,
                dit_cond_dict=dit_cond_dict,
                kv_cache=kv_cache[block_index] if kv_cache is not None else None,
                crossattn_cache=crossattn_cache[block_index] if crossattn_cache is not None else None,
                current_start=current_start,
                max_attention_size=max_attention_size,
                frame_seqlen=frame_seqlen,
                ref_kv=ref_kv[block_index] if ref_kv is not None else None,
                commit_ref=commit_ref,
                attend_ref=attend_ref,
                rope_offset_frames=rope_offset_frames,
                ref_temporal_pos=ref_temporal_pos,
                commit_chunk=commit_chunk,
            )
        x = self.head(x, e)
        if sp_on:
            x = all_gather(x, dim=1, group=sp_group)  # gather the full sequence
            if _saved_sp_meta is not None:
                parallel_dims.set_sp_sequence_info(_saved_sp_meta)
        x = self.unpatchify(x, grid_sizes)
        return [item.float() for item in x]

    def unpatchify(self, x, grid_sizes):
        outputs = []
        for item, grid_size in zip(x, grid_sizes.tolist()):
            item = item[: math.prod(grid_size)].view(*grid_size, *self.patch_size, self.out_dim)
            item = torch.einsum("fhwpqrc->cfphqwr", item)
            item = item.reshape(
                self.out_dim,
                *[grid * patch for grid, patch in zip(grid_size, self.patch_size)],
            )
            outputs.append(item)
        return outputs

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

        nn.init.xavier_uniform_(self.patch_embedding_wancamctrl.weight)
        nn.init.zeros_(self.patch_embedding_wancamctrl.bias)
        nn.init.xavier_uniform_(self.c2ws_hidden_states_layer1.weight)
        nn.init.zeros_(self.c2ws_hidden_states_layer1.bias)
        nn.init.xavier_uniform_(self.c2ws_hidden_states_layer2.weight)
        nn.init.zeros_(self.c2ws_hidden_states_layer2.bias)

        for block in self.blocks:
            nn.init.xavier_uniform_(block.cam_injector_layer1.weight)
            nn.init.zeros_(block.cam_injector_layer1.bias)
            nn.init.xavier_uniform_(block.cam_injector_layer2.weight)
            nn.init.zeros_(block.cam_injector_layer2.bias)
            nn.init.xavier_uniform_(block.cam_scale_layer.weight)
            nn.init.zeros_(block.cam_scale_layer.bias)
            nn.init.xavier_uniform_(block.cam_shift_layer.weight)
            nn.init.zeros_(block.cam_shift_layer.bias)
