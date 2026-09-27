import warnings
import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

try:
    # FlashAttention-4 (CuTeDSL, Hopper/Blackwell). Use the varlen entry point
    # so it slots into the same cu_seqlens path as FA3/FA2. A broad except
    # guards against partial/broken CuTe installs so they degrade to FA3/FA2.
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_varlen_func_v4
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    FLASH_ATTN_4_AVAILABLE = False

try:
    try:
        from flash_attn_3 import flash_attn_interface
    except ModuleNotFoundError:
        # Compatibility with earlier FA3 source installs that exposed the module
        # at the top level.
        import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
]


def _flex_attention(
    q,
    k,
    v,
    q_lens,
    k_lens,
    dropout_p,
    softmax_scale,
    q_scale,
    causal,
    window_size,
    dtype,
):
    """PyTorch Flex Attention fallback for ``[B, L, H, D]`` tensors."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "No FlashAttention backend is available and this PyTorch build does not "
            "provide torch.nn.attention.flex_attention."
        )
    if dropout_p != 0:
        raise ValueError("PyTorch Flex Attention fallback does not support attention dropout.")

    batch, q_max = q.shape[:2]
    k_max = k.shape[1]
    output_dtype = q.dtype
    q_lengths = [q_max] * batch if q_lens is None else [int(x) for x in q_lens]
    k_lengths = [k_max] * batch if k_lens is None else [int(x) for x in k_lens]
    output = q.new_zeros((batch, q_max, q.shape[2], v.shape[-1]))

    for index, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        query = q[index:index + 1, :q_length].transpose(1, 2).to(dtype)
        key = k[index:index + 1, :k_length].transpose(1, 2).to(dtype)
        value = v[index:index + 1, :k_length].transpose(1, 2).to(dtype)
        if q_scale is not None:
            query = query * q_scale
        left, right = window_size
        needs_mask = causal or left >= 0 or right >= 0
        block_mask = None
        if needs_mask:
            offset = k_length - q_length

            def mask_mod(batch_index, head_index, query_index, key_index):
                keep = query_index == query_index
                if causal:
                    keep = keep & (key_index <= query_index + offset)
                if left >= 0:
                    keep = keep & (key_index >= query_index + offset - left)
                if right >= 0:
                    keep = keep & (key_index <= query_index + offset + right)
                return keep

            block_mask = create_block_mask(
                mask_mod,
                B=1,
                H=None,
                Q_LEN=q_length,
                KV_LEN=k_length,
                device=query.device,
            )

        result = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=softmax_scale,
            enable_gqa=query.size(1) != key.size(1),
        )
        output[index, :q_length] = result.transpose(1, 2).to(output_dtype)
    return output


def _unpack_varlen_output(output, lengths, batch, max_length):
    lengths = [int(length) for length in lengths]
    if all(length == max_length for length in lengths):
        return output.unflatten(0, (batch, max_length))
    padded = output.new_zeros((batch, max_length, *output.shape[1:]))
    offset = 0
    for index, length in enumerate(lengths):
        padded[index, :length] = output[offset:offset + length]
        offset += length
    return padded


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.size(-1) <= 256

    if version not in (None, 2, 3, 4):
        raise ValueError(f"Unsupported FlashAttention version: {version}")
    max_version = 4 if version is None else version
    fa4_compatible = dropout_p == 0
    fa3_compatible = dropout_p == 0 and window_size == (-1, -1)
    if max_version >= 4 and FLASH_ATTN_4_AVAILABLE and fa4_compatible:
        backend = 4
    elif max_version >= 3 and FLASH_ATTN_3_AVAILABLE and fa3_compatible:
        backend = 3
    elif max_version >= 2 and FLASH_ATTN_2_AVAILABLE:
        backend = 2
    else:
        backend = 0

    if version is not None and backend != version:
        fallback = f"FlashAttention {backend}" if backend else "PyTorch Flex Attention"
        warnings.warn(f"FlashAttention {version} is unavailable; falling back to {fallback}.")

    if backend == 0:
        return _flex_attention(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
            dtype,
        )

    if q.device.type != "cuda":
        raise RuntimeError("FlashAttention backends require CUDA tensors.")

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # FlashAttention-4 (highest priority). Handled in its own early-return branch
    # so the FA3 / FA2 code below stays untouched.
    if backend == 4:
        # FA4 uses `None` (not -1) to denote an unbounded window side.
        fa4_window = tuple(None if s is None or s < 0 else s for s in window_size)
        x = flash_attn_varlen_func_v4(
            q,
            k,
            v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=fa4_window)
        # FA4's varlen entry point returns (out, softmax_lse); older builds
        # return the tensor directly.
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = _unpack_varlen_output(x, q_lens, b, lq)
        return x.type(out_dtype)

    # apply attention
    if backend == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        x = _unpack_varlen_output(x, q_lens, b, lq)
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic)
        x = _unpack_varlen_output(x, q_lens, b, lq)

    # output
    return x.type(out_dtype)

