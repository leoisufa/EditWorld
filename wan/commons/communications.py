# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Any, Tuple
import torch
import torch.distributed as dist

from torch.nn import functional as F

from wan.commons.parallel_states import get_parallel_state


def broadcast(input_: torch.Tensor, group: dist.ProcessGroup):
    src = dist.get_global_rank(group, 0)
    dist.broadcast(input_, src=src, group=group)


def _pad_to_shape(input_: torch.Tensor, target_shape):
    if tuple(input_.shape) == tuple(target_shape):
        return input_

    pad = []
    for current, target in zip(reversed(input_.shape), reversed(target_shape)):
        if current > target:
            raise ValueError(
                f"Cannot pad tensor from shape {tuple(input_.shape)} to smaller shape {tuple(target_shape)}"
            )
        pad.extend([0, target - current])
    return F.pad(input_, pad)


def _slice_to_shape(input_: torch.Tensor, target_shape):
    slices = tuple(slice(0, size) for size in target_shape)
    return input_[slices]


def _reduce_scatter_fallback(group, input_tensor_list, rank):
    stacked = torch.stack(input_tensor_list, dim=0)
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=group)
    return stacked[rank]


def _get_sp_sequence_meta(group: dist.ProcessGroup):
    try:
        parallel_state = get_parallel_state()
    except Exception:
        return None

    split_sizes = getattr(parallel_state, "sp_split_sizes", None)
    max_local_seq_len = getattr(parallel_state, "sp_max_local_seq_len", None)
    total_seq_len = getattr(parallel_state, "sp_total_seq_len", None)
    world_size = dist.get_world_size(group)

    if (
        split_sizes is None
        or max_local_seq_len is None
        or total_seq_len is None
        or len(split_sizes) != world_size
    ):
        return None

    return tuple(int(size) for size in split_sizes), int(max_local_seq_len), int(total_seq_len)


def _trim_padded_sequence_blocks(
    input_: torch.Tensor,
    split_sizes,
    padded_block_len: int,
    dim: int,
):
    pieces = []
    start = 0
    for size in split_sizes:
        end = start + size
        indices = [slice(None)] * input_.dim()
        indices[dim] = slice(start, end)
        pieces.append(input_[tuple(indices)])
        start += padded_block_len
    return torch.cat(pieces, dim=dim)


def _all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        seq_meta = _get_sp_sequence_meta(group)
        if seq_meta is not None:
            seq_lens, padded_shard_seqlen, total_seq_len = seq_meta
        else:
            seq_lens = [None] * seq_world_size
            dist.all_gather_object(seq_lens, input.shape[1], group)
            padded_shard_seqlen = max(seq_lens)
            total_seq_len = sum(seq_lens)

        local_rank = dist.get_group_rank(group, dist.get_rank())
        local_seq_len = seq_lens[local_rank]
        if input.shape[1] != local_seq_len:
            raise ValueError(
                f"Sequence-parallel shard mismatch on rank {local_rank}: "
                f"expected {local_seq_len}, got {input.shape[1]}"
            )
        if local_seq_len < padded_shard_seqlen:
            input = F.pad(input, (0, 0, 0, 0, 0, padded_shard_seqlen - local_seq_len))

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert (
            hc % seq_world_size == 0
        ), f"Invalid size: {hc}, which should be divisible by {seq_world_size}"
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if total_seq_len != seqlen:
            output = _trim_padded_sequence_blocks(output, seq_lens, shard_seqlen, dim=1)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        seq_meta = _get_sp_sequence_meta(group)
        if seq_meta is not None and seqlen == seq_meta[2]:
            seq_lens, shard_seqlen, total_seq_len = seq_meta
            if shard_seqlen * seq_world_size != total_seq_len:
                gap = shard_seqlen * seq_world_size - total_seq_len
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
                bs, seqlen, shard_hc, hs = input.shape
            else:
                gap = 0
        else:
            if seqlen % seq_world_size != 0:
                new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
                gap = new_seqlen - seqlen
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
                bs, seqlen, shard_hc, hs = input.shape
            else:
                gap = 0
            shard_seqlen = seqlen // seq_world_size
            seq_lens = None
            total_seq_len = seqlen - gap

        assert seqlen % seq_world_size == 0

        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        # (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if seq_lens is not None:
            local_rank = dist.get_group_rank(group, dist.get_rank())
            output = output[:, : seq_lens[local_rank]]
        elif (
            gap > 0
            and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1
        ):
            output = output[:, :-gap]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_list(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    device, dtype = input_.device, input_.dtype
    input_shape_list = [
        torch.tensor(t.shape, device=device)
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_shape_list = [torch.empty_like(input_shape_list[idx]) for idx in range(world_size)]
    dist.all_to_all(output_shape_list, input_shape_list, group=group)

    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty(output_shape_list[idx].tolist(), device=device, dtype=dtype) for idx in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)

    return output_list


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all_list(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)


class _Reduce_Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        rank = dist.get_group_rank(group, global_rank)
        seq_meta = _get_sp_sequence_meta(group)

        ctx.use_sequence_meta = False
        if (
            seq_meta is not None
            and dim == 1
            and input_.shape[1] == seq_meta[0][rank]
        ):
            split_sizes, max_size, _ = seq_meta
            ctx.use_sequence_meta = True
            ctx.shapes = [
                tuple(max_size if axis == dim else size for axis, size in enumerate(input_.shape))
                for _ in range(world_size)
            ]
            ctx.shapes = [
                tuple(split_sizes[idx] if axis == dim else shape[axis] for axis in range(len(shape)))
                for idx, shape in enumerate(ctx.shapes)
            ]
            ctx.max_shape = tuple(max_size if axis == dim else size for axis, size in enumerate(input_.shape))
        else:
            shapes = [None] * world_size
            dist.all_gather_object(shapes, tuple(input_.shape), group=group)
            ctx.shapes = [tuple(shape) for shape in shapes]
            ctx.max_shape = tuple(max(shape[i] for shape in ctx.shapes) for i in range(len(ctx.shapes[0])))

        input_padded = _pad_to_shape(input_.contiguous(), ctx.max_shape)
        tensor_list = [
            torch.empty(ctx.max_shape, dtype=input_.dtype, device=input_.device)
            for _ in range(world_size)
        ]
        dist.all_gather(tensor_list, input_padded, group=group)

        output = torch.cat(
            [_slice_to_shape(tensor_list[i], ctx.shapes[i]) for i in range(world_size)],
            dim=dim,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        global_rank = dist.get_rank()
        rank = dist.get_group_rank(group, global_rank)
        dim = ctx.dim
        shapes = ctx.shapes
        sizes = [shape[dim] for shape in shapes]
        grad_input_list = list(torch.split(grad_output, sizes, dim=dim))

        if len(set(shapes)) == 1:
            grad_input = grad_input_list[rank].contiguous()
            if dist.get_backend(group) == dist.Backend.GLOO:
                grad_input = _reduce_scatter_fallback(
                    group,
                    [item.contiguous() for item in grad_input_list],
                    rank,
                )
            else:
                grad_input = _Reduce_Scatter.apply(
                    dist.ReduceOp.SUM,
                    group,
                    grad_input,
                    *(item.contiguous() for item in grad_input_list),
                )
            return grad_input, None, None

        padded_inputs = [
            _pad_to_shape(item.contiguous(), ctx.max_shape) for item in grad_input_list
        ]
        if dist.get_backend(group) == dist.Backend.GLOO:
            grad_input = _reduce_scatter_fallback(group, padded_inputs, rank)
        else:
            grad_input = torch.empty(
                ctx.max_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            dist.reduce_scatter(grad_input, padded_inputs, op=dist.ReduceOp.SUM, group=group)
        grad_input = _slice_to_shape(grad_input, shapes[rank]).contiguous()
        return grad_input, None, None


@torch.compiler.disable
def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, group)
