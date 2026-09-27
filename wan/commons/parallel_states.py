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

import os
from typing import Optional, Tuple
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelDims:
    sp: int = 1
    world_size: int = -1
    dp_replicate: int = 1
    sp_split_sizes: Optional[Tuple[int, ...]] = None
    sp_max_local_seq_len: Optional[int] = None
    sp_total_seq_len: Optional[int] = None

    def __post_init__(self):
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.build_mesh("cuda")

    def build_mesh(self, device_type):
        assert self.world_size % self.sp == 0, "world_size must be divisible by sp"
        assert self.world_size % self.dp_replicate == 0, "world_size must be divisible by dp_replicate"

        self.world_mesh = init_device_mesh(
            device_type,
            [self.world_size // self.sp, self.sp],
            mesh_dim_names=["dp", "sp"],
        )
        self.fsdp_mesh = init_device_mesh(
            device_type, 
            [self.dp_replicate, self.world_size // self.dp_replicate],
            mesh_dim_names=["dp_replicate", "fsdp_shard"]
        )

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def sp_group(self):
        return self.world_mesh["sp"].get_group()

    @property
    def sp_mesh(self):
        return self.world_mesh["sp"]

    @property
    def sp_rank(self):
        if self.sp_enabled:
            return self.world_mesh["sp"].get_local_rank()
        else:
            return dist.get_rank()

    @property
    def dp_enabled(self):
        return self.sp > 1

    def set_sp_sequence_info(self, split_sizes):
        split_sizes = tuple(int(size) for size in split_sizes)
        if not split_sizes:
            self.sp_split_sizes = None
            self.sp_max_local_seq_len = None
            self.sp_total_seq_len = None
            return
        self.sp_split_sizes = split_sizes
        self.sp_max_local_seq_len = max(split_sizes)
        self.sp_total_seq_len = sum(split_sizes)

    def clear_sp_sequence_info(self):
        self.sp_split_sizes = None
        self.sp_max_local_seq_len = None
        self.sp_total_seq_len = None


__parallel_dims = None


def initialize_parallel_state(
    sp: int = 1,
    dp_replicate: int = 1,
):
    global __parallel_dims
    __parallel_dims = ParallelDims(sp=sp, dp_replicate=dp_replicate)
    return __parallel_dims


def get_parallel_state():
    if __parallel_dims is None:
        # create default parallel states (without enabling any parallelism)
        initialize_parallel_state()
    return __parallel_dims
