# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.distributed as dist

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

def get_data_parallel_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_data_parallel_group, "_global_group"):
            get_data_parallel_group._global_group = dist.new_group()
        return get_data_parallel_group._global_group
    else:
        return None

def get_rank(group):
    return dist.get_rank(group=group)

def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1

def get_data_parallel_rank():
    return get_rank(get_data_parallel_group())

def get_data_parallel_world_size():
    return get_world_size(get_data_parallel_group())


class Allgather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        world_size = get_data_parallel_world_size()
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype,
                            device=torch.cuda.current_device())
        torch.distributed._all_gather_base(output, input_.contiguous(),
                                        group=get_data_parallel_group())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_data_parallel_world_size()

        dim_size = list(grad_output.size())
        assert dim_size[0] % world_size == 0, \
            "First dimension of the tensor should be divisible by tensor parallel size"
        
        dim_size[0] = dim_size[0] // world_size
    
        output = torch.empty(dim_size, dtype=grad_output.dtype,
                            device=torch.cuda.current_device())
        
        torch.distributed._reduce_scatter_base(output, grad_output.contiguous(), 
                                            group=get_data_parallel_group())
        
        return output

all_gather_func = Allgather.apply
