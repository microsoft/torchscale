import torch.distributed as dist


def _find_my_group_index(grouped_ranks):
    my_rank = dist.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError

def get_moe_group(moe_expert_count=None):
    if dist.is_initialized():
        if not hasattr(get_moe_group, "_moe_groups"):
            world_size = dist.get_world_size()

            if world_size <= moe_expert_count:
                assert moe_expert_count % world_size == 0
                moe_groups = [[i] for i in range(world_size)]

            else:
                assert world_size % moe_expert_count == 0
                ranks_per_group = world_size // moe_expert_count
                moe_groups = [
                    [i + j * moe_expert_count for j in range(ranks_per_group)]
                    for i in range(moe_expert_count)
                ]

            get_moe_group._moe_expert_count = moe_expert_count
            get_moe_group._moe_group_idx = moe_groups
            get_moe_group._moe_groups = [dist.new_group(g) for g in moe_groups]

        my_group_idx = _find_my_group_index(get_moe_group._moe_group_idx)
        return my_group_idx, get_moe_group._moe_groups[my_group_idx]


def get_all2all_group(moe_expert_count):
    if dist.is_initialized():
        if not hasattr(get_all2all_group, "_all2all_groups"):
            world_size = dist.get_world_size()

            # more experts than world size
            if world_size <= moe_expert_count:
                assert moe_expert_count % world_size == 0
                all2all_groups = [[i for i in range(world_size)]]

            # larger world than num experts
            else:
                assert world_size % moe_expert_count == 0
                ranks_per_group = world_size // moe_expert_count
                all2all_groups = [
                    [i * moe_expert_count + j for j in range(moe_expert_count)]
                    for i in range(ranks_per_group)
                ]

            get_all2all_group._all2all_group_idx = all2all_groups
            get_all2all_group._all2all_groups = [
                dist.new_group(g) for g in all2all_groups
            ]

        my_group_idx = _find_my_group_index(get_all2all_group._all2all_group_idx)
        return get_all2all_group._all2all_groups[my_group_idx]




