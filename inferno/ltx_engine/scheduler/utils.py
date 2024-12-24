import torch


def append_dims(x: torch.Tensor, target_dims: int):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]