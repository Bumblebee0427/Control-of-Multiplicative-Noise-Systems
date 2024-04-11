import typing

import torch
from torch import Tensor
from numpy import ndarray


def to_numpy(x: Tensor) -> ndarray:
    return x.cpu().detach().numpy()


def extract_copy_tensors_from_iterable(param_iterable: typing.Iterable) -> Tensor:
    list_params: typing.List = []
    for param in param_iterable:
        list_params.append(torch.flatten(torch.clone(param.data)))
    params_tensor: Tensor = torch.cat(tensors=list_params)
    return params_tensor


__all__ = ["to_numpy", "extract_copy_tensors_from_iterable"]
