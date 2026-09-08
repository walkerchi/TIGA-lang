"""Optional Torch correctness oracle for framework-independent reducers.

Nothing in this file is a Tiga lowering or performance backend. Compiler
execution consumes reducer algebra from IR; this adapter exists only so the
Torch-based MessagePassing reference evaluator can provide differential tests.
"""

from __future__ import annotations

import torch

from ...reducer.core import (
    MeanReducer,
    OnlineSoftmaxItem,
    OnlineSoftmaxReducer,
    ProductReducer,
    Reducer,
    SumReducer,
)


def _torch_dtype(dtype: object | None, score: torch.Tensor) -> torch.dtype:
    if dtype is None:
        return (
            torch.float32
            if score.dtype in (torch.float16, torch.bfloat16)
            else score.dtype
        )
    if isinstance(dtype, torch.dtype):
        return dtype
    name = getattr(dtype, "name", None)
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise TypeError(
            "online_softmax accumulation_dtype must be a floating dtype"
        ) from error


def reduce_sum(
    reducer: SumReducer,
    message: torch.Tensor,
    destination: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    output = torch.full(
        (num_dst, *message.shape[1:]),
        reducer.identity(),
        dtype=message.dtype,
        device=message.device,
    )
    output.index_add_(0, destination, message)
    return output


def reduce_online_softmax(
    reducer: OnlineSoftmaxReducer,
    item: OnlineSoftmaxItem,
    destination: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    score, value = item.score, item.value
    if not isinstance(score, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("online_softmax score and value must be Tensors")
    # Trailing singleton score lanes are decorative: an (E, 1) score is a
    # scalar score per edge, matching a value with no lane prefix.
    while score.ndim > 1 and score.shape[-1] == 1:
        score = score.squeeze(-1)
    if score.ndim < 1 or value.ndim < score.ndim:
        raise ValueError(
            "online_softmax requires score [edges, *lanes] and value "
            "[edges, *lanes, *payload]")
    if score.shape[0] != destination.numel() or value.shape[0] != score.shape[0]:
        raise ValueError("online_softmax score/value leading dimension must be edges")
    if tuple(value.shape[1:score.ndim]) != tuple(score.shape[1:]):
        raise ValueError(
            "online_softmax value must begin with the score lane dimensions")
    if not score.is_floating_point() or not value.is_floating_point():
        raise TypeError("online_softmax score and value must be floating point")
    if score.device != value.device or destination.device != score.device:
        raise ValueError("online_softmax score, value and graph must share a device")

    accumulation_dtype = _torch_dtype(reducer.accumulation_dtype, score)
    if not torch.empty((), dtype=accumulation_dtype).is_floating_point():
        raise TypeError("online_softmax accumulation_dtype must be floating point")
    score_acc = score.to(accumulation_dtype)
    value_acc = value.to(accumulation_dtype)
    lanes = tuple(score.shape[1:])
    payload_rank = value.ndim - score.ndim

    if reducer.deterministic:
        result = torch.zeros(
            (num_dst, *value.shape[1:]),
            device=value.device, dtype=accumulation_dtype)
        for row in range(num_dst):
            selected = destination == row
            if not bool(selected.any()):
                continue
            row_score = score_acc[selected]
            row_value = value_acc[selected]
            row_maximum = row_score.max(dim=0).values
            row_scale = torch.exp(row_score - row_maximum)
            row_scale_value = row_scale.reshape(
                *row_scale.shape, *((1,) * payload_rank))
            row_denominator = row_scale.sum(dim=0).reshape(
                *lanes, *((1,) * payload_rank))
            result[row] = (
                row_value * row_scale_value).sum(dim=0) / row_denominator
        return result.to(value.dtype)

    maximum = torch.full(
        (num_dst, *lanes), -torch.inf,
        device=score.device, dtype=accumulation_dtype)
    score_index = destination.reshape(
        destination.shape[0], *((1,) * len(lanes))).expand_as(score_acc)
    maximum.scatter_reduce_(
        0, score_index, score_acc, reduce="amax", include_self=True)
    scale = torch.exp(score_acc - maximum[destination])
    denominator = torch.zeros_like(maximum)
    denominator.index_add_(0, destination, scale)
    value_scale = scale.reshape(*scale.shape, *((1,) * payload_rank))
    numerator = torch.zeros(
        (num_dst, *value.shape[1:]),
        device=value.device, dtype=accumulation_dtype)
    numerator.index_add_(0, destination, value_acc * value_scale)
    denominator = denominator.reshape(
        *denominator.shape, *((1,) * payload_rank))
    result = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(accumulation_dtype).tiny),
        torch.zeros_like(numerator),
    )
    return result.to(value.dtype)


def reduce_mean(
    reducer: MeanReducer,
    message: torch.Tensor,
    destination: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    total = torch.zeros(
        (num_dst, *message.shape[1:]),
        dtype=message.dtype,
        device=message.device,
    )
    total.index_add_(0, destination, message)
    count = torch.zeros(
        num_dst, dtype=message.dtype, device=message.device
    )
    count.index_add_(
        0, destination, torch.ones_like(destination, dtype=message.dtype))
    # finalize(identity) is 0 / 0: degree-0 rows are NaN, as documented.
    return total / count.reshape(num_dst, *((1,) * (message.ndim - 1)))


def reduce_prod(
    reducer: ProductReducer,
    message: torch.Tensor,
    destination: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    output = torch.full(
        (num_dst, *message.shape[1:]),
        reducer.identity(),
        dtype=message.dtype,
        device=message.device,
    )
    import warnings

    with warnings.catch_warnings():
        # index_reduce is the differentiable segment product; torch still
        # tags it beta, which is noise for Tiga callers.
        warnings.simplefilter("ignore", UserWarning)
        return output.index_reduce(0, destination, message, "prod")


def reduce(
    reducer: Reducer,
    message: object,
    destination: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    if isinstance(reducer, SumReducer):
        if not isinstance(message, torch.Tensor):
            raise TypeError("sum reducer message must be a Tensor")
        return reduce_sum(reducer, message, destination, num_dst)
    if isinstance(reducer, OnlineSoftmaxReducer):
        if not isinstance(message, OnlineSoftmaxItem):
            raise TypeError(
                "online_softmax edge() must return self.reducer(score, value)"
            )
        return reduce_online_softmax(reducer, message, destination, num_dst)
    if isinstance(reducer, MeanReducer):
        if not isinstance(message, torch.Tensor):
            raise TypeError("mean reducer message must be a Tensor")
        return reduce_mean(reducer, message, destination, num_dst)
    if isinstance(reducer, ProductReducer):
        if not isinstance(message, torch.Tensor):
            raise TypeError("prod reducer message must be a Tensor")
        return reduce_prod(reducer, message, destination, num_dst)
    raise NotImplementedError(
        f"Torch oracle has no evaluator for reducer {type(reducer).__name__}"
    )


__all__ = [
    "reduce",
    "reduce_mean",
    "reduce_online_softmax",
    "reduce_prod",
    "reduce_sum",
]
