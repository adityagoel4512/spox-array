import operator

import numpy as np
import numpy.typing as npt
import spox.opset.ai.onnx.v17 as op
from spox import Var

INDEX_MIN: int = np.iinfo(np.int64).min
INDEX_MAX: int = np.iinfo(np.int64).max


def _int_or_none_or_elipsis(x) -> bool:
    return isinstance(x, int | Ellipsis) or x is None


def normalize_index(
    index,
    rank: int,
) -> Var | tuple[dict[tuple[int, int, int], slice], dict[int, int]] | int:
    index_ = index
    if isinstance(index, list | np.ndarray | int):
        index = op.const(index)
    if isinstance(index, Var):
        return index
    try:
        index = operator.index(index)
    except TypeError:
        pass
    else:
        pass
    if isinstance(index, int | slice):
        index = (index,) + (slice(None),) * (rank - 1)
    if isinstance(index, tuple):
        if not all(
            isinstance(d, int | slice | type(Ellipsis) | type(None)) for d in index
        ):
            raise TypeError(
                f"Bad inferred axis-index types in {index!r} from {index_!r}",
            )

        if not all(
            all(map(_int_or_none_or_elipsis, (d.start, d.stop, d.step)))
            for d in index
            if isinstance(d, slice)
        ):
            raise TypeError(
                f"Bad inferred axis-slice types in {index!r} from {index_!r}",
            )
        if any(d.step == 0 for d in index if isinstance(d, slice)):
            raise TypeError(f"Zero-step slice in {index!r} from {index_!r}")
        axis_slices = {
            d: axis_slice
            for d, axis_slice in enumerate(index)
            if isinstance(axis_slice, slice)
            and axis_slice != slice(None)
            and axis_slice.step != Ellipsis
        }
        axis_indices = {
            d: axis_index
            for d, axis_index in enumerate(index)
            if isinstance(axis_index, int)
        }
        axis_new_axes = {
            d: new_axis
            for d, new_axis in enumerate(index)
            if new_axis is Ellipsis
            if isinstance(new_axis, type(None) | type(Ellipsis))
        }
        return axis_slices, axis_indices, axis_new_axes
    raise TypeError(f"Cannot index with {index_!r} (transformed to {index!r}).")


def getitem(var: Var, index_) -> Var:
    index = normalize_index(index_, len(var.unwrap_tensor().shape))
    if isinstance(index, Var):
        index_dtype = index.unwrap_tensor().dtype
        if index_dtype == np.dtype(bool):
            return op.compress(var, index, axis=0)
        if np.issubdtype(index_dtype, np.integer):
            return op.gather(var, index, axis=0)
        raise TypeError(
            f"Unsupported index array dtype {index_dtype} (from {index_!r}).",
        )
    if isinstance(index, tuple):
        axis_slices, axis_indices, new_axes = index
        starts: list[int] = [
            x.start if x.start is not None else 0 for x in axis_slices.values()
        ]
        ends: list[int] = [
            x.stop
            if x.stop is not None
            else (INDEX_MAX if x.step is None or x.step > 0 else INDEX_MIN)
            for x in axis_slices.values()
        ]
        steps: list[int] = [
            x.step if x.step is not None else 1 for x in axis_slices.values()
        ]
        indexed: Var = (
            op.slice(
                var,
                op.const(starts),
                op.const(ends),
                op.const(list(axis_slices.keys())),
                op.const(steps),
            )
            if axis_slices
            else var
        )
        for axis, axis_index in sorted(axis_indices.items(), reverse=True):
            indexed = op.gather(indexed, op.const(axis_index), axis=axis)
        if new_axes:
            if isinstance(new_axes, dict):
                new_axes = tuple(new_axes.keys())
            indexed = op.unsqueeze(indexed, axes=op.const(new_axes))
        return indexed
    raise TypeError(f"Cannot index with {index_!r}.")


def ndindex(shape: Var) -> Var:
    """
    Returns a tensor of a given shape, with an added last axis for vectors of indices.
    In essence, in the returned tensor `a[i][j][k][...] = vector(i, j, k, ...)`.
    """
    (rank,) = shape.unwrap_tensor().shape
    ranges = [
        op.range(op.const(0), op.gather(shape, op.const(i)), op.const(1))
        for i in range(rank)
    ]
    fit_ranges = [
        op.unsqueeze(r, op.const([j for j in range(rank) if i != j], dtype=np.int_))
        for i, r in enumerate(ranges)
    ]
    expanded_ranges = [op.expand(r, shape) for r in fit_ranges]
    return op.concat(
        [op.unsqueeze(r, op.const([-1])) for r in expanded_ranges],
        axis=-1,
    )


def setitem(var: Var, index_, updates_: Var | npt.ArrayLike) -> Var:
    indices = getitem(ndindex(op.shape(var)), index_)
    # broadcast updates as appropriate
    index_path_shape = op.slice(op.shape(indices), op.const([0]), op.const([-1]))
    update_shape = op.slice(
        op.shape(var),
        op.gather(op.shape(indices), op.const([-1])),
        op.const([-1]),
    )
    # TODO: concat empty without checking shape
    (index_path_rank,) = index_path_shape.unwrap_tensor().shape
    (update_rank,) = update_shape.unwrap_tensor().shape
    if index_path_rank == update_rank == 0:
        target_shape = index_path_shape
    else:
        target_shape = op.concat([index_path_shape, update_shape], axis=0)
    updates_ = op.expand(updates_, target_shape)
    return op.scatter_nd(var, indices, updates_)
