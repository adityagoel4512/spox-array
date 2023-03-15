import functools
import operator
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import numpy.lib.mixins
import numpy.typing as npt

import spox.opset.ai.onnx.v17 as op
from spox import Var

UFUNC_HANDLERS: dict[str, dict[str, Any]] = {}
FUNCTION_HANDLERS: dict[str, Any] = {}

INDEX_MIN: int = np.iinfo(np.int64).min
INDEX_MAX: int = np.iinfo(np.int64).max


def implements(target=None, *, name: str | None = None, method: str | None = None):
    def decorator(fun):
        nonlocal name
        if name is None:
            name = fun.__name__
        if method is not None:
            UFUNC_HANDLERS.setdefault(name, {})[method] = fun
        else:
            FUNCTION_HANDLERS[name] = fun
        return fun

    return decorator if target is None else decorator(target)


def const(value: npt.ArrayLike, dtype: npt.DTypeLike = None) -> Var:
    return op.constant(value=np.array(value, dtype))


class SpoxArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    _var: Var

    def __init__(self, var: Var):
        self._var = var
        if self._var.unwrap_tensor().shape is None:
            raise TypeError("Rank of a SpoxArray must be known.")

    def __var__(self, var: Var | None = None) -> Var:
        if var is not None:
            self._var = var
        return self._var

    @property
    def dtype(self) -> np.dtype:
        return self._var.unwrap_tensor().dtype

    @property
    def shape(self) -> tuple[int | str | None, ...]:
        return self._var.unwrap_tensor().shape

    def __repr__(self):
        return f"{self.__class__.__name__}({self._var})"

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if (
            ufunc.__name__ in UFUNC_HANDLERS
            and method in UFUNC_HANDLERS[ufunc.__name__]
        ):
            return UFUNC_HANDLERS[ufunc.__name__][method](*inputs, **kwargs)
        # raise NotImplementedError(f"{ufunc = }, {method = }, {inputs = }, {kwargs = }")
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ in FUNCTION_HANDLERS:
            return FUNCTION_HANDLERS[func.__name__](*args, **kwargs)
        # raise NotImplementedError(f"{func = }, {types = }, {args = }, {kwargs = }")
        return NotImplemented

    def __getitem__(self, index):
        index_ = index
        try:
            index = operator.index(index)
        except TypeError:
            pass
        else:
            pass
        if isinstance(index, slice):
            index = (index,) + (slice(None),) * (len(self.shape) - 1)
        if isinstance(index, tuple):
            axis_slices = {
                d: axis_slice
                for d, axis_slice in enumerate(index)
                if isinstance(axis_slice, slice) and axis_slice != slice(None)
            }
            axis_indices = {
                d: axis_index
                for d, axis_index in enumerate(index)
                if isinstance(axis_index, int)
            }
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
                    self.__var__(),
                    const(starts),
                    const(ends),
                    const(list(axis_slices.keys())),
                    const(steps),
                )
                if axis_slices
                else self.__var__()
            )
            for axis, axis_index in sorted(axis_indices.items(), reverse=True):
                indexed = op.gather(indexed, const(axis_index), axis=axis)
            return SpoxArray(indexed)
        raise TypeError(f"Cannot index SpoxArray with {index_!r}.")


def promote(
    *args: Var | SpoxArray | npt.ArrayLike,
    floating: bool = False,
    casting: str = "same_kind",
    dtype: Any = None,
) -> Sequence[SpoxArray]:
    """
    Apply constant promotion and type promotion to given parameters,
    creating constants and/or casting.
    """
    if dtype is None:
        target_type = result_type(*args)
        if floating and not issubclass(target_type.type, np.floating):
            target_type = np.float64
    else:
        target_type = dtype

    def _promote_target(obj: Var | npt.ArrayLike) -> Optional[Var]:
        to_cast = obj.dtype if isinstance(obj, SpoxArray) else obj
        if casting is not None and not np.can_cast(to_cast, target_type, casting):
            raise TypeError(
                f"Cannot cast {obj.dtype} to {target_type} with {casting=}."
            )
        if isinstance(obj, SpoxArray):
            return (
                op.cast(obj.__var__(), to=target_type)
                if obj.dtype != target_type
                else obj.__var__()
            )
        return const(obj, dtype=target_type)

    return tuple(SpoxArray(var) for var in map(_promote_target, args))


def _nested_structure(xs):
    if not isinstance(xs, Iterable) or isinstance(xs, numpy.ndarray):
        return [xs], lambda x: x
    sub = [_nested_structure(x) for x in xs]
    flat: list[Any] = sum((chunk for chunk, _ in sub), [])

    def restructure(*args):
        i = 0
        result = []
        for ch, re in sub:
            n = len(ch)
            result.append(re(*args[i : i + n]))
            i += n
        return result

    return flat, restructure


def promote_args(obj=None, *, floating=False):
    def wrapper(fun):
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            flat_args, restructure = _nested_structure(args)
            promoted_args = promote(
                *flat_args,
                casting=kwargs.pop("casting", None),
                dtype=kwargs.pop("dtype", None),
                floating=floating,
            )
            re_args = restructure(*promoted_args)
            return fun(*re_args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper


def handle_out(fun):
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        out = kwargs.pop("out", None)
        result: SpoxArray = fun(*args, **kwargs)
        if out is not None:
            if not isinstance(out, SpoxArray):
                raise TypeError(
                    f"Output for SpoxArrays must also be written to one, not {type(out).__name__}."
                )
            out.__var__(result.__var__())
            return out
        return result

    return inner


def unwrap_vars(fun):
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        flat_args, restructure = _nested_structure(args)
        re_args = restructure(
            *(arg.__var__() if isinstance(arg, SpoxArray) else arg for arg in flat_args)
        )
        return fun(*re_args, **kwargs)

    return inner


def wrap_var(fun):
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        return SpoxArray(fun(*args, **kwargs))

    return inner


@implements
def result_type(*args):
    targets: list[np.dtype | npt.ArrayLike] = [
        x.__var__().unwrap_tensor().dtype
        if isinstance(x, SpoxArray)
        else (x.unwrap_tensor().dtype if isinstance(x, Var) else x)
        for x in args
    ]
    return np.dtype(np.result_type(*targets))
