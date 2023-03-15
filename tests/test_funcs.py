from typing import cast

import numpy as np

from spox import Var
from spox_array import SpoxArray, const


def arr(var: Var) -> np.ndarray:
    return cast(np.ndarray, SpoxArray(var))  # type: ignore


def val(array: np.ndarray):
    if isinstance(array, SpoxArray):
        return array._var._get_value()  # noqa
    return np.asarray(array)


def assert_eq(got, expected):
    expected = np.array(expected)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected)


def test_concat():
    assert_eq(val(np.concatenate([arr(const([1])), arr(const([2]))])), [1, 2])