import inspect
import itertools
import pickle
import re
from contextlib import nullcontext as does_not_raise
from functools import partial
from typing import Literal

import jax.numpy
import numpy as np
import pynapple as nap
import pytest
from conftest import BasisFuncsTesting, CombinedBasis, list_all_basis_classes

import nemos._inspect_utils as inspect_utils
import nemos.basis.basis as basis
import nemos.convolve as convolve
from nemos.basis._basis import AdditiveBasis, Basis, MultiplicativeBasis, add_docstring
from nemos.basis._decaying_exponential import OrthExponentialBasis
from nemos.basis._raised_cosine_basis import (
    RaisedCosineBasisLinear,
    RaisedCosineBasisLog,
)
from nemos.basis._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from nemos.utils import pynapple_concatenate_numpy


def extra_decay_rates(cls, n_basis):
    name = cls.__name__
    if "OrthExp" in name:
        return dict(decay_rates=np.arange(1, n_basis + 1))
    return {}


def test_all_basis_are_tested() -> None:
    """Meta-test.

    Ensure that all concrete classes in the 'basis' module are tested.
    """
    # Get all classes from the current module.
    all_classes = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isclass
    )

    # Filter the classes that are subclasses of 'SuperClass'.
    subclasses = [
        cls
        for _, cls in all_classes
        if issubclass(cls, BasisFuncsTesting) and cls != BasisFuncsTesting
    ]

    # Create the set of basis function objects that are tested using the cls definition
    tested_bases = {
        test_cls.cls[mode]
        for mode in ["eval", "conv"]
        for test_cls in subclasses
        if test_cls != CombinedBasis
    }

    # Create the set of all the concrete basis classes
    all_bases = set(list_all_basis_classes())

    if all_bases != all_bases.intersection(tested_bases):
        raise ValueError(
            "Test should be implemented for each of the concrete classes in the basis module.\n"
            f"The following classes are not tested: {[bas.__qualname__ for bas in all_bases.difference(tested_bases)]}"
        )

    pytest_marks = getattr(TestSharedMethods, "pytestmark", [])

    # Find the parametrize mark for TestSharedMethods
    out = None
    for mark in pytest_marks:
        if mark.name == "parametrize":
            # Return the arguments of the parametrize mark
            out = mark.args[1]  # The second argument contains the list

    if out is None:
        raise ValueError("cannot fine parametrization.")

    basis_tested_in_shared_methods = {o[key] for key in ("eval", "conv") for o in out}
    all_one_dim_basis = set(
        list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    assert basis_tested_in_shared_methods == all_one_dim_basis


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize(
    "method_name, descr_match",
    [
        ("evaluate_on_grid", "The number of points in the uniformly spaced grid"),
        (
            "compute_features",
            "Apply the basis transformation to the input data|Convolve basis functions with input time series|Evaluate basis at sample points",
        ),
        (
            "split_by_feature",
            "Decompose an array along a specified axis into sub-arrays",
        ),
        (
            "set_input_shape",
            "Set the expected input shape for the basis object",
        ),
    ],
)
def test_example_docstrings_add(
    basis_cls, method_name, descr_match, basis_class_specific_params
):

    basis_instance = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    method = getattr(basis_instance, method_name)
    doc = method.__doc__
    examp_delim = "\n        Examples\n        --------"

    assert examp_delim in doc
    doc_components = doc.split(examp_delim)
    assert len(doc_components) == 2
    assert len(doc_components[0].strip()) > 0
    assert re.search(descr_match, doc_components[0])

    # check that the basis name is in the example
    if basis_cls not in [AdditiveBasis, MultiplicativeBasis]:
        assert basis_cls.__name__ in doc_components[1]

    # check that no other basis name is in the example (except for additive and multiplicative)
    for basis_name in basis.__dir__():
        if basis_cls in [AdditiveBasis, MultiplicativeBasis]:
            continue
        if basis_name == basis_instance.__class__.__name__:
            continue
        assert f" {basis_name}" not in doc_components[1]


def test_add_docstring():

    class CustomClass:
        def method(self):
            """My extra text."""
            pass

    custom_add_docstring = partial(add_docstring, cls=CustomClass)

    class CustomSubClass(CustomClass):
        @custom_add_docstring("method")
        def method(self):
            """My custom method."""
            pass

    assert CustomSubClass().method.__doc__ == "My extra text.\nMy custom method."
    with pytest.raises(AttributeError, match="CustomClass has no attribute"):

        class CustomSubClass2(CustomClass):
            @custom_add_docstring("unknown", cls=CustomClass)
            def method(self):
                """My custom method."""
                pass

        CustomSubClass2()


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10), BSplineBasis),
        (basis.BSplineConv(10, window_size=11), BSplineBasis),
        (basis.CyclicBSplineEval(10), CyclicBSplineBasis),
        (basis.CyclicBSplineConv(10, window_size=11), CyclicBSplineBasis),
        (basis.MSplineEval(10), MSplineBasis),
        (basis.MSplineConv(10, window_size=11), MSplineBasis),
        (basis.RaisedCosineLinearEval(10), RaisedCosineBasisLinear),
        (basis.RaisedCosineLinearConv(10, window_size=11), RaisedCosineBasisLinear),
        (basis.RaisedCosineLogEval(10), RaisedCosineBasisLog),
        (basis.RaisedCosineLogConv(10, window_size=11), RaisedCosineBasisLog),
        (basis.OrthExponentialEval(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.OrthExponentialConv(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
    ],
)
def test_expected_output_eval_on_grid(basis_instance, super_class):
    x, y = super_class.evaluate_on_grid(basis_instance, 100)
    xx, yy = basis_instance.evaluate_on_grid(100)
    np.testing.assert_equal(xx, x)
    np.testing.assert_equal(yy, y)


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10), BSplineBasis),
        (basis.BSplineConv(10, window_size=11), BSplineBasis),
        (basis.CyclicBSplineEval(10), CyclicBSplineBasis),
        (basis.CyclicBSplineConv(10, window_size=11), CyclicBSplineBasis),
        (basis.MSplineEval(10), MSplineBasis),
        (basis.MSplineConv(10, window_size=11), MSplineBasis),
        (basis.RaisedCosineLinearEval(10), RaisedCosineBasisLinear),
        (basis.RaisedCosineLinearConv(10, window_size=11), RaisedCosineBasisLinear),
        (basis.RaisedCosineLogEval(10), RaisedCosineBasisLog),
        (basis.RaisedCosineLogConv(10, window_size=11), RaisedCosineBasisLog),
        (basis.OrthExponentialEval(10, np.arange(1, 11)), OrthExponentialBasis),
        (
            basis.OrthExponentialConv(10, decay_rates=np.arange(1, 11), window_size=12),
            OrthExponentialBasis,
        ),
    ],
)
def test_expected_output_compute_features(basis_instance, super_class):
    x = super_class.compute_features(basis_instance, np.linspace(0, 1, 100))
    xx = basis_instance.compute_features(np.linspace(0, 1, 100))
    nans = np.isnan(x.sum(axis=1))
    assert np.all(np.isnan(xx[nans]))
    np.testing.assert_array_equal(xx[~nans], x[~nans])


@pytest.mark.parametrize(
    "basis_instance, super_class",
    [
        (basis.BSplineEval(10, label="label"), BSplineBasis),
        (basis.BSplineConv(10, window_size=11, label="label"), BSplineBasis),
        (basis.CyclicBSplineEval(10, label="label"), CyclicBSplineBasis),
        (
            basis.CyclicBSplineConv(10, window_size=11, label="label"),
            CyclicBSplineBasis,
        ),
        (basis.MSplineEval(10, label="label"), MSplineBasis),
        (basis.MSplineConv(10, window_size=11, label="label"), MSplineBasis),
        (basis.RaisedCosineLinearEval(10, label="label"), RaisedCosineBasisLinear),
        (
            basis.RaisedCosineLinearConv(10, window_size=11, label="label"),
            RaisedCosineBasisLinear,
        ),
        (basis.RaisedCosineLogEval(10, label="label"), RaisedCosineBasisLog),
        (
            basis.RaisedCosineLogConv(10, window_size=11, label="label"),
            RaisedCosineBasisLog,
        ),
        (
            basis.OrthExponentialEval(10, np.arange(1, 11), label="label"),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="label"
            ),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="a"
            )
            * basis.RaisedCosineLogConv(10, window_size=11, label="b"),
            OrthExponentialBasis,
        ),
        (
            basis.OrthExponentialConv(
                10, decay_rates=np.arange(1, 11), window_size=12, label="a"
            )
            + basis.RaisedCosineLogConv(10, window_size=11, label="b"),
            OrthExponentialBasis,
        ),
    ],
)
def test_expected_output_split_by_feature(basis_instance, super_class):
    inp = [np.linspace(0, 1, 100)] * basis_instance._n_input_dimensionality
    x = super_class.compute_features(basis_instance, *inp)
    xdict = super_class.split_by_feature(basis_instance, x)
    xxdict = basis_instance.split_by_feature(x)
    assert xdict.keys() == xxdict.keys()
    for k in xdict.keys():
        xx = xxdict[k]
        x = xdict[k]
        nans = np.isnan(x.sum(axis=(1, 2)))
        assert np.all(np.isnan(xx[nans]))
        np.testing.assert_array_equal(xx[~nans], x[~nans])


@pytest.mark.parametrize(
    "cls",
    [
        {"eval": basis.RaisedCosineLogEval, "conv": basis.RaisedCosineLogConv},
        {"eval": basis.RaisedCosineLinearEval, "conv": basis.RaisedCosineLinearConv},
        {"eval": basis.BSplineEval, "conv": basis.BSplineConv},
        {"eval": basis.CyclicBSplineEval, "conv": basis.CyclicBSplineConv},
        {"eval": basis.MSplineEval, "conv": basis.MSplineConv},
        {"eval": basis.OrthExponentialEval, "conv": basis.OrthExponentialConv},
    ],
)
class TestSharedMethods:

    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (0.5, 0, 1, does_not_raise()),
            (
                -0.5,
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (np.linspace(-1, 1, 10), 0, 1, does_not_raise()),
            (
                np.linspace(-1, 0, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
            (
                np.linspace(1, 2, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
        ],
    )
    def test_call_vmin_vmax(self, samples, vmin, vmax, expectation, cls):
        if "OrthExp" in cls["eval"].__name__ and not hasattr(samples, "shape"):
            return
        bas = cls["eval"](5, bounds=(vmin, vmax), **extra_decay_rates(cls["eval"], 5))
        with expectation:
            bas._evaluate(samples)

    @pytest.mark.parametrize("n_basis", [5, 6])
    @pytest.mark.parametrize("vmin, vmax", [(0, 1), (-1, 1)])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone_eval(self, cls, n_basis, vmin, vmax, inp_num):
        bas = cls["eval"](
            n_basis, bounds=(vmin, vmax), **extra_decay_rates(cls["eval"], n_basis)
        )
        bas.set_input_shape(inp_num)
        bas2 = bas.__sklearn_clone__()
        assert id(bas) != id(bas2)
        assert np.all(
            bas.__dict__.pop("decay_rates", True)
            == bas2.__dict__.pop("decay_rates", True)
        )
        assert bas.__dict__ == bas2.__dict__

    @pytest.mark.parametrize("n_basis", [5, 6])
    @pytest.mark.parametrize("ws", [10, 20])
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone_conv(self, cls, n_basis, ws, inp_num):
        bas = cls["conv"](
            n_basis, window_size=ws, **extra_decay_rates(cls["eval"], n_basis)
        )
        bas.set_input_shape(inp_num)
        bas2 = bas.__sklearn_clone__()
        assert id(bas) != id(bas2)
        assert np.all(
            bas.__dict__.pop("decay_rates", True)
            == bas2.__dict__.pop("decay_rates", True)
        )
        assert bas.__dict__ == bas2.__dict__

    @pytest.mark.parametrize(
        "attribute, value",
        [
            ("label", None),
            ("label", "label"),
            ("n_basis_input_", 1),
            ("n_output_features", 5),
        ],
    )
    def test_attr_setter(self, attribute, value, cls):
        bas = cls["eval"](n_basis_funcs=5, **extra_decay_rates(cls["eval"], 5))
        with pytest.raises(
            AttributeError, match=rf"can't set attribute|property '{attribute}' of"
        ):
            setattr(bas, attribute, value)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (2, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (3, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=10, **extra_decay_rates(cls["eval"], 5)
        )
        x = np.random.randn(20, 2)
        bas.compute_features(x)
        with expectation:
            bas.compute_features(np.random.randn(30, n_input))

    @pytest.mark.parametrize(
        "conv_kwargs, expectation",
        [
            (dict(), does_not_raise()),
            (
                dict(axis=0),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(axis=1),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (dict(shift=True), does_not_raise()),
            (
                dict(shift=True, axis=0),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(shifts=True),
                pytest.raises(ValueError, match="Unrecognized keyword arguments"),
            ),
            (dict(shift=True, predictor_causality="causal"), does_not_raise()),
            (
                dict(shift=True, time_series=np.arange(10)),
                pytest.raises(ValueError, match="Unrecognized keyword arguments"),
            ),
        ],
    )
    def test_init_conv_kwargs(self, conv_kwargs, expectation, cls):
        with expectation:
            cls["conv"](
                n_basis_funcs=5,
                window_size=200,
                conv_kwargs=conv_kwargs,
                **extra_decay_rates(cls["eval"], 5),
            )

    @pytest.mark.parametrize("label", [None, "label"])
    def test_init_label(self, label, cls):
        bas = cls["eval"](
            n_basis_funcs=5, label=label, **extra_decay_rates(cls["eval"], 5)
        )
        expected_label = str(label) if label is not None else cls["eval"].__name__
        assert bas.label == expected_label

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_output_features(self, n_input, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=10, **extra_decay_rates(cls["conv"], 5)
        )
        assert bas.n_output_features is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_output_features == n_input * bas.n_basis_funcs

    @pytest.mark.parametrize("n_input", [1, 2, 3])
    def test_set_num_basis_input(self, n_input, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=10, **extra_decay_rates(cls["conv"], 5)
        )
        assert bas.n_basis_input_ is None
        bas.compute_features(np.random.randn(20, n_input))
        assert bas.n_basis_input_ == (n_input,)
        assert bas._n_basis_input_ == (n_input,)

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, mn, mx",
        [
            (None, np.arange(5), [4], 0, 1),
            ((0, 3), np.arange(5), [4], 0, 3),
            ((1, 4), np.arange(5), [0], 1, 4),
            ((1, 3), np.arange(5), [0, 4], 1, 3),
        ],
    )
    def test_vmin_vmax_eval_on_grid_affects_x(
        self, bounds, samples, nan_idx, mn, mx, cls
    ):
        bas_no_range = cls["eval"](
            n_basis_funcs=5, bounds=None, **extra_decay_rates(cls["eval"], 5)
        )
        bas = cls["eval"](
            n_basis_funcs=5, bounds=bounds, **extra_decay_rates(cls["eval"], 5)
        )
        x1, _ = bas.evaluate_on_grid(10)
        x2, _ = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(x1, x2 * (mx - mn) + mn)

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_eval_on_grid_no_effect_on_eval(
        self, vmin, vmax, samples, nan_idx, cls
    ):
        # MSPline integrates to 1 on domain so must be excluded from this check
        if "MSpline" in cls["eval"].__name__:
            return
        bas_no_range = cls["eval"](
            n_basis_funcs=5, bounds=None, **extra_decay_rates(cls["eval"], 5)
        )
        bas = cls["eval"](
            n_basis_funcs=5, bounds=(vmin, vmax), **extra_decay_rates(cls["eval"], 5)
        )
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1, out2)

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (1, 2, 3),
                pytest.raises(
                    ValueError, match="The provided `bounds` must be of length two"
                ),
            ),
        ],
    )
    def test_vmin_vmax_init(self, bounds, expectation, cls):
        with expectation:
            bas = cls["eval"](
                n_basis_funcs=5, bounds=bounds, **extra_decay_rates(cls["eval"], 5)
            )
            assert bounds == bas.bounds if bounds else bas.bounds is None

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_basis_number(self, n_basis, mode, kwargs, cls):

        bas = cls[mode](
            n_basis_funcs=n_basis, **kwargs, **extra_decay_rates(cls[mode], n_basis)
        )
        x = np.linspace(0, 1, 10)
        assert bas._evaluate(x).shape[1] == n_basis

    @pytest.mark.parametrize("n_basis", [6])
    def test_call_equivalent_in_conv(self, n_basis, cls):
        bas_con = cls["conv"](
            n_basis_funcs=n_basis,
            window_size=10,
            **extra_decay_rates(cls["conv"], n_basis),
        )
        bas_eval = cls["eval"](
            n_basis_funcs=n_basis, **extra_decay_rates(cls["eval"], n_basis)
        )
        x = np.linspace(0, 1, 10)
        assert np.all(bas_con._evaluate(x) == bas_eval._evaluate(x))

    @pytest.mark.parametrize(
        "num_input, expectation",
        [
            (0, pytest.raises(TypeError, match="Input dimensionality mismatch")),
            (1, does_not_raise()),
            (2, pytest.raises(TypeError, match="Input dimensionality mismatch")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    @pytest.mark.parametrize("n_basis", [6])
    def test_call_input_num(self, num_input, n_basis, mode, kwargs, expectation, cls):
        bas = cls[mode](
            n_basis_funcs=n_basis, **kwargs, **extra_decay_rates(cls[mode], n_basis)
        )
        with expectation:
            bas._evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], does_not_raise()),
            (np.repeat(np.linspace(0, 1, 10), 10).reshape(10, 5, 2), does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("n_basis", [6])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_input_shape(self, inp, mode, kwargs, expectation, n_basis, cls):
        bas = cls[mode](
            n_basis_funcs=n_basis, **kwargs, **extra_decay_rates(cls[mode], n_basis)
        )
        with expectation:
            out = bas._evaluate(inp)
            out2 = bas.evaluate_on_grid(inp.shape[0])[1]
            assert np.all((out.reshape(out.shape[0], -1, n_basis) - out2[:, None]) == 0)
            assert out.shape == tuple((*inp.shape, n_basis))

    @pytest.mark.parametrize("n_basis", [6])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_nan_location(self, mode, kwargs, n_basis, cls):
        bas = cls[mode](
            n_basis_funcs=n_basis, **kwargs, **extra_decay_rates(cls[mode], n_basis)
        )
        inp = np.random.randn(10, 2, 3)
        inp[2, 0, [0, 2]] = np.nan
        inp[4, 1, 1] = np.nan
        out = bas._evaluate(inp)
        assert np.all(np.isnan(out[2, 0, [0, 2]]))
        assert np.all(np.isnan(out[4, 1, 1]))
        assert np.isnan(out).sum() == 3 * n_basis

    @pytest.mark.parametrize(
        "samples, expectation",
        [
            (np.array([0, 1, 2, 3, 4, 5]), does_not_raise()),
            (
                np.array(["a", "1", "2", "3", "4", "5"]),
                pytest.raises(TypeError, match="Input samples must"),
            ),
        ],
    )
    @pytest.mark.parametrize("n_basis", [6])
    def test_call_input_type(self, samples, expectation, n_basis, cls):
        bas = cls["eval"](
            n_basis_funcs=n_basis, **extra_decay_rates(cls["eval"], n_basis)
        )  # Only eval mode is relevant here
        with expectation:
            bas._evaluate(samples)

    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_nan(self, mode, kwargs, cls):
        bas = cls[mode](n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5))
        x = np.linspace(0, 1, 10)
        x[3] = np.nan
        assert all(np.isnan(bas._evaluate(x)[3]))

    @pytest.mark.parametrize("n_basis", [6, 7])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_non_empty(self, n_basis, mode, kwargs, cls):
        bas = cls[mode](
            n_basis_funcs=n_basis, **kwargs, **extra_decay_rates(cls[mode], n_basis)
        )
        with pytest.raises(ValueError, match="All sample provided must"):
            bas._evaluate(np.array([]))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_sample_axis(self, time_axis_shape, mode, kwargs, cls):
        bas = cls[mode](n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5))
        assert (
            bas._evaluate(np.linspace(0, 1, time_axis_shape)).shape[0]
            == time_axis_shape
        )

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_call_sample_range(self, mn, mx, expectation, mode, kwargs, cls):
        bas = cls[mode](n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5))
        with expectation:
            bas._evaluate(np.linspace(mn, mx, 10))

    @pytest.mark.parametrize(
        "kwargs, input1_shape, expectation",
        [
            (dict(), (10,), does_not_raise()),
            (
                dict(axis=0),
                (10,),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
            (
                dict(axis=1),
                (2, 10),
                pytest.raises(
                    ValueError, match="Setting the `axis` parameter is not allowed"
                ),
            ),
        ],
    )
    def test_compute_features_axis(self, kwargs, input1_shape, expectation, cls):
        with expectation:
            basis_obj = cls["conv"](
                n_basis_funcs=5,
                window_size=5,
                conv_kwargs=kwargs,
                **extra_decay_rates(cls["conv"], 5),
            )
            basis_obj.compute_features(np.ones(input1_shape))

    @pytest.mark.parametrize("n_basis_funcs", [4, 5])
    @pytest.mark.parametrize("time_scaling", [50, 70])
    @pytest.mark.parametrize("enforce_decay", [True, False])
    @pytest.mark.parametrize("window_size", [10, 15])
    @pytest.mark.parametrize("order", [3, 4])
    @pytest.mark.parametrize("width", [2, 3])
    @pytest.mark.parametrize(
        "input_shape, expected_n_input",
        [
            ((20,), 1),
            ((20, 1), 1),
            ((20, 2), 2),
            ((20, 1, 2), 2),
            ((20, 2, 1), 2),
            ((20, 2, 2), 4),
        ],
    )
    def test_compute_features_conv_input(
        self,
        n_basis_funcs,
        time_scaling,
        enforce_decay,
        window_size,
        input_shape,
        expected_n_input,
        order,
        width,
        cls,
        basis_class_specific_params,
    ):
        x = np.ones(input_shape)

        kwargs = dict(
            n_basis_funcs=n_basis_funcs,
            decay_rates=np.arange(1, n_basis_funcs + 1),
            time_scaling=time_scaling,
            window_size=window_size,
            enforce_decay_to_zero=enforce_decay,
            order=order,
            width=width,
        )

        # figure out which kwargs needs to be removed
        kwargs = inspect_utils.trim_kwargs(
            cls["conv"], kwargs, basis_class_specific_params
        )

        basis_obj = cls["conv"](**kwargs)
        out = basis_obj.compute_features(x)
        assert out.shape[1] == expected_n_input * basis_obj.n_basis_funcs

    @pytest.mark.parametrize(
        "eval_input", [0, [0], (0,), np.array([0]), jax.numpy.array([0])]
    )
    def test_compute_features_input(self, eval_input, cls):
        # orth exp needs more inputs (orthogonalizaiton impossible otherwise)
        if "OrthExp" in cls["eval"].__name__:
            return
        basis_obj = cls["eval"](n_basis_funcs=5)
        basis_obj.compute_features(eval_input)

    @pytest.mark.parametrize(
        "args, sample_size",
        [[{"n_basis_funcs": n_basis}, 100] for n_basis in [6, 10, 13]],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 30})]
    )
    def test_compute_features_returns_expected_number_of_basis(
        self, args, sample_size, mode, kwargs, cls
    ):
        basis_obj = cls[mode](
            **args, **kwargs, **extra_decay_rates(cls[mode], args["n_basis_funcs"])
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        assert eval_basis.shape[1] == args["n_basis_funcs"], (
            "Dimensions do not agree: The number of basis should match the first dimension "
            f"of the evaluated basis. The number of basis is {args['n_basis_funcs']}, but the "
            f"evaluated basis has dimension {eval_basis.shape[1]}"
        )

    @pytest.mark.parametrize(
        "samples, vmin, vmax, expectation",
        [
            (0.5, 0, 1, does_not_raise()),
            (
                -0.5,
                0,
                1,
                pytest.raises(ValueError, match="All the samples lie outside"),
            ),
            (np.linspace(-1, 1, 10), 0, 1, does_not_raise()),
            (
                np.linspace(-1, 0, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
            (
                np.linspace(1, 2, 10),
                0,
                1,
                pytest.warns(UserWarning, match="More than 90% of the samples"),
            ),
        ],
    )
    def test_compute_features_vmin_vmax(self, samples, vmin, vmax, expectation, cls):
        if "OrthExp" in cls["eval"].__name__ and not hasattr(samples, "shape"):
            return
        basis_obj = cls["eval"](
            5, bounds=(vmin, vmax), **extra_decay_rates(cls["eval"], 5)
        )
        with expectation:
            basis_obj.compute_features(samples)

    @pytest.mark.parametrize(
        "bounds, samples, exception",
        [
            (
                None,
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (0, 3),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (1, 4),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
            (
                (1, 3),
                np.arange(5),
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
        ],
    )
    def test_vmin_vmax_mode_conv(self, bounds, samples, exception, cls):
        with exception:
            cls["conv"](
                n_basis_funcs=5,
                window_size=10,
                bounds=bounds,
                **extra_decay_rates(cls["conv"], 5),
            )

    @pytest.mark.parametrize(
        "vmin, vmax, samples, nan_idx",
        [
            (None, None, np.arange(5), []),
            (0, 3, np.arange(5), [4]),
            (1, 4, np.arange(5), [0]),
            (1, 3, np.arange(5), [0, 4]),
        ],
    )
    def test_vmin_vmax_range(self, vmin, vmax, samples, nan_idx, cls):
        bounds = None if vmin is None else (vmin, vmax)
        bas = cls["eval"](
            n_basis_funcs=5, bounds=bounds, **extra_decay_rates(cls["eval"], 5)
        )
        out = bas.compute_features(samples)
        assert np.all(np.isnan(out[nan_idx]))
        valid_idx = list(set(samples).difference(nan_idx))
        assert np.all(~np.isnan(out[valid_idx]))

    @pytest.mark.parametrize(
        "bounds, expectation",
        [
            (None, does_not_raise()),
            ((None, 3), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, None), pytest.raises(TypeError, match=r"Could not convert")),
            ((1, 3), does_not_raise()),
            (("a", 3), pytest.raises(TypeError, match="Could not convert")),
            ((1, "a"), pytest.raises(TypeError, match="Could not convert")),
            (("a", "a"), pytest.raises(TypeError, match="Could not convert")),
            (
                (2, 1),
                pytest.raises(
                    ValueError, match=r"Invalid bound \(2, 1\). Lower bound is greater"
                ),
            ),
        ],
    )
    def test_vmin_vmax_setter(self, bounds, expectation, cls):
        bas = cls["eval"](
            n_basis_funcs=5, bounds=(1, 3), **extra_decay_rates(cls["eval"], 5)
        )
        with expectation:
            bas.set_params(bounds=bounds)
            assert bounds == bas.bounds if bounds else bas.bounds is None

    def test_conv_kwargs_error(self, cls):
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'test'"
        ):
            cls["eval"](n_basis_funcs=5, test="hi", **extra_decay_rates(cls["eval"], 5))

    def test_convolution_is_performed(self, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=10, **extra_decay_rates(cls["conv"], 5)
        )
        x = np.random.normal(size=100)
        conv = bas.compute_features(x)
        conv_2 = convolve.create_convolutional_predictor(bas.kernel_, x)
        valid = ~np.isnan(conv)
        assert np.all(conv[valid] == conv_2[valid])
        assert np.all(np.isnan(conv_2[~valid]))

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_evaluate_on_grid_basis_size(self, sample_size, mode, kwargs, cls):
        if "OrthExp" in cls["eval"].__name__:
            return
        basis_obj = cls[mode](
            n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5)
        )
        if sample_size <= 0:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            _, eval_basis = basis_obj.evaluate_on_grid(sample_size)
            assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_evaluate_on_grid_input_number(self, n_input, mode, kwargs, cls):
        basis_obj = cls[mode](
            n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5)
        )
        inputs = [10] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) missing 1 required positional argument",
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError,
                match=r"evaluate_on_grid\(\) takes [0-9] positional arguments but [0-9] were given",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [-1, 0, 1, 10, 11, 100])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_evaluate_on_grid_meshgrid_size(self, sample_size, mode, kwargs, cls):
        if "OrthExp" in cls["eval"].__name__:
            return
        basis_obj = cls[mode](
            n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5)
        )
        if sample_size <= 0:
            with pytest.raises(
                ValueError, match=r"All sample counts provided must be greater"
            ):
                basis_obj.evaluate_on_grid(sample_size)
        else:
            grid, _ = basis_obj.evaluate_on_grid(sample_size)
            assert grid.shape[0] == sample_size

    def test_fit_kernel(self, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=30, **extra_decay_rates(cls["conv"], 5)
        )
        bas.set_kernel()
        assert bas.kernel_ is not None

    def test_fit_kernel_shape(self, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=30, **extra_decay_rates(cls["conv"], 5)
        )
        bas.set_kernel()
        assert bas.kernel_.shape == (30, 5)

    @pytest.mark.parametrize(
        "mode, ws, expectation",
        [
            ("conv", 5, does_not_raise()),
            (
                "conv",
                -1,
                pytest.raises(
                    ValueError, match="`window_size` must be a positive integer"
                ),
            ),
            (
                "conv",
                None,
                pytest.raises(
                    ValueError,
                    match="You must provide a window_size",
                ),
            ),
            (
                "conv",
                1.5,
                pytest.raises(ValueError, match="`window_size` must be a positive "),
            ),
            (
                "eval",
                None,
                pytest.raises(
                    TypeError,
                    match=r"got an unexpected keyword argument 'window_size'",
                ),
            ),
            (
                "eval",
                10,
                pytest.raises(
                    TypeError,
                    match=r"got an unexpected keyword argument 'window_size'",
                ),
            ),
        ],
    )
    def test_init_window_size(self, mode, ws, expectation, cls):
        with expectation:
            cls[mode](
                n_basis_funcs=5, window_size=ws, **extra_decay_rates(cls[mode], 5)
            )

    @pytest.mark.parametrize("samples", [[], [0] * 10, [0] * 11])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_non_empty_samples(self, samples, mode, kwargs, cls):
        if "OrthExp" in cls["eval"].__name__:
            return
        if mode == "conv" and len(samples) == 1:
            return
        if len(samples) == 0:
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                cls[mode](
                    5, **kwargs, **extra_decay_rates(cls[mode], 5)
                ).compute_features(samples)
        else:
            cls[mode](5, **kwargs, **extra_decay_rates(cls[mode], 5)).compute_features(
                samples
            )

    @pytest.mark.parametrize("n_input", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 6})]
    )
    def test_number_of_required_inputs_compute_features(
        self, n_input, mode, kwargs, cls
    ):
        basis_obj = cls[mode](
            n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5)
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input == 0:
            expectation = pytest.raises(
                TypeError, match="missing 1 required positional argument"
            )
        elif n_input != basis_obj._n_input_dimensionality:
            expectation = pytest.raises(
                TypeError, match=r"takes 2 positional arguments but \d were given"
            )
        else:
            expectation = does_not_raise()

        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 8})]
    )
    def test_pynapple_support(self, mode, kwargs, cls):
        bas = cls[mode](n_basis_funcs=5, **kwargs, **extra_decay_rates(cls[mode], 5))
        x = np.linspace(0, 1, 10)
        x_nap = nap.Tsd(t=np.arange(10), d=x)
        y = bas._evaluate(x)
        y_nap = bas._evaluate(x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap.t)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis", [5])
    def test_pynapple_support_compute_features(self, n_basis, sample_size, cls):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        out = cls["eval"](
            n_basis_funcs=n_basis, **extra_decay_rates(cls["eval"], n_basis)
        ).compute_features(inp)
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_funcs", [5, 10, 80])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 90})]
    )
    def test_sample_size_of_compute_features_matches_that_of_input(
        self, n_basis_funcs, sample_size, mode, kwargs, cls
    ):
        basis_obj = cls[mode](
            n_basis_funcs=n_basis_funcs,
            **kwargs,
            **extra_decay_rates(cls[mode], n_basis_funcs),
        )
        eval_basis = basis_obj.compute_features(np.linspace(0, 1, sample_size))
        assert eval_basis.shape[0] == sample_size, (
            f"Dimensions do not agree: The sample size of the output should match the input sample size. "
            f"Expected {sample_size}, but got {eval_basis.shape[0]}."
        )

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("eval", does_not_raise()),
            (
                "conv",
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'bounds'"
                ),
            ),
        ],
    )
    def test_set_bounds(self, mode, expectation, cls):
        kwargs = {"bounds": (1, 2)}
        with expectation:
            cls[mode](n_basis_funcs=10, **kwargs, **extra_decay_rates(cls[mode], 10))

        if mode == "conv":
            bas = cls["conv"](
                n_basis_funcs=10, window_size=20, **extra_decay_rates(cls[mode], 10)
            )
            with pytest.raises(
                ValueError, match="Invalid parameter 'bounds' for estimator"
            ):
                bas.set_params(bounds=(1, 2))

    @pytest.mark.parametrize(
        "enforce_decay_to_zero, time_scaling, width, window_size, n_basis_funcs, bounds, mode, decay_rates",
        [
            (False, 15, 4, None, 10, (1, 2), "eval", np.arange(1, 11)),
            (False, 15, 4, 10, 10, None, "conv", np.arange(1, 11)),
        ],
    )
    @pytest.mark.parametrize(
        "order, conv_kwargs",
        [
            (10, dict(shift=True)),
        ],
    )
    def test_set_params(
        self,
        enforce_decay_to_zero,
        time_scaling,
        width,
        window_size,
        n_basis_funcs,
        bounds,
        mode: Literal["eval", "conv"],
        order,
        decay_rates,
        conv_kwargs,
        cls,
        basis_class_specific_params,
    ):
        """Test the read-only and read/write property of the parameters."""
        pars = dict(
            enforce_decay_to_zero=enforce_decay_to_zero,
            time_scaling=time_scaling,
            width=width,
            window_size=window_size,
            n_basis_funcs=n_basis_funcs,
            bounds=bounds,
            order=order,
            decay_rates=decay_rates,
            conv_kwargs=conv_kwargs,
        )
        pars = {
            key: value
            for key, value in pars.items()
            if key in basis_class_specific_params[cls[mode].__name__]
        }

        keys = list(pars.keys())
        bas = cls[mode](**pars)
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                par_set = {keys[i]: pars[keys[i]], keys[j]: pars[keys[j]]}
                bas = bas.set_params(**par_set)
                assert isinstance(bas, cls[mode])

    @pytest.mark.parametrize(
        "mode, expectation",
        [
            ("conv", does_not_raise()),
            (
                "eval",
                pytest.raises(
                    TypeError, match="got an unexpected keyword argument 'window_size'"
                ),
            ),
        ],
    )
    def test_set_window_size(self, mode, expectation, cls):
        kwargs = {"window_size": 10}
        with expectation:
            cls[mode](n_basis_funcs=10, **kwargs, **extra_decay_rates(cls[mode], 10))

        if mode == "conv":
            bas = cls["conv"](
                n_basis_funcs=10, window_size=10, **extra_decay_rates(cls["conv"], 10)
            )
            with pytest.raises(ValueError, match="You must provide a window_siz"):
                bas.set_params(window_size=None)

        if mode == "eval":
            bas = cls["eval"](n_basis_funcs=10, **extra_decay_rates(cls["eval"], 10))
            with pytest.raises(
                ValueError, match="Invalid parameter 'window_size' for estimator"
            ):
                bas.set_params(window_size=10)

    def test_transform_fails(self, cls):
        bas = cls["conv"](
            n_basis_funcs=5, window_size=5, **extra_decay_rates(cls["conv"], 5)
        )
        with pytest.raises(
            RuntimeError, match="You must call `setup_basis` before `_compute_features`"
        ):
            bas._compute_features(np.linspace(0, 1, 10))

    def test_transformer_get_params(self, cls):
        bas = cls["eval"](n_basis_funcs=5, **extra_decay_rates(cls["eval"], 5))
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
        bas_transformer = bas.to_transformer()
        params_transf = bas_transformer.get_params()
        params_transf.pop("_basis")
        params_basis = bas.get_params()
        rates_1 = params_basis.pop("decay_rates", 1)
        rates_2 = params_transf.pop("decay_rates", 1)
        assert params_transf == params_basis
        assert np.all(rates_1 == rates_2)

    @pytest.mark.parametrize(
        "x, inp_shape, expectation",
        [
            (np.ones((10,)), 1, does_not_raise()),
            (
                np.ones((10, 1)),
                1,
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (np.ones((10, 2)), 2, does_not_raise()),
            (
                np.ones((10, 1)),
                2,
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (
                np.ones((10, 2, 1)),
                2,
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (
                np.ones((10, 1, 2)),
                2,
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (np.ones((10, 1)), (1,), does_not_raise()),
            (np.ones((10,)), tuple(), does_not_raise()),
            (np.ones((10,)), np.zeros((12,)), does_not_raise()),
            (
                np.ones((10,)),
                (1,),
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (
                np.ones((10, 1)),
                (),
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (
                np.ones((10, 1)),
                np.zeros((12,)),
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
            (
                np.ones((10)),
                np.zeros((12, 1)),
                pytest.raises(ValueError, match="Input shape mismatch detected"),
            ),
        ],
    )
    def test_input_shape_validity(self, x, inp_shape, expectation, cls):
        bas = cls["eval"](n_basis_funcs=5, **extra_decay_rates(cls["eval"], 5))
        bas.set_input_shape(inp_shape)
        with expectation:
            bas.compute_features(x)

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            ((1, 1), does_not_raise()),
            (
                (1, 1.0),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (np.ones((1,)), does_not_raise()),
            (np.ones((1, 1)), does_not_raise()),
        ],
    )
    def test_set_input_value_types(self, inp_shape, expectation, cls):
        bas = cls["eval"](n_basis_funcs=5, **extra_decay_rates(cls["eval"], 5))
        with expectation:
            bas.set_input_shape(inp_shape)

    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 6})]
    )
    def test_iterate_over_component(self, mode, kwargs, cls):
        basis_obj = cls[mode](
            n_basis_funcs=5,
            **kwargs,
            **extra_decay_rates(cls[mode], 5),
        )

        out = tuple(basis_obj._iterate_over_components())
        assert len(out) == 1
        assert id(out[0]) == id(basis_obj)


class TestRaisedCosineLogBasis(BasisFuncsTesting):
    cls = {"eval": basis.RaisedCosineLogEval, "conv": basis.RaisedCosineLogConv}

    @pytest.mark.parametrize("width", [1.5, 2, 2.5])
    def test_decay_to_zero_basis_number_match(self, width):
        n_basis_funcs = 10
        _, ev = self.cls["conv"](
            n_basis_funcs=n_basis_funcs,
            width=width,
            enforce_decay_to_zero=True,
            window_size=5,
        ).evaluate_on_grid(2)
        assert ev.shape[1] == n_basis_funcs, (
            "Basis function number mismatch. "
            f"Expected {n_basis_funcs}, got {ev.shape[1]} instead!"
        )

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        if n_basis_funcs < 2:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 2 basis elements.",
            ):
                self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)
        else:
            self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (
                10.3,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (
                -10,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (None, pytest.raises(TypeError, match="'<=' not supported between")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_set_width(self, width, expectation, mode, kwargs):
        basis_obj = self.cls[mode](n_basis_funcs=5, **kwargs)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

    def test_time_scaling_property(self):
        time_scaling = [0.1, 10, 100]
        n_basis_funcs = 5
        _, lin_ev = basis.RaisedCosineLinearEval(n_basis_funcs).evaluate_on_grid(100)
        corr = np.zeros(len(time_scaling))
        for idx, ts in enumerate(time_scaling):
            basis_log = self.cls["eval"](
                n_basis_funcs=n_basis_funcs,
                time_scaling=ts,
                enforce_decay_to_zero=False,
            )
            _, log_ev = basis_log.evaluate_on_grid(100)
            corr[idx] = (lin_ev.flatten() @ log_ev.flatten()) / (
                np.linalg.norm(lin_ev.flatten()) * np.linalg.norm(log_ev.flatten())
            )
        assert np.all(
            np.diff(corr) < 0
        ), "As time scales increases, deviation from linearity should increase!"

    @pytest.mark.parametrize(
        "time_scaling, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="Only strictly positive time_scaling are allowed"
                ),
            ),
            (
                0,
                pytest.raises(
                    ValueError, match="Only strictly positive time_scaling are allowed"
                ),
            ),
            (0.1, does_not_raise()),
            (10, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_time_scaling_values(self, time_scaling, expectation, mode, kwargs):
        with expectation:
            self.cls[mode](n_basis_funcs=5, time_scaling=time_scaling, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (-1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0.5, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1.5, does_not_raise()),
            (2, does_not_raise()),
            (2.1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_width_values(self, width, expectation, mode, kwargs):
        with expectation:
            self.cls[mode](n_basis_funcs=5, width=width, **kwargs)


class TestRaisedCosineLinearBasis(BasisFuncsTesting):
    cls = {"eval": basis.RaisedCosineLinearEval, "conv": basis.RaisedCosineLinearConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        if n_basis_funcs < 2:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 2 basis elements.",
            ):
                self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)
        else:
            self.cls[mode](n_basis_funcs=n_basis_funcs, **kwargs)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (10, does_not_raise()),
            (10.5, does_not_raise()),
            (
                0.5,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
            (
                -10,
                pytest.raises(
                    ValueError,
                    match=r"Invalid raised cosine width\. 2\*width must be a positive",
                ),
            ),
        ],
    )
    def test_set_width(self, width, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=5)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)

    @pytest.mark.parametrize(
        "width, expectation",
        [
            (-1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (0.5, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
            (1.5, does_not_raise()),
            (2, does_not_raise()),
            (2.1, pytest.raises(ValueError, match="Invalid raised cosine width. ")),
        ],
    )
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_width_values(self, width, expectation, mode, kwargs):
        """
        Test allowable widths: integer multiple of 1/2, greater than 1.
        This test validates the behavior of both `eval` and `conv` modes.
        """
        basis_obj = self.cls[mode](n_basis_funcs=5, **kwargs)
        with expectation:
            basis_obj.width = width
        with expectation:
            basis_obj.set_params(width=width)


class TestMSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.MSplineEval, "conv": basis.MSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1)
        and order < #basis are enforced.
        """
        raise_exception = (order < 1) | (n_basis_funcs < 1) | (order > n_basis_funcs)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Spline order must be positive!|"
                rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs, order=order, **kwargs
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))

            # test the setter valuerror
            if (order > 1) & (n_basis_funcs > 1):
                basis_obj = self.cls[mode](n_basis_funcs=20, order=order, **kwargs)
                with pytest.raises(
                    ValueError,
                    match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
                ):
                    basis_obj.n_basis_funcs = n_basis_funcs
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs, order=order, **kwargs
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the order must be positive and less than or equal to the number of basis functions.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(self, sample_range):
        """
        Verifies that the compute_features() method can handle input range.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))

    @pytest.mark.parametrize(
        "bounds, samples, nan_idx, scaling",
        [
            (None, np.arange(5), [4], 1),
            ((1, 4), np.arange(5), [0], 3),
            ((1, 3), np.arange(5), [0, 4], 2),
        ],
    )
    def test_vmin_vmax_eval_on_grid_scaling_effect_on_eval(
        self, bounds, samples, nan_idx, scaling
    ):
        """
        Check that the MSpline has the expected scaling property.

        The MSpline must integrate to one. If the support is reduced, the height of the spline increases.
        """
        bas_no_range = self.cls["eval"](5, bounds=None)
        bas = self.cls["eval"](5, bounds=bounds)
        _, out1 = bas.evaluate_on_grid(10)
        _, out2 = bas_no_range.evaluate_on_grid(10)
        assert np.allclose(out1 * scaling, out2)


class TestOrthExponentialBasis(BasisFuncsTesting):
    cls = {"eval": basis.OrthExponentialEval, "conv": basis.OrthExponentialConv}

    @pytest.mark.parametrize(
        "window_size, n_basis, expectation",
        [
            (
                4,
                5,
                pytest.raises(
                    ValueError,
                    match="OrthExponentialConv basis requires at least a window_size",
                ),
            ),
            (5, 5, does_not_raise()),
        ],
    )
    def test_window_size_at_init(self, window_size, n_basis, expectation):
        decay_rates = np.asarray(np.arange(1, n_basis + 1), dtype=float)
        with expectation:
            self.cls["conv"](n_basis, decay_rates=decay_rates, window_size=window_size)

    def test_check_window_size_after_init(self):
        decay_rates = np.asarray(np.arange(1, 5 + 1), dtype=float)
        expectation = pytest.raises(
            ValueError,
            match="OrthExponentialConv basis requires at least a window_size",
        )
        bas = self.cls["conv"](5, decay_rates=decay_rates, window_size=10)
        with expectation:
            bas.window_size = 4

    @pytest.mark.parametrize(
        "window_size, n_basis, expectation",
        [
            (
                4,
                5,
                pytest.raises(
                    ValueError,
                    match="OrthExponentialConv basis requires at least a window_size",
                ),
            ),
            (5, 5, does_not_raise()),
        ],
    )
    def test_window_size_at_init(self, window_size, n_basis, expectation):
        decay_rates = np.asarray(np.arange(1, n_basis + 1), dtype=float)
        obj = self.cls["conv"](
            n_basis, decay_rates=decay_rates, window_size=n_basis + 1
        )
        with expectation:
            obj.window_size = window_size

        with expectation:
            obj.set_params(window_size=window_size)

    @pytest.mark.parametrize(
        "decay_rates", [[1, 2, 3], [0.01, 0.02, 0.001], [2, 1, 1, 2.4]]
    )
    def test_decay_rate_repetition(self, decay_rates):
        """
        Tests whether the class instance correctly processes the decay rates without repetition.
        A repeated rate causes linear algebra issues, and should raise a ValueError exception.
        """
        decay_rates = np.asarray(decay_rates, dtype=float)
        raise_exception = len(set(decay_rates)) != len(decay_rates)
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Two or more rates are repeated! Repeating rates will",
            ):
                self.cls["eval"](
                    n_basis_funcs=len(decay_rates), decay_rates=decay_rates
                )
        else:
            self.cls["eval"](n_basis_funcs=len(decay_rates), decay_rates=decay_rates)

    @pytest.mark.parametrize(
        "decay_rates", [[], [1], [1, 2, 3], [1, 0.01, 0.02, 0.001]]
    )
    @pytest.mark.parametrize("n_basis_funcs", [1, 2, 3, 4])
    def test_decay_rate_size_match_n_basis_funcs(self, decay_rates, n_basis_funcs):
        """
        Tests whether the size of decay rates matches the number of basis functions.
        """
        raise_exception = len(decay_rates) != n_basis_funcs
        decay_rates = np.asarray(decay_rates, dtype=float)
        if raise_exception:
            with pytest.raises(
                ValueError, match="The number of basis functions must match the"
            ):
                self.cls["eval"](n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)
        else:
            self.cls["eval"](n_basis_funcs=n_basis_funcs, decay_rates=decay_rates)

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 30})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, mode, kwargs
    ):
        """
        Tests whether the class instance has a minimum number of basis functions.
        """
        raise_exception = n_basis_funcs < 1
        decay_rates = np.arange(1, 1 + n_basis_funcs) if n_basis_funcs > 0 else []
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=f"Object class {self.cls[mode].__name__} requires >= 1 basis elements.",
            ):
                self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    decay_rates=decay_rates,
                    **kwargs,
                )
        else:
            self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                decay_rates=decay_rates,
                **kwargs,
            )


class TestBSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.BSplineEval, "conv": basis.BSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    **kwargs,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                order=order,
                **kwargs,
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 1, 2])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1) and
        order < #basis are enforced.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(
        self, sample_range: tuple
    ):
        """
        Verifies that the compute_features() method can handle input range.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))


class TestCyclicBSplineBasis(BasisFuncsTesting):
    cls = {"eval": basis.CyclicBSplineEval, "conv": basis.CyclicBSplineConv}

    @pytest.mark.parametrize("n_basis_funcs", [-1, 0, 1, 3, 10, 20])
    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    @pytest.mark.parametrize(
        "mode, kwargs", [("eval", {}), ("conv", {"window_size": 5})]
    )
    def test_minimum_number_of_basis_required_is_matched(
        self, n_basis_funcs, order, mode, kwargs
    ):
        """
        Verifies that the minimum number of basis functions and order required (i.e., at least 1)
        and order < #basis are enforced.
        """
        raise_exception = order > n_basis_funcs
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=rf"{self.cls[mode].__name__} `order` parameter cannot be larger than",
            ):
                basis_obj = self.cls[mode](
                    n_basis_funcs=n_basis_funcs,
                    order=order,
                    **kwargs,
                )
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls[mode](
                n_basis_funcs=n_basis_funcs,
                order=order,
                **kwargs,
            )
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_1_invalid(self, n_basis_funcs, order):
        """
        Verifies that order >= 2 is required for cyclic B-splines.
        """
        raise_exception = order == 1
        if raise_exception:
            with pytest.raises(
                ValueError, match=r"Order >= 2 required for cyclic B-spline"
            ):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [10])
    @pytest.mark.parametrize("order", [-1, 0, 2, 3])
    def test_order_is_positive(self, n_basis_funcs, order):
        """
        Verifies that the order is positive and < #basis.
        """
        raise_exception = order < 1
        if raise_exception:
            with pytest.raises(ValueError, match=r"Spline order must be positive!"):
                basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
                basis_obj.compute_features(np.linspace(0, 1, 10))
        else:
            basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=order)
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize("n_basis_funcs", [5])
    @pytest.mark.parametrize(
        "order, expectation",
        [
            (1.5, pytest.raises(ValueError, match=r"Spline order must be an integer")),
            (-1, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (0, pytest.raises(ValueError, match=r"Spline order must be positive")),
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                10,
                pytest.raises(
                    ValueError,
                    match=r"[a-z]+|[A-Z]+ `order` parameter cannot be larger",
                ),
            ),
        ],
    )
    def test_order_setter(self, n_basis_funcs, order, expectation):
        """
        Verifies that setting `order` validates the value correctly.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=n_basis_funcs, order=4)
        with expectation:
            basis_obj.order = order
            basis_obj.compute_features(np.linspace(0, 1, 10))

    @pytest.mark.parametrize(
        "sample_range", [(0, 1), (0.1, 0.9), (-0.5, 1), (0, 1.5), (-0.5, 1.5)]
    )
    def test_samples_range_matches_compute_features_requirements(
        self, sample_range: tuple
    ):
        """
        Verifies that the compute_features() method can handle input ranges.
        """
        basis_obj = self.cls["eval"](n_basis_funcs=5, order=3)
        basis_obj.compute_features(np.linspace(*sample_range, 100))


class TestAdditiveBasis(CombinedBasis):
    cls = {"eval": AdditiveBasis, "conv": AdditiveBasis}

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_iterate_over_component(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            6, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a_obj + basis_b_obj
        out = tuple(add._iterate_over_components())
        assert len(out) == add._n_input_dimensionality

        def get_ids(bas):

            if hasattr(bas, "basis1"):
                ids = get_ids(bas.basis1)
                ids += get_ids(bas.basis2)
            else:
                ids = [id(bas)]
            return ids

        id_list = get_ids(add)

        assert tuple(id(o) for o in out) == tuple(id_list)

    @pytest.mark.parametrize("samples", [[[0], []], [[], [0]], [[0, 0], [0, 0]]])
    @pytest.mark.parametrize("base_cls", [basis.BSplineEval, basis.BSplineConv])
    def test_non_empty_samples(self, base_cls, samples, basis_class_specific_params):
        kwargs = {"window_size": 2, "n_basis_funcs": 5}
        kwargs = inspect_utils.trim_kwargs(
            base_cls, kwargs, basis_class_specific_params
        )
        basis_obj = base_cls(**kwargs) + base_cls(**kwargs)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.compute_features(*samples)
        else:
            basis_obj.compute_features(*samples)

    @pytest.mark.parametrize(
        "eval_input",
        [
            [0, 0],
            [[0], [0]],
            [(0,), (0,)],
            [np.array([0]), [0]],
            [jax.numpy.array([0]), [0]],
        ],
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = basis.MSplineEval(5) + basis.MSplineEval(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [6])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("vmin, vmax", [(-1, 1)])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("inp_num", [1, 2])
    def test_sklearn_clone(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        vmin,
        vmax,
        inp_num,
        basis_class_specific_params,
    ):
        """Recursively check cloning."""
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a_obj = basis_a_obj.set_input_shape(
            *([inp_num] * basis_a_obj._n_input_dimensionality)
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=15
        )
        basis_b_obj = basis_b_obj.set_input_shape(
            *([inp_num] * basis_b_obj._n_input_dimensionality)
        )
        add = basis_a_obj + basis_b_obj

        def filter_attributes(obj, exclude_keys):
            return {
                key: val for key, val in obj.__dict__.items() if key not in exclude_keys
            }

        def compare(b1, b2):
            assert id(b1) != id(b2)
            assert b1.__class__.__name__ == b2.__class__.__name__
            if hasattr(b1, "basis1"):
                compare(b1.basis1, b2.basis1)
                compare(b1.basis2, b2.basis2)
                # add all params that are not parent or _basis1,_basis2
                d1 = filter_attributes(
                    b1, exclude_keys=["_basis1", "_basis2", "_parent"]
                )
                d2 = filter_attributes(
                    b2, exclude_keys=["_basis1", "_basis2", "_parent"]
                )
                assert d1 == d2
            else:
                decay_rates_b1 = b1.__dict__.get("_decay_rates", -1)
                decay_rates_b2 = b2.__dict__.get("_decay_rates", -1)
                assert np.array_equal(decay_rates_b1, decay_rates_b2)
                d1 = filter_attributes(b1, exclude_keys=["_decay_rates", "_parent"])
                d2 = filter_attributes(b2, exclude_keys=["_decay_rates", "_parent"])
                assert d1 == d2

        add2 = add.__sklearn_clone__()
        compare(add, add2)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("window_size", [10])
    def test_compute_features_returns_expected_number_of_basis(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the evaluation of the `AdditiveBasis` results in a number of basis
        that is the sum of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )

        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the output features."
                f"The number of basis is {n_basis_a + n_basis_b}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [100, 1000])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("window_size", [10])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the output sample size from `AdditiveBasis` compute_features function matches input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the "
                f"output features basis."
                f"The window size is {sample_size}",
                f"The second dimension of the output features basis is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("window_size", [10])
    def test_number_of_required_inputs_compute_features(
        self,
        n_input,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self,
        n_input,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj + basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_pynapple_support_compute_features(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_class_specific_params,
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_add = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        ) + self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        # compute_features the basis over pynapple Tsd objects
        out = basis_add.compute_features(*([inp] * basis_add._n_input_dimensionality))
        # check type
        assert isinstance(out, nap.TsdFrame)
        # check value
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(" window_size", [8])
    def test_call_input_num(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        with expectation:
            basis_obj._evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        window_size,
        expectation,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        with expectation:
            basis_obj._evaluate(*([inp] * basis_obj._n_input_dimensionality))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        time_axis_shape,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj._evaluate(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        if (
            basis_a == basis.OrthExponentialBasis
            or basis_b == basis.OrthExponentialBasis
        ):
            return
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj + basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj._evaluate(*inp)[3]))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=9
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=9
        )
        bas_eva = basis_a_obj + basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=8
        )
        bas_con = basis_a_obj + basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con._evaluate(*x) == bas_eva._evaluate(*x))

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas._evaluate(*x)
        y_nap = bas._evaluate(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
        assert (
            bas._evaluate(*x).shape[1]
            == basis_a_obj.n_basis_funcs + basis_b_obj.n_basis_funcs
        )

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas._evaluate(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_range(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        mn,
        mx,
        expectation,
        window_size,
        basis_class_specific_params,
    ):
        if expectation == "check":
            if (
                basis_a == basis.OrthExponentialBasis
                or basis_b == basis.OrthExponentialBasis
            ):
                expectation = pytest.raises(
                    ValueError, match="OrthExponentialBasis requires positive samples"
                )
            else:
                expectation = does_not_raise()
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj + basis_b_obj
        with expectation:
            bas._evaluate(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        bas.setup_basis(*([np.ones(10)] * bas._n_input_dimensionality))

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "_basis1"):
                has_kern += check_kernel(basis_obj._basis1)
                has_kern += check_kernel(basis_obj._basis2)
            else:
                has_kern += [
                    basis_obj.kernel_ is not None if basis_obj.mode == "conv" else True
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj + basis_b_obj
        if "Eval" in basis_a.__name__ and "Eval" in basis_b.__name__:
            context = does_not_raise()
        else:
            context = pytest.raises(
                RuntimeError,
                match="You must call `setup_basis` before `_compute_features`",
            )
        with context:
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(11, window_size=10)
        bas_add = bas1 + bas2
        assert bas_add.n_output_features is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_output_features == (n_basis_input1 * 10 + n_basis_input2 * 11)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_add = bas1 + bas2
        assert bas_add.n_basis_input_ is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_basis_input_ == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (3, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (4, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas = bas1 + bas2
        x = np.random.randn(20, 2), np.random.randn(20, 3)
        bas.compute_features(*x)
        with expectation:
            bas.compute_features(np.random.randn(30, 2), np.random.randn(30, n_input))

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [1, (), np.ones(3)])
    @pytest.mark.parametrize("shape_b", [1, (), np.ones(3)])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_1d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, *add_shape_a)), np.ones((10, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            add.compute_features(*x)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [2, (2,), np.ones((3, 2))])
    @pytest.mark.parametrize("shape_b", [3, (3,), np.ones((3, 3))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_2d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, *add_shape_a)), np.ones((10, 3, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            add.compute_features(*x)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [(2, 2), np.ones((3, 2, 2))])
    @pytest.mark.parametrize("shape_b", [(3, 1), np.ones((3, 3, 1))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_nd_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, 2, *add_shape_a)), np.ones((10, 3, 1, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b

        add.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            add.compute_features(*x)

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            (((1, 1), (1, 1)), does_not_raise()),
            (
                ((1, 1.0), (1, 1)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (
                ((1, 1), (1, 1.0)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_set_input_value_types(
        self, inp_shape, expectation, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b
        with expectation:
            add.set_input_shape(*inp_shape)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_deep_copy_basis(self, basis_a, basis_b, basis_class_specific_params):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b
        # test pointing to different objects
        assert id(add.basis1) != id(basis_a)
        assert id(add.basis1) != id(basis_b)
        assert id(add.basis2) != id(basis_a)
        assert id(add.basis2) != id(basis_b)

        # test attributes are not related
        basis_a.n_basis_funcs = 10
        basis_b.n_basis_funcs = 10
        assert add.basis1.n_basis_funcs == 5
        assert add.basis2.n_basis_funcs == 5

        add.basis1.n_basis_funcs = 6
        add.basis2.n_basis_funcs = 6
        assert basis_a.n_basis_funcs == 10
        assert basis_b.n_basis_funcs == 10

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_compute_n_basis_runtime(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        add = basis_a + basis_b
        add.basis1.n_basis_funcs = 10
        assert add.n_basis_funcs == 15
        add.basis2.n_basis_funcs = 10
        assert add.n_basis_funcs == 20

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_runtime_n_basis_out_compute(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a.set_input_shape(
            *([1] * basis_a._n_input_dimensionality)
        ).to_transformer()
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        basis_b.set_input_shape(
            *([1] * basis_b._n_input_dimensionality)
        ).to_transformer()
        add = basis_a + basis_b
        inps_a = [2] * basis_a._n_input_dimensionality
        add.basis1.set_input_shape(*inps_a)
        if isinstance(basis_a, MultiplicativeBasis):
            new_out_num = np.prod(inps_a) * add.basis1.n_basis_funcs
        else:
            new_out_num = inps_a[0] * add.basis1.n_basis_funcs
        assert add.n_output_features == new_out_num + add.basis2.n_basis_funcs
        inps_b = [3] * basis_b._n_input_dimensionality
        if isinstance(basis_b, MultiplicativeBasis):
            new_out_num_b = np.prod(inps_b) * add.basis2.n_basis_funcs
        else:
            new_out_num_b = inps_b[0] * add.basis2.n_basis_funcs
        add.basis2.set_input_shape(*inps_b)
        assert add.n_output_features == new_out_num + new_out_num_b


class TestMultiplicativeBasis(CombinedBasis):
    cls = {"eval": MultiplicativeBasis, "conv": MultiplicativeBasis}

    @pytest.mark.parametrize(
        "samples", [[[0], []], [[], [0]], [[0], [0]], [[0, 0], [0, 0]]]
    )
    @pytest.mark.parametrize(" ws", [3])
    def test_non_empty_samples(self, samples, ws):
        basis_obj = basis.MSplineEval(5) * basis.RaisedCosineLinearEval(5)
        if any(tuple(len(s) == 0 for s in samples)):
            with pytest.raises(
                ValueError, match="All sample provided must be non empty"
            ):
                basis_obj.compute_features(*samples)
        else:
            basis_obj.compute_features(*samples)

    @pytest.mark.parametrize(
        "eval_input",
        [
            [0, 0],
            [[0], [0]],
            [(0,), (0,)],
            [np.array([0]), [0]],
            [jax.numpy.array([0]), [0]],
        ],
    )
    def test_compute_features_input(self, eval_input):
        """
        Checks that the sample size of the output from the compute_features() method matches the input sample size.
        """
        basis_obj = basis.MSplineEval(5) * basis.MSplineEval(5)
        basis_obj.compute_features(*eval_input)

    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("sample_size", [10, 1000])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("window_size", [10])
    def test_compute_features_returns_expected_number_of_basis(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the evaluation of the `MultiplicativeBasis` results in a number of basis
        that is the product of the number of basis functions from two individual bases.
        """
        # define the two basis
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )

        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[1] != basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs:
            raise ValueError(
                "Dimensions do not agree: The number of basis should match the first dimension of the "
                "fit_transformed basis."
                f"The number of basis is {n_basis_a * n_basis_b}",
                f"The first dimension of the output features is {eval_basis.shape[1]}",
            )

    @pytest.mark.parametrize("sample_size", [12, 30, 35])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("window_size", [10])
    def test_sample_size_of_compute_features_matches_that_of_input(
        self,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the output sample size from the `MultiplicativeBasis` fit_transform function
        matches the input sample size.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.compute_features(
            *[np.linspace(0, 1, sample_size)] * basis_obj._n_input_dimensionality
        )
        if eval_basis.shape[0] != sample_size:
            raise ValueError(
                f"Dimensions do not agree: The window size should match the second dimension of the output features."
                f"The window size is {sample_size}",
                f"The second dimension of the output features is {eval_basis.shape[0]}",
            )

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_input", [0, 1, 2, 3, 10, 30])
    @pytest.mark.parametrize("n_basis_a", [5, 6])
    @pytest.mark.parametrize("n_basis_b", [5, 6])
    @pytest.mark.parametrize("window_size", [10])
    def test_number_of_required_inputs_compute_features(
        self,
        n_input,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        """
        Test whether the number of required inputs for the `compute_features` function matches
        the sum of the number of input samples from the two bases.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        inputs = [np.linspace(0, 1, 20)] * n_input
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.compute_features(*inputs)

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_meshgrid_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the resulting meshgrid size matches the sample size input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        res = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )
        for grid in res[:-1]:
            assert grid.shape[0] == sample_size

    @pytest.mark.parametrize("sample_size", [11, 20])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_basis_size(
        self,
        sample_size,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number sample size output by evaluate_on_grid matches the sample size of the input.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        eval_basis = basis_obj.evaluate_on_grid(
            *[sample_size] * basis_obj._n_input_dimensionality
        )[-1]
        assert eval_basis.shape[0] == sample_size

    @pytest.mark.parametrize("n_input", [0, 1, 2, 5, 6, 11, 30])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    def test_evaluate_on_grid_input_number(
        self,
        n_input,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        basis_class_specific_params,
    ):
        """
        Test whether the number of inputs provided to `evaluate_on_grid` matches
        the sum of the number of input samples required from each of the basis objects.
        """
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        basis_obj = basis_a_obj * basis_b_obj
        inputs = [20] * n_input
        required_dim = (
            basis_a_obj._n_input_dimensionality + basis_b_obj._n_input_dimensionality
        )
        if n_input != required_dim:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch."
            )
        else:
            expectation = does_not_raise()
        with expectation:
            basis_obj.evaluate_on_grid(*inputs)

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [6])
    @pytest.mark.parametrize("sample_size_a", [11, 12])
    @pytest.mark.parametrize("sample_size_b", [11, 12])
    def test_inconsistent_sample_sizes(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size_a,
        sample_size_b,
        basis_class_specific_params,
    ):
        """Test that the inputs of inconsistent sample sizes result in an exception when compute_features is called"""
        raise_exception = sample_size_a != sample_size_b
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        input_a = [
            np.linspace(0, 1, sample_size_a)
        ] * basis_a_obj._n_input_dimensionality
        input_b = [
            np.linspace(0, 1, sample_size_b)
        ] * basis_b_obj._n_input_dimensionality
        basis_obj = basis_a_obj * basis_b_obj
        if raise_exception:
            with pytest.raises(
                ValueError,
                match=r"Sample size mismatch\. Input elements have inconsistent",
            ):
                basis_obj.compute_features(*input_a, *input_b)
        else:
            basis_obj.compute_features(*input_a, *input_b)

    @pytest.mark.parametrize("sample_size", [30])
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_pynapple_support_compute_features(
        self,
        basis_a,
        basis_b,
        n_basis_a,
        n_basis_b,
        sample_size,
        basis_class_specific_params,
    ):
        iset = nap.IntervalSet(start=[0, 0.5], end=[0.49999, 1])
        inp = nap.Tsd(
            t=np.linspace(0, 1, sample_size),
            d=np.linspace(0, 1, sample_size),
            time_support=iset,
        )
        basis_prod = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        ) * self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        out = basis_prod.compute_features(*([inp] * basis_prod._n_input_dimensionality))
        assert isinstance(out, nap.TsdFrame)
        assert np.all(out.time_support.values == inp.time_support.values)

    # TEST CALL
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    @pytest.mark.parametrize("num_input", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize(" window_size", [8])
    def test_call_input_num(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        num_input,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        if num_input == basis_obj._n_input_dimensionality:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                TypeError, match="Input dimensionality mismatch"
            )
        with expectation:
            basis_obj._evaluate(*([np.linspace(0, 1, 10)] * num_input))

    @pytest.mark.parametrize(
        "inp, expectation",
        [
            (np.linspace(0, 1, 10), does_not_raise()),
            (np.linspace(0, 1, 10)[:, None], pytest.raises(ValueError)),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_input_shape(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        inp,
        window_size,
        expectation,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        with expectation:
            basis_obj._evaluate(*([inp] * basis_obj._n_input_dimensionality))

    @pytest.mark.parametrize("time_axis_shape", [10, 11, 12])
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_axis(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        time_axis_shape,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, time_axis_shape)] * basis_obj._n_input_dimensionality
        assert basis_obj._evaluate(*inp).shape[0] == time_axis_shape

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_nan(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        if (
            basis_a == basis.OrthExponentialBasis
            or basis_b == basis.OrthExponentialBasis
        ):
            return
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        basis_obj = basis_a_obj * basis_b_obj
        inp = [np.linspace(0, 1, 10)] * basis_obj._n_input_dimensionality
        for x in inp:
            x[3] = np.nan
        assert all(np.isnan(basis_obj._evaluate(*inp)[3]))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_equivalent_in_conv(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas_eva = basis_a_obj * basis_b_obj

        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=8
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=8
        )
        bas_con = basis_a_obj * basis_b_obj

        x = [np.linspace(0, 1, 10)] * bas_con._n_input_dimensionality
        assert np.all(bas_con._evaluate(*x) == bas_eva._evaluate(*x))

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_pynapple_support(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = np.linspace(0, 1, 10)
        x_nap = [nap.Tsd(t=np.arange(10), d=x)] * bas._n_input_dimensionality
        x = [x] * bas._n_input_dimensionality
        y = bas._evaluate(*x)
        y_nap = bas._evaluate(*x_nap)
        assert isinstance(y_nap, nap.TsdFrame)
        assert np.all(y == y_nap.d)
        assert np.all(y_nap.t == x_nap[0].t)

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [6, 7])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_basis_number(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
        assert (
            bas._evaluate(*x).shape[1]
            == basis_a_obj.n_basis_funcs * basis_b_obj.n_basis_funcs
        )

    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_non_empty(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        window_size,
        basis_class_specific_params,
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with pytest.raises(ValueError, match="All sample provided must"):
            bas._evaluate(*([np.array([])] * bas._n_input_dimensionality))

    @pytest.mark.parametrize(
        "mn, mx, expectation",
        [
            (0, 1, does_not_raise()),
            (-2, 2, does_not_raise()),
            (0.1, 2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize(" window_size", [8])
    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_call_sample_range(
        self,
        n_basis_a,
        n_basis_b,
        basis_a,
        basis_b,
        mn,
        mx,
        expectation,
        window_size,
        basis_class_specific_params,
    ):
        if expectation == "check":
            if (
                basis_a == basis.OrthExponentialBasis
                or basis_b == basis.OrthExponentialBasis
            ):
                expectation = pytest.raises(
                    ValueError, match="OrthExponentialBasis requires positive samples"
                )
            else:
                expectation = does_not_raise()
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=window_size
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=window_size
        )
        bas = basis_a_obj * basis_b_obj
        with expectation:
            bas._evaluate(*([np.linspace(mn, mx, 10)] * bas._n_input_dimensionality))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_fit_kernel(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        bas._set_input_independent_states()

        def check_kernel(basis_obj):
            has_kern = []
            if hasattr(basis_obj, "_basis1"):
                has_kern += check_kernel(basis_obj._basis1)
                has_kern += check_kernel(basis_obj._basis2)
            else:
                has_kern += [
                    basis_obj.kernel_ is not None if basis_obj.mode == "conv" else True
                ]
            return has_kern

        assert all(check_kernel(bas))

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    @pytest.mark.parametrize("n_basis_a", [5])
    @pytest.mark.parametrize("n_basis_b", [5])
    def test_transform_fails(
        self, n_basis_a, n_basis_b, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a_obj = self.instantiate_basis(
            n_basis_a, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b_obj = self.instantiate_basis(
            n_basis_b, basis_b, basis_class_specific_params, window_size=10
        )
        bas = basis_a_obj * basis_b_obj
        if "Eval" in basis_a.__name__ and "Eval" in basis_b.__name__:
            context = does_not_raise()
        else:
            context = pytest.raises(
                RuntimeError,
                match="You must call `setup_basis` before `_compute_features`",
            )
        with context:
            x = [np.linspace(0, 1, 10)] * bas._n_input_dimensionality
            bas._compute_features(*x)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_output_features(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(11, window_size=10)
        bas_add = bas1 * bas2
        assert bas_add.n_output_features is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_output_features == (n_basis_input1 * 10 * n_basis_input2 * 11)

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_set_num_basis_input(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_add = bas1 * bas2
        assert bas_add.n_basis_input_ is None
        bas_add.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_add.n_basis_input_ == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "n_input, expectation",
        [
            (3, does_not_raise()),
            (0, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (1, pytest.raises(ValueError, match="Input shape mismatch detected")),
            (4, pytest.raises(ValueError, match="Input shape mismatch detected")),
        ],
    )
    def test_expected_input_number(self, n_input, expectation):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas = bas1 * bas2
        x = np.random.randn(20, 2), np.random.randn(20, 3)
        bas.compute_features(*x)
        with expectation:
            bas.compute_features(np.random.randn(30, 2), np.random.randn(30, n_input))

    @pytest.mark.parametrize("n_basis_input1", [1, 2, 3])
    @pytest.mark.parametrize("n_basis_input2", [1, 2, 3])
    def test_n_basis_input_(self, n_basis_input1, n_basis_input2):
        bas1 = basis.RaisedCosineLinearConv(10, window_size=10)
        bas2 = basis.BSplineConv(10, window_size=10)
        bas_prod = bas1 * bas2
        bas_prod.compute_features(
            np.ones((20, n_basis_input1)), np.ones((20, n_basis_input2))
        )
        assert bas_prod.n_basis_input_ == (n_basis_input1, n_basis_input2)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [1, (), np.ones(3)])
    @pytest.mark.parametrize("shape_b", [1, (), np.ones(3)])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_1d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, *add_shape_a)), np.ones((10, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            mul.compute_features(*x)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [2, (2,), np.ones((3, 2))])
    @pytest.mark.parametrize("shape_b", [3, (3,), np.ones((3, 3))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_2d_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, *add_shape_a)), np.ones((10, 3, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            mul.compute_features(*x)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize("shape_a", [(2, 2), np.ones((3, 2, 2))])
    @pytest.mark.parametrize("shape_b", [(3, 1), np.ones((3, 3, 1))])
    @pytest.mark.parametrize("add_shape_a", [(), (1,)])
    @pytest.mark.parametrize("add_shape_b", [(), (1,)])
    def test_set_input_shape_type_nd_arrays(
        self,
        basis_a,
        basis_b,
        shape_a,
        shape_b,
        basis_class_specific_params,
        add_shape_a,
        add_shape_b,
    ):
        x = (np.ones((10, 2, 2, *add_shape_a)), np.ones((10, 3, 1, *add_shape_b)))
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b

        mul.set_input_shape(shape_a, shape_b)
        if add_shape_a == () and add_shape_b == ():
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match="Input shape mismatch detected"
            )
        with expectation:
            mul.compute_features(*x)

    @pytest.mark.parametrize(
        "inp_shape, expectation",
        [
            (((1, 1), (1, 1)), does_not_raise()),
            (
                ((1, 1.0), (1, 1)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
            (
                ((1, 1), (1, 1.0)),
                pytest.raises(
                    ValueError, match="The tuple provided contains non integer"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_set_input_value_types(
        self, inp_shape, expectation, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b
        with expectation:
            mul.set_input_shape(*inp_shape)

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_deep_copy_basis(self, basis_a, basis_b, basis_class_specific_params):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b
        # test pointing to different objects
        assert id(mul.basis1) != id(basis_a)
        assert id(mul.basis1) != id(basis_b)
        assert id(mul.basis2) != id(basis_a)
        assert id(mul.basis2) != id(basis_b)

        # test attributes are not related
        basis_a.n_basis_funcs = 10
        basis_b.n_basis_funcs = 10
        assert mul.basis1.n_basis_funcs == 5
        assert mul.basis2.n_basis_funcs == 5

        mul.basis1.n_basis_funcs = 6
        mul.basis2.n_basis_funcs = 6
        assert basis_a.n_basis_funcs == 10
        assert basis_b.n_basis_funcs == 10

    @pytest.mark.parametrize(
        "basis_a", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    @pytest.mark.parametrize(
        "basis_b", list_all_basis_classes("Eval") + list_all_basis_classes("Conv")
    )
    def test_compute_n_basis_runtime(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        mul = basis_a * basis_b
        mul.basis1.n_basis_funcs = 10
        assert mul.n_basis_funcs == 50
        mul.basis2.n_basis_funcs = 10
        assert mul.n_basis_funcs == 100

    @pytest.mark.parametrize("basis_a", list_all_basis_classes())
    @pytest.mark.parametrize("basis_b", list_all_basis_classes())
    def test_runtime_n_basis_out_compute(
        self, basis_a, basis_b, basis_class_specific_params
    ):
        basis_a = self.instantiate_basis(
            5, basis_a, basis_class_specific_params, window_size=10
        )
        basis_a.set_input_shape(
            *([1] * basis_a._n_input_dimensionality)
        ).to_transformer()
        basis_b = self.instantiate_basis(
            5, basis_b, basis_class_specific_params, window_size=10
        )
        basis_b.set_input_shape(
            *([1] * basis_b._n_input_dimensionality)
        ).to_transformer()
        mul = basis_a * basis_b
        inps_a = [2] * basis_a._n_input_dimensionality
        mul.basis1.set_input_shape(*inps_a)
        if isinstance(basis_a, MultiplicativeBasis):
            new_out_num = np.prod(inps_a) * mul.basis1.n_basis_funcs
        else:
            new_out_num = inps_a[0] * mul.basis1.n_basis_funcs
        assert mul.n_output_features == new_out_num * mul.basis2.n_basis_funcs
        inps_b = [3] * basis_b._n_input_dimensionality
        if isinstance(basis_b, MultiplicativeBasis):
            new_out_num_b = np.prod(inps_b) * mul.basis2.n_basis_funcs
        else:
            new_out_num_b = inps_b[0] * mul.basis2.n_basis_funcs
        mul.basis2.set_input_shape(*inps_b)
        assert mul.n_output_features == new_out_num * new_out_num_b


@pytest.mark.parametrize(
    "exponent", [-1, 0, 0.5, basis.RaisedCosineLogEval(4), 1, 2, 3]
)
@pytest.mark.parametrize("basis_class", list_all_basis_classes())
def test_power_of_basis(exponent, basis_class, basis_class_specific_params):
    """Test if the power behaves as expected."""
    raise_exception_type = not type(exponent) is int

    if not raise_exception_type:
        raise_exception_value = exponent <= 0
    else:
        raise_exception_value = False

    basis_obj = CombinedBasis.instantiate_basis(
        5, basis_class, basis_class_specific_params, window_size=10
    )

    if raise_exception_type:
        with pytest.raises(TypeError, match=r"Exponent should be an integer\!"):
            basis_obj**exponent
    elif raise_exception_value:
        with pytest.raises(
            ValueError, match=r"Exponent should be a non-negative integer\!"
        ):
            basis_obj**exponent
    else:
        basis_pow = basis_obj**exponent
        samples = np.linspace(0, 1, 10)
        eval_pow = basis_pow.compute_features(
            *[samples] * basis_pow._n_input_dimensionality
        )

        if exponent == 2:
            basis_obj = basis_obj * basis_obj
        elif exponent == 3:
            basis_obj = basis_obj * basis_obj * basis_obj

        non_nan = ~np.isnan(eval_pow)
        out = basis_obj.compute_features(*[samples] * basis_obj._n_input_dimensionality)
        assert np.allclose(
            eval_pow[non_nan],
            out[non_nan],
        )
        assert np.all(np.isnan(out[~non_nan]))


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_basis_to_transformer(basis_cls, basis_class_specific_params):
    n_basis_funcs = 5
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = bas.set_input_shape(
        *([1] * bas._n_input_dimensionality)
    ).to_transformer()

    assert isinstance(trans_bas, basis.TransformerBasis)

    # check that things like n_basis_funcs are the same as the original basis
    for k in bas.__dict__.keys():
        # skip for add and multiplicative.
        if basis_cls in [AdditiveBasis, MultiplicativeBasis]:
            continue
        assert np.all(getattr(bas, k) == getattr(trans_bas, k))


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(
            t=np.arange(100),
            d=np.arange(100),
            time_support=nap.IntervalSet(start=[0, 50], end=[20, 75]),
        )
    ],
)
@pytest.mark.parametrize(
    "window_size, shift, predictor_causality, nan_index",
    [
        (3, True, "causal", [0, 1, 2, 50, 51, 52]),
        (2, True, "causal", [0, 1, 50, 51]),
        (3, False, "causal", [0, 1, 50, 51]),
        (2, False, "causal", [0, 50]),
        (2, None, "causal", [0, 1, 50, 51]),
        (3, True, "anti-causal", [20, 19, 18, 75, 74, 73]),
        (2, True, "anti-causal", [20, 19, 75, 74]),
        (3, False, "anti-causal", [20, 19, 75, 74]),
        (2, False, "anti-causal", [20, 75]),
        (2, None, "anti-causal", [20, 19, 75, 74]),
        (3, False, "acausal", [0, 20, 50, 75]),
        (5, False, "acausal", [0, 1, 19, 20, 50, 51, 74, 75]),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv"),
)
def test_multi_epoch_pynapple_basis(
    basis_cls,
    tsd,
    window_size,
    shift,
    predictor_causality,
    nan_index,
    basis_class_specific_params,
):
    """Test nan location in multi-epoch pynapple tsd."""
    kwargs = dict(
        conv_kwargs=dict(shift=shift, predictor_causality=predictor_causality)
    )

    # require a ws of at least nbasis funcs.
    if "OrthExp" in basis_cls.__name__:
        nbasis = 2
    # splines requires at least 1 basis more than the order of the spline.
    else:
        nbasis = 5
    bas = CombinedBasis().instantiate_basis(
        nbasis,
        basis_cls,
        basis_class_specific_params,
        window_size=window_size,
        **kwargs,
    )

    n_input = bas._n_input_dimensionality

    res = bas.compute_features(*([tsd] * n_input))

    nan_index = np.sort(nan_index)
    times_nan_found = res[np.isnan(res.d[:, 0])].t
    assert len(times_nan_found) == len(nan_index)
    assert np.all(times_nan_found == np.array(nan_index))
    idx_nan = [np.where(res.t == k)[0][0] for k in nan_index]
    assert np.all(np.isnan(res.d[idx_nan]))


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(
            t=np.arange(100),
            d=np.arange(100),
            time_support=nap.IntervalSet(start=[0, 50], end=[20, 75]),
        )
    ],
)
@pytest.mark.parametrize(
    "window_size, shift, predictor_causality, nan_index",
    [
        (3, True, "causal", [0, 1, 2, 50, 51, 52]),
        (2, True, "causal", [0, 1, 50, 51]),
        (3, False, "causal", [0, 1, 50, 51]),
        (2, False, "causal", [0, 50]),
        (2, None, "causal", [0, 1, 50, 51]),
        (3, True, "anti-causal", [20, 19, 18, 75, 74, 73]),
        (2, True, "anti-causal", [20, 19, 75, 74]),
        (3, False, "anti-causal", [20, 19, 75, 74]),
        (2, False, "anti-causal", [20, 75]),
        (2, None, "anti-causal", [20, 19, 75, 74]),
        (3, False, "acausal", [0, 20, 50, 75]),
        (5, False, "acausal", [0, 1, 19, 20, 50, 51, 74, 75]),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv"),
)
def test_multi_epoch_pynapple_basis_transformer(
    basis_cls,
    tsd,
    window_size,
    shift,
    predictor_causality,
    nan_index,
    basis_class_specific_params,
):
    """Test nan location in multi-epoch pynapple tsd."""
    kwargs = dict(
        conv_kwargs=dict(shift=shift, predictor_causality=predictor_causality)
    )
    # require a ws of at least nbasis funcs.
    if "OrthExp" in basis_cls.__name__:
        nbasis = 2
    # splines requires at least 1 basis more than the order of the spline.
    else:
        nbasis = 5

    bas = CombinedBasis().instantiate_basis(
        nbasis,
        basis_cls,
        basis_class_specific_params,
        window_size=window_size,
        **kwargs,
    )

    n_input = bas._n_input_dimensionality

    # concat input
    X = pynapple_concatenate_numpy([tsd[:, None]] * n_input, axis=1)

    # run convolutions
    # pass through transformer
    bas.set_input_shape(X)
    bas = basis.TransformerBasis(bas)
    res = bas.fit_transform(X)

    # check nans
    nan_index = np.sort(nan_index)
    times_nan_found = res[np.isnan(res.d[:, 0])].t
    assert len(times_nan_found) == len(nan_index)
    assert np.all(times_nan_found == np.array(nan_index))
    idx_nan = [np.where(res.t == k)[0][0] for k in nan_index]
    assert np.all(np.isnan(res.d[idx_nan]))


@pytest.mark.parametrize(
    "bas1, bas2, bas3",
    list(itertools.product(*[list_all_basis_classes()] * 3)),
)
@pytest.mark.parametrize(
    "operator1, operator2, compute_slice",
    [
        (
            "__add__",
            "__add__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._n_basis_input_[0] * bas1.n_basis_funcs),
                "2": slice(
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input_[0] * bas2.n_basis_funcs,
                ),
                "3": slice(
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input_[0] * bas2.n_basis_funcs,
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input_[0] * bas2.n_basis_funcs
                    + bas3._n_basis_input_[0] * bas3.n_basis_funcs,
                ),
            },
        ),
        (
            "__add__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "1": slice(0, bas1._n_basis_input_[0] * bas1.n_basis_funcs),
                "(2 * 3)": slice(
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input_[0]
                    * bas2.n_basis_funcs
                    * bas3._n_basis_input_[0]
                    * bas3.n_basis_funcs,
                ),
            },
        ),
        (
            "__mul__",
            "__add__",
            lambda bas1, bas2, bas3: {
                # note that it doesn't respect algebra order but execute right to left (first add then multiplies)
                "(1 * (2 + 3))": slice(
                    0,
                    bas1._n_basis_input_[0]
                    * bas1.n_basis_funcs
                    * (
                        bas2._n_basis_input_[0] * bas2.n_basis_funcs
                        + bas3._n_basis_input_[0] * bas3.n_basis_funcs
                    ),
                ),
            },
        ),
        (
            "__mul__",
            "__mul__",
            lambda bas1, bas2, bas3: {
                "(1 * (2 * 3))": slice(
                    0,
                    bas1._n_basis_input_[0]
                    * bas1.n_basis_funcs
                    * bas2._n_basis_input_[0]
                    * bas2.n_basis_funcs
                    * bas3._n_basis_input_[0]
                    * bas3.n_basis_funcs,
                ),
            },
        ),
    ],
)
def test__get_splitter(
    bas1, bas2, bas3, operator1, operator2, compute_slice, basis_class_specific_params
):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return
    # define the basis
    n_basis = [5, 6, 7]
    n_input_basis = [1, 2, 3]

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        n_basis[0], bas1, basis_class_specific_params, window_size=10, label="1"
    )
    bas1_instance.set_input_shape(
        *([n_input_basis[0]] * bas1_instance._n_input_dimensionality)
    )
    bas2_instance = combine_basis.instantiate_basis(
        n_basis[1], bas2, basis_class_specific_params, window_size=10, label="2"
    )
    bas2_instance.set_input_shape(
        *([n_input_basis[1]] * bas2_instance._n_input_dimensionality)
    )
    bas3_instance = combine_basis.instantiate_basis(
        n_basis[2], bas3, basis_class_specific_params, window_size=10, label="3"
    )
    bas3_instance.set_input_shape(
        *([n_input_basis[2]] * bas3_instance._n_input_dimensionality)
    )

    func1 = getattr(bas1_instance, operator1)
    func2 = getattr(bas2_instance, operator2)
    bas23 = func2(bas3_instance)
    bas123 = func1(bas23)
    inps = [np.zeros((1, n)) if n > 1 else np.zeros((1,)) for n in n_input_basis]
    bas123.set_input_shape(*inps)
    splitter_dict, _ = bas123._get_feature_slicing(split_by_input=False)
    exp_slices = compute_slice(bas1_instance, bas2_instance, bas3_instance)
    assert exp_slices == splitter_dict


@pytest.mark.parametrize(
    "bas1, bas2",
    list(itertools.product(*[list_all_basis_classes()] * 2)),
)
@pytest.mark.parametrize(
    "operator, n_input_basis_1, n_input_basis_2, compute_slice",
    [
        (
            "__add__",
            1,
            1,
            lambda bas1, bas2: {
                "1": slice(0, bas1._n_basis_input_[0] * bas1.n_basis_funcs),
                "2": slice(
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs,
                    bas1._n_basis_input_[0] * bas1.n_basis_funcs
                    + bas2._n_basis_input_[0] * bas2.n_basis_funcs,
                ),
            },
        ),
        (
            "__mul__",
            1,
            1,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0,
                    bas1._n_basis_input_[0]
                    * bas1.n_basis_funcs
                    * bas2._n_basis_input_[0]
                    * bas2.n_basis_funcs,
                )
            },
        ),
        (
            "__add__",
            2,
            1,
            lambda bas1, bas2: {
                "1": {
                    "0": slice(0, bas1.n_basis_funcs),
                    "1": slice(bas1.n_basis_funcs, 2 * bas1.n_basis_funcs),
                },
                "2": slice(
                    2 * bas1.n_basis_funcs, 2 * bas1.n_basis_funcs + bas2.n_basis_funcs
                ),
            },
        ),
        (
            "__mul__",
            2,
            1,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0, bas1._n_basis_input_[0] * bas1.n_basis_funcs * bas2.n_basis_funcs
                )
            },
        ),
        (
            "__add__",
            1,
            2,
            lambda bas1, bas2: {
                "1": slice(0, bas1.n_basis_funcs),
                "2": {
                    "0": slice(
                        bas1.n_basis_funcs, bas1.n_basis_funcs + bas2.n_basis_funcs
                    ),
                    "1": slice(
                        bas1.n_basis_funcs + bas2.n_basis_funcs,
                        bas1.n_basis_funcs + 2 * bas2.n_basis_funcs,
                    ),
                },
            },
        ),
        (
            "__mul__",
            1,
            2,
            lambda bas1, bas2: {
                "(1 * 2)": slice(
                    0, bas2._n_basis_input_[0] * bas1.n_basis_funcs * bas2.n_basis_funcs
                )
            },
        ),
        (
            "__add__",
            2,
            2,
            lambda bas1, bas2: {
                "1": {
                    "0": slice(0, bas1.n_basis_funcs),
                    "1": slice(bas1.n_basis_funcs, 2 * bas1.n_basis_funcs),
                },
                "2": {
                    "0": slice(
                        2 * bas1.n_basis_funcs,
                        2 * bas1.n_basis_funcs + bas2.n_basis_funcs,
                    ),
                    "1": slice(
                        2 * bas1.n_basis_funcs + bas2.n_basis_funcs,
                        2 * bas1.n_basis_funcs + 2 * bas2.n_basis_funcs,
                    ),
                },
            },
        ),
        (
            "__mul__",
            2,
            2,
            lambda bas1, bas2: {
                "(1 * 2)": slice(0, 2 * bas1.n_basis_funcs * 2 * bas2.n_basis_funcs)
            },
        ),
    ],
)
def test__get_splitter_split_by_input(
    bas1,
    bas2,
    operator,
    n_input_basis_1,
    n_input_basis_2,
    compute_slice,
    basis_class_specific_params,
):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2]
    ):
        return
    # define the basis
    n_basis = [5, 6]
    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        n_basis[0], bas1, basis_class_specific_params, window_size=10, label="1"
    )
    bas1_instance.set_input_shape(
        *([n_input_basis_1] * bas1_instance._n_input_dimensionality)
    )

    bas2_instance = combine_basis.instantiate_basis(
        n_basis[1], bas2, basis_class_specific_params, window_size=10, label="2"
    )
    bas2_instance.set_input_shape(
        *([n_input_basis_2] * bas1_instance._n_input_dimensionality)
    )

    func1 = getattr(bas1_instance, operator)
    bas12 = func1(bas2_instance)

    inps = [
        np.zeros((1, n)) if n > 1 else np.zeros((1,))
        for n in (n_input_basis_1, n_input_basis_2)
    ]
    bas12.set_input_shape(*inps)
    splitter_dict, _ = bas12._get_feature_slicing()
    exp_slices = compute_slice(bas1_instance, bas2_instance)
    assert exp_slices == splitter_dict


@pytest.mark.parametrize(
    "bas1, bas2, bas3",
    list(itertools.product(*[list_all_basis_classes()] * 3)),
)
def test_duplicate_keys(bas1, bas2, bas3, basis_class_specific_params):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2, bas3]
    ):
        return

    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        5, bas1, basis_class_specific_params, window_size=10, label="label"
    )
    bas2_instance = combine_basis.instantiate_basis(
        5, bas2, basis_class_specific_params, window_size=10, label="label"
    )
    bas3_instance = combine_basis.instantiate_basis(
        5, bas3, basis_class_specific_params, window_size=10, label="label"
    )
    bas_obj = bas1_instance + bas2_instance + bas3_instance

    inps = [np.zeros((1,)) for n in range(3)]
    bas_obj.set_input_shape(*inps)
    slice_dict = bas_obj._get_feature_slicing()[0]
    assert tuple(slice_dict.keys()) == ("label", "label-1", "label-2")


@pytest.mark.parametrize(
    "bas1, bas2",
    list(itertools.product(*[list_all_basis_classes()] * 2)),
)
@pytest.mark.parametrize(
    "x, axis, expectation, exp_shapes",  # num output is 5*2 + 6*3 = 28
    [
        (np.ones((1, 28)), 1, does_not_raise(), [(1, 2, 5), (1, 3, 6)]),
        (np.ones((28,)), 0, does_not_raise(), [(2, 5), (3, 6)]),
        (np.ones((2, 2, 28)), 2, does_not_raise(), [(2, 2, 2, 5), (2, 2, 3, 6)]),
        (
            np.ones((2, 2, 27)),
            2,
            pytest.raises(
                ValueError, match=r"`x.shape\[axis\]` does not match the expected"
            ),
            [(2, 2, 2, 5), (2, 2, 3, 6)],
        ),
    ],
)
def test_split_feature_axis(
    bas1, bas2, x, axis, expectation, exp_shapes, basis_class_specific_params
):
    # skip nested
    if any(
        bas in (AdditiveBasis, MultiplicativeBasis, basis.TransformerBasis)
        for bas in [bas1, bas2]
    ):
        return
    # define the basis
    n_basis = [5, 6]
    combine_basis = CombinedBasis()
    bas1_instance = combine_basis.instantiate_basis(
        n_basis[0], bas1, basis_class_specific_params, window_size=10, label="1"
    )
    bas2_instance = combine_basis.instantiate_basis(
        n_basis[1], bas2, basis_class_specific_params, window_size=10, label="2"
    )

    bas = bas1_instance + bas2_instance
    bas.set_input_shape(np.zeros((1, 2)), np.zeros((1, 3)))
    with expectation:
        out = bas.split_by_feature(x, axis=axis)
        for i, itm in enumerate(out.items()):
            _, val = itm
            assert val.shape == exp_shapes[i]
