import pickle
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from conftest import CombinedBasis, list_all_basis_classes
from sklearn.base import clone as sk_clone

from nemos import basis
import nemos as nmo
from sklearn.pipeline import Pipeline

@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_has_the_same_public_attributes_as_basis(
    basis_cls, basis_class_specific_params
):
    n_basis_funcs = 5
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=10
    )

    public_attrs_basis = {attr for attr in dir(bas) if not attr.startswith("_")}
    public_attrs_transformerbasis = {
        attr
        for attr in dir(
            bas.set_input_shape(*([1] * bas._n_input_dimensionality)).to_transformer()
        )
        if not attr.startswith("_")
    }

    assert public_attrs_transformerbasis - public_attrs_basis == {
        "fit",
        "fit_transform",
        "transform",
        "basis",
    }

    assert public_attrs_basis - public_attrs_transformerbasis == set()


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv") + list_all_basis_classes("Eval"),
)
def test_to_transformer_and_constructor_are_equivalent(
    basis_cls, basis_class_specific_params
):
    n_basis_funcs = 5
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=10
    )
    bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    trans_bas_a = bas.to_transformer()
    trans_bas_b = basis.TransformerBasis(bas)

    # they both just have a _basis
    assert (
        list(trans_bas_a.__dict__.keys())
        == list(trans_bas_b.__dict__.keys())
        == ["_basis"]
    )
    # and those bases are the same
    assert np.all(
        trans_bas_a._basis.__dict__.pop("_decay_rates", 1)
        == trans_bas_b._basis.__dict__.pop("_decay_rates", 1)
    )
    assert trans_bas_a._basis.__dict__ == trans_bas_b._basis.__dict__


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_basis_to_transformer_makes_a_copy(basis_cls, basis_class_specific_params):
    bas_a = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas_a = bas_a.set_input_shape(
        *([1] * bas_a._n_input_dimensionality)
    ).to_transformer()

    # changing an attribute in bas should not change trans_bas
    if basis_cls in [basis.AdditiveBasis, basis.MultiplicativeBasis]:
        bas_a._basis1.n_basis_funcs = 10
        assert trans_bas_a._basis._basis1.n_basis_funcs == 5

        # changing an attribute in the transformer basis should not change the original
        bas_b = CombinedBasis().instantiate_basis(
            5, basis_cls, basis_class_specific_params, window_size=10
        )
        bas_b.set_input_shape(*([1] * bas_b._n_input_dimensionality))
        trans_bas_b = bas_b.to_transformer()
        trans_bas_b._basis._basis1.n_basis_funcs = 100
        assert bas_b._basis1.n_basis_funcs == 5
    else:
        bas_a.n_basis_funcs = 10
        assert trans_bas_a.n_basis_funcs == 5

        # changing an attribute in the transformer basis should not change the original
        bas_b = CombinedBasis().instantiate_basis(
            5, basis_cls, basis_class_specific_params, window_size=10
        )
        trans_bas_b = bas_b.set_input_shape(
            *([1] * bas_b._n_input_dimensionality)
        ).to_transformer()
        trans_bas_b.n_basis_funcs = 100
        assert bas_b.n_basis_funcs == 5


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize("n_basis_funcs", [5, 10, 20])
def test_transformerbasis_getattr(
    basis_cls, n_basis_funcs, basis_class_specific_params
):
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=30
    )
    trans_basis = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )
    if basis_cls in [basis.AdditiveBasis, basis.MultiplicativeBasis]:
        for bas in [
            getattr(trans_basis._basis, attr) for attr in ("_basis1", "_basis2")
        ]:
            assert bas.n_basis_funcs == n_basis_funcs
    else:
        assert trans_basis.n_basis_funcs == n_basis_funcs


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv") + list_all_basis_classes("Eval"),
)
@pytest.mark.parametrize("n_basis_funcs_init", [5])
@pytest.mark.parametrize("n_basis_funcs_new", [6, 10, 20])
def test_transformerbasis_set_params(
    basis_cls, n_basis_funcs_init, n_basis_funcs_new, basis_class_specific_params
):
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs_init, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_basis = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )
    trans_basis.set_params(n_basis_funcs=n_basis_funcs_new)

    assert trans_basis.n_basis_funcs == n_basis_funcs_new
    assert trans_basis._basis.n_basis_funcs == n_basis_funcs_new


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv") + list_all_basis_classes("Eval"),
)
def test_transformerbasis_setattr_basis(basis_cls, basis_class_specific_params):
    # setting the _basis attribute should change it
    bas = CombinedBasis().instantiate_basis(
        10, basis_cls, basis_class_specific_params, window_size=30
    )
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )

    bas = CombinedBasis().instantiate_basis(
        20, basis_cls, basis_class_specific_params, window_size=30
    )

    trans_bas.basis = bas.set_input_shape(*([1] * bas._n_input_dimensionality))

    assert trans_bas.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv") + list_all_basis_classes("Eval"),
)
def test_transformerbasis_setattr_basis_attribute(
    basis_cls, basis_class_specific_params
):
    # setting an attribute that is an attribute of the underlying _basis
    # should propagate setting it on _basis itself
    bas = CombinedBasis().instantiate_basis(
        10, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )
    trans_bas.n_basis_funcs = 20

    assert trans_bas.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv") + list_all_basis_classes("Eval"),
)
def test_transformerbasis_copy_basis_on_contsruct(
    basis_cls, basis_class_specific_params
):
    # modifying the transformerbasis's attributes shouldn't
    # touch the original basis that was used to create it
    orig_bas = CombinedBasis().instantiate_basis(
        10, basis_cls, basis_class_specific_params, window_size=10
    )
    orig_bas = orig_bas.set_input_shape(*([1] * orig_bas._n_input_dimensionality))
    trans_bas = basis.TransformerBasis(orig_bas)
    trans_bas.n_basis_funcs = 20

    assert orig_bas.n_basis_funcs == 10
    assert trans_bas._basis.n_basis_funcs == 20
    assert trans_bas._basis.n_basis_funcs == 20
    assert isinstance(trans_bas._basis, basis_cls)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformerbasis_setattr_illegal_attribute(
    basis_cls, basis_class_specific_params
):
    # changing an attribute that is not _basis or an attribute of _basis
    # is not allowed
    bas = CombinedBasis().instantiate_basis(
        10, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )

    with pytest.raises(
        ValueError,
        match="Only setting _basis or existing attributes of _basis is allowed.",
    ):
        trans_bas.random_attr = "random value"


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformerbasis_addition(basis_cls, basis_class_specific_params):
    n_basis_funcs_a = 5
    n_basis_funcs_b = n_basis_funcs_a * 2
    bas_a = CombinedBasis().instantiate_basis(
        n_basis_funcs_a, basis_cls, basis_class_specific_params, window_size=10
    )
    bas_a.set_input_shape(*([1] * bas_a._n_input_dimensionality))
    bas_b = CombinedBasis().instantiate_basis(
        n_basis_funcs_b, basis_cls, basis_class_specific_params, window_size=10
    )
    bas_b.set_input_shape(*([1] * bas_b._n_input_dimensionality))
    trans_bas_a = basis.TransformerBasis(bas_a)
    trans_bas_b = basis.TransformerBasis(bas_b)
    trans_bas_sum = trans_bas_a + trans_bas_b
    assert isinstance(trans_bas_sum, basis.TransformerBasis)
    assert isinstance(trans_bas_sum._basis, basis.AdditiveBasis)
    assert (
        trans_bas_sum.n_basis_funcs
        == trans_bas_a.n_basis_funcs + trans_bas_b.n_basis_funcs
    )
    assert (
        trans_bas_sum._n_input_dimensionality
        == trans_bas_a._n_input_dimensionality + trans_bas_b._n_input_dimensionality
    )
    if basis_cls not in [basis.AdditiveBasis, basis.MultiplicativeBasis]:
        assert trans_bas_sum._basis1.n_basis_funcs == n_basis_funcs_a
        assert trans_bas_sum._basis2.n_basis_funcs == n_basis_funcs_b


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformerbasis_multiplication(basis_cls, basis_class_specific_params):
    n_basis_funcs_a = 5
    n_basis_funcs_b = n_basis_funcs_a * 2
    bas1 = CombinedBasis().instantiate_basis(
        n_basis_funcs_a, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas_a = basis.TransformerBasis(
        bas1.set_input_shape(*([1] * bas1._n_input_dimensionality))
    )
    bas2 = CombinedBasis().instantiate_basis(
        n_basis_funcs_b, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas_b = basis.TransformerBasis(
        bas2.set_input_shape(*([1] * bas2._n_input_dimensionality))
    )
    trans_bas_prod = trans_bas_a * trans_bas_b
    assert isinstance(trans_bas_prod, basis.TransformerBasis)
    assert isinstance(trans_bas_prod._basis, basis.MultiplicativeBasis)
    assert (
        trans_bas_prod.n_basis_funcs
        == trans_bas_a.n_basis_funcs * trans_bas_b.n_basis_funcs
    )
    assert (
        trans_bas_prod._n_input_dimensionality
        == trans_bas_a._n_input_dimensionality + trans_bas_b._n_input_dimensionality
    )
    if basis_cls not in [basis.AdditiveBasis, basis.MultiplicativeBasis]:
        assert trans_bas_prod._basis1.n_basis_funcs == n_basis_funcs_a
        assert trans_bas_prod._basis2.n_basis_funcs == n_basis_funcs_b


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize(
    "exponent, error_type, error_message",
    [
        (2, does_not_raise, None),
        (5, does_not_raise, None),
        (0.5, TypeError, "Exponent should be an integer"),
        (-1, ValueError, "Exponent should be a non-negative integer"),
    ],
)
def test_transformerbasis_exponentiation(
    basis_cls, exponent: int, error_type, error_message, basis_class_specific_params
):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )

    if not isinstance(exponent, int):
        with pytest.raises(error_type, match=error_message):
            trans_bas_exp = trans_bas**exponent
            assert isinstance(trans_bas_exp, basis.TransformerBasis)
            assert isinstance(trans_bas_exp._basis, basis.MultiplicativeBasis)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformerbasis_dir(basis_cls, basis_class_specific_params):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )
    for attr_name in (
        "fit",
        "transform",
        "fit_transform",
        "n_basis_funcs",
        "mode",
        "window_size",
    ):
        if (
            attr_name == "window_size"
            and "Conv" not in trans_bas._basis.__class__.__name__
        ):
            continue
        assert attr_name in dir(trans_bas)


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes("Conv"),
)
def test_transformerbasis_sk_clone_kernel_noned(basis_cls, basis_class_specific_params):
    orig_bas = CombinedBasis().instantiate_basis(
        10, basis_cls, basis_class_specific_params, window_size=20
    )
    orig_bas.set_input_shape(*([1] * orig_bas._n_input_dimensionality))
    trans_bas = basis.TransformerBasis(orig_bas)

    # kernel should be saved in the object after fit
    trans_bas.fit(np.random.randn(100, 1))
    assert isinstance(trans_bas.kernel_, np.ndarray)

    # cloning should set kernel_ to None
    trans_bas_clone = sk_clone(trans_bas)

    # the original object should still have kernel_
    assert isinstance(trans_bas.kernel_, np.ndarray)
    # but the clone should not have one
    assert trans_bas_clone.kernel_ is None


@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize("n_basis_funcs", [5])
def test_transformerbasis_pickle(
    tmpdir, basis_cls, n_basis_funcs, basis_class_specific_params
):
    bas = CombinedBasis().instantiate_basis(
        n_basis_funcs, basis_cls, basis_class_specific_params, window_size=10
    )
    # the test that tries cross-validation with n_jobs = 2 already should test this
    trans_bas = basis.TransformerBasis(
        bas.set_input_shape(*([1] * bas._n_input_dimensionality))
    )
    filepath = tmpdir / "transformerbasis.pickle"
    with open(filepath, "wb") as f:
        pickle.dump(trans_bas, f)
    with open(filepath, "rb") as f:
        trans_bas2 = pickle.load(f)

    assert isinstance(trans_bas2, basis.TransformerBasis)
    if basis_cls in [basis.AdditiveBasis, basis.MultiplicativeBasis]:
        for bas in [
            getattr(trans_bas2._basis, attr) for attr in ("_basis1", "_basis2")
        ]:
            assert bas.n_basis_funcs == n_basis_funcs
    else:
        assert trans_bas2.n_basis_funcs == n_basis_funcs


@pytest.mark.parametrize(
    "set_input, expectation",
    [
        (True, does_not_raise()),
        (
            False,
            pytest.raises(
                RuntimeError,
                match="Cannot initialize TransformerBasis: the provided basis has no defined input shape",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "inp", [np.ones((10,)), np.ones((10, 1)), np.ones((10, 2)), np.ones((10, 2, 3))]
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_to_transformer_and_set_input(
    basis_cls, inp, set_input, expectation, basis_class_specific_params
):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    if set_input:
        bas.set_input_shape(*([inp] * bas._n_input_dimensionality))
    with expectation:
        bas.to_transformer()


@pytest.mark.parametrize(
    "inp, expectation",
    [
        (np.ones((10,)), pytest.raises(ValueError, match="X must be 2-")),
        (np.ones((10, 1)), does_not_raise()),
        (np.ones((10, 2)), does_not_raise()),
        (np.ones((10, 2, 3)), pytest.raises(ValueError, match="X must be 2-")),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_fit(basis_cls, inp, basis_class_specific_params, expectation):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.concatenate(
        [inp.reshape(inp.shape[0], -1)] * bas._n_input_dimensionality, axis=1
    )
    transformer.fit(X)
    if "Conv" in basis_cls.__name__:
        assert transformer.kernel_ is not None

    # try and pass segmented time series
    if isinstance(bas, (basis.AdditiveBasis, basis.MultiplicativeBasis)):
        if inp.ndim == 2:
            expectation = pytest.raises(ValueError, match="Input mismatch: expected ")

    with expectation:
        transformer.fit(*([inp] * bas._n_input_dimensionality))


@pytest.mark.parametrize(
    "inp",
    [
        np.ones((10, 1)),
        np.ones((10, 2)),
    ],
)
@pytest.mark.parametrize(
    "delta_input, expectation",
    [
        (0, does_not_raise()),
        (1, pytest.raises(ValueError, match="Input mismatch: expected ")),
        (-1, pytest.raises(ValueError, match="Input mismatch: expected ")),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_fit_input_shape_mismatch(
    basis_cls, delta_input, inp, basis_class_specific_params, expectation
):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.random.randn(10, int(sum(bas._n_basis_input_) + delta_input))
    with expectation:
        transformer.fit(X)


@pytest.mark.parametrize(
    "inp",
    [
        np.random.randn(
            10,
        ),
        np.random.randn(10, 1),
        np.random.randn(10, 2),
        np.random.randn(10, 2, 3),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_transform(basis_cls, inp, basis_class_specific_params):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.concatenate(
        [inp.reshape(inp.shape[0], -1)] * bas._n_input_dimensionality, axis=1
    )
    transformer.fit(X)

    out = transformer.transform(X)
    out2 = bas.compute_features(*([inp] * bas._n_input_dimensionality))

    assert np.array_equal(out, out2, equal_nan=True)


@pytest.mark.parametrize(
    "inp",
    [
        np.random.randn(
            10,
        ),
        np.random.randn(10, 1),
        np.random.randn(10, 2),
        np.random.randn(10, 2, 3),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_fit_transform(basis_cls, inp, basis_class_specific_params):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.concatenate(
        [inp.reshape(inp.shape[0], -1)] * bas._n_input_dimensionality, axis=1
    )

    out = transformer.fit_transform(X)
    out2 = bas.compute_features(*([inp] * bas._n_input_dimensionality))

    assert np.array_equal(out, out2, equal_nan=True)


@pytest.mark.parametrize(
    "inp",
    [
        np.ones((10, 1)),
        np.ones((10, 2)),
    ],
)
@pytest.mark.parametrize(
    "delta_input, expectation",
    [
        (0, does_not_raise()),
        (1, pytest.raises(ValueError, match="Input mismatch: expected ")),
        (-1, pytest.raises(ValueError, match="Input mismatch: expected ")),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_fit_transform_input_shape_mismatch(
    basis_cls, delta_input, inp, basis_class_specific_params, expectation
):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.random.randn(10, int(sum(bas._n_basis_input_) + delta_input))
    with expectation:
        transformer.fit_transform(X)


@pytest.mark.parametrize(
    "inp, expectation",
    [
        (np.ones((10,)), pytest.raises(ValueError, match="X must be 2-")),
        (np.ones((10, 1)), does_not_raise()),
        (np.ones((10, 2)), does_not_raise()),
        (np.ones((10, 2, 3)), pytest.raises(ValueError, match="X must be 2-")),
    ],
)
@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
def test_transformer_fit_transform_input_struct(
    basis_cls, inp, basis_class_specific_params, expectation
):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()
    X = np.concatenate(
        [inp.reshape(inp.shape[0], -1)] * bas._n_input_dimensionality, axis=1
    )
    transformer.fit_transform(X)

    if "Conv" in basis_cls.__name__:
        assert transformer.kernel_ is not None

    # try and pass a tuple of time series
    if isinstance(bas, (basis.AdditiveBasis, basis.MultiplicativeBasis)) and inp.ndim != 2:
        expectation = pytest.raises(ValueError, match="X must be 2-")
    elif isinstance(bas, (basis.AdditiveBasis, basis.MultiplicativeBasis)) and inp.ndim == 2:
        expectation = pytest.raises(ValueError, match="Input mismatch: expected")
    with expectation:
        transformer.fit(*([inp] * bas._n_input_dimensionality))

@pytest.mark.parametrize(
    "basis_cls",
    list_all_basis_classes(),
)
@pytest.mark.parametrize(
    "inp",
    [
        np.random.randn(100,),
        np.random.randn(100, 1),
        np.random.randn(100, 2),
        np.random.randn(100, 1, 2),
    ],
)
def test_transformer_in_pipeline(basis_cls, inp, basis_class_specific_params):
    bas = CombinedBasis().instantiate_basis(
        5, basis_cls, basis_class_specific_params, window_size=10
    )
    transformer = bas.set_input_shape(
        *([inp] * bas._n_input_dimensionality)
    ).to_transformer()

    # fit outside pipeline
    X = bas.compute_features(*([inp] * bas._n_input_dimensionality))
    log_mu = X.dot(0.005 * np.ones(X.shape[1]))
    y = np.full(X.shape[0], 0)
    y[~np.isnan(log_mu)] = np.random.poisson(np.exp(log_mu[~np.isnan(log_mu)] - np.nanmean(log_mu)))
    model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.001).fit(X, y)

    # pipeline
    pipe = Pipeline(
        [
            ("bas", transformer),
            ("glm", nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.001))
        ]
    )
    x = np.concatenate(
        [inp.reshape(inp.shape[0], -1)] * bas._n_input_dimensionality, axis=1
    )
    pipe.fit(x, y)
    np.testing.assert_allclose(pipe["glm"].coef_, model.coef_)

    # set basis & refit
    if isinstance(bas, (basis.AdditiveBasis, basis.MultiplicativeBasis)):
        pipe.set_params(bas__basis2__n_basis_funcs=4)
        assert bas.basis2.n_basis_funcs == 5  # make sure that the change did not affect bas
        X = bas.set_params(basis2__n_basis_funcs=4).compute_features(*([inp] * bas._n_input_dimensionality))
    else:
        pipe.set_params(bas__n_basis_funcs=4)
        assert bas.n_basis_funcs == 5  # make sure that the change did not affect bas
        X = bas.set_params(n_basis_funcs=4).compute_features(*([inp] * bas._n_input_dimensionality))
    pipe.fit(x, y)
    model.fit(X, y)
    np.testing.assert_allclose(pipe["glm"].coef_, model.coef_)
