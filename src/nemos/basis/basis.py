"""Bases classes."""

# required to get ArrayLike to render correctly
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis_mixin import EvalBasisMixin, ConvBasisMixin

from ._basis import Basis, check_transform_input, check_one_dimensional
from ._spline_basis import BSplineBasis, CyclicBSplineBasis, MSplineBasis
from ._raised_cosine_basis import RaisedCosineBasisLinear, RaisedCosineBasisLog

__all__ = [
    "EvalMSpline",
    "ConvMSpline",
    "EvalBSpline",
    "ConvBSpline",
    "EvalCyclicBSpline",
    "ConvCyclicBSpline",
    "EvalRaisedCosineLinear",
    "ConvRaisedCosineLinear",
    "OrthExponentialBasis",
]


def __dir__() -> list[str]:
    return __all__


class EvalBSpline(EvalBasisMixin, BSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalBSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        BSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )



class ConvBSpline(ConvBasisMixin, BSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvBSpline",
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        BSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalCyclicBSpline(EvalBasisMixin, CyclicBSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalCyclicBSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        CyclicBSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )


class ConvCyclicBSpline(ConvBasisMixin, CyclicBSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvCyclicBSpline",
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        CyclicBSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalMSpline(EvalBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalMSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )


class ConvMSpline(ConvBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvMSpline",
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )

class EvalMSpline(EvalBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            order: int = 4,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalMSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="eval",
            order=order,
            label=label,
        )


class ConvMSpline(ConvBasisMixin, MSplineBasis):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            order: int = 4,
            label: Optional[str] = "ConvMSpline",
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        MSplineBasis.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            order=order,
            label=label,
        )


class EvalRaisedCosineLinear(EvalBasisMixin, RaisedCosineBasisLinear):
    def __init__(
            self,
            n_basis_funcs: int,
            width: float = 2.0,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalMSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        RaisedCosineBasisLinear.__init__(
            self,
            n_basis_funcs,
            width=width,
            mode="eval",
            label=label,
        )


class ConvRaisedCosineLinear(ConvBasisMixin, RaisedCosineBasisLinear):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            width: float = 2.0,
            label: Optional[str] = "ConvMSpline",
            **conv_kwargs,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        RaisedCosineBasisLinear.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            width=width,
            label=label,
            **conv_kwargs,
        )

class EvalRaisedCosineLog(EvalBasisMixin, RaisedCosineBasisLog):
    def __init__(
            self,
            n_basis_funcs: int,
            width: float = 2.0,
            time_scaling: float = None,
            enforce_decay_to_zero: bool = True,
            bounds: Optional[Tuple[float, float]] = None,
            label: Optional[str] = "EvalMSpline",
    ):
        EvalBasisMixin.__init__(self, bounds=bounds)
        RaisedCosineBasisLog.__init__(
            self,
            n_basis_funcs,
            width=width,
            time_scaling=time_scaling,
            enforce_decay_to_zero=enforce_decay_to_zero,
            mode="eval",
            label=label,
        )


class ConvRaisedCosineLog(ConvBasisMixin, RaisedCosineBasisLog):
    def __init__(
            self,
            n_basis_funcs: int,
            window_size: int,
            width: float = 2.0,
            time_scaling: float = None,
            enforce_decay_to_zero: bool = True,
            label: Optional[str] = "ConvMSpline",
            **conv_kwargs,
    ):
        ConvBasisMixin.__init__(self, window_size=window_size)
        RaisedCosineBasisLog.__init__(
            self,
            n_basis_funcs,
            mode="conv",
            width=width,
            time_scaling=time_scaling,
            enforce_decay_to_zero=enforce_decay_to_zero,
            label=label,
            **conv_kwargs,
        )



class OrthExponentialBasis(Basis):
    """Set of 1D basis decaying exponential functions numerically orthogonalized.

    Parameters
    ----------
    n_basis_funcs
            Number of basis functions.
    decay_rates :
            Decay rates of the exponentials, shape ``(n_basis_funcs,)``.
    mode :
        The mode of operation. ``'eval'`` for evaluation at sample points,
        ``'conv'`` for convolutional operation.
    window_size :
        The window size for convolution. Required if mode is ``'conv'``.
    bounds :
        The bounds for the basis domain in ``mode="eval"``. The default ``bounds[0]`` and ``bounds[1]`` are the
        minimum and the maximum of the samples provided when evaluating the basis.
        If a sample is outside the bounds, the basis will return NaN.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.
    **kwargs :
        Additional keyword arguments passed to ``nemos.convolve.create_convolutional_predictor`` when
        ``mode='conv'``; These arguments are used to change the default behavior of the convolution.
        For example, changing the ``predictor_causality``, which by default is set to ``"causal"``.
        Note that one cannot change the default value for the ``axis`` parameter. Basis assumes
        that the convolution axis is ``axis=0``.

    Examples
    --------
    >>> from numpy import linspace
    >>> from nemos.basis import OrthExponentialBasis
    >>> X = np.random.normal(size=(1000, 1))
    >>> n_basis_funcs = 5
    >>> decay_rates = [0.01, 0.02, 0.03, 0.04, 0.05]  # sample decay rates
    >>> window_size=10
    >>> ortho_basis = OrthExponentialBasis(n_basis_funcs, decay_rates, "conv", window_size)
    >>> sample_points = linspace(0, 1, 100)
    >>> basis_functions = ortho_basis(sample_points)
    """

    def __init__(
        self,
        n_basis_funcs: int,
        decay_rates: NDArray[np.floating],
        mode="eval",
        window_size: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = None,
        label: Optional[str] = "OrthExponentialBasis",
        **kwargs,
    ):
        super().__init__(
            n_basis_funcs,
            mode=mode,
            window_size=window_size,
            bounds=bounds,
            label=label,
            **kwargs,
        )
        self.decay_rates = decay_rates
        self._check_rates()
        self._n_input_dimensionality = 1

    @property
    def decay_rates(self):
        """Decay rate.

        The rate of decay of the exponential functions. If :math:`f_i(t) = \exp{-\alpha_i t}` is the i-th decay
        exponential before orthogonalization, :math:`\alpha_i` is the i-th element of the ``decay_rate`` vector.
        """
        return self._decay_rates

    @decay_rates.setter
    def decay_rates(self, value: NDArray):
        """Decay rate setter."""
        value = np.asarray(value)
        if value.shape[0] != self.n_basis_funcs:
            raise ValueError(
                f"The number of basis functions must match the number of decay rates provided. "
                f"Number of basis functions provided: {self.n_basis_funcs}, "
                f"Number of decay rates provided: {value.shape[0]}"
            )
        self._decay_rates = value

    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Checks that the number of basis is at least 1.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        if self.n_basis_funcs < 1:
            raise ValueError(
                f"Object class {self.__class__.__name__} requires >= 1 basis elements. "
                f"{self.n_basis_funcs} basis elements specified instead"
            )

    def _check_rates(self) -> None:
        """
        Check if the decay rates list has duplicate entries.

        Raises
        ------
        ValueError
            If two or more decay rates are repeated, which would result in a linearly
            dependent set of functions for the basis.
        """
        if len(set(self._decay_rates)) != len(self._decay_rates):
            raise ValueError(
                "Two or more rate are repeated! Repeating rate will result in a "
                "linearly dependent set of function for the basis."
            )

    def _check_sample_size(self, *sample_pts: NDArray) -> None:
        """Check that the sample size is greater than the number of basis.

        This is necessary for the orthogonalization procedure,
        that otherwise will return (sample_size, ) basis elements instead of the expected number.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0, inf).

        Raises
        ------
        ValueError
            If the number of basis element is less than the number of samples.
        """
        if sample_pts[0].size < self.n_basis_funcs:
            raise ValueError(
                "OrthExponentialBasis requires at least as many samples as basis functions!\n"
                f"Class instantiated with {self.n_basis_funcs} basis functions "
                f"but only {sample_pts[0].size} samples provided!"
            )

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def __call__(
        self,
        sample_pts: NDArray,
    ) -> FeatureMatrix:
        """Generate basis functions with given spacing.

        Parameters
        ----------
        sample_pts
            Spacing for basis functions, holding elements on the interval [0,
            inf), shape (n_samples,).

        Returns
        -------
        basis_funcs
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape (n_samples, n_basis_funcs)

        """
        self._check_sample_size(sample_pts)
        sample_pts, _ = min_max_rescale_samples(sample_pts, self.bounds)
        valid_idx = ~np.isnan(sample_pts)
        # because of how scipy.linalg.orth works, have to create a matrix of
        # shape (n_pts, n_basis_funcs) and then transpose, rather than
        # directly computing orth on the matrix of shape (n_basis_funcs,
        # n_pts)
        exp_decay_eval = np.stack(
            [np.exp(-lam * sample_pts[valid_idx]) for lam in self._decay_rates], axis=1
        )
        # count the linear independent components (could be lower than n_basis_funcs for num precision).
        n_independent_component = np.linalg.matrix_rank(exp_decay_eval)
        # initialize output to nan
        basis_funcs = np.full(
            shape=(sample_pts.shape[0], n_independent_component), fill_value=np.nan
        )
        # orthonormalize on valid points
        basis_funcs[valid_idx] = scipy.linalg.orth(exp_decay_eval)
        return basis_funcs

    def evaluate_on_grid(self, n_samples: int) -> Tuple[NDArray, NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        Parameters
        ----------
        n_samples :
            The number of samples.

        Returns
        -------
        X :
            Array of shape (n_samples,) containing the equi-spaced sample
            points where we've evaluated the basis.
        basis_funcs :
            Evaluated exponentially decaying basis functions, numerically
            orthogonalized, shape (n_samples, n_basis_funcs)

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from nemos.basis import OrthExponentialBasis
        >>> n_basis_funcs = 5
        >>> decay_rates = [0.01, 0.02, 0.03, 0.04, 0.05] # sample decay rates
        >>> window_size=10
        >>> ortho_basis = OrthExponentialBasis(n_basis_funcs, decay_rates, "conv", window_size)
        >>> sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
        """
        return super().evaluate_on_grid(n_samples)
