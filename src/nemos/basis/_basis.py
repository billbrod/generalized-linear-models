# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import copy
from functools import wraps
from typing import Callable, Generator, Literal, Optional, Tuple, Union

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from ..base_class import Base
from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ..utils import row_wise_kron
from ..validation import check_fraction_valid_samples
from ._basis_mixin import BasisTransformerMixin


def add_docstring(method_name, cls):
    """Prepend super-class docstrings."""
    attr = getattr(cls, method_name, None)
    if attr is None:
        raise AttributeError(f"{cls.__name__} has no attribute {method_name}!")
    doc = attr.__doc__

    # Decorator to add the docstring
    def wrapper(func):
        func.__doc__ = "\n".join([doc, func.__doc__])  # Combine docstrings
        return func

    return wrapper


def check_transform_input(func: Callable) -> Callable:
    """Check input before calling basis.

    This decorator allows to raise an exception that is more readable
    when the wrong number of input is provided to _evaluate.
    """

    @wraps(func)
    def wrapper(self: Basis, *xi: ArrayLike, **kwargs) -> NDArray:
        xi = self._check_transform_input(*xi)
        return func(self, *xi, **kwargs)  # Call the basis

    return wrapper


def check_one_dimensional(func: Callable) -> Callable:
    """Check if the input is one-dimensional."""

    @wraps(func)
    def wrapper(self: Basis, *xi: ArrayLike, **kwargs):
        if any(x.ndim != 1 for x in xi):
            raise ValueError("Input sample must be one dimensional!")
        return func(self, *xi, **kwargs)

    return wrapper


def min_max_rescale_samples(
    sample_pts: NDArray,
    bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[NDArray, float]:
    """Rescale samples to [0,1].

    Parameters
    ----------
    sample_pts:
        The original samples.
    bounds:
        Sample bounds. `bounds[0]` and `bounds[1]` are mapped to 0 and 1, respectively.
        Default are `min(sample_pts), max(sample_pts)`.

    Warns
    -----
    UserWarning
        If more than 90% of the sample points contain NaNs or Infs.
    """
    sample_pts = sample_pts.astype(float)
    vmin = np.nanmin(sample_pts) if bounds is None else bounds[0]
    vmax = np.nanmax(sample_pts) if bounds is None else bounds[1]
    sample_pts[(sample_pts < vmin) | (sample_pts > vmax)] = np.nan
    sample_pts -= vmin
    # this passes if `samples_pts` contains a single value
    if vmin != vmax:
        scaling = vmax - vmin
        sample_pts /= scaling
    else:
        scaling = 1.0

    check_fraction_valid_samples(
        sample_pts,
        err_msg="All the samples lie outside the [vmin, vmax] range.",
        warn_msg="More than 90% of the samples lie outside the [vmin, vmax] range.",
    )

    return sample_pts, scaling


class Basis(Base, abc.ABC, BasisTransformerMixin):
    """
    Abstract base class for defining basis functions for feature transformation.

    Basis functions are mathematical constructs that can represent data in alternative,
    often more compact or interpretable forms. This class provides a template for such
    transformations, with specific implementations defining the actual behavior.

    Parameters
    ----------
    n_basis_funcs :
        The number of basis functions.
    mode :
        The mode of operation. 'eval' for evaluation at sample points,
        'conv' for convolutional operation.
    label :
        The label of the basis, intended to be descriptive of the task variable being processed.
        For example: velocity, position, spike_counts.

    Raises
    ------
    ValueError:
        If ``mode`` is not 'eval' or 'conv'.
    ValueError:
        If ``kwargs`` are not None and ``mode =="eval"``.
    ValueError:
        If ``kwargs`` include parameters not recognized or do not have
        default values in ``create_convolutional_predictor``.
    ValueError:
        If ``axis`` different from 0 is provided as a keyword argument (samples must always be in the first axis).
    """

    def __init__(
        self,
        n_basis_funcs: int,
        mode: Literal["eval", "conv"] = "eval",
        label: Optional[str] = None,
    ) -> None:
        self.n_basis_funcs = n_basis_funcs
        self._n_input_dimensionality = 0

        self._mode = mode

        self._n_basis_input = None

        # these parameters are going to be set at the first call of `compute_features`
        # since we cannot know a-priori how many features may be convolved
        self._n_output_features = None
        self._input_shape = None

        if label is None:
            self._label = self.__class__.__name__
        else:
            self._label = str(label)

        self.kernel_ = None

    @property
    def n_output_features(self) -> int | None:
        """
        Number of features returned by the basis.

        Notes
        -----
        The number of output features can be determined only when the number of inputs
        provided to the basis is known. Therefore, before the first call to ``compute_features``,
        this property will return ``None``. After that call, ``n_output_features`` will be available.
        """
        return self._n_output_features

    @property
    def label(self) -> str:
        """Label for the basis."""
        return self._label

    @property
    def n_basis_input(self) -> tuple | None:
        """Number of expected inputs.

        The number of inputs ``compute_feature`` expects.
        """
        return self._n_basis_input

    @property
    def n_basis_funcs(self):
        """Number of basis functions."""
        return self._n_basis_funcs

    @n_basis_funcs.setter
    def n_basis_funcs(self, value):
        orig_n_basis = copy.deepcopy(getattr(self, "_n_basis_funcs", None))
        self._n_basis_funcs = value
        try:
            self._check_n_basis_min()
        except ValueError as e:
            self._n_basis_funcs = orig_n_basis
            raise e

    @property
    def mode(self):
        """Mode of operation, either ``"conv"`` or ``"eval"``."""
        return self._mode

    @staticmethod
    def _apply_identifiability_constraints(X: NDArray):
        """Apply identifiability constraints to a design matrix `X`.

         Removes columns from `X` until `[1, X]` is full rank to ensure the uniqueness
         of the GLM (Generalized Linear Model) maximum-likelihood solution. This is particularly
         crucial for models using bases like BSplines and CyclicBspline, which, due to their
         construction, sum to 1 and can cause rank deficiency when combined with an intercept.

         For GLMs, this rank deficiency means that different sets of coefficients might yield
         identical predicted rates and log-likelihood, complicating parameter learning, especially
         in the absence of regularization.

        Parameters
        ----------
        X:
            The design matrix before applying the identifiability constraints.

        Returns
        -------
        :
            The adjusted design matrix with redundant columns dropped and columns mean-centered.
        """

        def add_constant(x):
            return np.hstack((np.ones((x.shape[0], 1)), x))

        rank = np.linalg.matrix_rank(add_constant(X))
        # mean center
        X = X - np.nanmean(X, axis=0)
        while rank < X.shape[1] + 1:
            # drop a column
            X = X[:, :-1]
            # recompute rank
            rank = np.linalg.matrix_rank(add_constant(X))
        return X

    @check_transform_input
    def compute_features(self, *xi: ArrayLike | Tsd | TsdFrame |  TsdTensor) -> FeatureMatrix:
        """
        Apply the basis transformation to the input data.

        This method is designed to be a high-level interface for transforming input
        data using the basis functions defined by the subclass. Depending on the basis'
        mode ('Eval' or 'Conv'), it either evaluates the basis functions at the sample
        points or performs a convolution operation between the input data and the
        basis functions.

        Parameters
        ----------
        *xi :
            Input data arrays to be transformed. The shape and content requirements
            depend on the subclass and mode of operation ('Eval' or 'Conv').

        Returns
        -------
        :
            Transformed features. In 'eval' mode, it corresponds to the basis functions
            evaluated at the input samples. In 'conv' mode, it consists of convolved
            input samples with the basis functions. The output shape varies based on
            the subclass and mode.

        Notes
        -----
        Subclasses should implement how to handle the transformation specific to their
        basis function types and operation modes.
        """
        self._set_num_output_features(*xi)
        self._set_kernel()
        return self._compute_features(*xi)

    @abc.abstractmethod
    def _compute_features(self, *xi: NDArray | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """Convolve or evaluate the basis."""
        pass

    @abc.abstractmethod
    def _set_kernel(self):
        """Set kernel for conv basis and return self or just return self for eval."""
        pass

    @abc.abstractmethod
    def _evaluate(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Abstract method to evaluate the basis functions at given points.

        This method must be implemented by subclasses to define the specific behavior
        of the basis transformation. The implementation depends on the type of basis
        (e.g., spline, raised cosine), and it should evaluate the basis functions at
        the specified points in the domain.

        Parameters
        ----------
        *xi :
            Variable number of arguments, each representing an array of points at which
            to evaluate the basis functions. The dimensions and requirements of these
            inputs vary depending on the specific basis implementation.

        Returns
        -------
        :
            An array containing the evaluated values of the basis functions at the input
            points. The shape and structure of this array are specific to the subclass
            implementation.
        """
        pass

    def _get_samples(self, *n_samples: int) -> Generator[NDArray]:
        """Get equi-spaced samples for all the input dimensions.

        This will be used to evaluate the basis on a grid of
        points derived by the samples.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of samples in each axis of the grid.

        Returns
        -------
        :
            A generator yielding numpy arrays of linspaces from 0 to 1 of sizes specified by ``n_samples``.
        """
        # handling of defaults when evaluating on a grid
        # (i.e. when we cannot use max and min of samples)
        bounds = getattr(self, "bounds", None)
        if bounds is None:
            mn, mx = 0, 1
        else:
            mn, mx = bounds
        return (np.linspace(mn, mx, n_samples[k]) for k in range(len(n_samples)))

    @support_pynapple(conv_type="numpy")
    def _check_transform_input(
        self, *xi: ArrayLike
    ) -> Tuple[Union[NDArray, Tsd, TsdFrame]]:
        """Check transform input.

        Parameters
        ----------
        xi[0],...,xi[n] :
            The input samples, each  with shape (number of samples, ).

        Raises
        ------
        ValueError
            - If the time point number is inconsistent between inputs.
            - If the number of inputs doesn't match what the Basis object requires.
            - At least one of the samples is empty.

        """
        # check that the input is array-like (i.e., whether we can cast it to
        # numeric arrays)
        try:
            # make sure array is at least 1d (so that we succeed when only
            # passed a scalar)
            xi = tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in xi)
        # ValueError here surfaces the exception with e.g., `x=np.array["a", "b"])`
        except (TypeError, ValueError):
            raise TypeError("Input samples must be array-like of floats!")

        # check for non-empty samples
        if self._has_zero_samples(tuple(len(x) for x in xi)):
            raise ValueError("All sample provided must be non empty.")

        # checks on input and outputs
        self._check_samples_consistency(*xi)
        self._check_input_dimensionality(xi)

        return xi

    def _check_has_kernel(self) -> None:
        """Check that the kernel is pre-computed."""
        if self.mode == "conv" and self.kernel_ is None:
            raise ValueError(
                "You must call `_set_kernel` before `_compute_features` when mode =`conv`."
            )

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
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
        """
        self._check_input_dimensionality(n_samples)

        if self._has_zero_samples(n_samples):
            raise ValueError("All sample counts provided must be greater than zero.")

        # get the samples
        sample_tuple = self._get_samples(*n_samples)
        Xs = np.meshgrid(*sample_tuple, indexing="ij")

        # evaluates the basis on a flat NDArray and reshape to match meshgrid output
        Y = self._evaluate(*tuple(grid_axis.flatten() for grid_axis in Xs)).reshape(
            (*n_samples, self.n_basis_funcs)
        )

        return *Xs, Y

    @staticmethod
    def _has_zero_samples(n_samples: Tuple[int, ...]) -> bool:
        return any([n <= 0 for n in n_samples])

    def _check_input_dimensionality(self, xi: Tuple) -> None:
        """
        Check that the number of inputs provided by the user matches the number of inputs required.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the number of inputs doesn't match what the Basis object requires.
        """
        if len(xi) != self._n_input_dimensionality:
            raise TypeError(
                f"Input dimensionality mismatch. This basis evaluation requires {self._n_input_dimensionality} inputs, "
                f"{len(xi)} inputs provided instead."
            )

    @staticmethod
    def _check_samples_consistency(*xi: NDArray) -> None:
        """
        Check that each input provided to the Basis object has the same number of time points.

        Parameters
        ----------
        xi[0], ..., xi[n] :
            The input samples, shape (number of samples, ).

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs.
        """
        sample_sizes = [sample.shape[0] for sample in xi]
        if any(elem != sample_sizes[0] for elem in sample_sizes):
            raise ValueError(
                "Sample size mismatch. Input elements have inconsistent sample sizes."
            )

    @abc.abstractmethod
    def _check_n_basis_min(self) -> None:
        """Check that the user required enough basis elements.

        Most of the basis work with at least 1 element, but some
        such as the RaisedCosineBasisLog requires a minimum of 2 basis to be well defined.

        Raises
        ------
        ValueError
            If an insufficient number of basis element is requested for the basis type
        """
        pass

    def __add__(self, other: Basis) -> AdditiveBasis:
        """
        Add two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to add.

        Returns
        -------
        : AdditiveBasis
            The resulting Basis object.
        """
        return AdditiveBasis(self, other)

    def __mul__(self, other: Basis) -> MultiplicativeBasis:
        """
        Multiply two Basis objects together.

        Parameters
        ----------
        other
            The other Basis object to multiply.

        Returns
        -------
        :
            The resulting Basis object.
        """
        return MultiplicativeBasis(self, other)

    def __pow__(self, exponent: int) -> MultiplicativeBasis:
        """Exponentiation of a Basis object.

        Define the power of a basis by repeatedly applying the method __multiply__.
        The exponent must be a positive integer.

        Parameters
        ----------
        exponent :
            Positive integer exponent

        Returns
        -------
        :
            The product of the basis with itself "exponent" times. Equivalent to ``self * self * ... * self``.

        Raises
        ------
        TypeError
            If the provided exponent is not an integer.
        ValueError
            If the integer is zero or negative.
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent should be an integer!")

        if exponent <= 0:
            raise ValueError("Exponent should be a non-negative integer!")

        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def _get_feature_slicing(
        self,
        n_inputs: Optional[tuple] = None,
        start_slice: Optional[int] = None,
        split_by_input: bool = True,
    ) -> Tuple[dict, int]:
        """
        Calculate and return the slicing for features based on the input structure.

        This method determines how to slice the features for different basis types.

        Parameters
        ----------
        n_inputs :
            The number of input basis for each component, by default it uses ``self._n_basis_input``.
        start_slice :
            The starting index for slicing, by default it starts from 0.
        split_by_input :
            Flag indicating whether to split the slicing by individual inputs or not.
            If ``False``, a single slice is generated for all inputs.

        Returns
        -------
        split_dict :
            Dictionary with keys as labels and values as slices representing
            the slicing for each input or additive component, if split_by_input equals to
            True or False respectively.
        start_slice :
            The updated starting index after slicing.

        See Also
        --------
        _get_default_slicing : Handles default slicing logic.
        _merge_slicing_dicts : Merges multiple slicing dictionaries, handling keys conflicts.
        """
        # Set default values for n_inputs and start_slice if not provided
        n_inputs = n_inputs or self._n_basis_input
        start_slice = start_slice or 0

        # Handle the default case for non-additive basis types
        # See overwritten method for recursion logic
        split_dict, start_slice = self._get_default_slicing(
            split_by_input=split_by_input, start_slice=start_slice
        )

        return split_dict, start_slice

    @staticmethod
    def _generate_unique_key(existing_dict: dict, key: str) -> str:
        """Generate a unique key if there is a conflict."""
        extra = 1
        new_key = f"{key}-{extra}"
        while new_key in existing_dict:
            extra += 1
            new_key = f"{key}-{extra}"
        return new_key

    def _get_default_slicing(
        self, split_by_input: bool, start_slice: int
    ) -> Tuple[dict, int]:
        """Handle default slicing logic."""
        if split_by_input:
            # should we remove this option?
            if self._n_basis_input[0] == 1 or isinstance(self, MultiplicativeBasis):
                split_dict = {
                    self.label: slice(
                        start_slice, start_slice + self._n_output_features
                    )
                }
            else:
                split_dict = {
                    self.label: {
                        f"{i}": slice(
                            start_slice + i * self.n_basis_funcs,
                            start_slice + (i + 1) * self.n_basis_funcs,
                        )
                        for i in range(self._n_basis_input[0])
                    }
                }
        else:
            split_dict = {
                self.label: slice(start_slice, start_slice + self._n_output_features)
            }
        start_slice += self._n_output_features
        return split_dict, start_slice

    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Decompose an array along a specified axis into sub-arrays based on the number of expected inputs.

        This function takes an array (e.g., a design matrix or model coefficients) and splits it along
        a designated axis.

        **How it works:**

        - If the basis expects an input shape ``(n_samples, n_inputs)``, then the feature axis length will
          be ``total_n_features = n_inputs * n_basis_funcs``. This axis is reshaped into dimensions
          ``(n_inputs, n_basis_funcs)``.

        - If the basis expects an input of shape ``(n_samples,)``, then the feature axis length will
          be ``total_n_features = n_basis_funcs``. This axis is reshaped into ``(1, n_basis_funcs)``.

        For example, if the input array ``x`` has shape ``(1, 2, total_n_features, 4, 5)``,
        then after applying this method, it will be reshaped into ``(1, 2, n_inputs, n_basis_funcs, 4, 5)``.

        The specified axis (``axis``) determines where the split occurs, and all other dimensions
        remain unchanged. See the example section below for the most common use cases.

        Parameters
        ----------
        x :
            The input array to be split, representing concatenated features, coefficients,
            or other data. The shape of ``x`` along the specified axis must match the total
            number of features generated by the basis, i.e., ``self.n_output_features``.

            **Examples:**

            - For a design matrix: ``(n_samples, total_n_features)``

            - For model coefficients: ``(total_n_features,)`` or ``(total_n_features, n_neurons)``.

        axis : int, optional
            The axis along which to split the features. Defaults to 1.
            Use ``axis=1`` for design matrices (features along columns) and ``axis=0`` for
            coefficient arrays (features along rows). All other dimensions are preserved.

        Raises
        ------
        ValueError
            If the shape of ``x`` along the specified axis does not match ``self.n_output_features``.

        Returns
        -------
        dict
            A dictionary where:
            - **Key**: Label of the basis.
            - **Value**: the array reshaped to: ``(..., n_inputs, n_basis_funcs, ...)``
        """
        if x.shape[axis] != self.n_output_features:
            raise ValueError(
                "`x.shape[axis]` does not match the expected number of features."
                f" `x.shape[axis] == {x.shape[axis]}`, while the expected number "
                f"of features is {self.n_output_features}"
            )

        # Get the slice dictionary based on predefined feature slicing
        slice_dict = self._get_feature_slicing(split_by_input=False)[0]

        # Helper function to build index tuples for each slice
        def build_index_tuple(slice_obj, axis: int, ndim: int):
            """Create an index tuple to apply a slice on the given axis."""
            index = [slice(None)] * ndim  # Initialize index for all dimensions
            index[axis] = slice_obj  # Replace the axis with the slice object
            return tuple(index)

        # Get the dict for slicing the correct axis
        index_dict = jax.tree_util.tree_map(
            lambda sl: build_index_tuple(sl, axis, x.ndim), slice_dict
        )

        # Custom leaf function to identify index tuples as leaves
        def is_leaf(val):
            # Check if it's a tuple, length matches ndim, and all elements are slice objects
            if isinstance(val, tuple) and len(val) == x.ndim:
                return all(isinstance(v, slice) for v in val)
            return False

        # Apply the slicing using the custom leaf function
        out = jax.tree_util.tree_map(lambda sl: x[sl], index_dict, is_leaf=is_leaf)

        # reshape the arrays to spilt by n_basis_input
        reshaped_out = dict()
        for i, vals in enumerate(out.items()):
            key, val = vals
            shape = list(val.shape)
            reshaped_out[key] = val.reshape(
                shape[:axis] + [self._n_basis_input[i], -1] + shape[axis + 1 :]
            )
        return reshaped_out

    def _check_input_shape_consistency(self, x: NDArray):
        """Check input consistency across calls."""
        # remove sample axis
        shape = x.shape[1:]
        if self._input_shape is not None and self._input_shape != shape:
            expected_shape_str = "(n_samples, " + f"{self._input_shape}"[1:]
            expected_shape_str = expected_shape_str.replace(",)", ")")
            raise ValueError(
                f"Input shape mismatch detected.\n\n"
                f"The basis `{self.__class__.__name__}` with label '{self.label}' expects inputs with "
                f"a consistent shape (excluding the sample axis). Specifically, the shape should be:\n"
                f"  Expected: {expected_shape_str}\n"
                f"  But got:  {x.shape}.\n\n"
                "Note: The number of samples (`n_samples`) can vary between calls of `compute_features`, "
                "but all other dimensions must remain the same. If you need to process inputs with a "
                "different shape, please create a new basis instance."
            )

    def _set_num_output_features(self, *xi: NDArray) -> Basis:
        """
        Pre-compute the number of inputs and output features.

        This function computes the number of inputs that are provided to the basis and uses
        that number, and the n_basis_funcs to calculate the number of output features that
        ``self.compute_features`` will return. These quantities and the input shape (excluding the sample axis)
        are stored in ``self._n_basis_input`` and ``self._n_output_features``, and ``self._input_shape``
        respectively.

        Parameters
        ----------
        xi:
            The input arrays.

        Returns
        -------
        :
            The basis itself, for chaining.

        Raises
        ------
        ValueError:
            If the number of inputs do not match ``self._n_basis_input``, if  ``self._n_basis_input`` was
            not None.

        Notes
        -----
        Once a ``compute_features`` is called, we enforce that for all subsequent calls of the method,
        the input that the basis receives preserves the shape of all axes, except for the sample axis.
        This condition guarantees the consistency of the feature axis, and therefore that
         ``self.split_by_feature`` behaves appropriately.

        """
        # Check that the input shape matches expectation
        # Note that this method is reimplemented in AdditiveBasis and MultiplicativeBasis
        # so we can assume that len(xi) == 1
        xi = xi[0]
        self._check_input_shape_consistency(xi)

        # remove sample axis (samples are allowed to vary)
        shape = xi.shape[1:]

        self._input_shape = shape

        # remove sample axis & get the total input number
        n_inputs = (1,) if xi.ndim == 1 else (np.prod(shape),)

        self._n_basis_input = n_inputs
        self._n_output_features = self.n_basis_funcs * self._n_basis_input[0]
        return self


class AdditiveBasis(Basis):
    """
    Class representing the addition of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to add.
    basis2 :
        Second basis object to add.

    Attributes
    ----------
    n_basis_funcs : int
        Number of basis functions.

    Examples
    --------
    >>> # Generate sample data
    >>> import numpy as np
    >>> import nemos as nmo
    >>> X = np.random.normal(size=(30, 2))

    >>> # define two basis objects and add them
    >>> basis_1 = nmo.basis.BSplineEval(10)
    >>> basis_2 = nmo.basis.RaisedCosineLinearEval(15)
    >>> additive_basis = basis_1 + basis_2

    >>> # can add another basis to the AdditiveBasis object
    >>> X = np.random.normal(size=(30, 3))
    >>> basis_3 = nmo.basis.RaisedCosineLogEval(100)
    >>> additive_basis_2 = additive_basis + basis_3
    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs + basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs, mode="eval")
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._n_basis_input = None
        self._n_output_features = None
        self._label = "(" + basis1.label + " + " + basis2.label + ")"
        self._basis1 = basis1
        self._basis2 = basis2

    def _set_num_output_features(self, *xi: NDArray) -> Basis:
        self._n_basis_input = (
            *self._basis1._set_num_output_features(
                *xi[: self._basis1._n_input_dimensionality]
            )._n_basis_input,
            *self._basis2._set_num_output_features(
                *xi[self._basis1._n_input_dimensionality :]
            )._n_basis_input,
        )
        self._n_output_features = (
            self._basis1.n_output_features + self._basis2.n_output_features
        )
        return self

    def _check_n_basis_min(self) -> None:
        pass

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def _evaluate(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Examples
        --------
        >>> # Generate sample data
        >>> import numpy as np
        >>> import nemos as nmo
        >>> x, y = np.random.normal(size=(2, 30))

        >>> # define two basis objects and add them
        >>> basis_1 = nmo.basis.BSplineEval(10)
        >>> basis_2 = nmo.basis.RaisedCosineLinearEval(15)
        >>> additive_basis = basis_1 + basis_2

        >>> # call the basis.
        >>> out = additive_basis._evaluate(x, y)

        """
        X = np.hstack(
            (
                self._basis1._evaluate(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2._evaluate(*xi[self._basis1._n_input_dimensionality :]),
            )
        )
        return X

    @add_docstring("compute_features", Basis)
    def compute_features(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        r"""
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval, RaisedCosineLogConv
        >>> from nemos.glm import GLM
        >>> basis1 = BSplineEval(n_basis_funcs=5, label="one_input")
        >>> basis2 = RaisedCosineLogConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> basis_add = basis1 + basis2
        >>> X_multi = basis_add.compute_features(np.random.randn(20), np.random.randn(20, 2))
        >>> print(X_multi.shape) # num_features: 17 = 5 + 2*6
        (20, 17)

        """
        return super().compute_features(*xi)

    def _compute_features(self, *xi: NDArray | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Compute features for added bases and concatenate.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The features, shape (n_samples, n_basis_funcs)

        """
        # the numpy conversion is important, there is some in-place
        # array modification in basis.
        hstack_pynapple = support_pynapple(conv_type="numpy")(np.hstack)
        X = hstack_pynapple(
            (
                self._basis1._compute_features(
                    *xi[: self._basis1._n_input_dimensionality]
                ),
                self._basis2._compute_features(
                    *xi[self._basis1._n_input_dimensionality :]
                ),
            ),
        )
        return X

    def _set_kernel(self, *xi: ArrayLike) -> Basis:
        """Call fit on the added basis.

        If any of the added basis is in "conv" mode, it will prepare its kernels for the convolution.

        Parameters
        ----------
        *xi:
            The sample inputs. Unused, necessary to conform to ``scikit-learn`` API.

        Returns
        -------
        :
            The AdditiveBasis ready to be evaluated.
        """
        self._basis1._set_kernel()
        self._basis2._set_kernel()
        return self

    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        r"""
        Decompose an array along a specified axis into sub-arrays based on the basis components.

        This function takes an array (e.g., a design matrix or model coefficients) and splits it along
        a designated axis. Each split corresponds to a different additive component of the basis,
        preserving all dimensions except the specified axis.

        **How It Works:**

        Suppose the basis is made up of **m components**, each with :math:`b_i` basis functions and :math:`n_i` inputs.
        The total number of features, :math:`N`, is calculated as:

        .. math::
            N = b_1 \cdot n_1 + b_2 \cdot n_2 + \ldots + b_m \cdot n_m

        This method splits any axis of length :math:`N` into sub-arrays, one for each basis component.

        The sub-array for the i-th basis component is reshaped into dimensions
        :math:`(n_i, b_i)`.

        For example, if the array shape is :math:`(1, 2, N, 4, 5)`, then each split sub-array will have shape:

        .. math::
            (1, 2, n_i, b_i, 4, 5)

        where:

        - :math:`n_i` represents the number of inputs associated with the i-th component,
        - :math:`b_i` represents the number of basis functions in that component.

        The specified axis (``axis``) determines where the split occurs, and all other dimensions
        remain unchanged. See the example section below for the most common use cases.

        Parameters
        ----------
        x :
            The input array to be split, representing concatenated features, coefficients,
            or other data. The shape of ``x`` along the specified axis must match the total
            number of features generated by the basis, i.e., ``self.n_output_features``.

            **Examples:**
            - For a design matrix: ``(n_samples, total_n_features)``
            - For model coefficients: ``(total_n_features,)`` or ``(total_n_features, n_neurons)``.

        axis : int, optional
            The axis along which to split the features. Defaults to 1.
            Use ``axis=1`` for design matrices (features along columns) and ``axis=0`` for
            coefficient arrays (features along rows). All other dimensions are preserved.

        Raises
        ------
        ValueError
            If the shape of ``x`` along the specified axis does not match ``self.n_output_features``.

        Returns
        -------
        dict
            A dictionary where:
            - **Keys**: Labels of the additive basis components.
            - **Values**: Sub-arrays corresponding to each component. Each sub-array has the shape:

            .. math::
                (..., n_i, b_i, ...)

            - ``n_i``: The number of inputs processed by the i-th basis component.
            - ``b_i``: The number of basis functions for the i-th basis component.

            These sub-arrays are reshaped along the specified axis, with all other dimensions
            remaining the same.

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineConv
        >>> from nemos.glm import GLM
        >>> # Define an additive basis
        >>> basis = (
        ...     BSplineConv(n_basis_funcs=5, window_size=10, label="feature_1") +
        ...     BSplineConv(n_basis_funcs=6, window_size=10, label="feature_2")
        ... )
        >>> # Generate a sample input array and compute features
        >>> x1, x2 = np.random.randn(20), np.random.randn(20)
        >>> X = basis.compute_features(x1, x2)
        >>> # Split the feature matrix along axis 1
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        feature_1: shape (20, 1, 5)
        feature_2: shape (20, 1, 6)
        >>> # If one of the basis components accepts multiple inputs, the resulting dictionary will be nested:
        >>> multi_input_basis = BSplineConv(n_basis_funcs=6, window_size=10,
        ... label="multi_input")
        >>> X_multi = multi_input_basis.compute_features(np.random.randn(20, 2))
        >>> split_features_multi = multi_input_basis.split_by_feature(X_multi, axis=1)
        >>> for feature, sub_dict in split_features_multi.items():
        ...        print(f"{feature}, shape {sub_dict.shape}")
        multi_input, shape (20, 2, 6)
        >>> # the method can be used to decompose the glm coefficients in the various features
        >>> counts = np.random.poisson(size=20)
        >>> model = GLM().fit(X, counts)
        >>> split_coef = basis.split_by_feature(model.coef_, axis=0)
        >>> for feature, coef in split_coef.items():
        ...     print(f"{feature}: shape {coef.shape}")
        feature_1: shape (1, 5)
        feature_2: shape (1, 6)

        """
        return super().split_by_feature(x, axis=axis)

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        The i-th axis of the grid will be sampled with ``n_samples[i]`` equi-spaced points.
        The method uses numpy.meshgrid with ``indexing="ij"``, returning matrix indexing
        instead of the default cartesian indexing, see Notes.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of points in the uniformly spaced grid. The length of
            n_samples must equal the number of combined bases.

        Returns
        -------
        *Xs :
            A tuple of arrays containing the meshgrid values, one element for each of the n dimension of the grid,
            where n equals to the number of inputs.
            The size of ``Xs[i]`` is ``(n_samples[0], ... , n_samples[n])``.
        Y :
            The basis function evaluated at the samples,
            shape ``(n_samples[0], ... , n_samples[n], number of basis)``.

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs or if the number of inputs doesn't match what
            the Basis object requires.
        ValueError
            If one of the n_samples is <= 0.

        Notes
        -----
        Setting ``indexing = 'ij'`` returns a meshgrid with matrix indexing. In the N-D case with inputs of size
        :math:`M_1,...,M_N`, outputs are of shape :math:`(M_1, M_2, M_3, ....,M_N)`.
        This differs from the numpy.meshgrid default, which uses Cartesian indexing.
        For the same input, Cartesian indexing would return an output of shape :math:`(M_2, M_1, M_3, ....,M_N)`.

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo

        >>> # define two basis objects and add them
        >>> basis_1 = nmo.basis.BSplineEval(10)
        >>> basis_2 = nmo.basis.RaisedCosineLinearEval(15)
        >>> additive_basis = basis_1 + basis_2

        >>> # evaluate on a grid of 10 x 10 equi-spaced samples
        >>> X, Y, Z = additive_basis.evaluate_on_grid(10, 10)

        """
        return super().evaluate_on_grid(*n_samples)

    def _get_feature_slicing(
        self,
        n_inputs: Optional[tuple] = None,
        start_slice: Optional[int] = None,
        split_by_input: bool = True,
    ) -> Tuple[dict, int]:
        """
        Calculate and return the slicing for features based on the input structure.

        This method determines how to slice the features for different basis types.

        Parameters
        ----------
        n_inputs :
            The number of input basis for each component, by default it uses ``self._n_basis_input``.
        start_slice :
            The starting index for slicing, by default it starts from 0.
        split_by_input :
            Flag indicating whether to split the slicing by individual inputs or not.
            If ``False``, a single slice is generated for all inputs.

        Returns
        -------
        split_dict :
            Dictionary with keys as labels and values as slices representing
            the slicing for each input or additive component, if split_by_input equals to
            True or False respectively.
        start_slice :
            The updated starting index after slicing.

        See Also
        --------
        _get_default_slicing : Handles default slicing logic.
        _merge_slicing_dicts : Merges multiple slicing dictionaries, handling keys conflicts.
        """
        # Set default values for n_inputs and start_slice if not provided
        n_inputs = n_inputs or self._n_basis_input
        start_slice = start_slice or 0

        # If the instance is of AdditiveBasis type, handle slicing for the additive components

        split_dict, start_slice = self._basis1._get_feature_slicing(
            n_inputs[: len(self._basis1._n_basis_input)],
            start_slice,
            split_by_input=split_by_input,
        )
        sp2, start_slice = self._basis2._get_feature_slicing(
            n_inputs[len(self._basis1._n_basis_input) :],
            start_slice,
            split_by_input=split_by_input,
        )
        split_dict = self._merge_slicing_dicts(split_dict, sp2)
        return split_dict, start_slice

    def _merge_slicing_dicts(self, dict1: dict, dict2: dict) -> dict:
        """Merge two slicing dictionaries, handling key conflicts."""
        for key, val in dict2.items():
            if key in dict1:
                new_key = self._generate_unique_key(dict1, key)
                dict1[new_key] = val
            else:
                dict1[key] = val
        return dict1


class MultiplicativeBasis(Basis):
    """
    Class representing the multiplication (external product) of two Basis objects.

    Parameters
    ----------
    basis1 :
        First basis object to multiply.
    basis2 :
        Second basis object to multiply.

    Attributes
    ----------
    n_basis_funcs :
        Number of basis functions.

    Examples
    --------
    >>> # Generate sample data
    >>> import numpy as np
    >>> import nemos as nmo
    >>> X = np.random.normal(size=(30, 3))

    >>> # define two basis and multiply
    >>> basis_1 = nmo.basis.BSplineEval(10)
    >>> basis_2 = nmo.basis.RaisedCosineLinearEval(15)
    >>> multiplicative_basis = basis_1 * basis_2

    >>> # Can multiply or add another basis to the AdditiveBasis object
    >>> # This will cause the number of output features of the result basis to grow accordingly
    >>> basis_3 = nmo.basis.RaisedCosineLogEval(100)
    >>> multiplicative_basis_2 = multiplicative_basis * basis_3
    """

    def __init__(self, basis1: Basis, basis2: Basis) -> None:
        self.n_basis_funcs = basis1.n_basis_funcs * basis2.n_basis_funcs
        super().__init__(self.n_basis_funcs, mode="eval")
        self._n_input_dimensionality = (
            basis1._n_input_dimensionality + basis2._n_input_dimensionality
        )
        self._n_basis_input = None
        self._n_output_features = None
        self._label = "(" + basis1.label + " * " + basis2.label + ")"
        self._basis1 = basis1
        self._basis2 = basis2

    def _check_n_basis_min(self) -> None:
        pass

    def _set_kernel(self, *xi: NDArray) -> Basis:
        """Call fit on the multiplied basis.

        If any of the added basis is in "conv" mode, it will prepare its kernels for the convolution.

        Parameters
        ----------
        *xi:
            The sample inputs. Unused, necessary to conform to ``scikit-learn`` API.

        Returns
        -------
        :
            The MultiplicativeBasis ready to be evaluated.
        """
        self._basis1._set_kernel()
        self._basis2._set_kernel()
        return self

    @support_pynapple(conv_type="numpy")
    @check_transform_input
    @check_one_dimensional
    def _evaluate(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Evaluate the basis at the input samples.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The basis function evaluated at the samples, shape (n_samples, n_basis_funcs)

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> mult_basis = nmo.basis.BSplineEval(5) * nmo.basis.RaisedCosineLinearEval(6)
        >>> x, y = np.random.randn(2, 30)
        >>> X = mult_basis._evaluate(x, y)
        """
        X = np.asarray(
            row_wise_kron(
                self._basis1._evaluate(*xi[: self._basis1._n_input_dimensionality]),
                self._basis2._evaluate(*xi[self._basis1._n_input_dimensionality :]),
                transpose=False,
            )
        )
        return X

    def _compute_features(self, *xi: NDArray | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Compute the features for the multiplied bases, and compute their outer product.

        Parameters
        ----------
        xi[0], ..., xi[n] : (n_samples,)
            Tuple of input samples, each with the same number of samples. The
            number of input arrays must equal the number of combined bases.

        Returns
        -------
        :
            The  features, shape (n_samples, n_basis_funcs)

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> mult_basis = nmo.basis.BSplineEval(5) * nmo.basis.RaisedCosineLinearEval(6)
        >>> x, y = np.random.randn(2, 30)
        >>> X = mult_basis.compute_features(x, y)
        """
        kron = support_pynapple(conv_type="numpy")(row_wise_kron)
        X = kron(
            self._basis1._compute_features(*xi[: self._basis1._n_input_dimensionality]),
            self._basis2._compute_features(*xi[self._basis1._n_input_dimensionality :]),
            transpose=False,
        )
        return X

    def _set_num_output_features(self, *xi: NDArray) -> Basis:
        self._n_basis_input = (
            *self._basis1._set_num_output_features(
                *xi[: self._basis1._n_input_dimensionality]
            )._n_basis_input,
            *self._basis2._set_num_output_features(
                *xi[self._basis1._n_input_dimensionality :]
            )._n_basis_input,
        )
        self._n_output_features = (
            self._basis1.n_output_features * self._basis2.n_output_features
        )
        return self

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Evaluate the basis set on a grid of equi-spaced sample points.

        The i-th axis of the grid will be sampled with ``n_samples[i]`` equi-spaced points.
        The method uses numpy.meshgrid with ``indexing="ij"``, returning matrix indexing
        instead of the default cartesian indexing, see Notes.

        Parameters
        ----------
        n_samples[0],...,n_samples[n]
            The number of points in the uniformly spaced grid. The length of
            n_samples must equal the number of combined bases.

        Returns
        -------
        *Xs :
            A tuple of arrays containing the meshgrid values, one element for each of the n dimension of the grid,
            where n equals to the number of inputs.
            The size of ``Xs[i]`` is ``(n_samples[0], ... , n_samples[n])``.
        Y :
            The basis function evaluated at the samples,
            shape ``(n_samples[0], ... , n_samples[n], number of basis)``.

        Raises
        ------
        ValueError
            If the time point number is inconsistent between inputs or if the number of inputs doesn't match what
            the Basis object requires.
        ValueError
            If one of the n_samples is <= 0.

        Notes
        -----
        Setting ``indexing = 'ij'`` returns a meshgrid with matrix indexing. In the N-D case with inputs of size
        :math:`M_1,...,M_N`, outputs are of shape :math:`(M_1, M_2, M_3, ....,M_N)`.
        This differs from the numpy.meshgrid default, which uses Cartesian indexing.
        For the same input, Cartesian indexing would return an output of shape :math:`(M_2, M_1, M_3, ....,M_N)`.

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> mult_basis = nmo.basis.BSplineEval(4) * nmo.basis.RaisedCosineLinearEval(5)
        >>> X, Y, Z = mult_basis.evaluate_on_grid(10, 10)
        """
        return super().evaluate_on_grid(*n_samples)

    @add_docstring("compute_features", Basis)
    def compute_features(self, *xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval, RaisedCosineLogConv
        >>> from nemos.glm import GLM
        >>> basis1 = BSplineEval(n_basis_funcs=5, label="one_input")
        >>> basis2 = RaisedCosineLogConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> basis_mul = basis1 * basis2
        >>> X_multi = basis_mul.compute_features(np.random.randn(20), np.random.randn(20, 2))
        >>> print(X_multi.shape) # num_features: 60 = 5 * 2 * 6
        (20, 60)

        """
        return super().compute_features(*xi)

    @add_docstring("split_by_feature", Basis)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import BSplineEval, RaisedCosineLogConv
        >>> from nemos.glm import GLM
        >>> basis1 = BSplineEval(n_basis_funcs=5, label="one_input")
        >>> basis2 = RaisedCosineLogConv(n_basis_funcs=6, window_size=10, label="two_inputs")
        >>> basis_mul = basis1 * basis2
        >>> X_multi = basis_mul.compute_features(np.random.randn(20), np.random.randn(20, 2))
        >>> print(X_multi.shape) # num_features: 60 = 5 * 2 * 6
        (20, 60)

        >>> # The multiplicative basis is a single 2D component.
        >>> split_features = basis_mul.split_by_feature(X_multi, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        (one_input * two_inputs): shape (20, 1, 60)

        """
        return super().split_by_feature(x, axis=axis)
