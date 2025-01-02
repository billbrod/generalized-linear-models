---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Basis Functions

## Introduction

Collection of (usually non-linear) maps. 

Used for parametrize a non-linear function as a linear (weighted sum) combination of simple elements.

Why this is useful: 
1. Capture non-linear effects. 
2. Keep the parameter fitting simple. (convex GLM, single solution).

For a list of the available bases, check out the [package documentation](table_basis).

## Basis in NeMoS 

### Define A 1D Basis

In NeMoS, you can define a basis with the following syntax,

```{code-cell} ipython3
import nemos as nmo

# construct a BSpline basis with 10 elements
bas = nmo.basis.BSplineEval(n_basis_funcs=10)
```

Let's plot each basis element. We provide the convenience function `evaluate_on_grid`, creates a grid of equi-spaced 
sample points, and evaluate each basis element at the samples.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# evaluate the basis on 100 samples
x, basis_elements = bas.evaluate_on_grid(100)

print(basis_elements.shape)

plt.plot(x, y)
```

### Changing Basis Parameters

A basis object is characterized by set of (hyper-)parameters. These parameters can be provided at object definition, most of them have defaults, while some (as `n_basis_funcs` in the example above) must be provided explicitly.

You can retrieve all the available parameters with the `get_params` method, that returns a dictionary of parameters.

```{code-cell} ipython3

# get a dictionary storing the parameters
bas.get_params()
```

You cam set the parameters by:

1. Calling the `set_params` method.
2. Setting the attribute directly (`bas.attribute_name = new_value`).


Set the  spline `order` to 3 with `set_params` and the number of basis to 8 by changing the attribute directly.

```{code-cell} ipython3

# change basis parameters
bas.set_params(order=3)
bas.n_basis_funcs = 8

# check the new settings
bas.get_params()
```

Plot the basis under different configurations.
 
```{code-cell} ipython3
```

All the parameters except `label` can be modified after the basis is instantiated. Those parameters define the functional form of the elements and their number, and constitutes the nobs that one can play with when modeling a neural response. 

On the other hand,`label` should a descriptive ID for the basis, indicating the type of variable being modelled. 

An attempt to change the `label` will result in an error:

```{code-cell} ipython3

bas = nmo.basis.BSplineEval(n_basis_funcs=10, label="my label")

try:
    bas.label = "new label"
except AttributeError as e:
    print(repr(e))
```

### Approximating 1D Non-Linearities

By weighting and summing together the basis elements, one can approximate well smoothly changing non-linearities.

Let's see how this works.

Here I am evaluating two smoothly changing functions (a gaussian and a logarithm), and one that changing abruptly
(a step function) over a set of equi-spaced points. For each function I am providing a set of 10 weights, one per basis element.

```{code-cell} ipython3

# define a basis
bas = nmo.basis.BSplineEval(n_basis_funcs=10)

gauss = np.exp(-np.linspace(-2, 2, 100)**2)
weights_gauss = [0.018, 0.034, 0.082, 0.441, 1.02 , 1.02 , 0.441, 0.082, 0.034, 0.018]

log = np.log(0.1 + np.linspace(0, 1, 100))
weights_log = [-2.299, -1.852, -1.367, -0.93 , -0.626, -0.391, -0.201, -0.04 , 0.052,  0.095]


step = np.floor(np.arange(100) / 50)
weights_step = [-0.025,  0.081, -0.144,  0.196, -0.404,  1.404,  0.804,  1.144, 0.919,  1.025]

```

If you multipy each basis element by the corresponding weight, and sum over the elements, we are approximating the original non-linearity.

As an exercise, compute these weighted sums and store the results in `approx_gauss`, `approx_log`, `approx_step` respectively.

```{code-cell} ipython3

# weight and sum the basis elements
approx_gauss = np.sum(basis_elements * weights_gauss, axis=1)
approx_log = np.sum(basis_elements * weights_log, axis=1)
approx_step = np.sum(basis_elements * weights_step, axis=1)

```

Plot the original and approximated non-linearities.

```{code-cell} ipython3

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.plot(gauss, lw=2)
plt.plot(approx_gauss, ls='--')

plt.subplot(132)
plt.plot(log, lw=2)
plt.plot(approx_log, ls='--')

plt.subplot(133)
plt.plot(step, lw=2)
plt.plot(approx_step, ls='--')
plt.tight_layout()
```

### Basis for Angular Variables

To model periodic variables (angles) you can define a cyclic-BSpline. 

Define, evaluate and plot a `CyclicBSplineEval` object:


```{code-cell} ipython3

bas_cyclic = nmo.basis.CyclicBSplineEval(10)

x, cyclic_basis_element = bas_cyclic.evaluate_on_grid(100)

plt.plot(x, cyclic_basis_element)
```

### Multi-dimensional Bases

With the same approach, we can approximate functions of higher dimension. This can be useful when we want to characterize the 
neural response to a 2D variable such as the position of an animal in an arena. 

In NeMoS, you can combine two or more 1D bases to obtain a multidimensional bases using the multiplication operator.

... add code

The drawback is that the number of parameters grows exponentially with the dimensionality, so be aware of that!


### Non-linear Mapping And Linear Temporal Effects

So far, we have shown how we can approximate a non-linear function with a basis and a set weights but we did not talk about how to use the approximation in a model.

Here we will show two common applications,

### Basis Pynapple Time-Series


### Composite Bases







