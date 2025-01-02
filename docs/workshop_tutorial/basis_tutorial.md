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

### Approximating 1D Non-Linearities

By weighting and summing this elements, we can approximate well smoothly changing non-linearities.

Let's see how this works.

Here I am evaluating two smoothly changing functions (a gaussian and a logarithm), and one that changing abruptly
(a step function) over a set of equi-spaced points. For each function I am providing a set of 10 weights, one per basis element.

```{code-cell} ipython3

gauss = np.exp(-np.linspace(-2, 2, 100)**2)
weights_gauss = [0.018, 0.034, 0.082, 0.441, 1.02 , 1.02 , 0.441, 0.082, 0.034, 0.018]

log = np.log(0.1 + np.linspace(0, 1, 100))
weights_log = [-2.299, -1.852, -1.367, -0.93 , -0.626, -0.391, -0.201, -0.04 , 0.052,  0.095]


step = np.floor(np.arange(100) / 50)
weights_step = [-0.025,  0.081, -0.144,  0.196, -0.404,  1.404,  0.804,  1.144, 0.919,  1.025]

```

If you multipy each basis element by the corresponding weight, and sum over the elements, you can create approximate the original non-linearity.

As an exercise, compute these weighted sums and call the results `approx_gauss`, `approx_log`, `approx_step` respectively.

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
