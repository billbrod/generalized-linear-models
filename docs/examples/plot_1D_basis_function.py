# -*- coding: utf-8 -*-

"""
# One-Dimensional Basis

## Defining a 1D Basis Object

We'll start by defining a 1D basis function object of the type `MSplineBasis`.
The hyperparameters required to initialize this class are:

- The number of basis functions, which should be a positive integer.
- The order of the spline, which should be an integer greater than 1.
"""

import matplotlib.pylab as plt
import numpy as np

import nemos as nmo

# Initialize hyperparameters
order = 4
n_basis = 10

# Define the 1D basis function object
mspline_basis = nmo.basis.MSplineBasis(n_basis_funcs=n_basis, order=order)

# %%
# Evaluating a Basis
# ------------------------------------
# The `Basis.evaluate` method enables us to evaluate a basis function. For `SplineBasis`, the domain is defined by
# the samples that we input to the `evaluate` method. This results in an equi-spaced set of knots, which spans
# the range from the smallest to the largest sample. These knots are then used to construct a uniformly spaced basis.

# Generate an array of sample points
samples = np.random.uniform(0, 10, size=1000)

# Evaluate the basis at the sample points
eval_basis = mspline_basis.evaluate(samples)

# Output information about the evaluated basis
print(f"Evaluated M-spline of order {order} with {eval_basis.shape[1]} "
      f"basis element and {eval_basis.shape[0]} samples.")

# %%
# Plotting the Basis Function Elements:
# --------------------------------------
# We suggest visualizing the basis post-instantiation by evaluating each element on a set of equi-spaced sample points
# and then plotting the result. The method `Basis.evaluate_on_grid` is designed for this, as it generates and returns
# the equi-spaced samples along with the evaluated basis functions. The benefits of using Basis.evaluate_on_grid become
# particularly evident when working with multidimensional basis functions. You can find more details and visual
# examples in the [2D basis elements plotting section](../plot_2D_basis_function/#2d-basis-elements-plotting).

# Call evaluate on grid on 100 sample points to generate samples and evaluate the basis at those samples
n_samples = 100
equispaced_samples, eval_basis = mspline_basis.evaluate_on_grid(n_samples)

# Plot each basis element
plt.figure()
plt.title(f"M-spline basis with {eval_basis.shape[1]} elements\nevaluated at {eval_basis.shape[0]} sample points")
plt.plot(equispaced_samples, eval_basis)
plt.show()

# %%
# Other Basis Types
# -----------------
# Each basis type may necessitate specific hyperparameters for instantiation. For a comprehensive description,
# please refer to the  [Code References](../../../reference/nemos/basis). After instantiation, all classes
# share the same syntax for basis evaluation.
#
# ### The Log-Spaced Raised Cosine Basis
# The following is an example of how to instantiate and  evaluate a log-spaced cosine raised function basis.

# Instantiate the basis noting that the `RaisedCosineBasisLog` does not require an `order` parameter
raised_cosine_log = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=10, width=1.5, time_scaling=50)

# Evaluate the raised cosine basis at the equi-spaced sample points
# (same method in all Basis elements)
samples, eval_basis = raised_cosine_log.evaluate_on_grid(100)

# Plot the evaluated log-spaced raised cosine basis
plt.figure()
plt.title(f"Log-spaced Raised Cosine basis with {eval_basis.shape[1]} elements")
plt.plot(samples, eval_basis)
plt.show()

# %%
# ### The Fourier Basis
# Another type of basis available is the Fourier Basis. Fourier basis are ideal to capture periodic and
# quasi-periodic patterns. Such oscillatory, rhythmic behavior is a common signature of many neural signals.
# Additionally, the Fourier basis has the advantage of being orthogonal, which simplifies the estimation and
# interpretation of the model parameters, each of which will represent the relative contribution of a specific
# oscillation frequency to the overall signal.
#
# A Fourier basis can be instantiated with the following syntax:
# the user can provide the maximum frequency of the cosine and negative
# sine pairs by setting the `max_freq` parameter.
# The sinusoidal basis elements will have frequencies from 0 to `max_freq`.


fourier_basis = nmo.basis.FourierBasis(max_freq=3)

# evaluate on equi-spaced samples
samples, eval_basis = fourier_basis.evaluate_on_grid(1000)

# plot the `sin` and `cos` separately
plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.title("Cos")
plt.plot(samples, eval_basis[:, :4])
plt.subplot(122)
plt.title("Sin")
plt.plot(samples, eval_basis[:, 4:])
plt.tight_layout()

# %%
# ## Fourier Basis Convolution and Fourier Transform
# The Fourier transform of a signal $ s(t) $ restricted to a temporal window $ [t_0,\;t_1] $ is
# $$ \\hat{x}(\\omega) = \\int_{t_0}^{t_1} s(\\tau) e^{-j\\omega \\tau} d\\tau. $$
# where $ e^{-j\\omega \\tau} = \\cos(\\omega \\tau) - j \\sin (\\omega \\tau) $.
#
# When computing the cross-correlation of a signal with the Fourier basis functions,
# we essentially measure how well the signal correlates with sinusoids of different frequencies,
# within a specified temporal window. This process mirrors the operation performed by the Fourier transform.
# Therefore, it becomes clear that computing the cross-correlation of a signal with the Fourier basis defined here
# is equivalent to computing the discrete Fourier transform on a sliding window of the same size
# as that of the basis.


n_samples = 1000
max_freq = 20

# define a signal
signal = np.random.normal(size=n_samples)

# evaluate the basis
_, eval_basis = nmo.basis.FourierBasis(max_freq=max_freq).evaluate_on_grid(n_samples)

# compute the cross-corr with the signal and the basis
# Note that we are inverting the time axis of the basis because we are aiming
# for a cross-correlation, while np.convolve compute a convolution which would flip the time axis.
xcorr = np.array(
    [
        np.convolve(eval_basis[::-1, k], signal, mode="valid")[0]
        for k in range(2 * max_freq + 1)
    ]
)

# compute the power (add back sin(0 * t) = 0)
fft_complex = np.fft.fft(signal)
fft_amplitude = np.abs(fft_complex[:max_freq + 1])
fft_phase = np.angle(fft_complex[:max_freq + 1])
# compute the phase and amplitude from the convolution
xcorr_phase = np.arctan2(np.hstack([[0], xcorr[max_freq+1:]]), xcorr[:max_freq+1])
xcorr_aplitude = np.sqrt(xcorr[:max_freq+1] ** 2 + np.hstack([[0], xcorr[max_freq+1:]]) ** 2)

fig, ax = plt.subplots(1, 2)
ax[0].set_aspect("equal")
ax[0].set_title("Signal amplitude")
ax[0].scatter(fft_amplitude, xcorr_aplitude)
ax[0].set_xlabel("FFT")
ax[0].set_ylabel("cross-correlation")

ax[1].set_aspect("equal")
ax[1].set_title("Signal phase")
ax[1].scatter(fft_phase, xcorr_phase)
ax[1].set_xlabel("FFT")
ax[1].set_ylabel("cross-correlation")
plt.tight_layout()

print(f"Max Error Amplitude: {np.abs(fft_amplitude - xcorr_aplitude).max()}")
print(f"Max Error Phase: {np.abs(fft_phase - xcorr_phase).max()}")
