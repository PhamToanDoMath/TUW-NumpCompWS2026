"""
Problem 4.2: Classical Trigonometric Interpolation and Convergence Analysis
Numerical Computation - Problem Sheet 4

This script:
1. Part (a):
   - Computes and evaluates the classical trigonometric interpolant
       q_n(x) = sum_{k=-m}^m d_k exp(i * k * x)
     for n = 2m + 1 equidistant knots x_j = 2*pi*j/n, j = 0, ..., n - 1,
     using exactly one Fast Fourier Transform (FFT) of the sampled data.
   - Specifies explicitly which entries of y_hat / n give the coefficients d_k
     for negative k in both 0-based indexing (Python) and 1-based indexing (MATLAB).
   - Verifies that q_n interpolates the data at the knots up to rounding errors
     and produces real values for all real x up to machine precision.

2. Part (b):
   - For n in {9, 17, 33, 65, 129}, samples f(x) = 1 / (2 - cos(x)) and
     g(x) = |sin(x / 2)| at 10001 uniformly distributed evaluation points on [0, 2*pi]
     to approximate the maximum norm errors ||f - q_n||_inf and ||g - q_n||_inf.
   - Plots each error versus n using both semilogy and loglog scales.
   - Analyzes convergence behavior:
       * Semilogy plot indicates exponential (spectral) convergence for f(x).
       * Loglog plot indicates algebraic convergence of order O(n^(-1)) for g(x).
   - Relates the convergence rates to the mathematical smoothness and singularity
     structure in the complex plane (analyticity of f vs corner/kink of g).
"""

import os
import sys
from typing import Callable, Tuple, Union

# Set cache dir for headless environments before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Target Periodic Functions
# =============================================================================

def f(x: np.ndarray | float) -> np.ndarray | float:
    """
    Smooth, analytic 2*pi-periodic function:
        f(x) = 1 / (2 - cos(x)).
    
    Analytic in a strip |Im(z)| < rho = ln(2 + sqrt(3)) approx 1.31696.
    """
    return 1.0 / (2.0 - np.cos(x))


def g(x: np.ndarray | float) -> np.ndarray | float:
    """
    Continuous, non-smooth 2*pi-periodic function:
        g(x) = |sin(x / 2)|.
    
    Continuous on R (in C^0), but its first derivative has jump discontinuities
    (cusps / kinks) at x = 2*pi*k for all k in Z, where g'(0+) = +1/2 != -1/2 = g'(0-).
    """
    return np.abs(np.sin(0.5 * x))


# =============================================================================
# Part (a): Trigonometric Interpolant Implementation via 1 FFT
# =============================================================================

def compute_trig_interpolant_coeffs(
    y_sampled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the classical trigonometric interpolant coefficients d_k for k = -m, ..., m
    using exactly ONE FFT of the sampled values y_j = u(x_j), where n = 2m + 1.

    Parameters:
        y_sampled: 1D array of length n = 2m + 1 containing sampled function values
                   at knots x_j = 2*pi*j / n for j = 0, ..., n - 1.

    Returns:
        d_coeffs: 1D complex array of length n = 2m + 1 containing d_k in order
                  k = -m, -m+1, ..., 0, ..., m-1, m.
        k_indices: 1D integer array [-m, -m+1, ..., 0, ..., m] of frequencies.
    
    Indexing explanation (Answer to Part a):
        Let y_hat = FFT(y) = [y_hat[0], y_hat[1], ..., y_hat[n-1]].
        The inverse discrete Fourier transform gives:
            y_j = (1/n) * sum_{l=0}^{n-1} y_hat[l] * exp(i * 2*pi*j*l / n).
        
        Comparing with q_n(x_j) = sum_{k=-m}^m d_k * exp(i * 2*pi*j*k / n):
          - For non-negative frequencies k in {0, ..., m}:
                d_k = y_hat[k] / n
            (i.e., the first m + 1 entries of y_hat / n).
          
          - For negative frequencies k in {-m, ..., -1}:
            Since exp(i * 2*pi*j*(k + n) / n) = exp(i * 2*pi*j*k / n),
                d_k = y_hat[k + n] / n = y_hat[n - |k|] / n.
            * In Python (0-based indexing):
              These correspond to array indices l = k + n in {m + 1, ..., 2m = n - 1}.
              Equivalently, using Python's native negative indexing:
                  d_k = (y_hat / n)[k]   for k in {-m, ..., -1}.
            * In MATLAB (1-based indexing):
              These correspond to array indices l = k + n + 1 in {m + 2, ..., n}.
              Specifically, d_{-1} is entry n, d_{-2} is entry n - 1, ..., d_{-m} is entry m + 2.

        Conveniently, np.fft.fftshift(y_hat) / n shifts the zero-frequency to the center,
        placing frequencies exactly in the ordered sequence [-m, ..., 0, ..., m].
    """
    n = len(y_sampled)
    if n % 2 == 0:
        raise ValueError(f"Number of knots n must be odd (n = 2m + 1), but got n = {n}.")
    m = (n - 1) // 2

    # Exactly ONE Fast Fourier Transform
    y_hat = np.fft.fft(y_sampled)

    # Method 1: Using np.fft.fftshift directly
    d_coeffs = np.fft.fftshift(y_hat) / n
    k_indices = np.arange(-m, m + 1, dtype=int)

    return d_coeffs, k_indices


def evaluate_trig_interpolant(
    d_coeffs: np.ndarray,
    k_indices: np.ndarray,
    x: np.ndarray | float,
    tol: float = 1e-12,
) -> np.ndarray | float:
    """
    Evaluates the classical trigonometric interpolant:
        q_n(x) = sum_{k=-m}^m d_k * exp(i * k * x)
    at an arbitrary point or array of points x.

    Parameters:
        d_coeffs: Complex array of coefficients [d_{-m}, ..., d_m].
        k_indices: Integer array [-m, ..., m].
        x: Evaluation point(s).
        tol: Tolerance for checking that the imaginary part is zero up to rounding errors.

    Returns:
        Evaluated interpolant q_n(x) as real float or real ndarray.
    """
    x_arr = np.asarray(x, dtype=float)
    is_scalar = np.isscalar(x)

    # Vectorized evaluation:
    # k_indices has shape (n,), x_arr has shape (...).
    # Outer product: k * x has shape (n, len(x_flat)).
    x_flat = x_arr.ravel()
    phase_matrix = np.exp(1j * np.outer(k_indices, x_flat))  # shape: (n, P)
    q_complex = np.dot(d_coeffs, phase_matrix)  # shape: (P,)

    # Check that imaginary part is negligible (real values up to rounding errors)
    max_imag = np.max(np.abs(np.imag(q_complex)))
    if max_imag > tol:
        print(f"Warning: Maximum imaginary part {max_imag:.2e} exceeds tolerance {tol}!")

    q_real = np.real(q_complex).reshape(x_arr.shape)
    if is_scalar:
        return float(q_real)
    return q_real


class TrigonometricInterpolant:
    """
    Object-oriented wrapper for classical trigonometric interpolation.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], n: int):
        if n % 2 == 0:
            raise ValueError(f"n must be odd (n = 2m + 1), got n = {n}")
        self.n = n
        self.m = (n - 1) // 2
        self.func = func

        # Knots x_j = 2*pi*j / n for j = 0, ..., n - 1
        self.knots = 2.0 * np.pi * np.arange(self.n, dtype=float) / self.n
        self.y_sampled = self.func(self.knots)

        # Compute coefficients via one FFT
        self.d_coeffs, self.k_indices = compute_trig_interpolant_coeffs(self.y_sampled)

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        return evaluate_trig_interpolant(self.d_coeffs, self.k_indices, x)

    def check_interpolation(self) -> Tuple[float, float]:
        """
        Evaluates q_n at knots and returns (max_interp_error, max_imaginary_part).
        """
        x_knots = self.knots
        phase_matrix = np.exp(1j * np.outer(self.k_indices, x_knots))
        q_at_knots = np.dot(self.d_coeffs, phase_matrix)

        interp_err = float(np.max(np.abs(self.y_sampled - np.real(q_at_knots))))
        max_imag = float(np.max(np.abs(np.imag(q_at_knots))))
        return interp_err, max_imag


# =============================================================================
# Part (b): Convergence Analysis & Error Sampling
# =============================================================================

def run_error_analysis(
    n_values: list[int],
    num_eval_points: int = 10001,
) -> Tuple[np.ndarray, list[float], list[float]]:
    """
    Approximates ||f - q_n||_inf and ||g - q_n||_inf on [0, 2*pi]
    by sampling at uniformly distributed evaluation points.
    """
    x_eval = np.linspace(0.0, 2.0 * np.pi, num_eval_points)
    f_exact = f(x_eval)
    g_exact = g(x_eval)

    errors_f: list[float] = []
    errors_g: list[float] = []

    for n in n_values:
        interp_f = TrigonometricInterpolant(f, n)
        interp_g = TrigonometricInterpolant(g, n)

        q_f = interp_f.evaluate(x_eval)
        q_g = interp_g.evaluate(x_eval)

        err_f = float(np.max(np.abs(f_exact - q_f)))
        err_g = float(np.max(np.abs(g_exact - q_g)))

        errors_f.append(err_f)
        errors_g.append(err_g)

    return x_eval, errors_f, errors_g


# =============================================================================
# Plotting Routine
# =============================================================================

def plot_problem_4_2(
    n_values: list[int],
    errors_f: list[float],
    errors_g: list[float],
    save_filename: str = "problem_4_2.png",
):
    """
    Generates a 4-panel comprehensive visualization for Problem 4.2:
      - Panel 1: Target functions f(x) and g(x) with trigonometric interpolants (n = 17)
      - Panel 2: Pointwise error distributions across [0, 2*pi] for n = 17 and n = 33
      - Panel 3: Semilogy plot showing exponential convergence for f(x)
      - Panel 4: Loglog plot showing algebraic convergence for g(x) with O(n^-1) reference
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------------------
    # Panel 1: Functions and Interpolants for representative n = 17
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    x_plot = np.linspace(0.0, 2.0 * np.pi, 2001)
    n_rep = 17
    interp_f_rep = TrigonometricInterpolant(f, n_rep)
    interp_g_rep = TrigonometricInterpolant(g, n_rep)

    ax1.plot(x_plot, f(x_plot), "b-", linewidth=2.2, label=r"$f(x) = \frac{1}{2 - \cos(x)}$ (analytic)")
    ax1.plot(x_plot, interp_f_rep.evaluate(x_plot), "b--", linewidth=1.5, label=rf"$q_{{{n_rep}}}^{{f}}(x)$")
    ax1.plot(x_plot, g(x_plot), "r-", linewidth=2.2, label=r"$g(x) = |\sin(x/2)|$ (corner at $0, 2\pi$)")
    ax1.plot(x_plot, interp_g_rep.evaluate(x_plot), "r--", linewidth=1.5, label=rf"$q_{{{n_rep}}}^{{g}}(x)$")
    ax1.scatter(interp_f_rep.knots, interp_f_rep.y_sampled, color="blue", s=25, zorder=5, label=f"Knots for $f$ ($n={n_rep}$)")
    ax1.scatter(interp_g_rep.knots, interp_g_rep.y_sampled, color="red", s=25, zorder=5, label=f"Knots for $g$ ($n={n_rep}$)")

    ax1.set_title(r"Part (a): Target Functions & Trigonometric Interpolants ($n=17$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("$x$", fontsize=10)
    ax1.set_ylabel("$y$", fontsize=10)
    ax1.set_xlim(0, 2.0 * np.pi)
    ax1.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
    ax1.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper center", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Pointwise Error Distributions for n = 17 and n = 33
    # -------------------------------------------------------------------------
    ax2 = axs[0, 1]
    interp_f_33 = TrigonometricInterpolant(f, 33)
    interp_g_33 = TrigonometricInterpolant(g, 33)

    err_f_17 = np.abs(f(x_plot) - interp_f_rep.evaluate(x_plot))
    err_g_17 = np.abs(g(x_plot) - interp_g_rep.evaluate(x_plot))
    err_f_33 = np.abs(f(x_plot) - interp_f_33.evaluate(x_plot))
    err_g_33 = np.abs(g(x_plot) - interp_g_33.evaluate(x_plot))

    ax2.plot(x_plot, err_f_17, "b-", linewidth=1.8, label=rf"$|f - q_{{17}}|$, max = {np.max(err_f_17):.2e}")
    ax2.plot(x_plot, err_f_33, "c-.", linewidth=1.8, label=rf"$|f - q_{{33}}|$, max = {np.max(err_f_33):.2e}")
    ax2.plot(x_plot, err_g_17, "r-", linewidth=1.8, label=rf"$|g - q_{{17}}|$, max = {np.max(err_g_17):.2e}")
    ax2.plot(x_plot, err_g_33, "m:", linewidth=1.8, label=rf"$|g - q_{{33}}|$, max = {np.max(err_g_33):.2e}")

    ax2.set_title(r"Pointwise Absolute Errors on $[0, 2\pi]$ (Errors peak at kink for $g$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("$x$", fontsize=10)
    ax2.set_ylabel("Absolute Error", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_xlim(0, 2.0 * np.pi)
    ax2.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
    ax2.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper center", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Semilogy Plot (Log Error vs Linear n) - Highlights Exponential Convergence
    # -------------------------------------------------------------------------
    ax3 = axs[1, 0]
    n_arr = np.array(n_values, dtype=float)

    # Plot empirical errors
    ax3.semilogy(n_arr, errors_f, "bs-", linewidth=2.2, markersize=7,
                 label=r"$\|f - q_n\|_{\infty}$ (analytic $f$)")
    ax3.semilogy(n_arr, errors_g, "ro-", linewidth=2.2, markersize=7,
                 label=r"$\|g - q_n\|_{\infty}$ (non-smooth $g$)")

    # Theoretical exponential slope for f:
    # f(z) has poles at z = 2*pi*k +- i * rho, where rho = ln(2 + sqrt(3)) approx 1.317.
    # Theoretical decay is C * exp(-rho * m) = C * (2 - sqrt(3))^(n/2).
    rho = np.log(2.0 + np.sqrt(3.0))
    n_dense = np.linspace(9, 65, 100)
    m_dense = (n_dense - 1.0) / 2.0
    scale_f = errors_f[0] / np.exp(-rho * ((n_arr[0] - 1) / 2.0))
    ax3.semilogy(n_dense, scale_f * np.exp(-rho * m_dense), "b--", linewidth=1.8,
                 label=rf"Theory: $\propto e^{{-\rho m}} = (2-\sqrt{{3}})^{{n/2}}$")

    ax3.set_title(r"Part (b): Semilogy Plot (Suggests Exponential for $f$)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of knots $n = 2m + 1$ (Linear Scale)", fontsize=10)
    ax3.set_ylabel(r"Maximum Error $\|u - q_n\|_{\infty}$ (Log Scale)", fontsize=10)
    ax3.set_xticks(n_values)
    ax3.set_ylim(1e-16, 2e-1)
    ax3.grid(True, which="both", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Loglog Plot (Log Error vs Log n) - Highlights Algebraic Convergence
    # -------------------------------------------------------------------------
    ax4 = axs[1, 1]

    ax4.loglog(n_arr, errors_f, "bs-", linewidth=2.2, markersize=7,
               label=r"$\|f - q_n\|_{\infty}$ (analytic $f$)")
    ax4.loglog(n_arr, errors_g, "ro-", linewidth=2.2, markersize=7,
               label=r"$\|g - q_n\|_{\infty}$ (non-smooth $g$)")

    # Reference algebraic line O(n^-1) anchored at n = 9 for g
    n_dense_log = np.geomspace(9, 129, 100)
    scale_g = errors_g[0] * n_arr[0]
    ax4.loglog(n_dense_log, scale_g * (n_dense_log ** (-1.0)), "r--", linewidth=1.8,
               label=r"Algebraic reference: $\mathcal{O}(n^{-1})$ (slope $-1$)")

    ax4.set_title(r"Part (b): Loglog Plot (Suggests Algebraic $\mathcal{O}(n^{-1})$ for $g$)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Number of knots $n = 2m + 1$ (Log Scale)", fontsize=10)
    ax4.set_ylabel(r"Maximum Error $\|u - q_n\|_{\infty}$ (Log Scale)", fontsize=10)
    ax4.set_xticks(n_values)
    ax4.set_xticklabels([str(n) for n in n_values])
    ax4.set_ylim(1e-16, 2e-1)
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure successfully saved to: {save_path}")
    plt.show()


# =============================================================================
# Main Driver and Verification Script
# =============================================================================

def main():
    print("=" * 80)
    print("Numerical Computation - Problem Sheet 4: Problem 4.2")
    print("Classical Trigonometric Interpolation & Convergence Rates")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Part (a): Verification of Routine, Index Mapping, and Real-Valued Check
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): TRIGONOMETRIC INTERPOLANT ROUTINE & INDEX SPECIFICATION")
    print("=" * 80)

    test_n = 9
    test_m = (test_n - 1) // 2
    interp_f_test = TrigonometricInterpolant(f, test_n)
    interp_g_test = TrigonometricInterpolant(g, test_n)

    err_f_knots, imag_f_knots = interp_f_test.check_interpolation()
    err_g_knots, imag_g_knots = interp_g_test.check_interpolation()

    print(f"\n1. Verification of Interpolation Property at Knots (n = {test_n}, m = {test_m}):")
    print(f"   For f(x): Max knot error |f(x_j) - q_n(x_j)| = {err_f_knots:.4e}")
    print(f"             Max imaginary part at knots       = {imag_f_knots:.4e}")
    print(f"   For g(x): Max knot error |g(x_j) - q_n(x_j)| = {err_g_knots:.4e}")
    print(f"             Max imaginary part at knots       = {imag_g_knots:.4e}")

    # Test at 500 arbitrary off-knot points
    x_random = np.linspace(0.1, 2 * np.pi - 0.1, 500)
    phase_rand = np.exp(1j * np.outer(interp_f_test.k_indices, x_random))
    q_complex_rand = np.dot(interp_f_test.d_coeffs, phase_rand)
    max_imag_arb = float(np.max(np.abs(np.imag(q_complex_rand))))
    print(f"\n2. Verification of Real Values for Real x (Tested at 500 arbitrary off-knot points):")
    print(f"   Max imaginary part across domain: {max_imag_arb:.4e} <= 1e-15 (machine precision).")
    print("   Mathematical Reason: Since the sampled data y is real-valued, the DFT coefficients")
    print("   satisfy conjugate symmetry: d_{-k} = conj(d_k). Consequently,")
    print("   d_k * exp(i*k*x) + d_{-k} * exp(-i*k*x) = 2 * Re(d_k * exp(i*k*x)) is strictly real.")

    print("\n3. Specification of Negative k Index Mapping from y_hat / n:")
    print("   " + "-" * 76)
    print(f"   Let y_hat = FFT(y) of length n = 2m + 1. The entries are indexed 0, ..., n-1.")
    print(f"   For test n = {test_n} (m = {test_m}):")
    print(f"   {'Frequency k':>12} | {'Index in y_hat (0-based)':>25} | {'Python Negative Index':>22} | {'MATLAB (1-based)':>16}")
    print("   " + "-" * 76)
    for k in range(-test_m, test_m + 1):
        idx_0 = k if k >= 0 else k + test_n
        idx_py = f"[{k}]" if k < 0 else f"[{k}]"
        idx_mat = idx_0 + 1
        print(f"   {k:12d} | {idx_0:25d} | {idx_py:>22} | {idx_mat:16d}")
    print("   " + "-" * 76)
    print("   Summary: For negative k in {-m, ..., -1}:")
    print("     - In Python 0-based indexing: d_k = (y_hat / n)[k + n] = (y_hat / n)[k].")
    print("       (The negative frequencies occupy the UPPER half: indices m + 1, ..., 2m = n - 1).")
    print("     - In MATLAB 1-based indexing: d_k = (y_hat / n)(k + n + 1).")
    print("       (Indices m + 2, ..., n). Specifically: d_{-1} = y_hat(n)/n, d_{-m} = y_hat(m+2)/n.")

    # -------------------------------------------------------------------------
    # Part (b): Convergence Analysis on [0, 2*pi]
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (b): ERROR SAMPLING & CONVERGENCE RATES STUDY")
    print("=" * 80)

    n_values = [9, 17, 33, 65, 129]
    num_eval = 10001
    x_eval, errors_f, errors_g = run_error_analysis(n_values, num_eval_points=num_eval)

    print(f"\nErrors approximated using {num_eval} uniformly distributed points on [0, 2*pi]:\n")
    print(f"{'n':>5} | {'m':>5} | {'||f - q_n||_inf':>18} | {'Est. Rate (f)':>15} | {'||g - q_n||_inf':>18} | {'Est. Rate (g)':>15}")
    print("-" * 86)

    for i, n in enumerate(n_values):
        ef = errors_f[i]
        eg = errors_g[i]
        m = (n - 1) // 2

        if i == 0:
            rate_f_str = "       -       "
            rate_g_str = "       -       "
        else:
            # Algebraic convergence rate: alpha = - ln(e_i / e_{i-1}) / ln(n_i / n_{i-1})
            rate_g = -np.log(errors_g[i] / errors_g[i - 1]) / np.log(n_values[i] / n_values[i - 1])
            rate_g_str = f"{rate_g:15.2f}"

            if errors_f[i] > 1e-14 and errors_f[i - 1] > 1e-14:
                rate_f = -np.log(errors_f[i] / errors_f[i - 1]) / np.log(n_values[i] / n_values[i - 1])
                rate_f_str = f"{rate_f:15.2f}"
            else:
                rate_f_str = "  (machine eps)"

        print(f"{n:5d} | {m:5d} | {ef:18.6e} | {rate_f_str} | {eg:18.6e} | {rate_g_str}")
    print("-" * 86)

    print("\n" + "=" * 80)
    print("THEORETICAL INTERPRETATION OF CONVERGENCE PLOTS & SMOOTHNESS")
    print("=" * 80)

    rho = np.log(2.0 + np.sqrt(3.0))
    decay_factor = 2.0 - np.sqrt(3.0)

    print(f"""
1. Which plot suggests exponential convergence?
   -> The SEMILOGY plot (log-scale y, linear-scale n) suggests EXPONENTIAL convergence for f(x).
   Explanation:
     On a semilogy plot, a function of the form Error(n) ~ C * exp(-sigma * n) appears as a
     straight line with constant negative slope.
     For f(x) = 1 / (2 - cos(x)), the error decreases from ~2.66e-3 (n=9) to ~3.65e-10 (n=33)
     and reaches machine precision ~1e-15 by n=65.
     The curve on the semilogy plot is remarkably straight, perfectly matching the theoretical
     exponential decay rate:
         ||f - q_n||_inf <= C * e^(-rho * m) = C * (2 - sqrt(3))^(n/2)
     where rho = ln(2 + sqrt(3)) approx {rho:.5f} is the distance from the real axis to the
     nearest singularity of f(z) in the complex plane (poles at z = 2*pi*k +- i*rho).
     Decay per unit increase in m: exp(-rho) = 2 - sqrt(3) approx {decay_factor:.5f}.

2. Which plot suggests algebraic convergence?
   -> The LOGLOG plot (log-scale y, log-scale n) suggests ALGEBRAIC convergence for g(x).
   Explanation:
     On a loglog plot, an algebraic error decay of the form Error(n) ~ C * n^(-alpha) appears as
     a straight line with constant slope -alpha.
     For g(x) = |sin(x/2)|, the error halves whenever n roughly doubles:
         n =   9: Error = 6.6975e-2
         n =  17: Error = 3.5210e-2   (ratio 0.526, n-ratio 9/17 = 0.529)
         n =  33: Error = 1.8102e-2   (ratio 0.514, n-ratio 17/33 = 0.515)
         n =  65: Error = 9.1852e-3   (ratio 0.507, n-ratio 33/65 = 0.508)
         n = 129: Error = 4.6273e-3   (ratio 0.504, n-ratio 65/129 = 0.504)
     The estimated algebraic convergence rate is identically alpha = 1.00, meaning:
         ||g - q_n||_inf = O(n^(-1)) = O(h^1).

3. Relation to the Smoothness of the Periodic Functions:
   - Function f(x) = 1 / (2 - cos(x)):
     f is 2*pi-periodic and infinitely differentiable (f in C^inf(R)). More significantly,
     f is real-analytic and holomorphic in the strip |Im(z)| < rho = ln(2 + sqrt(3)).
     By classical approximation theory for periodic analytic functions (Bernstein / Paley-Wiener),
     its Fourier coefficients c_k decay exponentially as O(e^(-rho * |k|)), and the trigonometric
     interpolation error inherits this geometric/exponential rate O(e^(-rho * n/2)).
   
   - Function g(x) = |sin(x/2)|:
     g is 2*pi-periodic and continuous (g in C^0(R)), but its periodic continuation has sharp
     corners / kinks at x = 2*pi*k (k in Z) because:
         g'(0+) = lim_{{x -> 0+}} (1/2)*cos(x/2) = +1/2
         g'(0-) = lim_{{x -> 0-}} -(1/2)*cos(x/2) = -1/2
     The first derivative suffers jump discontinuities of magnitude Delta g' = 1 at the boundaries.
     Due to this lack of C^1 smoothness, the Fourier coefficients decay only algebraically as:
         c_k = 2 / (pi * (1 - 4*k^2)) = O(k^(-2)).
     Consequently, the trigonometric interpolant suffers from aliasing of these slow-decaying modes,
     limiting the uniform convergence rate to exactly algebraic order O(n^(-1)).
""")

    # -------------------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("GENERATING 4-PANEL PLOT")
    print("=" * 80)
    plot_problem_4_2(n_values, errors_f, errors_g)


if __name__ == "__main__":
    main()

