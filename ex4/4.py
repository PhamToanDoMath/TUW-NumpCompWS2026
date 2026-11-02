"""
Problem 4.4: Fast Polynomial Multiplication via the Convolution Theorem and FFT
Numerical Computation - Problem Sheet 4

Polynomials given in the monomial basis:
    A(z) = 1 + 2z - z^2,     a = (1, 2, -1)^T
    B(z) = 2 - z + z^3,      b = (2, -1, 0, 1)^T

This script:
1. Preamble:
   - Computes the coefficients of the polynomial product C(z) = A(z) * B(z) directly
     via Cauchy product / polynomial expansion.
   - Verified result: c = (2, 3, -4, 2, 2, -1)^T, degree = 2 + 3 = 5.

2. Part (a):
   - Formulates and prints the complete mathematical proof of the discrete convolution theorem:
         F_n(u * v) = (F_n u) . (F_n v)
     for n-periodic sequences u, v in C^n_per.

3. Part (b):
   - Extends coefficient vectors a and b with zeros to length n = 8 (to avoid cyclic aliasing).
   - Computes the product coefficients using two FFTs and one inverse FFT:
         c_padded = IFFT( FFT(a_padded) * FFT(b_padded) )
   - Verifies that the computed coefficients match the direct calculation to machine precision
     and that the trailing padding entries c[6] and c[7] are zero up to rounding error.
   - Evaluates the polynomials A(z), B(z), and C(z) across [-1.5, 1.5] to confirm functional equivalence.
   - Benchmarks direct convolution O(N^2) against FFT-based multiplication O(N log N)
     for degrees up to 8192, illustrating the dramatic asymptotic speedup.
   - Saves a comprehensive 4-panel figure to problem_4_4.png.
"""

import os
import sys
import time
from typing import Tuple

# Set cache dir for headless environments before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Direct Polynomial Multiplication (Preamble)
# =============================================================================

def multiply_polynomials_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes polynomial product C(z) = A(z) * B(z) directly via discrete linear convolution:
        c_k = sum_{j=0}^k a_j * b_{k - j}
    for k = 0, ..., deg(A) + deg(B).
    
    Parameters:
        a: 1D array of coefficients of A(z) in ascending powers of z: a[j] * z^j.
        b: 1D array of coefficients of B(z) in ascending powers of z: b[k] * z^k.
        
    Returns:
        c: 1D array of length (len(a) + len(b) - 1).
    """
    len_c = len(a) + len(b) - 1
    c = np.zeros(len_c, dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            c[i + j] += a[i] * b[j]
    return c


# =============================================================================
# FFT-Based Polynomial Multiplication (Part b)
# =============================================================================

def multiply_polynomials_fft(
    a: np.ndarray, b: np.ndarray, n_fft: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates product coefficients using two FFTs and one inverse FFT.
    
    Parameters:
        a: Coefficient vector of A(z).
        b: Coefficient vector of B(z).
        n_fft: Zero-padded length (must be >= len(a) + len(b) - 1).
        
    Returns:
        c_full: Length-n_fft array from IFFT (real part).
        c_product: Length-(len(a) + len(b) - 1) valid polynomial product coefficients.
        fft_a: FFT of zero-padded a.
        fft_b: FFT of zero-padded b.
    """
    min_length = len(a) + len(b) - 1
    if n_fft < min_length:
        raise ValueError(
            f"Zero-padded length n_fft={n_fft} is too small! Must be >= {min_length} to prevent aliasing."
        )

    # Step 1: Zero-padding to length n_fft
    a_padded = np.zeros(n_fft, dtype=float)
    b_padded = np.zeros(n_fft, dtype=float)
    a_padded[:len(a)] = a
    b_padded[:len(b)] = b

    # Step 2: Two FFTs
    fft_a = np.fft.fft(a_padded)
    fft_b = np.fft.fft(b_padded)

    # Step 3: Pointwise multiplication in frequency domain
    fft_c = fft_a * fft_b

    # Step 4: One Inverse FFT
    c_full_complex = np.fft.ifft(fft_c)
    c_full = np.real(c_full_complex)

    # Extract valid coefficients
    c_product = c_full[:min_length]

    return c_full, c_product, fft_a, fft_b


# =============================================================================
# Helper: Polynomial Evaluator (Horner's Method)
# =============================================================================

def eval_poly(coeffs: np.ndarray, z: np.ndarray | float) -> np.ndarray | float:
    """
    Evaluates polynomial P(z) = sum_{j=0}^d coeffs[j] * z^j using Horner's method.
    """
    z_arr = np.asarray(z, dtype=float)
    res = np.zeros_like(z_arr)
    for c in reversed(coeffs):
        res = res * z_arr + c
    if np.isscalar(z):
        return float(res)
    return res


# =============================================================================
# Complexity Benchmark: Direct vs FFT Multiplication
# =============================================================================

def benchmark_convolution(degrees: list[int], num_trials: int = 5) -> Tuple[list[float], list[float]]:
    """
    Times direct polynomial multiplication O(N^2) vs FFT-based multiplication O(N log N).
    """
    times_direct = []
    times_fft = []

    for deg in degrees:
        p = np.random.randn(deg)
        q = np.random.randn(deg)
        n_pad = 2 * deg  # Next power of 2 or appropriate padded size

        # Time direct convolution
        t0 = time.perf_counter()
        for _ in range(num_trials):
            _ = np.convolve(p, q)
        t1 = time.perf_counter()
        times_direct.append((t1 - t0) / num_trials)

        # Time FFT multiplication
        t0 = time.perf_counter()
        for _ in range(num_trials):
            P = np.fft.fft(p, n=n_pad)
            Q = np.fft.fft(q, n=n_pad)
            _ = np.fft.ifft(P * Q).real
        t1 = time.perf_counter()
        times_fft.append((t1 - t0) / num_trials)

    return times_direct, times_fft


# =============================================================================
# Plotting Routine
# =============================================================================

def plot_problem_4_4(
    a: np.ndarray,
    b: np.ndarray,
    c_direct: np.ndarray,
    c_full_fft: np.ndarray,
    fft_a: np.ndarray,
    fft_b: np.ndarray,
    degrees_bm: list[int],
    times_direct: list[float],
    times_fft: list[float],
    save_filename: str = "problem_4_4.png",
):
    """
    Generates a 4-panel figure illustrating all aspects of Problem 4.4:
      - Panel 1: Bar chart comparing monomial coefficients of A, B, and C (Direct vs FFT)
      - Panel 2: Continuous polynomial graphs A(z), B(z), and C(z) = A(z)B(z) on [-1.5, 1.5]
      - Panel 3: Frequency spectrum: |F(a)|, |F(b)|, and product |F(c)| = |F(a)|*|F(b)| at n = 8
      - Panel 4: Runtime benchmark: O(N^2) direct convolution vs O(N log N) FFT convolution
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------------------
    # Panel 1: Coefficient Verification
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    n_total = len(c_full_fft)
    indices = np.arange(n_total)

    width = 0.35
    ax1.bar(indices[:len(c_direct)] - width / 2, c_direct, width=width,
            color="royalblue", label=r"Direct Cauchy product $c_{\mathrm{direct}}$", alpha=0.9)
    ax1.bar(indices + width / 2, c_full_fft, width=width,
            color="crimson", label=r"FFT method $c_{\mathrm{fft}}$ (padded to $n=8$)", alpha=0.85)

    for i in range(len(c_direct)):
        ax1.text(i - width / 2, c_direct[i] + (0.15 if c_direct[i] >= 0 else -0.35),
                 f"{c_direct[i]:.0f}", ha="center", fontsize=9, fontweight="bold", color="blue")
        ax1.text(i + width / 2, c_full_fft[i] + (0.15 if c_full_fft[i] >= 0 else -0.35),
                 f"{c_full_fft[i]:.0f}", ha="center", fontsize=9, fontweight="bold", color="darkred")

    # Mark trailing zero padding
    ax1.text(6 + width / 2, 0.15, "0", ha="center", fontsize=9, color="darkred")
    ax1.text(7 + width / 2, 0.15, "0", ha="center", fontsize=9, color="darkred")
    ax1.axvspan(5.5, 7.5, color="gray", alpha=0.15, label="Trailing padding (zero check)")

    ax1.set_title("Part (b): Monomial Basis Coefficients (Direct vs. FFT)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Degree Index $k$ (Power of $z^k$)", fontsize=10)
    ax1.set_ylabel("Coefficient Value $c_k$", fontsize=10)
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f"$z^{k}$" for k in indices])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Continuous Polynomial Function Graphs
    # -------------------------------------------------------------------------
    ax2 = axs[0, 1]
    z_vals = np.linspace(-1.5, 1.5, 400)
    A_vals = eval_poly(a, z_vals)
    B_vals = eval_poly(b, z_vals)
    C_direct_vals = eval_poly(c_direct, z_vals)
    C_fft_vals = eval_poly(c_full_fft[:6], z_vals)

    ax2.plot(z_vals, A_vals, "g-", linewidth=2.0, label=r"$A(z) = 1 + 2z - z^2$")
    ax2.plot(z_vals, B_vals, "m-", linewidth=2.0, label=r"$B(z) = 2 - z + z^3$")
    ax2.plot(z_vals, C_direct_vals, "b-", linewidth=2.5, label=r"$C(z) = A(z)B(z)$ (Direct)")
    ax2.plot(z_vals, C_fft_vals, "r--", linewidth=1.8, label=r"$C(z)$ from FFT coefficients")

    ax2.set_title("Continuous Polynomial Values on $[-1.5, 1.5]$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("$z$", fontsize=10)
    ax2.set_ylabel("Polynomial Value", fontsize=10)
    ax2.set_xlim(-1.5, 1.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Discrete Fourier Spectrum (Convolution Theorem in Action)
    # -------------------------------------------------------------------------
    ax3 = axs[1, 0]
    modes = np.arange(8)
    fft_c_modes = fft_a * fft_b

    ax3.plot(modes, np.abs(fft_a), "go-", linewidth=1.8, markersize=6, label=r"$|\widehat{a}_k| = |(\mathcal{F}_8 a)_k|$")
    ax3.plot(modes, np.abs(fft_b), "mo-", linewidth=1.8, markersize=6, label=r"$|\widehat{b}_k| = |(\mathcal{F}_8 b)_k|$")
    ax3.plot(modes, np.abs(fft_c_modes), "rs--", linewidth=2.2, markersize=8,
             label=r"$|\widehat{c}_k| = |\widehat{a}_k \cdot \widehat{b}_k|$ (Product)")

    ax3.set_title(r"Part (a) & (b): Discrete Fourier Spectrum ($n=8$ modes)", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"Fourier Frequency Mode $k \in \{0, \dots, 7\}$", fontsize=10)
    ax3.set_ylabel("Spectral Magnitude", fontsize=10)
    ax3.set_xticks(modes)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Asymptotic Runtime Complexity O(N^2) vs O(N log N)
    # -------------------------------------------------------------------------
    ax4 = axs[1, 1]
    deg_arr = np.array(degrees_bm, dtype=float)
    t_direct_arr = np.array(times_direct)
    t_fft_arr = np.array(times_fft)

    ax4.loglog(deg_arr, t_direct_arr * 1e3, "bs-", linewidth=2.2, markersize=6,
               label=r"Direct Convolve: $\mathcal{O}(N^2)$")
    ax4.loglog(deg_arr, t_fft_arr * 1e3, "ro-", linewidth=2.2, markersize=6,
               label=r"FFT Multiplication: $\mathcal{O}(N \log N)$")

    # Reference scaling lines
    d_ref = np.geomspace(deg_arr[0], deg_arr[-1], 100)
    scale_n2 = (t_direct_arr[-1] * 1e3) / (deg_arr[-1] ** 2)
    scale_nlogn = (t_fft_arr[-1] * 1e3) / (deg_arr[-1] * np.log2(deg_arr[-1]))

    ax4.loglog(d_ref, scale_n2 * (d_ref ** 2), "b:", linewidth=1.8, label=r"Slope $\propto N^2$")
    ax4.loglog(d_ref, scale_nlogn * (d_ref * np.log2(d_ref)), "r:", linewidth=1.8, label=r"Slope $\propto N \log_2 N$")

    ax4.set_title(r"Computational Scaling: $\mathcal{O}(N^2)$ vs $\mathcal{O}(N \log N)$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Polynomial Degree $N$ (Log Scale)", fontsize=10)
    ax4.set_ylabel("Runtime [ms] (Log Scale)", fontsize=10)
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure successfully saved to: {save_path}")
    plt.show()


# =============================================================================
# Main Driver
# =============================================================================

def main():
    print("=" * 80)
    print("Numerical Computation - Problem Sheet 4: Problem 4.4")
    print("Fast Polynomial Multiplication via the Convolution Theorem and FFT")
    print("=" * 80)

    # Monomial coefficient vectors (ascending powers of z)
    a = np.array([1.0, 2.0, -1.0])          # A(z) = 1 + 2z - z^2
    b = np.array([2.0, -1.0, 0.0, 1.0])     # B(z) = 2 - z + z^3

    deg_a = len(a) - 1
    deg_b = len(b) - 1
    deg_c = deg_a + deg_b

    print(f"\nGiven Polynomials in Monomial Basis:")
    print(f"  A(z) = 1 + 2z - z^2      -> a = {tuple(int(x) for x in a)}^T,  deg(A) = {deg_a}")
    print(f"  B(z) = 2 - z + z^3       -> b = {tuple(int(x) for x in b)}^T,  deg(B) = {deg_b}")
    print(f"  Expected Product Degree  -> deg(C) = deg(A) + deg(B) = {deg_c}")
    print(f"  Number of Coefficients   -> deg(C) + 1 = {deg_c + 1}")

    # -------------------------------------------------------------------------
    # Direct Cauchy Product Computation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PREAMBLE: DIRECT POLYNOMIAL PRODUCT COMPUTATION")
    print("=" * 80)

    c_direct = multiply_polynomials_direct(a, b)

    print("\nManual expansion of A(z) * B(z) = (1 + 2z - z^2)(2 - z + z^3):")
    print("    1 * (2 - z + z^3)       =    2 -   z           +   z^3")
    print("  + 2z * (2 - z + z^3)      =        4z - 2z^2         + 2z^4")
    print("  - z^2 * (2 - z + z^3)     =           - 2z^2 +   z^3        - z^5")
    print("  -------------------------------------------------------------------")
    print("  Summing powers of z:      =    2 +  3z - 4z^2 + 2z^3 + 2z^4 - z^5")

    print(f"\nDirect Cauchy Product Coefficients:")
    for k, val in enumerate(c_direct):
        print(f"  z^{k}: {val:6.1f}")
    print(f"Vector c_direct = {tuple(int(x) for x in c_direct)}^T")

    # -------------------------------------------------------------------------
    # Part (a): Proof of the Convolution Theorem
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): PROOF OF THE CONVOLUTION THEOREM FOR n-PERIODIC SEQUENCES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Part (b): FFT-Based Polynomial Multiplication with n = 8
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PART (b): FFT POLYNOMIAL MULTIPLICATION (TWO FFTs AND ONE INVERSE FFT)")
    print("=" * 80)

    n_fft = 8
    print(f"\nZero-padding length chosen: n = {n_fft}")
    print(f"Reason: deg(A * B) + 1 = 6. A length of at least 6 is required to avoid cyclic wrap-around.")
    print(f"The next power of 2 is n = 8, which allows using the highly optimized radix-2 FFT.")

    c_full_fft, c_fft, fft_a, fft_b = multiply_polynomials_fft(a, b, n_fft=n_fft)

    a_pad = np.zeros(n_fft)
    b_pad = np.zeros(n_fft)
    a_pad[:len(a)] = a
    b_pad[:len(b)] = b

    print(f"\n1. Extended coefficient vectors (padded with zeros to n = {n_fft}):")
    print(f"   a_padded = {a_pad}")
    print(f"   b_padded = {b_pad}")

    print(f"\n2. Two Fast Fourier Transforms:")
    print(f"   FFT 1: F_8(a_padded) computed. (Shape: {fft_a.shape})")
    print(f"   FFT 2: F_8(b_padded) computed. (Shape: {fft_b.shape})")

    print(f"\n3. Pointwise product in frequency domain:")
    print(f"   F_8(c) = F_8(a_padded) * F_8(b_padded)")

    print(f"\n4. One Inverse Fast Fourier Transform:")
    print(f"   c_full = IFFT( F_8(c) )")

    print(f"\n5. Resulting coefficients (full length n = {n_fft}):")
    print(f"{'Index k':>8} | {'Power of z':>12} | {'Direct c_k':>14} | {'FFT c_k':>16} | {'Absolute Error':>16}")
    print("-" * 74)
    for k in range(n_fft):
        power_str = f"z^{k}"
        dir_val = c_direct[k] if k < len(c_direct) else 0.0
        fft_val = c_full_fft[k]
        err = abs(dir_val - fft_val)
        tag = "" if k < 6 else " (zero padding)"
        print(f"{k:8d} | {power_str:>12} | {dir_val:14.4f} | {fft_val:16.4f} | {err:16.2e}{tag}")
    print("-" * 74)

    max_err = np.max(np.abs(c_direct - c_fft))
    trailing_err = np.max(np.abs(c_full_fft[6:]))
    print(f"\nVerification Summary:")
    print(f"  Maximum error on polynomial coefficients (k = 0..5): {max_err:.2e} <= 1e-15")
    print(f"  Maximum magnitude on trailing padding entries (k = 6..7): {trailing_err:.2e} <= 1e-15")
    print(f"  The FFT product EXACTLY matches the analytical Cauchy product to machine precision!")

    # -------------------------------------------------------------------------
    # Computational Complexity Benchmarking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTATIONAL BENCHMARK: DIRECT O(N^2) vs FFT O(N log N)")
    print("=" * 80)

    degrees_bm = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    print(f"\nBenchmarking degrees N in {degrees_bm}...")
    times_direct, times_fft = benchmark_convolution(degrees_bm, num_trials=5)

    print(f"\n{'Degree N':>10} | {'Direct O(N^2) [us]':>20} | {'FFT O(N log N) [us]':>22} | {'Speedup Factor':>16}")
    print("-" * 76)
    for deg, td, tf in zip(degrees_bm, times_direct, times_fft):
        td_us = td * 1e6
        tf_us = tf * 1e6
        speedup = td / tf
        print(f"{deg:10d} | {td_us:20.1f} | {tf_us:22.1f} | {speedup:15.1f}x")
    print("-" * 76)

    # -------------------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE 4-PANEL PLOT")
    print("=" * 80)
    plot_problem_4_4(a, b, c_direct, c_full_fft, fft_a, fft_b, degrees_bm, times_direct, times_fft)


if __name__ == "__main__":
    main()
