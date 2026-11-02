"""
Problem 4.5: Frequency Analysis, Low-Pass Filtering, Parseval Identity, and Aliasing
Numerical Computation - Problem Sheet 4

Periodic signal sampled at x_j = 2*pi*j / n, j = 0, ..., n - 1:
    y_j = s(x_j) + eta(x_j)
    s(x)   = sin(3x) + (1/2)*cos(7x)
    eta(x) = (3/10)*sin(25x) + (1/5)*cos(40x)

This script:
1. Part (a):
   - Samples the signal for n = 128.
   - Computes the amplitude spectrum |y_hat_k| / n via FFT.
   - Labels the frequency axis by the signed integers 0, ..., n/2, -n/2 + 1, ..., -1.
   - Identifies all nonzero modes and their exact theoretical amplitudes.

2. Part (b):
   - Implements an ideal low-pass filter keeping only frequencies |k| <= 10.
   - Inverts the filtered spectrum via IFFT to obtain y_tilde_j.
   - Derives the Discrete Parseval Identity from Theorem 1.45.
   - Uses Parseval's identity to predict the root mean square error (RMSE) before filtering,
     and compares with the sample RMSE before and after filtering.

3. Part (c):
   - Repeats the analysis for the undersampled case n = 32 (Nyquist limit = 16).
   - Demonstrates that high-frequency noise modes alias into the filter passband:
         k = 25  aliases to  k = 25 - 32 = -7
         k = -25 aliases to  k = -25 + 32 = +7
         k = 40  aliases to  k = 40 - 32 = +8
         k = -40 aliases to  k = -40 + 32 = -8
   - Explains why low-pass filtering fails completely when undersampling occurs.
   - Generates a comprehensive 4-panel figure saved to problem_4_5.png.
"""

import os
import sys
from typing import Tuple

# Set cache dir for headless environments before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Signal and Noise Functions
# =============================================================================

def s(x: np.ndarray | float) -> np.ndarray | float:
    """
    True periodic signal containing low frequencies:
        s(x) = sin(3x) + 0.5 * cos(7x).
    Frequencies present: k = +-3 (amplitude 0.5) and k = +-7 (amplitude 0.25).
    """
    return np.sin(3.0 * x) + 0.5 * np.cos(7.0 * x)


def eta(x: np.ndarray | float) -> np.ndarray | float:
    """
    High-frequency noise perturbation:
        eta(x) = 0.3 * sin(25x) + 0.2 * cos(40x).
    Frequencies present: k = +-25 (amplitude 0.15) and k = +-40 (amplitude 0.10).
    """
    return 0.3 * np.sin(25.0 * x) + 0.2 * np.cos(40.0 * x)


def y(x: np.ndarray | float) -> np.ndarray | float:
    """
    Noisy sampled signal:
        y(x) = s(x) + eta(x).
    """
    return s(x) + eta(x)


# =============================================================================
# Frequency Analysis & Filtering Routines
# =============================================================================

def get_signed_frequencies(n: int) -> np.ndarray:
    """
    Returns signed integer frequencies matching standard FFT layout:
        [0, 1, ..., n/2, -n/2 + 1, ..., -1]
    for an even integer n.
    """
    if n % 2 != 0:
        raise ValueError(f"n must be even, but got n = {n}.")
    signed_k = np.zeros(n, dtype=int)
    signed_k[: n // 2 + 1] = np.arange(0, n // 2 + 1)
    signed_k[n // 2 + 1 :] = np.arange(-n // 2 + 1, 0)
    return signed_k


def compute_spectrum(yj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the FFT, amplitude spectrum |y_hat_k| / n, and signed frequency axis.
    """
    n = len(yj)
    y_hat = np.fft.fft(yj)
    amp_spectrum = np.abs(y_hat) / n
    signed_k = get_signed_frequencies(n)
    return y_hat, amp_spectrum, signed_k


def apply_low_pass_filter(
    y_hat: np.ndarray, signed_k: np.ndarray, cutoff: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies an ideal low-pass filter keeping frequencies |k| <= cutoff
    and setting all other modes to zero. Returns (y_hat_filtered, y_filtered).
    """
    y_hat_filtered = np.copy(y_hat)
    y_hat_filtered[np.abs(signed_k) > cutoff] = 0.0
    y_filtered = np.fft.ifft(y_hat_filtered).real
    return y_hat_filtered, y_filtered


# =============================================================================
# Plotting Routine
# =============================================================================

def plot_problem_4_5(
    n_128: int = 128,
    n_32: int = 32,
    save_filename: str = "problem_4_5.png",
):
    """
    Generates a 4-panel comprehensive visualization for Problem 4.5:
      - Panel 1: Part (a) Sampled signal y_j vs continuous s(x) for n = 128
      - Panel 2: Part (a) Amplitude spectrum |y_hat|/n for n = 128 with signed frequency axis
      - Panel 3: Part (b) Low-pass filtered samples vs true signal s(x) for n = 128
      - Panel 4: Part (c) Undersampling n = 32: Aliased spectrum and failed recovery
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Dense grid for plotting continuous curves
    x_dense = np.linspace(0.0, 2.0 * np.pi, 2001)
    s_dense = s(x_dense)
    y_dense = y(x_dense)

    # -------------------------------------------------------------------------
    # Panel 1: Samples and Continuous Signals for n = 128
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    j_128 = np.arange(n_128)
    x_128 = 2.0 * np.pi * j_128 / n_128
    y_128 = y(x_128)

    ax1.plot(x_dense, y_dense, "lightgray", linewidth=1.2, label=r"Continuous noisy $y(x) = s(x) + \eta(x)$")
    ax1.plot(x_dense, s_dense, "b-", linewidth=2.2, label=r"True signal $s(x) = \sin(3x) + \frac{1}{2}\cos(7x)$")
    ax1.scatter(x_128, y_128, color="darkred", s=14, zorder=5, label=rf"Noisy samples $y_j$ ($n={n_128}$)")

    ax1.set_title(r"Part (a): Signal Samples $y_j$ and Ground Truth $s(x)$ ($n = 128$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("$x$", fontsize=10)
    ax1.set_ylabel("$y$", fontsize=10)
    ax1.set_xlim(0, 2.0 * np.pi)
    ax1.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
    ax1.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Amplitude Spectrum for n = 128 with Signed Frequency Labels
    # -------------------------------------------------------------------------
    ax2 = axs[0, 1]
    y_hat_128, amp_128, signed_k_128 = compute_spectrum(y_128)

    # Sort frequencies for intuitive continuous visualization: -63, ..., 0, ..., 64
    sort_idx_128 = np.argsort(signed_k_128)
    k_sorted_128 = signed_k_128[sort_idx_128]
    amp_sorted_128 = amp_128[sort_idx_128]

    ax2.stem(k_sorted_128, amp_sorted_128, linefmt="k-", markerfmt="ko", basefmt="gray")
    
    # Highlight passband |k| <= 10
    ax2.axvspan(-10, 10, color="green", alpha=0.15, label=r"Low-pass passband ($|k| \leq 10$)")

    # Annotate signal and noise peaks
    ax2.annotate(r"$k=\pm 3$ (amp 0.50)", xy=(3, 0.5), xytext=(8, 0.53),
                 arrowprops=dict(facecolor="blue", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="blue", fontweight="bold")
    ax2.annotate(r"$k=\pm 7$ (amp 0.25)", xy=(7, 0.25), xytext=(12, 0.32),
                 arrowprops=dict(facecolor="blue", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="blue", fontweight="bold")
    ax2.annotate(r"$k=\pm 25$ (noise 0.15)", xy=(25, 0.15), xytext=(28, 0.22),
                 arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="darkred")
    ax2.annotate(r"$k=\pm 40$ (noise 0.10)", xy=(40, 0.10), xytext=(42, 0.16),
                 arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="darkred")

    ax2.set_title(r"Part (a): Amplitude Spectrum $|\widehat{y}_k|/n$ ($n = 128$, Nyquist limit = $64$)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Signed Frequency Mode $k \in \{0, \dots, n/2, -n/2+1, \dots, -1\}$", fontsize=10)
    ax2.set_ylabel(r"Amplitude $|\widehat{y}_k|/n$", fontsize=10)
    ax2.set_xlim(-65, 65)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Low-Pass Filtered Samples for n = 128 (Part b)
    # -------------------------------------------------------------------------
    ax3 = axs[1, 0]
    _, y_filtered_128 = apply_low_pass_filter(y_hat_128, signed_k_128, cutoff=10)
    s_at_knots_128 = s(x_128)

    rmse_before_128 = np.sqrt(np.mean((y_128 - s_at_knots_128) ** 2))
    rmse_after_128 = np.sqrt(np.mean((y_filtered_128 - s_at_knots_128) ** 2))

    ax3.plot(x_dense, s_dense, "b-", linewidth=2.5, label=r"Exact signal $s(x)$")
    ax3.plot(x_128, y_filtered_128, "r--", linewidth=1.8, label=r"Filtered samples $\widetilde{y}_j$ ($|k| \leq 10$)")
    ax3.scatter(x_128, y_filtered_128, color="red", s=25, zorder=5, label=r"Filtered grid values")

    info_box = (
        f"RMSE before filtering: {rmse_before_128:.5f}\n"
        f"Predicted via Parseval: {np.sqrt(0.065):.5f}\n"
        f"RMSE after filtering:  {rmse_after_128:.2e} (machine precision!)"
    )
    ax3.text(0.03, 0.08, info_box, transform=ax3.transAxes, fontsize=8.5,
             verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.7))

    ax3.set_title(r"Part (b): Filtered Samples $\widetilde{y}_j$ vs Ground Truth $s(x)$ ($n = 128$)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("$x$", fontsize=10)
    ax3.set_ylabel("$y$", fontsize=10)
    ax3.set_xlim(0, 2.0 * np.pi)
    ax3.set_xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
    ax3.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Part (c) Undersampling with n = 32 (Aliasing Disaster)
    # -------------------------------------------------------------------------
    ax4 = axs[1, 1]
    j_32 = np.arange(n_32)
    x_32 = 2.0 * np.pi * j_32 / n_32
    y_32 = y(x_32)
    s_at_knots_32 = s(x_32)

    y_hat_32, amp_32, signed_k_32 = compute_spectrum(y_32)
    _, y_filtered_32 = apply_low_pass_filter(y_hat_32, signed_k_32, cutoff=10)

    sort_idx_32 = np.argsort(signed_k_32)
    k_sorted_32 = signed_k_32[sort_idx_32]
    amp_sorted_32 = amp_32[sort_idx_32]

    ax4.stem(k_sorted_32, amp_sorted_32, linefmt="r-", markerfmt="ro", basefmt="gray",
             label=r"Undersampled spectrum ($n=32$, Nyquist limit = $16$)")
    ax4.axvspan(-10, 10, color="green", alpha=0.15, label=r"Filter passband ($|k| \leq 10$)")

    ax4.annotate("Aliased noise:\n$25 \\equiv -7$\ncorrupts signal!", xy=(7, amp_32[7]), xytext=(9, 0.40),
                 arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="darkred", fontweight="bold")
    ax4.annotate("Aliased noise:\n$40 \\equiv +8$\nleaks into passband!", xy=(8, amp_32[8]), xytext=(10, 0.20),
                 arrowprops=dict(facecolor="purple", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, color="purple", fontweight="bold")

    rmse_before_32 = np.sqrt(np.mean((y_32 - s_at_knots_32) ** 2))
    rmse_after_32 = np.sqrt(np.mean((y_filtered_32 - s_at_knots_32) ** 2))

    alias_box = (
        f"Nyquist limit: n/2 = 16 < 25, 40 (Aliasing occurs!)\n"
        f"Noise k=25 -> aliases to k=-7 (inside passband!)\n"
        f"Noise k=40 -> aliases to k=+8 (inside passband!)\n"
        f"RMSE after filter: {rmse_after_32:.5f} (= before: {rmse_before_32:.5f})\n"
        f"Filter removed ZERO noise!"
    )
    ax4.text(0.03, 0.52, alias_box, transform=ax4.transAxes, fontsize=8.0,
             verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.5", facecolor="pink", alpha=0.8))

    ax4.set_title(r"Part (c): Undersampling ($n = 32$): Aliasing Folds Noise into Passband",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel(r"Signed Frequency Mode $k \in \{-15, \dots, 0, \dots, 16\}$", fontsize=10)
    ax4.set_ylabel(r"Amplitude $|\widehat{y}_k|/n$", fontsize=10)
    ax4.set_xlim(-17, 17)
    ax4.set_ylim(0, 0.6)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

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
    print("Numerical Computation - Problem Sheet 4: Problem 4.5")
    print("Frequency Analysis, Low-Pass Filtering, Parseval Identity, and Aliasing")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Part (a): Frequency Analysis for n = 128
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): SAMPLING AND AMPLITUDE SPECTRUM FOR n = 128")
    print("=" * 80)

    n_128 = 128
    j_128 = np.arange(n_128)
    x_128 = 2.0 * np.pi * j_128 / n_128
    y_128 = y(x_128)

    y_hat_128, amp_128, signed_k_128 = compute_spectrum(y_128)

    print(f"Sampling parameters: n = {n_128}, Nyquist frequency k_Nyquist = n/2 = {n_128 // 2}")
    print(f"Signal components:")
    print(f"  s(x)   = sin(3x) + 0.5*cos(7x)          -> True signal frequencies: k = +-3, +-7")
    print(f"  eta(x) = 0.3*sin(25x) + 0.2*cos(40x)    -> Noise frequencies:       k = +-25, +-40")
    print(f"  Max frequency in signal: k_max = 40 < k_Nyquist = 64 (No aliasing for n = 128!)")

    print(f"\nIdentification of Nonzero Fourier Modes in Amplitude Spectrum |y_hat_k| / n:")
    print(f"{'FFT Index':>10} | {'Signed Mode k':>14} | {'Complex Coeff d_k':>24} | {'Amplitude':>12} | {'Identity'}")
    print("-" * 80)

    nonzero_indices = np.where(amp_128 > 1e-10)[0]
    for idx in nonzero_indices:
        k = signed_k_128[idx]
        coeff = y_hat_128[idx] / n_128
        amp = amp_128[idx]

        if abs(k) == 3:
            ident = "Signal: sin(3x) term (amp 0.50)"
        elif abs(k) == 7:
            ident = "Signal: (1/2)*cos(7x) term (amp 0.25)"
        elif abs(k) == 25:
            ident = "Noise:  (3/10)*sin(25x) term (amp 0.15)"
        elif abs(k) == 40:
            ident = "Noise:  (1/5)*cos(40x) term (amp 0.10)"
        else:
            ident = "Other"

        print(f"{idx:10d} | {k:14d} | {coeff.real:10.4f} + {coeff.imag:9.4f}j | {amp:12.4f} | {ident}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Part (b): Low-Pass Filtering and Parseval Identity
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (b): LOW-PASS FILTERING & DISCRETE PARSEVAL IDENTITY")
    print("=" * 80)

    print(r"""
1. Derivation of the Discrete Parseval Identity:
   Theorem 1.45 states that for uniform knots x_j = 2*pi*j / n, the normalized
   DFT matrix (1 / sqrt(n)) * V_n is unitary, where V_n = (omega_n^{j * k})_{j,k=0}^{n-1}.
   
   A unitary matrix preserves the Euclidean (2-norm) of any vector v in C^n:
       || (1 / sqrt(n)) * V_n * v ||_2^2 = || v ||_2^2.
   
   Since (V_n * v)_k = (F_n v)_k, the left-hand side expands to:
       || (1 / sqrt(n)) * F_n v ||_2^2 = (1 / n) * sum_{k=0}^{n-1} |(F_n v)_k|^2.
   
   And the right-hand side is simply the sum of squared magnitudes of v:
       || v ||_2^2 = sum_{j=0}^{n-1} |v_j|^2.
   
   Equating both sides yields the Discrete Parseval Identity:
       sum_{j=0}^{n-1} |v_j|^2 = (1 / n) * sum_{k=0}^{n-1} |(F_n v)_k|^2.   (Q.E.D.)

2. Prediction of the Root Mean Square Error (RMSE) before filtering:
   The error before filtering is the noise perturbation: v_j = y_j - s(x_j) = eta(x_j).
   Dividing the Parseval identity by n:
       (1 / n) * sum_{j=0}^{n-1} |eta(x_j)|^2 = sum_{k=0}^{n-1} |(F_n eta)_k / n|^2.
   
   The Fourier modes of eta are k = +-25 (amplitude 0.15) and k = +-40 (amplitude 0.10):
       sum_k |d_k|^2 = |d_{25}|^2 + |d_{-25}|^2 + |d_{40}|^2 + |d_{-40}|^2
                     = (0.15)^2 + (0.15)^2 + (0.10)^2 + (0.10)^2
                     = 2 * (0.0225) + 2 * (0.0100) = 0.0450 + 0.0200 = 0.0650.
   
   Taking the square root gives the PREDICTED root mean square error:
       RMSE_predicted = sqrt(0.0650) = sqrt(13 / 200) approx 0.25495098.
""")

    # Numerical verification of RMSE
    s_128 = s(x_128)
    rmse_before = np.sqrt(np.mean((y_128 - s_128) ** 2))
    rmse_predicted = np.sqrt(2 * (0.15 ** 2) + 2 * (0.10 ** 2))

    y_hat_filt_128, y_filt_128 = apply_low_pass_filter(y_hat_128, signed_k_128, cutoff=10)
    rmse_after = np.sqrt(np.mean((y_filt_128 - s_128) ** 2))

    print(f"Numerical Error Comparison for n = 128:")
    print(f"  - Computed RMSE before filtering:  {rmse_before:.12f}")
    print(f"  - Predicted RMSE via Parseval:      {rmse_predicted:.12f}")
    print(f"  - Discrepancy (Roundoff error):     {abs(rmse_before - rmse_predicted):.2e} <= 1e-15")
    print(f"  - Computed RMSE after filtering:   {rmse_after:.2e} (machine precision!)")
    print(f"  Conclusion: The ideal low-pass filter (|k| <= 10) completely removed the noise")
    print(f"  modes (k = +-25, +-40) without touching the signal modes (k = +-3, +-7).")

    # -------------------------------------------------------------------------
    # Part (c): Undersampling with n = 32 and Aliasing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (c): UNDERSAMPLING WITH n = 32 AND ALIASING ANALYSIS")
    print("=" * 80)

    n_32 = 32
    j_32 = np.arange(n_32)
    x_32 = 2.0 * np.pi * j_32 / n_32
    y_32 = y(x_32)
    s_32 = s(x_32)

    y_hat_32, amp_32, signed_k_32 = compute_spectrum(y_32)
    _, y_filt_32 = apply_low_pass_filter(y_hat_32, signed_k_32, cutoff=10)

    print(f"Undersampling parameters: n = {n_32}, Nyquist limit k_Nyquist = n/2 = {n_32 // 2}")
    print(f"Frequencies present in signal and noise:")
    print(f"  - Signal modes: k = +-3, +-7. Both <= 16 (within Nyquist band).")
    print(f"  - Noise modes:  k = +-25, +-40. Both > 16 (EXCEED Nyquist limit!).")
    print(f"\nAliasing Derivation (modulo n = 32):")
    print(r"  Because exp(i * k * x_j) = exp(i * (k + p*32) * x_j) for all integers p,")
    print(r"  frequencies outside [-16, 16] fold (alias) back into the principal band:")
    print(f"    * Mode k = +25 aliases to: 25 - 32 = -7")
    print(f"    * Mode k = -25 aliases to: -25 + 32 = +7")
    print(f"    * Mode k = +40 aliases to: 40 - 32 = +8")
    print(f"    * Mode k = -40 aliases to: -40 + 32 = -8")

    print(f"\nNonzero Modes Observed in Spectrum for n = 32:")
    print(f"{'FFT Index':>10} | {'Signed Mode k':>14} | {'Measured Coeff':>24} | {'Amplitude':>12} | {'Cause'}")
    print("-" * 80)
    for idx in np.where(amp_32 > 1e-10)[0]:
        k = signed_k_32[idx]
        coeff = y_hat_32[idx] / n_32
        amp = amp_32[idx]
        if abs(k) == 3:
            cause = "Pure signal sin(3x)"
        elif abs(k) == 7:
            cause = "Signal cos(7x) CORRUPTED by aliased noise sin(25x)!"
        elif abs(k) == 8:
            cause = "Pure aliased noise from cos(40x)!"
        else:
            cause = "Other"
        print(f"{idx:10d} | {k:14d} | {coeff.real:10.4f} + {coeff.imag:9.4f}j | {amp:12.4f} | {cause}")
    print("-" * 80)

    rmse_before_32 = np.sqrt(np.mean((y_32 - s_32) ** 2))
    rmse_after_32 = np.sqrt(np.mean((y_filt_32 - s_32) ** 2))

    print(f"\nConsequences for Low-Pass Filtering (|k| <= 10):")
    print(f"  - All aliased noise modes (k = +-7 and k = +-8) lie INSIDE the filter passband |k| <= 10!")
    print(f"  - Therefore, the low-pass filter is completely INCAPABLE of distinguishing noise from signal.")
    print(f"  - RMSE before filtering (n = 32): {rmse_before_32:.8f}")
    print(f"  - RMSE after filtering  (n = 32): {rmse_after_32:.8f}")
    print(f"  - The error is IDENTICAL before and after filtering: 0% noise was removed!")

    # -------------------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE 4-PANEL PLOT")
    print("=" * 80)
    plot_problem_4_5(n_128, n_32)


if __name__ == "__main__":
    main()
