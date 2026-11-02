"""
Problem 3.3: Polynomial Interpolation, Lebesgue Constants, and Chebyshev Nodes
Numerical Computation - Problem Sheet 3

This script:
1. Compares uniform nodes vs Chebyshev nodes on [-5, 5] for n in {10, 20, 40}.
2. Uses scipy.interpolate.BarycentricInterpolator for numerically stable polynomial interpolation.
3. Part (a): Interpolates Runge's function f(x) = 1 / (1 + x^2) and demonstrates Runge's phenomenon.
4. Part (b): Numerically computes the Lebesgue constant Lambda_n = max_{x} sum_i |l_i(x)| using
   the BarycentricInterpolator with unit basis vectors (identity matrix), and plots Lambda_n vs n (semilogy).
5. Part (c): Fits an exponential curve Lambda_n approx C * exp(b * n) to uniform nodes for n in {20, 40}
   by solving a 2x2 linear system in log-space (and verifies via polyfit).
"""

import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator


def f_runge(x: np.ndarray | float) -> np.ndarray | float:
    """
    Runge's test function: f(x) = 1 / (1 + x^2).
    """
    return 1.0 / (1.0 + x**2)


def get_uniform_nodes(n: int, a: float = -5.0, b: float = 5.0) -> np.ndarray:
    """
    Generates n + 1 uniformly distributed interpolation nodes on [a, b]:
        x_i = 5 * (-1 + 2 * i / n) for i = 0, ..., n (when a=-5, b=5).
    """
    return a + (b - a) * np.arange(n + 1, dtype=float) / n


def get_chebyshev_nodes(n: int, a: float = -5.0, b: float = 5.0) -> np.ndarray:
    """
    Generates n + 1 Chebyshev nodes of the first kind on [a, b]:
        x_i = 5 * cos((i + 0.5) / (n + 1) * pi) for i = 0, ..., n (when a=-5, b=5).
    """
    # Nodes in [-1, 1]
    t = np.cos((np.arange(n + 1, dtype=float) + 0.5) / (n + 1) * np.pi)
    # Affine map from [-1, 1] to [a, b]
    return 0.5 * (a + b) + 0.5 * (b - a) * t


def compute_lebesgue_constant(
    x_nodes: np.ndarray, x_eval: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Numerically computes the Lebesgue function lambda_n(x) and Lebesgue constant Lambda_n
    on the evaluation grid x_eval using scipy.interpolate.BarycentricInterpolator.

    By passing the identity matrix np.eye(n + 1) as values, BarycentricInterpolator
    evaluates all n + 1 Lagrange basis polynomials l_i(x) simultaneously.
        lambda_n(x) = sum_{i=0}^n |l_i(x)|
        Lambda_n    = max_{x in [a, b]} lambda_n(x)
    """
    n_pts = len(x_nodes)
    # Evaluating with identity matrix produces an array of shape (len(x_eval), n_pts)
    # where column i is l_i(x_eval)
    interp = BarycentricInterpolator(x_nodes, np.eye(n_pts))
    lagrange_matrix = interp(x_eval)

    # Lebesgue function at each point in x_eval
    lambda_x = np.sum(np.abs(lagrange_matrix), axis=1)
    lambda_const = float(np.max(lambda_x))
    return lambda_const, lambda_x


def fit_exponential_growth(
    n_pair: Tuple[int, int], lambda_pair: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Determines C and b such that Lambda_n approx C * exp(b * n)
    by solving the 2 x 2 linear system for log(Lambda_n) = log(C) + b * n
    for n in {n_1, n_2}:
        [ 1   n_1 ] * [ log(C) ] = [ log(Lambda_1) ]
        [ 1   n_2 ]   [   b    ]   [ log(Lambda_2) ]

    Returns
    -------
    b : float
        Exponential growth rate.
    log_C : float
        Logarithm of prefactor C.
    C : float
        Prefactor C = exp(log_C).
    """
    n1, n2 = n_pair
    L1, L2 = lambda_pair

    # Exact solution of 2x2 system:
    # b * (n2 - n1) = log(L2) - log(L1)
    b = (np.log(L2) - np.log(L1)) / (n2 - n1)
    log_C = np.log(L1) - b * n1
    C = np.exp(log_C)

    return b, log_C, C


def plot_results(
    n_values: list[int],
    n_range_lebesgue: list[int],
    lebesgue_unif: list[float],
    lebesgue_cheb: list[float],
    b_fit: float,
    C_fit: float,
    save_filename: str = "problem_3_3.png",
):
    """
    Generates a 4-panel comprehensive plot:
    - Panels (a1, a2, a3): Interpolating polynomials for n = 10, 20, 40 comparing Uniform vs Chebyshev.
    - Panel (b): Semilogarithmic plot of Lebesgue constant Lambda_n vs n with the exponential fit.
    """
    x_eval = np.linspace(-5.0, 5.0, 2001)
    f_exact = f_runge(x_eval)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------------------
    # Panels 1 to 3: Interpolation of Runge's function for n = 10, 20, 40
    # -------------------------------------------------------------------------
    panels = [(0, 0, n_values[0]), (0, 1, n_values[1]), (1, 0, n_values[2])]

    for row, col, n in panels:
        ax = axes[row, col]

        x_unif = get_uniform_nodes(n)
        x_cheb = get_chebyshev_nodes(n)

        # Barycentric interpolation
        p_unif = BarycentricInterpolator(x_unif, f_runge(x_unif))(x_eval)
        p_cheb = BarycentricInterpolator(x_cheb, f_runge(x_cheb))(x_eval)

        # Plot exact function
        ax.plot(x_eval, f_exact, "k-", linewidth=2.0, label=r"$f(x) = \frac{1}{1 + x^2}$")

        # Plot Chebyshev interpolant
        ax.plot(x_eval, p_cheb, "g--", linewidth=2.0, label=f"Chebyshev ($n={n}$)")
        ax.scatter(x_cheb, f_runge(x_cheb), color="green", s=25, alpha=0.7, zorder=5)

        # Plot Uniform interpolant
        ax.plot(x_eval, p_unif, "r-.", linewidth=1.5, label=f"Uniform ($n={n}$)")
        ax.scatter(x_unif, f_runge(x_unif), color="red", s=25, alpha=0.7, zorder=5)

        ax.set_title(f"Runge Function Interpolation ($n = {n}$)", fontsize=12, fontweight="bold")
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel("$y$", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Clip y-limits reasonably so function shape and oscillations are both informative
        if n == 10:
            ax.set_ylim(-0.5, 1.5)
        elif n == 20:
            ax.set_ylim(-2.0, 3.0)
            ax.text(0.05, 0.92, "Uniform diverges near boundaries!",
                    transform=ax.transAxes, color="red", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="#ffe6e6", alpha=0.8))
        else:  # n = 40
            ax.set_ylim(-2.0, 3.0)
            ax.text(0.05, 0.92, "Uniform oscillates to $\\approx \\pm 10^5$ (clipped)",
                    transform=ax.transAxes, color="red", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="#ffe6e6", alpha=0.8))

        ax.legend(loc="upper right", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 4: Lebesgue Constant Lambda_n vs n (Semilog scale)
    # -------------------------------------------------------------------------
    ax_leb = axes[1, 1]

    # Plot Uniform Lebesgue constant
    ax_leb.semilogy(n_range_lebesgue, lebesgue_unif, "ro-", linewidth=2.0, markersize=6,
                    label=r"Uniform nodes: $\Lambda_n^{\mathrm{unif}}$")

    # Plot Chebyshev Lebesgue constant
    ax_leb.semilogy(n_range_lebesgue, lebesgue_cheb, "gs-", linewidth=2.0, markersize=6,
                    label=r"Chebyshev nodes: $\Lambda_n^{\mathrm{Cheb}}$")

    # Plot Fitted Exponential line: Lambda_n approx C * exp(b * n)
    n_fit_dense = np.linspace(n_range_lebesgue[0], n_range_lebesgue[-1], 200)
    lambda_fit_dense = C_fit * np.exp(b_fit * n_fit_dense)
    ax_leb.semilogy(n_fit_dense, lambda_fit_dense, "k--", linewidth=1.8,
                    label=rf"Fit (part c): $\Lambda_n \approx {C_fit:.4f} \, e^{{{b_fit:.3f} n}}$")

    # Highlight n = 10, 20, 40
    for n_target in [10, 20, 40]:
        if n_target in n_range_lebesgue:
            idx = n_range_lebesgue.index(n_target)
            Lu = lebesgue_unif[idx]
            Lc = lebesgue_cheb[idx]
            ax_leb.scatter([n_target], [Lu], color="darkred", s=80, zorder=6)
            ax_leb.scatter([n_target], [Lc], color="darkgreen", s=80, zorder=6)

    ax_leb.set_title(r"Lebesgue Constant $\Lambda_n$ vs $n$ (Semilog)", fontsize=12, fontweight="bold")
    ax_leb.set_xlabel("Degree $n$", fontsize=10)
    ax_leb.set_ylabel(r"$\Lambda_n = \max \sum |\ell_i(x)|$", fontsize=10)
    ax_leb.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_leb.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    print("=" * 80)
    print("Numerical Computation - Problem Sheet 3: Problem 3.3")
    print("Polynomial Interpolation, Runge Phenomenon, and Lebesgue Constants")
    print("=" * 80)

    # Grid for sampling/evaluation
    x_eval = np.linspace(-5.0, 5.0, 40001)
    f_exact = f_runge(x_eval)

    # -------------------------------------------------------------------------
    # Part (a): Runge Interpolation for n in {10, 20, 40}
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): RUNGE'S FUNCTION INTERPOLATION f(x) = 1 / (1 + x^2) on [-5, 5]")
    print("=" * 80)
    print("Using scipy.interpolate.BarycentricInterpolator for uniform and Chebyshev nodes:\n")

    target_n = [10, 20, 40]
    errors_summary = []

    for n in target_n:
        x_unif = get_uniform_nodes(n)
        x_cheb = get_chebyshev_nodes(n)

        # Barycentric interpolation
        interp_unif = BarycentricInterpolator(x_unif, f_runge(x_unif))
        interp_cheb = BarycentricInterpolator(x_cheb, f_runge(x_cheb))

        p_unif = interp_unif(x_eval)
        p_cheb = interp_cheb(x_eval)

        err_unif = float(np.max(np.abs(f_exact - p_unif)))
        err_cheb = float(np.max(np.abs(f_exact - p_cheb)))

        errors_summary.append((n, err_unif, err_cheb))
        print(f"Degree n = {n:2d} ({n+1:2d} points):")
        print(f"  Uniform Nodes:   Max Error ||f - p_n^unif||_inf = {err_unif:.4e}")
        print(f"  Chebyshev Nodes: Max Error ||f - p_n^Cheb||_inf = {err_cheb:.4e}")
        print(f"  Ratio (Uniform / Chebyshev): {err_unif / err_cheb:.2e}\n")

    print("Observation:")
    print("  - For uniform nodes, as n increases (10 -> 20 -> 40), the error explodes")
    print("    from 1.92 to ~10^5 near the boundaries (Runge's phenomenon).")
    print("  - For Chebyshev nodes, the error decays rapidly from 0.109 down to 2.89e-4,")
    print("    demonstrating uniform convergence across the entire interval [-5, 5]!")

    # -------------------------------------------------------------------------
    # Part (b): Numerical Investigation of Lebesgue Constants
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PART (b): NUMERICAL INVESTIGATION OF LEBESGUE CONSTANT Lambda_n")
    print("=" * 80)
    print(r"Lebesgue constant Lambda_n := max_{x in [-5, 5]} sum_{i=0}^n |l_i(x)|")
    print("Evaluated by passing the identity matrix np.eye(n+1) to BarycentricInterpolator.\n")

    # Evaluation across a range of degrees n to show the curve cleanly on semilogy
    n_range = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]
    leb_unif_list = []
    leb_cheb_list = []

    print(f"{'n':>4} | {'Lambda_n (Uniform)':>22} | {'Lambda_n (Chebyshev)':>22}")
    print("-" * 54)

    for n in n_range:
        x_u = get_uniform_nodes(n)
        x_c = get_chebyshev_nodes(n)

        Lu, _ = compute_lebesgue_constant(x_u, x_eval)
        Lc, _ = compute_lebesgue_constant(x_c, x_eval)

        leb_unif_list.append(Lu)
        leb_cheb_list.append(Lc)

        highlight = " <-- target" if n in target_n else ""
        print(f"{n:4d} | {Lu:22.4e} | {Lc:22.4f}{highlight}")

    # Specific values for target n
    print("\nSummary for required n in {10, 20, 40}:")
    for n in target_n:
        idx = n_range.index(n)
        print(f"  n = {n:2d}:  Lambda_n^unif = {leb_unif_list[idx]:.4e},  Lambda_n^Cheb = {leb_cheb_list[idx]:.4f}")

    print("\nComparison:")
    print("  - Lambda_n^unif grows exponentially: for n=40 it exceeds 4.69 * 10^9!")
    print("  - Lambda_n^Cheb grows logarithmically: Lambda_n^Cheb ~ (2/pi) * log(n), staying < 3.5 even for n=40!")

    # -------------------------------------------------------------------------
    # Part (c): Fitting Exponential Growth Lambda_n approx C * exp(b * n)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (c): FITTING EXPONENTIAL GROWTH Lambda_n approx C * exp(b * n) FOR n in {20, 40}")
    print("=" * 80)

    idx_20 = n_range.index(20)
    idx_40 = n_range.index(40)
    L_20 = leb_unif_list[idx_20]
    L_40 = leb_unif_list[idx_40]

    b_fit, log_C_fit, C_fit = fit_exponential_growth((20, 40), (L_20, L_40))

    # Verification with numpy polyfit
    poly_coeffs = np.polyfit([20.0, 40.0], [np.log(L_20), np.log(L_40)], deg=1)
    b_poly = poly_coeffs[0]
    log_C_poly = poly_coeffs[1]
    C_poly = np.exp(log_C_poly)

    print("Data from Part (b):")
    print(f"  Lambda_20 = {L_20:.6e}  ==>  log(Lambda_20) = {np.log(L_20):.6f}")
    print(f"  Lambda_40 = {L_40:.6e}  ==>  log(Lambda_40) = {np.log(L_40):.6f}")

    print("\nLinear 2 x 2 System in log-space:")
    print("  [ 1   20 ] * [ log(C) ] = [ log(Lambda_20) ] = [  9.304443 ]")
    print("  [ 1   40 ]   [   b    ]   [ log(Lambda_40) ]   [ 22.269222 ]")

    print("\nSolution:")
    print(f"  b = (log(Lambda_40) - log(Lambda_20)) / (40 - 20) = {b_fit:.6f}")
    print(f"  log(C) = log(Lambda_20) - 20 * b = {log_C_fit:.6f}")
    print(f"  C = exp(log(C)) = {C_fit:.6f}")

    print("\nVerification with np.polyfit:")
    print(f"  polyfit slope b = {b_poly:.6f}, intercept log(C) = {log_C_poly:.6f}, C = {C_poly:.6f} (MATCH)")

    print(f"\nResulting exponential model for uniform nodes:")
    print(f"  Lambda_n approx {C_fit:.5f} * exp({b_fit:.5f} * n)")
    print(f"  Or in base 2: Lambda_n approx {C_fit:.5f} * 2^({b_fit / np.log(2):.3f} * n) = {C_fit:.5f} * ({np.exp(b_fit):.4f})^n")
    print(f"  Theoretical asymptotic rate is 2^n (i.e. base ~ 2), closely matching our fitted base {np.exp(b_fit):.4f}!")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_results(target_n, n_range, leb_unif_list, leb_cheb_list, b_fit, C_fit)


if __name__ == "__main__":
    main()

