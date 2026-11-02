"""
Problem 2.4: Best Approximation, Chebyshev Interpolation, and Taylor Polynomials
Numerical Computation - Problem Sheet 2
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Target function f(x) = 1 / (4 - x^2)."""
    return 1.0 / (4.0 - x**2)


def chebyshev_nodes(n: int) -> np.ndarray:
    """
    Computes the n + 1 Chebyshev nodes of the 1st kind on [-1, 1] (Theorem 1.21):
        x_i = cos((2*i + 1) * pi / (2*n + 2)),  i = 0, ..., n.
    """
    indices = np.arange(n + 1)
    return np.cos((2.0 * indices + 1.0) * np.pi / (2.0 * (n + 1)))


def neville_eval_vector(x_nodes: np.ndarray, f_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluates the polynomial interpolating (x_nodes, f_nodes) at a vector of points x_eval
    using the Aitken-Neville scheme.
    """
    n = len(x_nodes) - 1
    m_pts = len(x_eval)

    # Initialize tableau values: P[j] initially holds f(x_j) across all evaluation points
    # P has shape (n + 1, m_pts)
    P = np.tile(f_nodes[:, np.newaxis], (1, m_pts))

    for m in range(1, n + 1):
        for j in range(n - m + 1):
            denom = x_nodes[j + m] - x_nodes[j]
            term1 = (x_eval - x_nodes[j]) * P[j + 1]
            term2 = (x_eval - x_nodes[j + m]) * P[j]
            P[j] = (term1 - term2) / denom

    return P[0]


def taylor_2n_eval(x_eval: np.ndarray, n: int) -> np.ndarray:
    """
    Evaluates the Taylor polynomial T_{2n} f about x = 0 of degree 2n:
        T_{2n} f(x) = sum_{k=0}^n x^(2k) / 4^(k+1)
    """
    y = np.zeros_like(x_eval, dtype=float)
    x2 = x_eval**2
    term = 1.0 / 4.0  # for k = 0
    y += term
    for k in range(1, n + 1):
        term = term * (x2 / 4.0)
        y += term
    return y


def best_approx_upper_estimate(n: int) -> float:
    """
    Upper estimate for min_{q in P_n} ||f - q||_inf,[-1, 1] using Taylor polynomial T_n:
        ||f - T_n||_inf = 1 / (3 * 4^(floor(n/2) + 1))
    """
    return 1.0 / (3.0 * (4.0 ** (n // 2 + 1)))


if __name__ == "__main__":
    print("=" * 70)
    print("Problem 2.4: Best Approximation, Chebyshev Interpolation & Taylor")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Theoretical Part: Upper Estimate for Best Approximation Error
    # -------------------------------------------------------------------------
    print("\n--- Theoretical Upper Estimate ---")
    print("Function f(x) = 1 / (4 - x^2) on [-1, 1].")
    print("Taylor expansion around x0 = 0 (geometric series for |x| < 2):")
    print("  f(x) = (1/4) * sum_{k=0}^inf (x^2 / 4)^k = sum_{k=0}^inf x^(2k) / 4^(k+1)")
    print("\nChoosing q to be the Taylor polynomial T_n of degree n:")
    print("  T_n(x) = sum_{k=0}^{floor(n/2)} x^(2k) / 4^(k+1)")
    print("The remainder is:")
    print("  f(x) - T_n(x) = sum_{k=floor(n/2)+1}^inf x^(2k) / 4^(k+1)")
    print("For all x in [-1, 1], |x| <= 1, so the maximum is attained at x = +-1:")
    print("  ||f - T_n||_inf,[-1, 1] = sum_{k=floor(n/2)+1}^inf 1 / 4^(k+1)")
    print("                          = (1 / 4^(floor(n/2)+2)) / (1 - 1/4)")
    print("                          = 1 / (3 * 4^(floor(n/2)+1))")
    print("Hence, an upper estimate for min_{q in P_n} ||f - q||_inf,[-1, 1] is:")
    print("  min_{q in P_n} ||f - q||_inf <= 1 / (3 * 4^(floor(n/2)+1))")

    # -------------------------------------------------------------------------
    # Numerical Comparison: Chebyshev vs Taylor on 100 uniform evaluation points
    # -------------------------------------------------------------------------
    print("\n--- Computing Errors for n in {1, ..., 20} ---")
    eval_points = np.linspace(-1.0, 1.0, 100)
    f_true = f(eval_points)

    degrees = np.arange(1, 21)
    cheb_errors = []
    taylor_errors = []
    best_bounds = []

    for n in degrees:
        # 1. Chebyshev interpolation of degree n
        x_cheb = chebyshev_nodes(n)
        f_cheb = f(x_cheb)
        p_cheb = neville_eval_vector(x_cheb, f_cheb, eval_points)
        err_c = np.max(np.abs(f_true - p_cheb))
        cheb_errors.append(err_c)

        # 2. Taylor polynomial T_{2n} of degree 2n
        p_taylor = taylor_2n_eval(eval_points, n)
        err_t = np.max(np.abs(f_true - p_taylor))
        taylor_errors.append(err_t)

        # Best approximation bound for degree n
        best_bounds.append(best_approx_upper_estimate(n))

    # Print table of sample values
    print(f"{'n':<4} | {'||f - I_n^Cheb f||_inf':<24} | {'||f - T_{2n} f||_inf':<24} | {'Best approx bound (P_n)':<22}")
    print("-" * 80)
    for i in range(0, 20, 2):
        print(f"{degrees[i]:<4} | {cheb_errors[i]:<24.4e} | {taylor_errors[i]:<24.4e} | {best_bounds[i]:<22.4e}")
    print(f"{degrees[-1]:<4} | {cheb_errors[-1]:<24.4e} | {taylor_errors[-1]:<24.4e} | {best_bounds[-1]:<22.4e}")

    # -------------------------------------------------------------------------
    # Discussion & Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison of Approximation Quality:")
    print("=" * 70)
    print("1. Both Chebyshev interpolation and the Taylor polynomial exhibit")
    print("   exponential (geometric) convergence O(rho^(-n)).")
    print("   This is because f(z) = 1/(4 - z^2) is analytic inside the Bernstein")
    print("   ellipse with foci +-1 and semi-axis parameter rho = 2 + sqrt(3) ~ 3.732.")
    print("\n2. Degree comparison:")
    print("   - Chebyshev interpolation I_n^{Cheb} f uses polynomial degree n.")
    print("   - The Taylor polynomial T_{2n} f uses polynomial degree 2n.")
    print("   Even though T_{2n} uses TWICE the polynomial degree of I_n^{Cheb},")
    print("   Chebyshev of degree n achieves error comparable to T_{2n}!")
    print("   This highlights the superior efficiency of Chebyshev interpolation")
    print("   across the entire interval [-1, 1].")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.semilogy(degrees, cheb_errors, "bo-", linewidth=2, markersize=6,
                 label=r"Chebyshev: $\|f - I_n^{Cheb} f\|_{\infty, [-1, 1]}$ (degree $n$)")
    plt.semilogy(degrees, taylor_errors, "rs--", linewidth=2, markersize=6,
                 label=r"Taylor: $\|f - T_{2n} f\|_{\infty, [-1, 1]}$ (degree $2n$)")
    plt.semilogy(degrees, best_bounds, "g:", linewidth=1.8,
                 label=r"Taylor bound for $P_n$: $\frac{1}{3 \cdot 4^{\lfloor n/2 \rfloor + 1}}$")

    plt.xlabel("Degree parameter $n$", fontsize=12)
    plt.ylabel(r"Maximal Error $\|\cdot\|_{\infty, [-1, 1]}$", fontsize=12)
    plt.title(r"Approximation of $f(x) = \frac{1}{4 - x^2}$ on $[-1, 1]$", fontsize=13)
    plt.xticks(np.arange(1, 21))
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10.5)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "problem_2_4.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()

