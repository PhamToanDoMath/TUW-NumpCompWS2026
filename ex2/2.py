"""
Problem 2.2: Quadratic Interpolation Error Estimation
Numerical Computation - Problem Sheet 2
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Target function f(x) = exp(x)."""
    return np.exp(x)


def quadratic_interpolant(x_eval: np.ndarray | float, x_nodes: np.ndarray, f_nodes: np.ndarray):
    """
    Evaluates the quadratic interpolating polynomial passing through (x_0, f_0), (x_1, f_1), (x_2, f_2)
    using the Lagrange form:
        p(x) = f_0 * L_0(x) + f_1 * L_1(x) + f_2 * L_2(x)
    """
    x = np.asarray(x_eval, dtype=float)
    x0, x1, x2 = x_nodes
    f0, f1, f2 = f_nodes

    L0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
    L1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
    L2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))

    p = f0 * L0 + f1 * L1 + f2 * L2
    if np.isscalar(x_eval):
        return float(p)
    return p


def pointwise_error_bound(x_eval: np.ndarray | float) -> np.ndarray | float:
    """
    Computes the upper bound for the pointwise interpolation error |f(x) - p(x)|
    using Theorem 1.17 (Theorem 1.15 in lecture notes):
        |f(x) - p(x)| <= (max_{xi in [0, 2]} |f'''(xi)| / 3!) * |(x - x_0)(x - x_1)(x - x_2)|
    For f(x) = exp(x) on [0, 2]:
        f'''(xi) = exp(xi) <= exp(2) = e^2.
    Thus:
        |f(x) - p(x)| <= (e^2 / 6) * |x * (x - 1) * (x - 2)|
    """
    x = np.asarray(x_eval, dtype=float)
    bound = (np.exp(2.0) / 6.0) * np.abs(x * (x - 1.0) * (x - 2.0))
    if np.isscalar(x_eval):
        return float(bound)
    return bound


def maximal_error_bound() -> float:
    """
    Computes the upper bound for the maximal error ||f - p||_inf,[0, 2].
    ||f - p||_inf <= (e^2 / 6) * max_{x in [0, 2]} |x(x - 1)(x - 2)|
    g(x) = x(x - 1)(x - 2) has extrema at x = 1 +- 1/sqrt(3), where |g(x)| = 2*sqrt(3) / 9.
    Therefore:
        ||f - p||_inf <= (e^2 / 6) * (2*sqrt(3) / 9) = (sqrt(3) * e^2) / 27.
    """
    return (np.sqrt(3.0) * np.exp(2.0)) / 27.0


if __name__ == "__main__":
    print("=" * 60)
    print("Problem 2.2: Quadratic Interpolation of f(x) = exp(x)")
    print("=" * 60)

    # Interpolation nodes and values
    x_nodes = np.array([0.0, 1.0, 2.0])
    f_nodes = np.array([1.0, math.e, math.e**2])

    print("Interpolation nodes: (0, 1), (1, e), (2, e^2)")

    # 1) Error at x = 1.2
    x_target = 1.2
    f_target = f(x_target)
    p_target = quadratic_interpolant(x_target, x_nodes, f_nodes)
    actual_err_target = abs(f_target - p_target)
    bound_target = pointwise_error_bound(x_target)

    print(f"\n--- Pointwise Evaluation at x = {x_target} ---")
    print(f"Exact f({x_target})      = {f_target:.8f}")
    print(f"Polynomial p({x_target}) = {p_target:.8f}")
    print(f"Actual error |f(1.2) - p(1.2)| = {actual_err_target:.8f}")
    print(f"Theoretical pointwise upper estimate = {bound_target:.8f}")
    print(f"Check: actual error <= upper estimate? {actual_err_target <= bound_target} "
          f"(Ratio bound/error = {bound_target / actual_err_target:.2f})")

    # 2) General Pointwise Error Formula
    print("\n--- Pointwise Error Formula (Thm. 1.17) ---")
    print("For any x in [0, 2], there exists xi in (0, 2) such that:")
    print("  f(x) - p(x) = (f'''(xi) / 3!) * (x - 0)(x - 1)(x - 2)")
    print("Since f'''(xi) = exp(xi) <= exp(2) = e^2 for all xi in [0, 2]:")
    print("  |f(x) - p(x)| <= (e^2 / 6) * |x(x - 1)(x - 2)|")

    # 3) Upper Estimate for Maximal Error ||f - p||_inf,[0, 2]
    max_bound = maximal_error_bound()

    # Numerical verification on a fine grid
    x_fine = np.linspace(0.0, 2.0, 10001)
    f_fine = f(x_fine)
    p_fine = quadratic_interpolant(x_fine, x_nodes, f_nodes)
    err_fine = np.abs(f_fine - p_fine)

    actual_max_err = np.max(err_fine)
    x_max_idx = np.argmax(err_fine)
    x_max_actual = x_fine[x_max_idx]

    print("\n--- Maximal Error Estimate on [0, 2] ---")
    print(f"Maximum of |omega_3(x)| = |x(x-1)(x-2)| on [0, 2] is 2*sqrt(3)/9 approx {2*np.sqrt(3)/9:.6f}")
    print(f"Theoretical upper bound for ||f - p||_inf = (sqrt(3)*e^2)/27 approx {max_bound:.8f}")
    print(f"Numerically observed max error ||f - p||_inf = {actual_max_err:.8f} at x = {x_max_actual:.4f}")
    print(f"Check: observed max <= theoretical bound? {actual_max_err <= max_bound} "
          f"(Ratio bound/max_err = {max_bound / actual_max_err:.2f})")

    # 4) Plotting
    plt.figure(figsize=(11, 5))

    # Subplot 1: Functions
    plt.subplot(1, 2, 1)
    plt.plot(x_fine, f_fine, "b-", label=r"$f(x) = \exp(x)$", linewidth=2)
    plt.plot(x_fine, p_fine, "r--", label=r"$p(x)$ (quadratic)", linewidth=2)
    plt.scatter(x_nodes, f_nodes, color="black", s=50, zorder=5, label="Nodes $(x_i, e^{x_i})$")
    plt.scatter([x_target], [p_target], color="darkviolet", marker="s", s=60, zorder=6,
                label=f"$p(1.2) = {p_target:.3f}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Approximation of $\exp(x)$ on $[0, 2]$")
    plt.legend()
    plt.grid(True, linestyle=":")

    # Subplot 2: Errors
    pw_bounds = pointwise_error_bound(x_fine)
    plt.subplot(1, 2, 2)
    plt.plot(x_fine, err_fine, "k-", label=r"Actual error $|f(x) - p(x)|$", linewidth=2)
    plt.plot(x_fine, pw_bounds, "m--", label=r"Pointwise bound $\frac{e^2}{6}|\omega_3(x)|$", linewidth=2)
    plt.axhline(max_bound, color="red", linestyle=":", label=f"Maximal bound: {max_bound:.4f}")
    plt.scatter([x_target], [actual_err_target], color="blue", s=50, zorder=5,
                label=f"Error at 1.2: {actual_err_target:.4f}")
    plt.scatter([x_target], [bound_target], color="magenta", marker="^", s=60, zorder=5,
                label=f"Bound at 1.2: {bound_target:.4f}")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title(r"Pointwise Error and Upper Bounds on $[0, 2]$")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "problem_2_2.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()

