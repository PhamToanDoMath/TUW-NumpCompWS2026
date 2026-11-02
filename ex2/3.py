"""
Problem 2.3: Neville Scheme and Extrapolation of Symmetric Difference Quotient
Numerical Computation - Problem Sheet 2
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def neville_scheme(x: np.ndarray, f: np.ndarray, x_eval: float = 0.0) -> np.ndarray:
    """
    Computes the full Neville scheme matrix at a given evaluation point x_eval.
    (Algorithm from Section 1.2 / Theorem 1.4 of the lecture notes).

    Parameters
    ----------
    x : np.ndarray
        Vector of knots of length n + 1.
    f : np.ndarray
        Vector of function values of length n + 1.
    x_eval : float, optional
        Point at which the interpolating polynomials are evaluated. Default is 0.0 (extrapolation to 0).

    Returns
    -------
    N : np.ndarray of shape (n + 1, n + 1)
        Matrix containing the columns of the Neville scheme.
        Column k contains the values p_{j, k}(x_eval) for j = 0, ..., n - k.
        Unused entries (j > n - k) are filled with np.nan.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(x) - 1

    N = np.full((n + 1, n + 1), np.nan, dtype=float)
    # Column 0: p_{j, 0}(x) = f_j
    N[:, 0] = f

    # Recurrence relation:
    # p_{j, m}(x) = ((x - x_j) * p_{j+1, m-1}(x) - (x - x_{j+m}) * p_{j, m-1}(x)) / (x_{j+m} - x_j)
    for m in range(1, n + 1):
        for j in range(n - m + 1):
            N[j, m] = ((x_eval - x[j]) * N[j + 1, m - 1] - (x_eval - x[j + m]) * N[j, m - 1]) / (x[j + m] - x[j])

    return N


def D_sym(func, x0: float, h: float | np.ndarray):
    """
    Symmetric difference quotient:
        D_sym(h) = (func(x0 + h) - func(x0 - h)) / (2 * h)
    """
    return (func(x0 + h) - func(x0 - h)) / (2.0 * h)


if __name__ == "__main__":
    print("=" * 65)
    print("Problem 2.3: Neville Scheme & Extrapolation of Difference Quotient")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Part (a): Verification of Neville Scheme
    # -------------------------------------------------------------------------
    print("\n--- Part (a) Verification ---")
    # Quick sanity check with known polynomial: e.g. f(x) = x^2, knots [0, 1, 2]
    # At x = 3, p(3) = 9
    x_test = np.array([0.0, 1.0, 2.0])
    f_test = x_test**2
    N_test = neville_scheme(x_test, f_test, x_eval=3.0)
    print(f"Test on f(x) = x^2, knots [0, 1, 2], evaluating at x = 3:")
    print(f"Extrapolated p(3) = {N_test[0, 2]:.4f} (Expected: 9.0000)")

    # -------------------------------------------------------------------------
    # Part (b): Extrapolation of D_sym(h) for f(x) = tan(x) at x0 = pi/4
    # -------------------------------------------------------------------------
    print("\n--- Part (b) Extrapolation ---")
    x0 = np.pi / 4.0
    exact_derivative = 2.0  # f'(x) = 1/cos^2(x) => f'(pi/4) = 1 / (1/sqrt(2))^2 = 2
    n_max = 10

    # Knot vector: h_i = 2^(-i), i = 0, ..., n_max
    indices = np.arange(n_max + 1)
    h_nodes = 2.0**(-indices)
    f_vals = D_sym(np.tan, x0, h_nodes)

    # Generate Neville scheme for extrapolation at h = 0
    N = neville_scheme(h_nodes, f_vals, x_eval=0.0)

    print(f"Knots h_i: {h_nodes[:4]} ... {h_nodes[-1]}")
    print(f"D_sym(h_i): {f_vals[:4]} ...")
    print(f"Exact limit f'({x0:.4f}) = {exact_derivative}")

    # Compute errors for the first 4 columns (k = 0, 1, 2, 3)
    errors = {}
    h_cols = {}
    rates = {}

    print("\nConvergence Rates for the First 4 Columns:")
    print("-" * 65)
    for k in range(4):
        valid_rows = n_max - k + 1
        h_cols[k] = h_nodes[:valid_rows]
        errors[k] = np.abs(N[:valid_rows, k] - exact_derivative)

        # Empirical convergence rates: log2(err_i / err_{i+1}) since h_{i+1} = h_i / 2
        col_rates = np.log2(errors[k][:-1] / errors[k][1:])
        rates[k] = col_rates
        print(f"Column {k} (Polynomial degree {k}):")
        print(f"  First 4 errors: {errors[k][:4]}")
        print(f"  Asymptotic rate (last 3 steps): {col_rates[-3:].round(3)} -> Rate ~ {round(col_rates[-1])}")

    # -------------------------------------------------------------------------
    # Discussion of Observed Rates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Observations & Explanation:")
    print("=" * 65)
    print("1. Column 0 (k = 0): Shows O(h^2) convergence (rate ~ 2).")
    print("   Since D_sym(h) is an even function of h:")
    print("     D_sym(h) = f'(x_0) + c_2*h^2 + c_4*h^4 + O(h^6).")
    print("   The leading error term is c_2 * h^2.")
    print("\n2. Column 1 (k = 1): Shows O(h^2) convergence (rate ~ 2).")
    print("   Linear extrapolation eliminates O(h) terms, but D_sym(h) has NO odd powers!")
    print("   Hence, the error remains O(h^2), though the constant is reduced.")
    print("\n3. Column 2 (k = 2): Shows O(h^4) convergence (rate ~ 4).")
    print("   Quadratic extrapolation eliminates both O(h) and O(h^2) terms.")
    print("   The next non-zero term in the Taylor expansion is O(h^4).")
    print("\n4. Column 3 (k = 3): Shows O(h^4) convergence (rate ~ 4).")
    print("   Cubic extrapolation eliminates up to O(h^3), but there is NO O(h^3) term.")
    print("   Thus, the error remains O(h^4) (rate ~ 4).")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    markers = ["o", "s", "^", "d"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for k in range(4):
        plt.loglog(h_cols[k], errors[k], marker=markers[k], color=colors[k],
                   linewidth=1.8, label=f"Column {k} (degree {k})")

    # Auxiliary reference lines h^2, h^3, h^4
    # Scaled to be clearly visible on the plot
    h_ref = np.logspace(-3, 0, 50)
    plt.loglog(h_ref, 2.0 * h_ref**2, "k--", alpha=0.7, label=r"Auxiliary $\propto h^2$")
    plt.loglog(h_ref, 0.5 * h_ref**3, "m-.", alpha=0.7, label=r"Auxiliary $\propto h^3$")
    plt.loglog(h_ref, 0.1 * h_ref**4, "c:", linewidth=2, alpha=0.9, label=r"Auxiliary $\propto h^4$")

    plt.xlabel("Step size $h$", fontsize=12)
    plt.ylabel(r"Error $|N_{j, k} - f'(x_0)|$", fontsize=12)
    plt.title(r"Extrapolation of $D_{sym}(h)$ for $f(x) = \tan(x)$ at $x_0 = \pi/4$", fontsize=13)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "problem_2_3.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()

