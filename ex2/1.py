"""
Problem 2.1: Newton Interpolation and Horner Scheme
Numerical Computation - Problem Sheet 2
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def newton_coefficients(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Computes the Newton divided difference coefficients d_j = f[x_0, ..., x_j].

    Parameters
    ----------
    x : np.ndarray
        Interpolation nodes (knots) of length n + 1.
    f : np.ndarray
        Function values at the nodes of length n + 1.

    Returns
    -------
    d : np.ndarray
        Newton coefficients [d_0, d_1, ..., d_n] where d_j = f[x_0, ..., x_j].
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(f, dtype=float).copy()
    n = len(x) - 1

    # In-place divided differences table computation
    for j in range(1, n + 1):
        for i in range(n, j - 1, -1):
            d[i] = (d[i] - d[i - 1]) / (x[i] - x[i - j])

    return d


def horner_eval(x_nodes: np.ndarray, d: np.ndarray, x_eval: np.ndarray | float) -> np.ndarray:
    """
    Evaluates the Newton interpolation polynomial at evaluation point(s) x_eval
    using the generalized Horner scheme (Algorithm 2 from lecture notes).

    p(x) = d_0 + (x - x_0) * [d_1 + (x - x_1) * [ ... [d_{n-1} + (x - x_{n-1}) * d_n] ... ]]

    Parameters
    ----------
    x_nodes : np.ndarray
        Knots x_0, ..., x_n of length n + 1.
    d : np.ndarray
        Newton coefficients d_0, ..., d_n of length n + 1.
    x_eval : np.ndarray or float
        Point(s) at which the polynomial is evaluated.

    Returns
    -------
    y : np.ndarray or float
        Evaluated polynomial values at x_eval.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    d = np.asarray(d, dtype=float)
    x_e = np.asarray(x_eval, dtype=float)

    n = len(d) - 1
    y = np.full_like(x_e, d[n], dtype=float)

    for j in range(n - 1, -1, -1):
        y = d[j] + (x_e - x_nodes[j]) * y

    if np.isscalar(x_eval):
        return float(y)
    return y


def compute_divided_differences_table(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Computes and returns the full divided differences table matrix.
    Table entry T[i, j] corresponds to f[x_{i-j}, ..., x_i].
    """
    n = len(x)
    table = np.full((n, n), np.nan)
    table[:, 0] = f
    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = (table[i, j - 1] - table[i - 1, j - 1]) / (x[i] - x[i - j])
    return table


if __name__ == "__main__":
    def f_func(x):
        return 1.0 / (1.0 + x)

    print("=" * 60)
    print("Problem 2.1: Newton Interpolation and Horner Scheme")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Part (a): Analytical and numerical divided differences
    # -------------------------------------------------------------------------
    print("\n--- Part (a) ---")
    x_nodes_2 = np.array([0.0, 1.0, 2.0])
    f_nodes_2 = f_func(x_nodes_2)

    table_2 = compute_divided_differences_table(x_nodes_2, f_nodes_2)
    print("Nodes: x0 = 0, x1 = 1, x2 = 2")
    print(f"f[x0] = {table_2[0, 0]:.4f} (1)")
    print(f"f[x1] = {table_2[1, 0]:.4f} (1/2)")
    print(f"f[x2] = {table_2[2, 0]:.4f} (1/3)")
    print(f"f[x0, x1] = {table_2[1, 1]:.4f} (-1/2)")
    print(f"f[x1, x2] = {table_2[2, 1]:.4f} (-1/6)")
    print(f"f[x0, x1, x2] = {table_2[2, 2]:.4f} (1/6)")

    d_quad = newton_coefficients(x_nodes_2, f_nodes_2)
    print(f"Quadratic Newton coefficients [d0, d1, d2]: {d_quad}")
    print("p2(x) = 1 - 0.5 * x + (1/6) * x * (x - 1)")

    # Adding the data point (4, f(4))
    x_nodes_3 = np.array([0.0, 1.0, 2.0, 4.0])
    f_nodes_3 = f_func(x_nodes_3)
    table_3 = compute_divided_differences_table(x_nodes_3, f_nodes_3)

    print("\nAdding data point (4, f(4)) = (4, 1/5):")
    print(f"f[x2, x3] = {table_3[3, 1]:.6f} (-1/15)")
    print(f"f[x1, x2, x3] = {table_3[3, 2]:.6f} (1/30)")
    print(f"f[x0, x1, x2, x3] = {table_3[3, 3]:.6f} (-1/30)")

    d_cubic = newton_coefficients(x_nodes_3, f_nodes_3)
    print(f"Cubic Newton coefficients [d0, d1, d2, d3]: {d_cubic}")
    print("p3(x) = p2(x) - (1/30) * x * (x - 1) * (x - 2)")

    # -------------------------------------------------------------------------
    # Part (b): Horner scheme evaluation at x = 3
    # -------------------------------------------------------------------------
    print("\n--- Part (b) ---")
    print("Nested form of p3(x):")
    print("p3(x) = 1 + x * (-1/2 + (x - 1) * (1/6 + (x - 2) * (-1/30)))")

    # Manual Horner steps at x = 3:
    x_val = 3.0
    y3 = -1.0 / 30.0
    y2 = 1.0 / 6.0 + (x_val - 2.0) * y3
    y1 = -0.5 + (x_val - 1.0) * y2
    y0 = 1.0 + (x_val - 0.0) * y1

    p3_at_3 = horner_eval(x_nodes_3, d_cubic, x_val)
    f_at_3 = f_func(x_val)
    err_at_3 = abs(f_at_3 - p3_at_3)

    print(f"Horner evaluation step-by-step at x = 3:")
    print(f"  Step 3: y = d3 = {y3:.6f} (-1/30)")
    print(f"  Step 2: y = d2 + (3 - 2) * y = {y2:.6f} (2/15)")
    print(f"  Step 1: y = d1 + (3 - 1) * y = {y1:.6f} (-7/30)")
    print(f"  Step 0: y = d0 + (3 - 0) * y = {y0:.6f} (3/10 = 0.3)")
    print(f"p3(3) from horner_eval = {p3_at_3:.6f}")
    print(f"f(3) = 1 / (1 + 3) = {f_at_3:.6f}")
    print(f"Error |f(3) - p3(3)| = |{f_at_3} - {p3_at_3}| = {err_at_3:.6f} (1/20 = 0.05)")

    # -------------------------------------------------------------------------
    # Part (c): Sampling and maximum error on [0, 4]
    # -------------------------------------------------------------------------
    print("\n--- Part (c) ---")
    x_eval_points = np.linspace(0.0, 4.0, 1001)
    f_vals = f_func(x_eval_points)
    p3_vals = horner_eval(x_nodes_3, d_cubic, x_eval_points)

    errors = np.abs(f_vals - p3_vals)
    max_error = np.max(errors)
    max_error_idx = np.argmax(errors)
    x_max_error = x_eval_points[max_error_idx]

    print(f"Maximum error ||f - p3||_inf,[0,4] = {max_error:.6f}")
    print(f"Achieved at x approx = {x_max_error:.4f}")

    # Plot f(x) and p3(x)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_eval_points, f_vals, "b-", label=r"$f(x) = \frac{1}{1+x}$", linewidth=2)
    plt.plot(x_eval_points, p3_vals, "r--", label=r"$p_3(x)$ (Newton)", linewidth=2)
    plt.scatter(x_nodes_3, f_nodes_3, color="black", zorder=5, label="Interpolation nodes")
    plt.scatter([3.0], [p3_at_3], color="magenta", marker="s", s=60, zorder=6, label=f"$p_3(3) = {p3_at_3:.2f}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function $f(x)$ and Interpolation Polynomial $p_3(x)$")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.subplot(1, 2, 2)
    plt.plot(x_eval_points, errors, "k-", label=r"$|f(x) - p_3(x)|$", linewidth=2)
    plt.scatter([x_max_error], [max_error], color="red", zorder=5,
                label=f"Max error = {   max_error:.4f} at x={x_max_error:.2f}")
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.title(r"Interpolation Error $|f(x) - p_3(x)|$ on $[0, 4]$")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "problem_2_1.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()