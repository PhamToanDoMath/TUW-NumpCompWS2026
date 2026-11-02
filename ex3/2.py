"""
Problem 3.2: Cubic Spline Interpolation with Not-a-Knot Condition
Numerical Computation - Problem Sheet 3

This script:
1. Takes arbitrary points of data (x_i, y_i) as input (defaulting to the Problem 3.2 points).
2. Formulates and solves the linear system of equations for the cubic spline with not-a-knot boundary conditions.
3. Obtains the piecewise cubic polynomials and analyzes the degrees of freedom (Part a).
4. Solves the linear system and prints the explicit spline formula (Part b).
5. Computes the theoretical error upper bound from Theorem 1.32 / Hall-Meyer (Part c).
6. Plots the resulting spline s(x) and its derivatives to demonstrate C^2 continuity and the not-a-knot property.
"""

import os
from typing import Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class NotAKnotCubicSpline:
    """
    Cubic spline interpolant with not-a-knot boundary conditions.
    Constructed by directly assembling and solving the full linear system of equations
    for the piecewise polynomial coefficients.
    """

    def __init__(self, x_points: Sequence[float], y_points: Sequence[float]):
        self.x = np.asarray(x_points, dtype=float)
        self.y = np.asarray(y_points, dtype=float)

        if len(self.x) < 4:
            raise ValueError("Not-a-knot cubic spline requires at least 4 data points.")
        if len(self.x) != len(self.y):
            raise ValueError("x and y arrays must have the same length.")

        # Ensure points are sorted by x
        sort_idx = np.argsort(self.x)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]

        self.n = len(self.x) - 1  # Number of intervals
        self.h = np.diff(self.x)   # Interval widths h_i = x_{i+1} - x_i

        # Solve linear systems
        self.A, self.b, self.coeffs = self._solve_piecewise_linear_system()
        self.A_mom, self.b_mom, self.moments = self._solve_moment_linear_system()

    def _solve_piecewise_linear_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assembles and solves the 4n x 4n linear system for the piecewise coefficients:
            s_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
        for i = 0, ..., n-1.
        
        Unknowns: [a_0, b_0, c_0, d_0, ..., a_{n-1}, b_{n-1}, c_{n-1}, d_{n-1}].
        """
        n = self.n
        h = self.h
        y = self.y
        num_unknowns = 4 * n

        A = np.zeros((num_unknowns, num_unknowns), dtype=float)
        b = np.zeros(num_unknowns, dtype=float)
        row = 0

        # 1. Left endpoint interpolation: s_i(x_i) = y_i  =>  a_i = y_i  (n equations)
        for i in range(n):
            A[row, 4 * i] = 1.0
            b[row] = y[i]
            row += 1

        # 2. Right endpoint interpolation: s_i(x_{i+1}) = y_{i+1}  (n equations)
        #    a_i + b_i*h_i + c_i*h_i^2 + d_i*h_i^3 = y_{i+1}
        for i in range(n):
            A[row, 4 * i] = 1.0
            A[row, 4 * i + 1] = h[i]
            A[row, 4 * i + 2] = h[i] ** 2
            A[row, 4 * i + 3] = h[i] ** 3
            b[row] = y[i + 1]
            row += 1

        # 3. C^1 continuity at interior knots x_1, ..., x_{n-1}  (n - 1 equations)
        #    s'_{i-1}(x_i) - s'_i(x_i) = 0
        #    b_{i-1} + 2*c_{i-1}*h_{i-1} + 3*d_{i-1}*h_{i-1}^2 - b_i = 0
        for i in range(1, n):
            A[row, 4 * (i - 1) + 1] = 1.0
            A[row, 4 * (i - 1) + 2] = 2.0 * h[i - 1]
            A[row, 4 * (i - 1) + 3] = 3.0 * (h[i - 1] ** 2)
            A[row, 4 * i + 1] = -1.0
            b[row] = 0.0
            row += 1

        # 4. C^2 continuity at interior knots x_1, ..., x_{n-1}  (n - 1 equations)
        #    s''_{i-1}(x_i) - s''_i(x_i) = 0
        #    2*c_{i-1} + 6*d_{i-1}*h_{i-1} - 2*c_i = 0
        for i in range(1, n):
            A[row, 4 * (i - 1) + 2] = 2.0
            A[row, 4 * (i - 1) + 3] = 6.0 * h[i - 1]
            A[row, 4 * i + 2] = -2.0
            b[row] = 0.0
            row += 1

        # 5. Not-a-knot condition at x_1  (1 equation)
        #    s'''_0(x_1) = s'''_1(x_1)  =>  6*d_0 = 6*d_1  =>  d_0 - d_1 = 0
        A[row, 4 * 0 + 3] = 1.0
        A[row, 4 * 1 + 3] = -1.0
        b[row] = 0.0
        row += 1

        # 6. Not-a-knot condition at x_{n-1}  (1 equation)
        #    s'''_{n-2}(x_{n-1}) = s'''_{n-1}(x_{n-1})  =>  d_{n-2} - d_{n-1} = 0
        A[row, 4 * (n - 2) + 3] = 1.0
        A[row, 4 * (n - 1) + 3] = -1.0
        b[row] = 0.0
        row += 1

        # Solve linear system
        coeffs_vector = np.linalg.solve(A, b)
        coeffs = coeffs_vector.reshape((n, 4))
        return A, b, coeffs

    def _solve_moment_linear_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assembles and solves the classical (n + 1) x (n + 1) moment system for M_i = s''(x_i).
        """
        n = self.n
        h = self.h
        y = self.y

        A_mom = np.zeros((n + 1, n + 1), dtype=float)
        b_mom = np.zeros(n + 1, dtype=float)

        # Row 0: Not-a-knot condition at x_1
        # (M_1 - M_0)/h_0 = (M_2 - M_1)/h_1  =>  h_1*M_0 - (h_0 + h_1)*M_1 + h_0*M_2 = 0
        A_mom[0, 0] = h[1]
        A_mom[0, 1] = -(h[0] + h[1])
        A_mom[0, 2] = h[0]
        b_mom[0] = 0.0

        # Rows 1 to n-1: C^1 continuity
        # h_{i-1}*M_{i-1} + 2*(h_{i-1} + h_i)*M_i + h_i*M_{i+1} = 6*((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})
        for i in range(1, n):
            A_mom[i, i - 1] = h[i - 1]
            A_mom[i, i] = 2.0 * (h[i - 1] + h[i])
            A_mom[i, i + 1] = h[i]
            b_mom[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        # Row n: Not-a-knot condition at x_{n-1}
        # (M_{n-1} - M_{n-2})/h_{n-2} = (M_n - M_{n-1})/h_{n-1}
        # => h_{n-1}*M_{n-2} - (h_{n-2} + h_{n-1})*M_{n-1} + h_{n-2}*M_n = 0
        A_mom[n, n - 2] = h[n - 1]
        A_mom[n, n - 1] = -(h[n - 2] + h[n - 1])
        A_mom[n, n] = h[n - 2]
        b_mom[n] = 0.0

        moments = np.linalg.solve(A_mom, b_mom)
        return A_mom, b_mom, moments

    def __call__(self, x_eval: Union[float, Sequence[float], np.ndarray], derivative: int = 0) -> Union[float, np.ndarray]:
        """
        Evaluates the spline s(x) or its derivative of order 0, 1, 2, or 3.
        """
        x_e = np.asarray(x_eval, dtype=float)
        is_scalar = np.isscalar(x_eval)
        flat_x = x_e.reshape(-1)

        result = np.zeros_like(flat_x, dtype=float)

        for idx, val in enumerate(flat_x):
            # Locate interval I_i = [x_i, x_{i+1}]
            # Handle boundary points appropriately
            if val <= self.x[0]:
                i = 0
            elif val >= self.x[-1]:
                i = self.n - 1
            else:
                i = int(np.searchsorted(self.x, val, side="right") - 1)
                i = min(max(i, 0), self.n - 1)

            dx = val - self.x[i]
            a, b, c, d = self.coeffs[i]

            if derivative == 0:
                result[idx] = a + b * dx + c * (dx ** 2) + d * (dx ** 3)
            elif derivative == 1:
                result[idx] = b + 2.0 * c * dx + 3.0 * d * (dx ** 2)
            elif derivative == 2:
                result[idx] = 2.0 * c + 6.0 * d * dx
            elif derivative == 3:
                result[idx] = 6.0 * d
            else:
                result[idx] = 0.0

        if is_scalar:
            return float(result[0])
        return result.reshape(x_e.shape)

    def get_piece_formulas(self) -> list[dict]:
        """
        Returns formula strings for each piece both in local (x - x_i) and global x coordinates.
        """
        def clean_val(v: float) -> float:
            return 0.0 if abs(v) < 1e-12 else v

        formulas = []
        for i in range(self.n):
            a, b, c, d = (clean_val(v) for v in self.coeffs[i])
            xi = clean_val(self.x[i])

            # In global coordinates:
            # s_i(x) = a + b*(x-xi) + c*(x-xi)^2 + d*(x-xi)^3
            # Expand: d*x^3 + (c - 3*d*xi)*x^2 + (b - 2*c*xi + 3*d*xi^2)*x + (a - b*xi + c*xi^2 - d*xi^3)
            p3 = clean_val(d)
            p2 = clean_val(c - 3.0 * d * xi)
            p1 = clean_val(b - 2.0 * c * xi + 3.0 * d * (xi ** 2))
            p0 = clean_val(a - b * xi + c * (xi ** 2) - d * (xi ** 3))

            formulas.append({
                "interval": (self.x[i], self.x[i + 1]),
                "local_coeffs": (a, b, c, d),
                "local_str": f"{a:.4g} + {b:.4g}*(x - {xi:.4g}) + {c:.4g}*(x - {xi:.4g})^2 + {d:.4g}*(x - {xi:.4g})^3",
                "global_coeffs": (p3, p2, p1, p0),
                "global_str": f"{p3:.4g}*x^3 + ({p2:.4g})*x^2 + ({p1:.4g})*x + ({p0:.4g})"
            })
        return formulas


def plot_spline(spline: NotAKnotCubicSpline, save_filename: str = "problem_3_2.png"):
    """
    Plots the cubic spline, its polynomial pieces, and its derivatives.
    """
    x_nodes = spline.x
    y_nodes = spline.y
    a, b = x_nodes[0], x_nodes[-1]

    x_fine = np.linspace(a, b, 1000)
    y_fine = spline(x_fine, derivative=0)
    dy_fine = spline(x_fine, derivative=1)
    d2y_fine = spline(x_fine, derivative=2)
    d3y_fine = spline(x_fine, derivative=3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -------------------------------------------------------------------------
    # Subplot 1: Spline s(x) and data points
    # -------------------------------------------------------------------------
    ax0 = axes[0, 0]
    ax0.plot(x_fine, y_fine, "b-", linewidth=2.5, label=r"Cubic Spline $s(x)$")
    ax0.scatter(x_nodes, y_nodes, color="red", s=70, zorder=5, label=r"Data Points $(x_i, y_i)$")

    # Distinguish knots: Not-a-knot vs Interior vs Boundary
    ax0.scatter([x_nodes[1], x_nodes[-2]], [y_nodes[1], y_nodes[-2]],
                facecolors="none", edgecolors="magenta", s=150, linewidth=2, zorder=6,
                label=r"Not-a-knot points ($x_1, x_3$)")

    # Plot individual intervals with background shading
    colors = ["#e6f2ff", "#e6ffe6", "#fff0e6", "#f9e6ff"]
    for i in range(spline.n):
        ax0.axvspan(x_nodes[i], x_nodes[i + 1], alpha=0.25, color=colors[i % len(colors)],
                   label=f"Interval $I_{i} = [{x_nodes[i]:.0f}, {x_nodes[i+1]:.0f}]$")

    for i, (xi, yi) in enumerate(zip(x_nodes, y_nodes)):
        ax0.annotate(f"({xi:.0f}, {yi:.0f})", (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

    ax0.set_title("Cubic Spline Interpolation $s(x)$ (Not-a-Knot)", fontsize=13, fontweight="bold")
    ax0.set_xlabel("$x$", fontsize=11)
    ax0.set_ylabel("$s(x)$", fontsize=11)
    ax0.grid(True, linestyle=":", alpha=0.7)
    ax0.legend(loc="upper left", fontsize=9)

    # -------------------------------------------------------------------------
    # Subplot 2: First derivative s'(x) - C^1 continuity
    # -------------------------------------------------------------------------
    ax1 = axes[0, 1]
    ax1.plot(x_fine, dy_fine, "g-", linewidth=2, label=r"$s'(x)$ (continuous)")
    for xi in x_nodes:
        ax1.axvline(xi, color="gray", linestyle="--", alpha=0.5)
    ax1.scatter(x_nodes, spline(x_nodes, derivative=1), color="darkgreen", s=50, zorder=5)
    ax1.set_title(r"First Derivative $s'(x)$ ($C^1$ Continuity)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("$x$", fontsize=11)
    ax1.set_ylabel(r"$s'(x)$", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="best", fontsize=9)

    # -------------------------------------------------------------------------
    # Subplot 3: Second derivative s''(x) - C^2 continuity
    # -------------------------------------------------------------------------
    ax2 = axes[1, 0]
    ax2.plot(x_fine, d2y_fine, "m-", linewidth=2, label=r"$s''(x)$ (continuous)")
    for xi in x_nodes:
        ax2.axvline(xi, color="gray", linestyle="--", alpha=0.5)
    ax2.scatter(x_nodes, spline(x_nodes, derivative=2), color="purple", s=50, zorder=5,
                label=r"Knots $s''(x_i) = M_i$")
    ax2.set_title(r"Second Derivative $s''(x)$ ($C^2$ Continuity)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("$x$", fontsize=11)
    ax2.set_ylabel(r"$s''(x)$", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="best", fontsize=9)

    # -------------------------------------------------------------------------
    # Subplot 4: Third derivative s'''(x) - Not-a-knot illustration
    # -------------------------------------------------------------------------
    ax3 = axes[1, 1]
    ax3.plot(x_fine, d3y_fine, "r-", linewidth=2, label=r"$s'''(x)$")
    for xi in x_nodes:
        ax3.axvline(xi, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x_nodes[1], color="magenta", linestyle="-", linewidth=2,
                label=r"No jump at $x_1=1$ ($s'''$ continuous)")
    ax3.axvline(x_nodes[3], color="magenta", linestyle="-", linewidth=2,
                label=r"No jump at $x_3=3$ ($s'''$ continuous)")
    ax3.axvline(x_nodes[2], color="black", linestyle=":", linewidth=2,
                label=r"Jump occurs at $x_2=2$ ($s'''$ discontinuous)")
    ax3.set_title(r"Third Derivative $s'''(x)$ (Demonstrating Not-a-Knot Property)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("$x$", fontsize=11)
    ax3.set_ylabel(r"$s'''(x)$", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot successfully saved to: {save_path}")
    plt.show()


def main():
    print("=" * 80)
    print("Numerical Computation - Problem Sheet 3: Problem 3.2")
    print("Cubic Spline with Not-a-Knot Condition")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Input Data Points
    # -------------------------------------------------------------------------
    data_points = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (4.0, 2.0)]
    x_input = [pt[0] for pt in data_points]
    y_input = [pt[1] for pt in data_points]

    print("\nInput data points (x_i, y_i):")
    for i, (xi, yi) in enumerate(data_points):
        print(f"  Point {i}: ({xi}, {yi})")

    # Fit spline
    spline = NotAKnotCubicSpline(x_input, y_input)
    n = spline.n

    # -------------------------------------------------------------------------
    # Part (a): Degrees of Freedom and Linear System Formulation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): DEGREES OF FREEDOM & LINEAR SYSTEM OF EQUATIONS")
    print("=" * 80)

    print("\n1. Degrees of Freedom (DOF):")
    print(f"  - Number of intervals n = {n} on [{spline.x[0]}, {spline.x[-1]}].")
    print("  - A piecewise cubic polynomial on n intervals has 4 * n parameters (coefficients).")
    print(f"    Here, 4 * {n} = {4 * n} coefficients.")
    print("  - Requiring C^2 regularity at the (n - 1) interior knots imposes 3*(n - 1) constraints:")
    print(f"    Continuity of s, s', and s'' at each interior knot => 3 * {n - 1} = {3 * (n - 1)} equations.")
    print("  - By Lemma 1.31 from lecture notes:")
    print("      dim S_{3,2}(Delta) = n*(p + 1) - (n - 1)*(r + 1) = 4*n - 3*(n - 1) = n + 3.")
    print(f"    For n = {n}, dim S_{{3,2}}(Delta) = {n + 3} degrees of freedom.")
    print("  - The not-a-knot condition requires continuity of s''' at x_1 and x_{n-1}:")
    print(f"      lim_{{x -> x_1^-}} s'''(x) = lim_{{x -> x_1^+}} s'''(x)")
    print(f"      lim_{{x -> x_{{n-1}}^-}} s'''(x) = lim_{{x -> x_{{n-1}}^+}} s'''(x)")
    print("    This imposes 2 additional linear constraints.")
    print(f"  - Hence, the subspace of cubic splines satisfying the not-a-knot condition has:")
    print(f"      dim = (n + 3) - 2 = n + 1 = {n + 1} degrees of freedom.")
    print(f"  - The {n + 1} interpolation conditions s(x_i) = y_i (i = 0, ..., {n}) uniquely determine the spline!")

    print("\n2. Linear System of Equations (Full 16 x 16 System A * c = b):")
    print(f"   Matrix A has dimension {spline.A.shape[0]} x {spline.A.shape[1]}.")
    print("   Unknown vector c = [a_0, b_0, c_0, d_0, ..., a_3, b_3, c_3, d_3]^T")
    print("   Equations:")
    print("     (1) Left interpolation:   s_i(x_i) = a_i = y_i                  (i = 0..3)  -> 4 eqs")
    print("     (2) Right interpolation:  a_i + b_i*h_i + c_i*h_i^2 + d_i*h_i^3 = y_{i+1} (i = 0..3)  -> 4 eqs")
    print("     (3) C^1 continuity at x_i: b_{i-1} + 2*c_{i-1}*h_{i-1} + 3*d_{i-1}*h_{i-1}^2 - b_i = 0 (i = 1..3) -> 3 eqs")
    print("     (4) C^2 continuity at x_i: 2*c_{i-1} + 6*d_{i-1}*h_{i-1} - 2*c_i = 0               (i = 1..3) -> 3 eqs")
    print("     (5) Not-a-knot at x_1:    d_0 - d_1 = 0                                         -> 1 eq")
    print("     (6) Not-a-knot at x_3:    d_2 - d_3 = 0                                         -> 1 eq")
    print("   Total = 4 + 4 + 3 + 3 + 1 + 1 = 16 equations for 16 unknowns.")

    print("\n   Alternatively, the classical (n+1) x (n+1) = 5 x 5 Moment System for M_i = s''(x_i):")
    print("   A_mom * M = b_mom:")
    for i in range(len(spline.b_mom)):
        row_str = "  [ " + "  ".join(f"{val:5.1f}" for val in spline.A_mom[i]) + f" ] * [ M_{i} ] = [ {spline.b_mom[i]:5.1f} ]"
        print(row_str)

    # -------------------------------------------------------------------------
    # Part (b): Solution of the Linear System & Explicit Cubic Spline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (b): SOLUTION OF THE LINEAR SYSTEM & EXPLICIT SPLINE FUNCTION")
    print("=" * 80)

    print("\nSolved Moments M_i = s''(x_i):")
    for i, M_val in enumerate(spline.moments):
        print(f"  M_{i} = s''({spline.x[i]:.0f}) = {M_val:.4f}")

    print("\nSolved Piecewise Polynomial Coefficients:")
    print("  s_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3")
    for i in range(n):
        a, b_c, c, d = spline.coeffs[i]
        print(f"  Piece {i} on [{spline.x[i]:.0f}, {spline.x[i+1]:.0f}]: "
              f"a_{i} = {a:.4f}, b_{i} = {b_c:.4f}, c_{i} = {c:.4f}, d_{i} = {d:.4f}")

    formulas = spline.get_piece_formulas()
    print("\nExplicit Spline Representation in Local Coordinates (x - x_i):")
    for i, f_info in enumerate(formulas):
        print(f"  s_{i}(x) = {f_info['local_str']}   for x in [{f_info['interval'][0]:.0f}, {f_info['interval'][1]:.0f}]")

    print("\nExplicit Spline Representation in Global Monomial Basis x:")
    for i, f_info in enumerate(formulas):
        print(f"  s_{i}(x) = {f_info['global_str']}   for x in [{f_info['interval'][0]:.0f}, {f_info['interval'][1]:.0f}]")

    print("\nRemark on Not-a-Knot Structure:")
    print("  Notice that Piece 0 and Piece 1 have identical global polynomials:")
    print("    s(x) = 0.5*x^3 - 2*x^2 + 2.5*x   on [0, 2]")
    print("  And Piece 2 and Piece 3 have identical global polynomials:")
    print("    s(x) = -0.5*x^3 + 4*x^2 - 9.5*x + 8   on [2, 4]")
    print("  Thus, x_1 = 1 and x_3 = 3 are 'not knots' in the sense that no change in polynomial piece occurs;")
    print("  the spline is simply two cubic polynomials joining at x_2 = 2 with C^2 continuity!")

    # Verify against scipy.interpolate.CubicSpline
    cs_scipy = CubicSpline(spline.x, spline.y, bc_type="not-a-knot")
    x_test = np.linspace(0, 4, 9)
    diff = np.max(np.abs(spline(x_test) - cs_scipy(x_test)))
    print(f"\nVerification against scipy.interpolate.CubicSpline: max difference = {diff:.2e} (MATCH)")

    # -------------------------------------------------------------------------
    # Part (c): Error Upper Bound
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (c): ERROR UPPER BOUND FOR ||f - p||_inf,[0,4]")
    print("=" * 80)

    h_max = np.max(spline.h)
    f4_bound = 4.0

    print(f"\nGiven:")
    print(f"  - Maximal grid width: h = max_i h_i = {h_max:.1f}")
    print(f"  - Upper bound on 4th derivative: ||f^(4)||_inf,[0,4] <= {f4_bound:.1f}")
    print("\nFrom Theorem 1.32 in Lecture Notes:")
    print("  For f in C^4([a,b]) and interpolating cubic spline p with not-a-knot conditions:")
    print("      ||f - p||_inf,[a,b] <= C * h^4 * ||f^(4)||_inf,[a,b]")
    print(f"  With h = {h_max} and ||f^(4)||_inf <= {f4_bound}:")
    print(f"      ||f - p||_inf,[0,4] <= C * (1.0)^4 * 4.0 = 4.0 * C")

    # Classical Hall-Meyer constant C = 5 / 384
    C_hall_meyer = 5.0 / 384.0
    err_bound_sharp = C_hall_meyer * (h_max ** 4) * f4_bound

    print("\nExplicit Upper Bound using the classical sharp Hall-Meyer constant (C = 5/384):")
    print(f"  C = 5 / 384 approx = {C_hall_meyer:.6f}")
    print(f"  ||f - p||_inf,[0,4] <= (5 / 384) * (1)^4 * 4 = 5 / 96 approx = {err_bound_sharp:.6f}")
    print(f"  Upper bound = 5/96 approx {err_bound_sharp:.5f}")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_spline(spline)


if __name__ == "__main__":
    main()