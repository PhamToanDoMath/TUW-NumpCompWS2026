"""
Problem 3.4: Natural vs Clamped Cubic Spline Interpolation
Numerical Computation - Problem Sheet 3

This script:
1. Part (a): Computes the natural cubic spline s_nat and clamped cubic spline s_cl
   for f(x) = cos(pi*x) + 0.25*cos(3*pi*x) on [-1, 1] with knots x = [-1, 0, 1].
2. Part (b): Numerically determines the L_infinity errors ||f - s_nat|| and ||f - s_cl||,
   identifies which spline has smaller error, explains Theorem 1.32 applicability,
   and computes f''(-1), f''(1) to compare with natural boundary conditions.
3. Part (c): For g(x) = exp(-100*(x - 0.2)^2) + 0.2*sin(6*pi*x) on [0, 1] with
   n in {8, 16, 32, 64, 128, 256}, computes natural and clamped cubic splines,
   samples maximum errors at 10001 uniform points, and plots errors vs h = 1/n
   in a loglog plot with auxiliary lines h^2 and h^4 to analyze convergence rates.
"""

import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# =============================================================================
# Functions for Part (a) and (b)
# =============================================================================

def f(x: np.ndarray | float) -> np.ndarray | float:
    """f(x) = cos(pi*x) + 0.25*cos(3*pi*x) on [-1, 1]."""
    return np.cos(np.pi * x) + 0.25 * np.cos(3.0 * np.pi * x)


def f_prime(x: np.ndarray | float) -> np.ndarray | float:
    """f'(x) = -pi*sin(pi*x) - 0.75*pi*sin(3*pi*x)."""
    return -np.pi * np.sin(np.pi * x) - 0.75 * np.pi * np.sin(3.0 * np.pi * x)


def f_double_prime(x: np.ndarray | float) -> np.ndarray | float:
    """f''(x) = -pi^2*cos(pi*x) - 2.25*pi^2*cos(3*pi*x)."""
    return -(np.pi ** 2) * np.cos(np.pi * x) - 2.25 * (np.pi ** 2) * np.cos(3.0 * np.pi * x)


def s_nat_piecewise(x: np.ndarray | float) -> np.ndarray | float:
    """
    Exact analytical formula for natural cubic spline on [-1, 1]:
      On [-1, 0]: s0(x) = -1.25*x^3 - 3.75*x^2 + 1.25
      On [ 0, 1]: s1(x) =  1.25*x^3 - 3.75*x^2 + 1.25
    """
    x_arr = np.asarray(x, dtype=float)
    is_scalar = np.isscalar(x)
    res = np.where(
        x_arr <= 0.0,
        -1.25 * (x_arr ** 3) - 3.75 * (x_arr ** 2) + 1.25,
        1.25 * (x_arr ** 3) - 3.75 * (x_arr ** 2) + 1.25,
    )
    if is_scalar:
        return float(res)
    return res


def s_cl_piecewise(x: np.ndarray | float) -> np.ndarray | float:
    """
    Exact analytical formula for clamped cubic spline on [-1, 1]:
      On [-1, 0]: s0(x) = -5*x^3 - 7.5*x^2 + 1.25
      On [ 0, 1]: s1(x) =  5*x^3 - 7.5*x^2 + 1.25
    """
    x_arr = np.asarray(x, dtype=float)
    is_scalar = np.isscalar(x)
    res = np.where(
        x_arr <= 0.0,
        -5.0 * (x_arr ** 3) - 7.5 * (x_arr ** 2) + 1.25,
        5.0 * (x_arr ** 3) - 7.5 * (x_arr ** 2) + 1.25,
    )
    if is_scalar:
        return float(res)
    return res


# =============================================================================
# Functions for Part (c)
# =============================================================================

def g(x: np.ndarray | float) -> np.ndarray | float:
    """
    Target function for part (c):
    g(x) = exp(-100*(x - 0.2)^2) + 0.2*sin(6*pi*x) on [0, 1].
    """
    return np.exp(-100.0 * (x - 0.2) ** 2) + 0.2 * np.sin(6.0 * np.pi * x)


def g_prime(x: np.ndarray | float) -> np.ndarray | float:
    """
    Exact first derivative:
    g'(x) = -200*(x - 0.2)*exp(-100*(x - 0.2)^2) + 1.2*pi*cos(6*pi*x).
    """
    return -200.0 * (x - 0.2) * np.exp(-100.0 * (x - 0.2) ** 2) + 1.2 * np.pi * np.cos(6.0 * np.pi * x)


def g_double_prime(x: np.ndarray | float) -> np.ndarray | float:
    """
    Exact second derivative:
    g''(x) = [-200 + 40000*(x - 0.2)^2]*exp(-100*(x - 0.2)^2) - 7.2*pi^2*sin(6*pi*x).
    """
    return (-200.0 + 40000.0 * (x - 0.2) ** 2) * np.exp(-100.0 * (x - 0.2) ** 2) - 7.2 * (np.pi ** 2) * np.sin(6.0 * np.pi * x)


# =============================================================================
# Plotting Routine
# =============================================================================

def plot_all(
    x_nodes_ab: np.ndarray,
    n_values: list[int],
    h_values: np.ndarray,
    err_nat_list: list[float],
    err_cl_list: list[float],
    save_filename: str = "problem_3_4.png",
):
    """
    Generates a 4-panel figure illustrating all aspects of Problem 3.4.
    """
    fig = plt.figure(figsize=(16, 12))

    # -------------------------------------------------------------------------
    # Panel 1: Part (a) & (b) - f(x), s_nat(x), s_cl(x) on [-1, 1]
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    x_dense = np.linspace(-1.0, 1.0, 1001)
    f_dense = f(x_dense)
    snat_dense = s_nat_piecewise(x_dense)
    scl_dense = s_cl_piecewise(x_dense)

    ax1.plot(x_dense, f_dense, "k-", linewidth=2.5, label=r"$f(x) = \cos(\pi x) + \frac{1}{4}\cos(3\pi x)$")
    ax1.plot(x_dense, snat_dense, "b--", linewidth=2.0, label=r"Natural spline $s_{\mathrm{nat}}(x)$")
    ax1.plot(x_dense, scl_dense, "r-.", linewidth=2.0, label=r"Clamped spline $s_{\mathrm{cl}}(x)$")
    ax1.scatter(x_nodes_ab, f(x_nodes_ab), color="black", s=80, zorder=5, label=r"Knots $x \in \{-1, 0, 1\}$")

    ax1.set_title(r"Part (a): Splines $s_{\mathrm{nat}}$ and $s_{\mathrm{cl}}$ for $f(x)$ on $[-1, 1]$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("$x$", fontsize=10)
    ax1.set_ylabel("$y$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower center", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Part (b) - Pointwise Error Comparison
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    err_nat_dense = np.abs(f_dense - snat_dense)
    err_cl_dense = np.abs(f_dense - scl_dense)

    ax2.plot(x_dense, err_nat_dense, "b-", linewidth=2.0,
             label=rf"$|f(x) - s_{{\mathrm{{nat}}}}(x)|$ (max = {np.max(err_nat_dense):.4f})")
    ax2.plot(x_dense, err_cl_dense, "r-", linewidth=2.0,
             label=rf"$|f(x) - s_{{\mathrm{{cl}}}}(x)|$ (max = {np.max(err_cl_dense):.4f})")
    ax2.axvline(-1, color="gray", linestyle=":")
    ax2.axvline(0, color="gray", linestyle=":")
    ax2.axvline(1, color="gray", linestyle=":")

    ax2.set_title("Part (b): Error Comparison on $[-1, 1]$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("$x$", fontsize=10)
    ax2.set_ylabel("Absolute Error", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper center", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 3: Part (c) - Function g(x) and Splines for n = 8 and n = 32
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    x_c_dense = np.linspace(0.0, 1.0, 2001)
    g_c_dense = g(x_c_dense)

    # Splines for n = 8
    x_8 = np.linspace(0.0, 1.0, 8 + 1)
    cs_nat_8 = CubicSpline(x_8, g(x_8), bc_type="natural")
    cs_cl_8 = CubicSpline(x_8, g(x_8), bc_type=((1, g_prime(0.0)), (1, g_prime(1.0))))

    # Splines for n = 32
    x_32 = np.linspace(0.0, 1.0, 32 + 1)
    cs_cl_32 = CubicSpline(x_32, g(x_32), bc_type=((1, g_prime(0.0)), (1, g_prime(1.0))))

    ax3.plot(x_c_dense, g_c_dense, "k-", linewidth=2.2, label=r"Exact $g(x)$")
    ax3.plot(x_c_dense, cs_nat_8(x_c_dense), "b--", linewidth=1.6, label="Natural ($n=8$)")
    ax3.plot(x_c_dense, cs_cl_8(x_c_dense), "r-.", linewidth=1.6, label="Clamped ($n=8$)")
    ax3.plot(x_c_dense, cs_cl_32(x_c_dense), "g:", linewidth=2.0, label="Clamped ($n=32$)")

    ax3.set_title("Part (c): Target Function $g(x)$ and Splines", fontsize=12, fontweight="bold")
    ax3.set_xlabel("$x$", fontsize=10)
    ax3.set_ylabel("$g(x)$", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 4: Part (c) - Convergence Rates Loglog Plot
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)

    # Errors
    ax4.loglog(h_values, err_nat_list, "bs-", linewidth=2.2, markersize=7,
               label=r"Natural Spline: $\|g - s_{\mathrm{nat}}\|_{\infty}$ ($\sim \mathcal{O}(h^2)$)")
    ax4.loglog(h_values, err_cl_list, "ro-", linewidth=2.2, markersize=7,
               label=r"Clamped Spline: $\|g - s_{\mathrm{cl}}\|_{\infty}$ ($\sim \mathcal{O}(h^4)$)")

    # Auxiliary lines h |-> h^2 and h |-> h^4 normalized for clean display
    h_aux = np.geomspace(h_values[-1], h_values[0], 100)
    # Anchor h^2 near natural error at h = 1/64
    scale_h2 = err_nat_list[3] / (h_values[3] ** 2)
    # Anchor h^4 near clamped error at h = 1/64
    scale_h4 = err_cl_list[3] / (h_values[3] ** 4)

    ax4.loglog(h_aux, scale_h2 * (h_aux ** 2), "b:", linewidth=1.8, label=r"Auxiliary line $\propto h^2$")
    ax4.loglog(h_aux, scale_h4 * (h_aux ** 4), "r:", linewidth=1.8, label=r"Auxiliary line $\propto h^4$")

    ax4.set_title(r"Part (c): Error vs $h = 1/n$ (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"Grid spacing $h = 1/n$", fontsize=10)
    ax4.set_ylabel(r"Maximum Error $\|g - s\|_{\infty,[0,1]}$", fontsize=10)
    ax4.grid(True, which="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), save_filename)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    print("=" * 80)
    print("Numerical Computation - Problem Sheet 3: Problem 3.4")
    print("Comparison of Natural and Clamped Cubic Spline Interpolation")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Part (a): Analytical Derivation of Natural and Clamped Splines
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (a): COMPUTATION OF NATURAL AND CLAMPED CUBIC SPLINES")
    print("=" * 80)

    x_knots = np.array([-1.0, 0.0, 1.0])
    y_knots = f(x_knots)

    print(f"Knots: x_0 = -1, x_1 = 0, x_2 = 1 (h_0 = h_1 = 1)")
    print(f"Function values at knots:")
    print(f"  f(-1) = cos(-pi) + 0.25*cos(-3*pi) = -1 - 0.25 = {y_knots[0]:.2f} (-5/4)")
    print(f"  f( 0) = cos(0)   + 0.25*cos(0)     =  1 + 0.25 =  {y_knots[1]:.2f} ( 5/4)")
    print(f"  f( 1) = cos(pi)  + 0.25*cos(3*pi)  = -1 - 0.25 = {y_knots[2]:.2f} (-5/4)")

    print(f"\nEndpoint derivatives:")
    print(f"  f'(-1) = -pi*sin(-pi) - (3*pi/4)*sin(-3*pi) = 0")
    print(f"  f'( 1) = -pi*sin(pi)  - (3*pi/4)*sin(3*pi)  = 0")

    print("\n1. Natural Cubic Spline s_nat(x) [s''_nat(-1) = s''_nat(1) = 0]:")
    print("   Moments M_0 = 0, M_2 = 0.")
    print("   Interior equation at x_1 = 0: h_0*M_0 + 2*(h_0 + h_1)*M_1 + h_1*M_2 = 6*((y_2-y_1)/h_1 - (y_1-y_0)/h_0)")
    print("     1*(0) + 2*(1 + 1)*M_1 + 1*(0) = 6*(-2.5 - 2.5) = -30")
    print("     4*M_1 = -30  ==>  M_1 = -7.5 = -15/2")
    print("   Polynomial pieces in standard powers of x:")
    print("     On [-1, 0]: s_nat,0(x) = -(5/4)*x^3 - (15/4)*x^2 + 5/4")
    print("     On [ 0, 1]: s_nat,1(x) =  (5/4)*x^3 - (15/4)*x^2 + 5/4")

    print("\n2. Clamped/Complete Cubic Spline s_cl(x) [s'_cl(-1) = f'(-1) = 0, s'_cl(1) = f'(1) = 0]:")
    print("   Moments satisfy:")
    print("     2*M_0 + M_1 = 6*((y_1 - y_0)/1 - 0) = 6*(2.5) = 15")
    print("     M_0 + 4*M_1 + M_2 = -30")
    print("     M_1 + 2*M_2 = 6*(0 - (y_2 - y_1)/1) = 6*(2.5) = 15")
    print("   Solving yields: M_0 = 15, M_1 = -15, M_2 = 15")
    print("   Polynomial pieces in standard powers of x:")
    print("     On [-1, 0]: s_cl,0(x) = -5*x^3 - (15/2)*x^2 + 5/4")
    print("     On [ 0, 1]: s_cl,1(x) =  5*x^3 - (15/2)*x^2 + 5/4")

    # -------------------------------------------------------------------------
    # Part (b): Error Analysis, Extrema, and Comparison with Theorem 1.32
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (b): ERROR DETERMINATION AND THEOREM 1.32 DISCUSSION")
    print("=" * 80)

    x_dense = np.linspace(-1.0, 1.0, 1000001)
    f_vals = f(x_dense)
    snat_vals = s_nat_piecewise(x_dense)
    scl_vals = s_cl_piecewise(x_dense)

    err_nat_arr = np.abs(f_vals - snat_vals)
    err_cl_arr = np.abs(f_vals - scl_vals)

    max_err_nat = float(np.max(err_nat_arr))
    max_err_cl = float(np.max(err_cl_arr))
    argmax_nat = x_dense[np.argmax(err_nat_arr)]
    argmax_cl = x_dense[np.argmax(err_cl_arr)]

    print(f"\n1. Maximum Errors on [-1, 1]:")
    print(f"   ||f - s_nat||_inf = {max_err_nat:.6f}  (achieved at x approx +/- {abs(argmax_nat):.4f})")
    print(f"   ||f - s_cl||_inf  = {max_err_cl:.6f}  (achieved at x approx +/- {abs(argmax_cl):.4f})")

    print("\n2. Comparison:")
    print(f"   The clamped spline s_cl has significantly smaller error ({max_err_cl:.4f} vs {max_err_nat:.4f}),")
    print(f"   achieving an error reduction of approx {max_err_nat / max_err_cl:.2f}x!")

    print("\n3. Applicability of Theorem 1.32:")
    print("   Theorem 1.32 states: ||f - s||_inf <= C * h^4 * ||f^(4)||_inf for:")
    print("     (i) Complete / clamped spline with s'(endpoints) = f'(endpoints)")
    print("     (ii) Periodic spline")
    print("     (iii) Not-a-knot spline")
    print("   Thus, ONLY the complete/clamped boundary condition is covered by Theorem 1.32.")

    f2_endpoints = float(f_double_prime(1.0))
    print(f"\n4. Second Derivative at Endpoints:")
    print(f"   f''(-1) = f''(1) = pi^2 + (9/4)*pi^2 = 13*pi^2/4 approx {f2_endpoints:.6f}")
    print(f"   Natural boundary condition imposes: s''_nat(-1) = s''_nat(1) = 0.0")
    print(f"   Discrepancy at endpoints: |f''(+/-1) - s''_nat''(+/-1)| = 13*pi^2/4 approx 32.0762 != 0!")
    print("   Because f'' does NOT vanish at the boundaries, natural splines suffer from a boundary")
    print("   layer error that reduces the overall asymptotic convergence rate to O(h^2)!")

    # -------------------------------------------------------------------------
    # Part (c): Localized Peak Function g(x) and Asymptotic Convergence Rates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART (c): LOCALIZED PEAK FUNCTION g(x) AND CONVERGENCE STUDY")
    print("=" * 80)

    n_values = [8, 16, 32, 64, 128, 256]
    h_values = 1.0 / np.array(n_values, dtype=float)

    # 10001 evaluation points on [0, 1] as requested
    x_sample = np.linspace(0.0, 1.0, 10001)
    g_sample = g(x_sample)

    err_nat_list = []
    err_cl_list = []

    print(f"{'n':>4} | {'h':>8} | {'||g - s_nat||_inf':>18} | {'Rate (nat)':>10} | {'||g - s_cl||_inf':>18} | {'Rate (cl)':>10}")
    print("-" * 80)

    for i, n in enumerate(n_values):
        x_nodes = np.linspace(0.0, 1.0, n + 1)
        y_nodes = g(x_nodes)

        # Spline construction with explicitly specified boundary conditions
        cs_nat = CubicSpline(x_nodes, y_nodes, bc_type="natural")
        cs_cl = CubicSpline(x_nodes, y_nodes, bc_type=((1, g_prime(0.0)), (1, g_prime(1.0))))

        e_nat = float(np.max(np.abs(g_sample - cs_nat(x_sample))))
        e_cl = float(np.max(np.abs(g_sample - cs_cl(x_sample))))

        err_nat_list.append(e_nat)
        err_cl_list.append(e_cl)

        if i == 0:
            rate_nat_str = "    -    "
            rate_cl_str = "    -    "
        else:
            r_nat = np.log(err_nat_list[i] / err_nat_list[i - 1]) / np.log(h_values[i] / h_values[i - 1])
            r_cl = np.log(err_cl_list[i] / err_cl_list[i - 1]) / np.log(h_values[i] / h_values[i - 1])
            rate_nat_str = f"{r_nat:10.2f}"
            rate_cl_str = f"{r_cl:10.2f}"

        print(f"{n:4d} | {h_values[i]:8.5f} | {e_nat:18.4e} | {rate_nat_str} | {e_cl:18.4e} | {rate_cl_str}")

    print("\nObserved Convergence Rates:")
    print("  - Clamped spline: Rate -> 4.00, confirming O(h^4) convergence predicted by Theorem 1.32.")
    print("  - Natural spline: Rate -> 2.00, dropping to O(h^2) convergence.")

    print("\nExplanation of the Difference via Endpoint Conditions:")
    g2_0 = g_double_prime(0.0)
    g2_1 = g_double_prime(1.0)
    print(f"  Exact second derivatives of g: g''(0) = {g2_0:.4f},  g''(1) approx {g2_1:.2e}")
    print("  - Natural spline forces s''_nat(0) = 0, but g''(0) approx 25.6419 != 0.")
    print("  - This introduces an O(1) error in s''(0), which via Taylor expansion yields an O(h^2)")
    print("    boundary layer error in the spline interpolation near x = 0.")
    print("  - Clamped spline matches the true first derivatives s'_cl'(0) = g'(0) and s'_cl'(1) = g'(1),")
    print("    avoiding boundary pollution and maintaining full O(h^4) accuracy throughout [0, 1].")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_all(x_knots, n_values, h_values, err_nat_list, err_cl_list)


if __name__ == "__main__":
    main()
