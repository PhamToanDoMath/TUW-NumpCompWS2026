"""
Problem 5.1: Composite Trapezoidal Rule and Convergence Behavior
Numerical Computation - Problem Sheet 5
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import matplotlib.pyplot as plt


def composite_trapezoidal(f, N: int, a: float, b: float) -> float:
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def run_experiment():
    a, b = -1.0, 1.0

    # 1. Integrands
    integrands = {
        "f1": (r"$f_1(x) = x^2$", lambda x: x**2, 2.0 / 3.0),
        "f2": (r"$f_2(x) = |x|$", lambda x: np.abs(x), 1.0),
        "f3": (
            r"$f_3(x)$ (jump at $1/3$)",
            lambda x: np.where(x < 1.0 / 3.0, 0.5 * np.exp(x), np.exp(x)),
            np.e - 0.5 * np.exp(1.0 / 3.0) - 0.5 * np.exp(-1.0),
        ),
        "f4": (r"$f_4(x) = \sin(\pi x)$", lambda x: np.sin(np.pi * x), 0.0),
        "f5": (r"$f_5(x) = \sin(4\pi x)$", lambda x: np.sin(4.0 * np.pi * x), 0.0),
    }

    levels = np.arange(1, 21)
    h_values = (b - a) * (2.0 ** (-levels))

    errors = {name: [] for name in integrands}
    zero_reports = {name: [] for name in integrands}

    ZERO_TOL = 1e-15  # threshold to distinguish structural zero from numerical approximation

    for i, h in zip(levels, h_values):
        N = 2**int(i)
        for name, (_, f, exact) in integrands.items():
            approx = composite_trapezoidal(f, N, a, b)
            err = abs(approx - exact)
            if err <= ZERO_TOL:
                zero_reports[name].append((i, h, err))
                errors[name].append(0.0)
            else:
                errors[name].append(err)

    # 2. Print concise report
    print("=" * 70)
    print("Exercise 5.1: Composite Trapezoidal Rule on [-1, 1]")
    print("=" * 70)
    print("\n[Zero Error Report]")
    for name, (_, _, _) in integrands.items():
        z_count = len(zero_reports[name])
        if z_count > 0:
            print(f"  - {name}: Reported zero error for {z_count}/20 values (err <= {ZERO_TOL:1.0e}).")
        else:
            print(f"  - {name}: No zero errors (all errors > {ZERO_TOL:1.0e}).")

    print("\n[Sample Non-zero Absolute Errors]")
    print(f"{'i':>2} | {'h':>10} | {'f1 (x^2)':>12} | {'f2 (|x|)':>12} | {'f3 (jump)':>12}")
    print("-" * 60)
    for idx in [0, 4, 9, 14, 19]:
        i = levels[idx]
        h = h_values[idx]
        e1_str = f"{errors['f1'][idx]:12.4e}"
        e2_str = "0 (exact)" if errors['f2'][idx] == 0.0 else f"{errors['f2'][idx]:12.4e}"
        e3_str = f"{errors['f3'][idx]:12.4e}"
        print(f"{i:2d} | {h:10.4e} | {e1_str} | {e2_str} | {e3_str}")

    # 3. Plot non-zero errors in log-log scale
    plt.figure(figsize=(9, 6))

    styles = {
        "f1": ("o-", "royalblue", 2.0),
        "f2": ("s--", "darkorange", 1.5),
        "f3": ("^-", "crimson", 2.0),
        "f4": ("x:", "forestgreen", 1.5),
        "f5": ("d:", "purple", 1.5),
    }

    for name, (label, _, _) in integrands.items():
        err_arr = np.array(errors[name])
        mask = err_arr > ZERO_TOL
        if np.any(mask):
            fmt, color, lw = styles[name]
            plt.loglog(h_values[mask], err_arr[mask], fmt, label=label, color=color, linewidth=lw, markersize=5)

    # Reference convergence lines O(h^2) and O(h)
    ref_h = np.array([h_values[15], h_values[2]])
    plt.loglog(ref_h, 0.4 * ref_h**2, "k--", alpha=0.7, label=r"$\mathcal{O}(h^2)$ reference")
    plt.loglog(ref_h, 0.05 * ref_h, "k:", alpha=0.7, label=r"$\mathcal{O}(h)$ reference")

    plt.xlabel("Step size $h = (b - a) 2^{-i}$", fontsize=11)
    plt.ylabel("Absolute error $|I - T(h)|$", fontsize=11)
    plt.title("Exercise 5.1: Composite Trapezoidal Rule Convergence on $[-1, 1]$", fontsize=12)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), "problem_5_1.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot successfully saved to: {save_path}")


if __name__ == "__main__":
    run_experiment()
