"""
Problem 2.5: Efficient Approximation of the Harmonic Series
Numerical Computation - Problem Sheet 2
"""

import time
import numpy as np


def S(N: int, chunk_size: int = 10_000_000) -> float:
    """
    Evaluates the harmonic sum S(N) = sum_{n=1}^N (1 / n).
    Uses chunked NumPy summation to ensure low memory footprint (O(chunk_size))
    and high performance via C-level vectorization.

    Parameters
    ----------
    N : int
        Upper summation limit.
    chunk_size : int, optional
        Size of memory chunk for array allocation (default: 10,000,000).

    Returns
    -------
    total : float
        Value of the harmonic sum S(N).
    """
    total = 0.0
    for start in range(1, N + 1, chunk_size):
        end = min(start + chunk_size, N + 1)
        total += np.sum(1.0 / np.arange(start, end, dtype=np.float64))
    return float(total)


def S_approx(N: float | np.ndarray, a0: float, a1: float, a2: float):
    """
    Approximates S(N) using the asymptotic expansion (Equation (1)):
        S(N) approx ln(N) + a0 + a1 / N + a2 / N^2
    """
    N_arr = np.asarray(N, dtype=float)
    val = np.log(N_arr) + a0 + a1 / N_arr + a2 / (N_arr**2)
    if np.isscalar(N):
        return float(val)
    return val


if __name__ == "__main__":
    print("=" * 70)
    print("Problem 2.5: Asymptotic Approximation of Harmonic Series S(N)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1) Set up linear system for N = 10, 100, 1000
    # -------------------------------------------------------------------------
    print("\n--- 1 & 2) Solving for Coefficients a0, a1, a2 ---")
    N_nodes = np.array([10, 100, 1000], dtype=np.float64)

    # Compute S(N) for N = 10, 100, 1000
    S_nodes = np.array([S(int(n)) for n in N_nodes], dtype=np.float64)

    # Matrix A and RHS vector b:
    # a0 + a1 / N + a2 / N^2 = S(N) - ln(N)
    A = np.column_stack([np.ones_like(N_nodes), 1.0 / N_nodes, 1.0 / (N_nodes**2)])
    b = S_nodes - np.log(N_nodes)

    # Solve linear system using numpy.linalg.solve
    coeffs = np.linalg.solve(A, b)
    a0, a1, a2 = coeffs

    gamma_exact = 0.5772156649015328606  # Euler-Mascheroni constant

    print(f"Nodes N: {N_nodes.astype(int)}")
    print(f"S(N):    {S_nodes}")
    print("\nLinear system matrix A:")
    print(A)
    print("RHS vector b (S(N) - ln(N)):")
    print(b)
    print("\nSolved coefficients:")
    print(f"  a0 = {a0:.15f}  (cf. Euler-Mascheroni constant gamma = {gamma_exact:.15f})")
    print(f"  a1 = {a1:.15f}  (cf. Theoretical Euler-Maclaurin a1 = 0.5)")
    print(f"  a2 = {a2:.15f}  (cf. Theoretical Euler-Maclaurin a2 = -1/12 = {-1/12:.15f})")

    # -------------------------------------------------------------------------
    # 3) Error of approximation for N = 10^6 and N = 10^8
    # -------------------------------------------------------------------------
    print("\n--- 3) Errors of Approximation for N = 10^6 and N = 10^8 ---")
    for N_test in [10**6, 10**8]:
        exact_val = S(N_test)
        approx_val = S_approx(N_test, a0, a1, a2)
        abs_err = abs(exact_val - approx_val)
        rel_err = abs_err / exact_val
        print(f"N = 10^{int(np.log10(N_test))}:")
        print(f"  Exact S(N)      = {exact_val:.12f}")
        print(f"  Approx S(N)     = {approx_val:.12f}")
        print(f"  Absolute Error  = {abs_err:.4e}")
        print(f"  Relative Error  = {rel_err:.4e}")

    # -------------------------------------------------------------------------
    # 4) Run time comparison for N = 10^8 and N = 10^9
    # -------------------------------------------------------------------------
    print("\n--- 4) Run Time Comparison for N = 10^8 and N = 10^9 ---")

    # Run time of approximation
    t0 = time.time()
    approx_10_8 = S_approx(10**8, a0, a1, a2)
    time_approx_10_8 = time.time() - t0

    t0 = time.time()
    approx_10_9 = S_approx(10**9, a0, a1, a2)
    time_approx_10_9 = time.time() - t0

    # Run time of direct evaluation S(10^8)
    print("Evaluating S(10^8) directly (using time.time())...")
    t0 = time.time()
    exact_10_8 = S(10**8)
    time_exact_10_8 = time.time() - t0

    # Run time of direct evaluation S(10^9)
    print("Evaluating S(10^9) directly (using time.time())...")
    t0 = time.time()
    exact_10_9 = S(10**9)
    time_exact_10_9 = time.time() - t0

    print("\n" + "=" * 70)
    print("Summary of Run Times & Results:")
    print("=" * 70)
    print(f"{'N':<8} | {'Exact S(N)':<18} | {'Approx S(N)':<18} | {'Exact Time':<12} | {'Approx Time':<12} | {'Speedup':<10}")
    print("-" * 88)
    speedup_8 = time_exact_10_8 / max(time_approx_10_8, 1e-9)
    speedup_9 = time_exact_10_9 / max(time_approx_10_9, 1e-9)
    print(f"{'10^8':<8} | {exact_10_8:<18.12f} | {approx_10_8:<18.12f} | {time_exact_10_8:<10.4f} s | {time_approx_10_8:<10.6f} s | {speedup_8:,.0f}x")
    print(f"{'10^9':<8} | {exact_10_9:<18.12f} | {approx_10_9:<18.12f} | {time_exact_10_9:<10.4f} s | {time_approx_10_9:<10.6f} s | {speedup_9:,.0f}x")
    print("=" * 70)
    print(f"Error at N = 10^8: {abs(exact_10_8 - approx_10_8):.4e}")
    print(f"Error at N = 10^9: {abs(exact_10_9 - approx_10_9):.4e}")

