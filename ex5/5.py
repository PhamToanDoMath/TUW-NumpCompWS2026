import numpy as np


def composite_trapezoidal(f, a: float, b: float, N: int) -> float:
    """Composite trapezoidal rule with N subintervals."""
    x = np.linspace(a, b, N + 1)
    y = f(x)
    h = (b - a) / N
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def composite_simpson(f, a: float, b: float, N: int) -> float:
    """Standard composite Simpson's rule with N quadratic pieces (2N sub-intervals)."""
    H = (b - a) / (2 * N)
    x = np.linspace(a, b, 2 * N + 1)
    y = f(x)
    return (H / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])


def run():
    f = np.exp
    a, b = 0.0, 1.0

    print("Exercise 5.5: Neville Scheme Columns m = 0 and m = 1")
    print(f"{'i':>2} | {'h_i':>8} | {'m = 0: T(h_i)':>16} | {'m = 1: N_i':>16} | {'Simpson S(h_i)':>16} | {'|N_i - S(h_i)|':>14}")
    print("-" * 84)

    # Compute trapezoidal values for i = 0, ..., 10
    T_vals = [composite_trapezoidal(f, a, b, 2**i) for i in range(11)]

    for i in range(10):
        h_i = (b - a) * (2.0 ** (-i))
        T_hi = T_vals[i]
        T_next = T_vals[i + 1]

        # Column m = 1: N_i
        N_i = T_next + (1.0 / 3.0) * (T_next - T_hi)

        # Composite Simpson with step size h_i
        S_hi = composite_simpson(f, a, b, 2**i)

        diff = abs(N_i - S_hi)
        print(f"{i:2d} | {h_i:8.4f} | {T_hi:16.10f} | {N_i:16.10f} | {S_hi:16.10f} | {diff:14.2e}")


if __name__ == "__main__":
    run()
