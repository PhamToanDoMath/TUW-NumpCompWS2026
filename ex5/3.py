"""
Problem 5.3: Analytical and Numerical Quadrature Comparison
Numerical Computation - Problem Sheet 5
"""

from fractions import Fraction


def f(x):
    return Fraction(1) - x - 4 * x**3 + 2 * x**5


def compute_all():
    a = Fraction(-2)
    b = Fraction(4)

    # Analytical integral
    # F(x) = x - x^2/2 - x^4 + x^6/3
    def F(x):
        return x - x**2 / 2 - x**4 + x**6 / 3

    I = F(b) - F(a)

    # (a) Trapezoidal rule
    Q_a = (b - a) * (f(a) + f(b)) / 2

    # (b) Midpoint rule
    m = (a + b) / 2
    Q_b = (b - a) * f(m)

    # (c) Simpson rule
    Q_c = (b - a) * (f(a) + 4 * f(m) + f(b)) / 6

    # (d) Milne rule (closed Newton-Cotes n = 4)
    h_d = (b - a) / 4
    x_d = [a + i * h_d for i in range(5)]
    w_d = [Fraction(7, 90), Fraction(32, 90), Fraction(12, 90), Fraction(32, 90), Fraction(7, 90)]
    Q_d = (b - a) * sum(w * f(x) for w, x in zip(w_d, x_d))

    # (e) Composite trapezoidal N = 4
    h_e = (b - a) / 4
    x_e = [a + i * h_e for i in range(5)]
    Q_e = h_e * (f(x_e[0]) / 2 + sum(f(x_e[i]) for i in range(1, 4)) + f(x_e[4]) / 2)

    # (f) Composite Simpson N = 3
    h_f = (b - a) / 3
    Q_f = Fraction(0)
    for i in range(3):
        xl = a + i * h_f
        xr = a + (i + 1) * h_f
        xm = (xl + xr) / 2
        Q_f += (h_f / 6) * (f(xl) + 4 * f(xm) + f(xr))

    methods = [
        ("(a) Trapezoidal rule", Q_a),
        ("(b) Midpoint rule", Q_b),
        ("(c) Simpson rule", Q_c),
        ("(d) Milne rule", Q_d),
        ("(e) Comp. trapezoidal (N=4)", Q_e),
        ("(f) Comp. Simpson (N=3)", Q_f),
    ]

    print(f"Exact integral I = {I}\n")
    print(f"{'Method':<28} | {'Approximation Q':>16} | {'Relative Error':>15}")
    print("-" * 65)
    for name, Q in methods:
        rel_err = abs(I - Q) / abs(I)
        print(f"{name:<28} | {float(Q):16.4f} | {float(rel_err):15.6e}")


if __name__ == "__main__":
    compute_all()
