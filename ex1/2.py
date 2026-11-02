import math
import matplotlib.pyplot as plt


def trapezoidal(N, f: callable, a: float = 0, b: float = 1):
    h = (b-a) / N
    I = f(a) + f(b)
    for i in range(1, N):
        I += 2* f(a + i * h)
    I *= h/2
    return I


def trapezoidal_with_estimator(N: int, f, a: float = 0.0, b: float = 1.0):
    T_h = trapezoidal(N, f, a, b)
    T_h2 = trapezoidal(2 * N, f, a, b)
    E_h = (4/3) * (T_h2 - T_h)
    return T_h, E_h


if __name__ == "__main__":
    def exp(x: float):
        return math.exp(x)

    def N(i: int):
        return 2**(i + 1)

    exact_integral = math.e - 1

    h_values = [1.0 / N(i) for i in range(10)]
    ratios = []

    for i in range(10):
        n_i = N(i)
        h_i = 1.0 / n_i
        
        T_h, E_h = trapezoidal_with_estimator(n_i, exp)
        err_i = exact_integral - T_h
        r_i = err_i / E_h

        ratios.append(r_i)

    plt.figure(figsize=(8, 5))
    plt.semilogx(h_values, ratios, marker="o", linestyle="-", color="b", label="r(h)")
    plt.xlabel("h")
    plt.ylabel("r(h)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()