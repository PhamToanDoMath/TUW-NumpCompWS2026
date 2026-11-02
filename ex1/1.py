import matplotlib.pyplot as plt 
import math

import numpy as np
def trapezoidal(N, f: callable, a: float = 0, b: float = 1):
    h = (b-a) / N
    I = f(a) + f(b)
    for i in range(1, N):
        I += 2* f(a + i * h)
    I *= h/2
    return I


def box(N, f: callable, a: float = 0, b: float = 1):
    h = (b-a) / N
    I = 0
    for i in range(N):
        I += f(a + i * h)
    I *= h
    return I



if __name__ == "__main__":

    # Question 1a
    def f(x: float):
        return np.exp(x)
    print(trapezoidal(2000, f))
    print(box(2000, f))

    # Question 1b
    def exp(x: float):
        return math.exp(x)

    def N(i: int):
        return 2**(i+1)
    
    h_values = [1/N(i) for i in range(10)]
    trapezoidal_error = [abs(math.e - 1 - trapezoidal(N(i),f)) for i in range(10)]
    box_error = [abs(math.exp(1) - 1 - box(N(i),f)) for i in range(10)]
    plt.loglog(h_values, trapezoidal_error,label="Trapezoidal", marker = "o")
    plt.loglog(h_values, box_error, label="Box", marker = "x")
    plt.legend()
    plt.show()
    