import numpy as np

# given data
x = np.array([0, 1, 4], dtype=float)
f = np.array([0, 2, 8], dtype=float)

# --------------------------------------------------
# Part (a): Lagrange interpolation
# --------------------------------------------------

def lagrange_poly(x_eval):
    L0 = ((x_eval - 1) * (x_eval - 4)) / ((0 - 1) * (0 - 4))
    L1 = ((x_eval - 0) * (x_eval - 4)) / ((1 - 0) * (1 - 4))
    L2 = ((x_eval - 0) * (x_eval - 1)) / ((4 - 0) * (4 - 1))

    return 0 * L0 + 2 * L1 + 8 * L2


print("p(2) from Lagrange =", lagrange_poly(2))

# --------------------------------------------------
# Part (b): Solve Vandermonde system
# --------------------------------------------------

A = np.array([
    [1, 0, 0**2],
    [1, 1, 1**2],
    [1, 4, 4**2]
], dtype=float)

a = np.linalg.solve(A, f)

a0, a1, a2 = a

print("\nCoefficients:")
print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)

def p(x):
    return a0 + a1*x + a2*x**2

print("p(2) from coefficients =", p(2))

# --------------------------------------------------
# Part (c): polyfit / polyval
# --------------------------------------------------

coef = np.polyfit(x, f, 2)

print("\npolyfit coefficients:")
print(coef)

p2 = np.polyval(coef, 2)

print("p(2) from polyval =", p2)