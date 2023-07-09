
import numpy as np


def test_qp(X, t):
    # Function
    f_x = X[0]**2 + X[1]**2 + (X[2]+1)**2

    # Gradient
    g_x = np.zeros(3, dtype=np.float64)
    g_x[0] = 2*t*X[0] - 1/X[0]
    g_x[1] = 2*t*X[1] - 1/X[1]
    g_x[2] = 2*t*(X[2] + 1) - 1/X[2]

    # Hessian
    h_x = np.zeros((3, 3), dtype=np.float64)
    h_x[0][0] = 2*t + 1/X[0]**2
    h_x[1][1] = 2*t + 1/X[1]**2
    h_x[2][2] = 2*t + 1/X[2]**2

    return f_x, g_x, h_x


def test_lp(X, t):
    # Function
    f_x = - X[0] - X[1]

    # Gradient
    g_x = np.zeros(2, dtype=np.float64)
    g_x[0] = - t + 1/(1-X[0]-X[1]) - 1/(X[0]-2)
    g_x[1] = - t + 1/(1-X[0]-X[1]) - 1/(X[1]-1) - 1/X[1]

    # Hessian
    h_x = np.zeros((2, 2), dtype=np.float64)
    h_val = (1 / (1-X[0]-X[1]))**2
    h_x[0][0] = h_val + (1/(X[0]-2))**2
    h_x[0][1] = h_val
    h_x[1][0] = h_val
    h_x[1][1] = h_val + (1/(X[1]-1))**2 + (1/X[1])**2

    return f_x, g_x, h_x


