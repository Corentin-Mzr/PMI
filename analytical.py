import numpy as np
from parameters import c, length, nx, nt, dt
from conditions import u0_1d, g_1d, u0_1d_vec, f_1d_integral, u0_2d, g_2d, f_2d_integral


def u_1d(t: float, x: float) -> float:
    """
    :param t: Time
    :param x: Position
    :return: Value of u at the time and position given
    """
    if x - c * t > 0:
        return u0_1d(x - c * t) + f_1d_integral(t, x)
    else:
        return g_1d(t - x / c)


u_1d_vec = np.vectorize(u_1d, otypes=[float])


def get_analytical_solution_1d() -> np.ndarray:
    """
    :return: The analytical solution for 1D case as U,
        matrix of shape (Nt, Nx) containing all the values for each time step
    """
    x = np.linspace(0, length, 10 * nx)

    u_mat = np.zeros((nt, 10 * nx))
    u_mat[0] = u0_1d_vec(x)

    for n in range(nt - 1):
        u_mat[n + 1] = u_1d_vec((n + 1) * dt, x)

    return u_mat


def u_2d(t: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param t: Time
    :param x: Position on x-axis
    :param y: Position on y-axis
    :return: Value of u at the time and position given
    """
    d = x ** 2 + y ** 2
    return np.where(d - c * t > 0, u0_2d(x, y) + f_2d_integral(t, x, y), g_2d(t - d / c))


def get_analytical_solution_2d() -> np.ndarray:
    """
    :return: The analytical solution for 2d case as U,
        matrix of shape (Nx, Ny, Nt) containing all the values for each time step
    """
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, nx)
    x_grid, y_grid = np.meshgrid(x, y)

    u_mat = np.zeros((nx, nx, nt))
    u_mat[:, :, 0] = u0_2d(x_grid, y_grid)

    for n in range(nt - 1):
        u_mat[:, :, n + 1] = u_2d(dt * (n + 1), x_grid, y_grid)

    return u_mat
