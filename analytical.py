import numpy as np
from typing import Callable
from parameters import c, length, nx, nt, dt


def get_analytical_solution_1d(size: str, u0: Callable, g: Callable, f: Callable) -> np.ndarray:
    """
    :param size: 'same' for nx = nx or 'accurate' for nx = 10 * nx
    :param u0:
    :param g:
    :param f:
    :return: The analytical solution for 1D case as U,
        matrix of shape (Nt, Nx) containing all the values for each time step
    """
    if size not in ['same', 'accurate']:
        raise Exception('Size must be "accurate" or "same"')

    n = 0
    if size == 'same':
        n = nx
    if size == 'accurate':
        n = 10 * nx
    x = np.linspace(0, length, n)

    def u_1d(t: float) -> np.ndarray:
        """
        :param t: Time
        :return: Value of u everywhere on the 1d space at the time given
        """
        return np.where(x - c * t > 0, u0(x - c * t) + f(t, x) * t, g(t - x / c))

    u_mat = np.zeros((nt, n))
    u_mat[0] = np.vectorize(u0, otypes=[float])(x)

    # Compute solution at each time step
    for n in range(nt - 1):
        u_mat[n + 1] = u_1d((n + 1) * dt)

    return u_mat


def get_analytical_solution_2d(u0: Callable, g: Callable, f: Callable) -> np.ndarray:
    """
    :return: The analytical solution for 2d case as U,
        matrix of shape (Nx, Ny, Nt) containing all the values for each time step
    """
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, nx)
    x_grid, y_grid = np.meshgrid(x, y)

    def u_2d(t: float) -> np.ndarray:
        """
        :param t: Time
        :return: Value of u everywhere in space at the time given
        """
        d = x_grid ** 2 + y_grid ** 2
        return np.where(d - c * t > 0, u0(x_grid - c * t, y_grid - c * t) + f(t, x_grid, y_grid) * t, g(t - d / c))

    u_mat = np.zeros((nx, nx, nt))
    u_mat[:, :, 0] = u0(x_grid, y_grid)

    # Compute solution at each time step
    for n in range(nt - 1):
        u_mat[:, :, n + 1] = u_2d(dt * (n + 1))

    return u_mat
