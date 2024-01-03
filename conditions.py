import numpy as np
from scipy.integrate import dblquad, tplquad, quad


def u0_1d(x: float) -> float:
    """
    :param x: Position
    :return: Value of u when x - ct > 0
    """
    return x


u0_1d_vec = np.vectorize(u0_1d, otypes=[float])


def g_1d(t: float) -> float:
    """
    :param t: Time
    :return: Value of u when x - ct < 0
    """
    return 4 * np.abs(np.sin(8 * np.pi * t))


g_1d_vec = np.vectorize(g_1d, otypes=[float])


def f_1d(t: float, x: float) -> float:
    """
    :param t: Time
    :param x: Position
    :return: Return the right-hand side of the equation for the time and position given
    """
    return 2


f_1d_vec = np.vectorize(f_1d, otypes=[float])


def f_1d_integral(t: float, x: float) -> float:
    """
    :param t: Time
    :param x: Position
    :return: Return the integral of f from [0,t] [0,x]
    """
    return f_1d(t, x) * t


def u0_2d(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """
    :param x: Position on x-axis
    :param y: Position on y-axis
    :return: Value of u when (||(x,y)||_2)² > ct
    """
    return x + y


def g_2d(t: float) -> float:
    """
    :param t: Time
    :return: Value of u when (||(x,y)||_2)² > ct
    """
    return np.sin(5 * np.pi * t)


def f_2d(t: float, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """
    :param t: Time
    :param x: Position on x-axis
    :param y: Position on y-axis
    :return: Source term in 2D
    """
    return 0


def f_2d_integral(t: float, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """
    :param t: Time
    :param x: Position on x-axis
    :param y: Position on y-axis
    :return: Return the integral of f from [0,t] [0,x] [0,y]
    """
    return f_2d(t, x, y) * t
