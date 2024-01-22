import numpy as np
from typing import Callable
from parameters import c, dt, dx, nx, nt, length, h


def get_numerical_solution_1d(method: str, u0_vec: Callable, g_vec: Callable, f_vec: Callable) -> np.ndarray:
    """
    :param method: Method used (explicit-upwind, explicit-centered, implicit-upwind, implicit-centered, crank-nicholson)
    :param u0_vec:
    :param g_vec:
    :param f_vec:
    :return: The numerical solution as U, matrix of shape (Nt, Nx) containing all the values for each time step
    """
    if method not in ['explicit-upwind', 'implicit-upwind', 'implicit-centered', 'crank-nicholson']:
        raise Exception("Method must be explicit-upwind, implicit-upwind, implicit-centered or crank-nicholson")

    # Initialization of the space
    x = np.linspace(0, length, nx)

    # Initialization of the matrix
    u = np.zeros((nt, nx))
    u[0] = u0_vec(x)

    # U^n+1 = A U^n + dt F^n
    if method == "explicit-upwind":
        # Computation of the matrix
        k = c * dt / dx
        a_mat = np.eye(nx) + k * (-np.eye(nx, k=0) + np.eye(nx, k=-1))

        # Scheme
        for n in range(nt - 1):
            u[n + 1] = a_mat.dot(u[n]) + dt * f_vec(n * dt, u[n])

            # Boundary conditions
            idx = np.where(x - c * (n + 1) * dt < 0)[0]
            if len(idx) != 0:
                u[n + 1, idx] = g_vec((n + 1) * dt - x[idx] / c)

    # A U^(n+1) = C U^n + dt * F^n => U^(n+1) = A^-1 * (C * U^n + dt * F^n)
    if "implicit" in method:
        # Computation of the matrices
        inv_a_mat = np.eye(nx)
        c_mat = np.eye(nx)

        if "upwind" in method:
            k = c * dt / dx
            c_mat = np.eye(nx)
            inv_a_mat = np.linalg.inv(np.eye(nx) + k * (np.eye(nx, k=0) - np.eye(nx, k=-1)))

        if "centered" in method:
            k = c * dt / (2 * dx)
            c_mat = np.eye(nx)
            inv_a_mat = np.linalg.inv(np.eye(nx) + k * (-np.eye(nx, k=1) + np.eye(nx, k=-1)))

        # Scheme
        for n in range(nt - 1):
            u[n + 1] = inv_a_mat @ (c_mat.dot(u[n]) + dt * f_vec(n * dt, u[n]))

            # Boundary conditions
            idx = np.where(x - c * (n + 1) * dt < 0)[0]
            if len(idx) != 0:
                u[n + 1, idx] = g_vec((n + 1) * dt - x[idx] / c)

    # A * U^n+1 = C * U^n + dt / 2 * F^n + dt / 2 * F^n+1
    if method == "crank-nicholson":
        k = c * dt / (2 * dx)
        inv_a_mat = np.linalg.inv(np.eye(nx) + k * (np.eye(nx, k=0) - np.eye(nx, k=-1)))
        c_mat = np.eye(nx) + k * (-np.eye(nx, k=0) + np.eye(nx, k=-1))

        for n in range(nt - 1):
            u[n + 1] = inv_a_mat @ (c_mat.dot(u[n])
                                    + dt / 2 * f_vec(n * dt, u[n])
                                    + dt / 2 * f_vec((n + 1) * dt, u[n]))

            # Boundary conditions
            idx = np.where(x - c * (n + 1) * dt < 0)[0]
            if len(idx) != 0:
                u[n + 1, idx] = g_vec((n + 1) * dt - x[idx] / c)

    return u


def get_numerical_solution_2d(method: str, u0: Callable, g: Callable, f: Callable) -> np.ndarray:
    """
    :return: The numerical solution as U, matrix of shape (Nx, Ny, Nt) containing all the values for each time step
    """
    if method not in ['fdm-upwind', 'fdm-centered', 'rk4', 'euler-maruyama']:
        raise Exception('Method must be fdm-upwind, fdm-centered, rk4, or euler-maruyama')

    def apply_boundary(mat: np.ndarray) -> np.ndarray:
        """
        :param mat: Matrix at step n
        :return: Matrix with applied boundary conditions
        """
        new_mat = np.copy(mat)
        new_mat[0, :] = 0
        new_mat[-1, :] = 0
        new_mat[:, 0] = 0
        new_mat[:, -1] = 0
        return new_mat

    # Initialization of the space
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, nx)
    x_grid, y_grid = np.meshgrid(x, y)
    d = x_grid ** 2 + y_grid ** 2

    # Initialization of the matrix
    u = np.zeros((nx, nx, nt))
    u[:, :, 0] = u0(x_grid, y_grid)
    u[:, :, 0] = apply_boundary(u[:, :, 0])

    if method == 'rk4':
        def equation(t: float, un: np.ndarray) -> np.ndarray:
            dudx = np.zeros(un.shape)
            dudx[:, 1:] = np.roll(un[:, 1:], -1, axis=1)

            dudy = np.zeros(un.shape)
            dudy[1:, :] = np.roll(un[1:, :], -1, axis=0)

            dudx = dudx - un
            dudy = dudy - un

            return f(t, x_grid, y_grid) - c * (dudx + dudy)

        # Scheme
        for n in range(nt - 1):
            # Compute the coefficients for RK4
            u_n = u[:, :, n]
            t_n = (n + 1) * dt
            k1 = equation(t_n, u_n)
            k2 = equation(t_n + dt / 2, u_n + dt / 2 * k1)
            k3 = equation(t_n + dt / 2, u_n + dt / 2 * k2)
            k4 = equation(t_n + dt, u_n + dt * k3)

            # New value
            u[:, :, n + 1] = u[:, :, n] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # When ||(x,y)||^2 - c * t < 0
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g((n + 1) * dt - dij / c)

            # Boundary conditions
            u[:, :, n + 1] = apply_boundary(u[:, :, n + 1])

    if 'fdm' in method:
        func = None
        k = 0

        def upwind(un: np.ndarray) -> np.ndarray:
            """
            Upwind scheme + Null boundary conditions
            :param un:
            """
            dudx = np.zeros((nx, nx))
            dudx[:, 1:] = (un[:, 1:] - un[:, :-1]) / dx

            dudy = np.zeros((nx, nx))
            dudy[1:, :] = (un[1:, :] - un[:-1, :]) / dx

            return dudx + dudy

        def centered(un: np.ndarray) -> np.ndarray:
            """
            Centered scheme + Null boundary conditions
            :param un:
            """
            dudx = np.zeros((nx, nx))
            dudx[:, 1:-1] = (un[:, 2:] - un[:, :-2]) / (2 * dx)

            dudy = np.zeros((nx, nx))
            dudy[1:-1, :] = (un[2:, :] - un[:-2, :]) / (2 * dx)

            return dudx + dudy

        if 'upwind' in method:
            func = upwind
            k = c * dt
        if 'centered' in method:
            func = centered
            k = c * dt / 2

        # Loop
        for n in range(nt - 1):
            # Scheme + Null boundary conditions
            u_n = u[:, :, n]
            u[:, :, n + 1] = u[:, :, n] - k * func(u_n) + dt * f(n * dt, x_grid, y_grid)

            # When ||(x,y)||^2 - c * t < 0
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g((n + 1) * dt - dij / c)

            # Boundary conditions
            u[:, :, n + 1] = apply_boundary(u[:, :, n + 1])

    if method == "euler-maruyama":
        # Define randomness
        dw = np.random.normal(loc=0.0, scale=dt, size=(nx, nx, nt))

        def grad_dx(mat: np.ndarray) -> np.ndarray:
            return mat - np.roll(mat, 1, axis=1)

        # u_i,j - u_i,j-1
        def grad_dy(mat: np.ndarray) -> np.ndarray:
            return mat - np.roll(mat, 1, axis=0)

        def upwind(mat: np.ndarray) -> np.ndarray:
            return grad_dx(mat) + grad_dy(mat)

        for n in range(nt - 1):
            u[:, :, n + 1] = (u[:, :, n]
                              - c * dt / dx * upwind(u[:, :, n])
                              + h * dw[:, :, n + 1]
                              + dt * f((n + 1) * dt, x_grid, y_grid))

            # When ||(x,y)||^2 - c * t < 0
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g((n + 1) * dt - dij / c)

            # Boundary conditions
            u[:, :, n + 1] = apply_boundary(u[:, :, n + 1])

    return u
