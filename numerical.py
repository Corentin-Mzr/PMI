import numpy as np
from parameters import c, dt, dx, nx, nt, length
from conditions import u0_1d_vec, g_1d_vec, f_1d_vec, u0_2d, g_2d, f_2d


def get_numerical_solution_1d() -> np.ndarray:
    """
    :return: The numerical solution as U, matrix of shape (Nt, Nx) containing all the values for each time step
    """
    # Initialization of the space
    x = np.linspace(0, length, nx)

    # Computation of the matrix
    k = c * dt / dx
    a_mat = np.eye(nx) + k * (-np.eye(nx, k=0) + np.eye(nx, k=-1))

    # Initialization of the matrix
    u = np.zeros((nt, nx))
    u[0] = u0_1d_vec(x)

    # Scheme
    for n in range(nt - 1):
        u[n + 1] = a_mat.dot(u[n]) + dt * f_1d_vec(n * dt, u[n])

        # Boundary conditions
        idx = np.where(x - c * (n + 1) * dt < 0)[0]
        if len(idx) != 0:
            u[n + 1, idx] = g_1d_vec((n + 1) * dt - x[idx] / c)

    return u


def get_numerical_solution_2d(method: str) -> np.ndarray:
    """
    :return: The numerical solution as U, matrix of shape (Nx, Ny, Nt) containing all the values for each time step
    """
    # Initialization of the space
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, nx)
    x_grid, y_grid = np.meshgrid(x, y)
    d = x_grid ** 2 + y_grid ** 2

    # Initialization of the matrix
    u = np.zeros((nx, nx, nt))
    u[:, :, 0] = u0_2d(x_grid, y_grid)

    if method == 'RK4':
        def equation(t: float, un: np.ndarray) -> np.ndarray:
            dudx = np.zeros(un.shape)
            dudx[:, 1:] = np.roll(un[:, 1:], -1, axis=1)

            dudy = np.zeros(un.shape)
            dudy[1:, :] = np.roll(un[1:, :], -1, axis=0)

            dudx = dudx - un
            dudy = dudy - un

            return f_2d(t, x_grid, y_grid) - c * (dudx + dudy)

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

            # Boundary conditions
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g_2d((n + 1) * dt - dij / c)

        return u

    if 'FDM' in method:
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

        if 'UP' in method:
            func = upwind
            k = c * dt
        elif 'CENTER' in method:
            func = centered
            k = c * dt / 2
        else:
            raise Exception('FDM Invalid scheme')

        # Loop
        for n in range(nt - 1):
            # Scheme + Null boundary conditions
            u_n = u[:, :, n]
            u[:, :, n + 1] = u[:, :, n] - k * func(u_n) + dt * f_2d(n * dt, x_grid, y_grid)

            # Conditions
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g_2d((n + 1) * dt - dij / c)

        return u

    if method == 'SIMPLE':
        for n in range(nt - 1):
            un = u[:, :, n]
            u[:, :, n + 1] = (1 - 2 * c * dt / dx) * un + dt * f_2d(n * dt, x_grid, y_grid)
            u[1:, :, n + 1] += c * dt / dx * un[1:, :]
            u[:, 1:, n + 1] += c * dt / dx * un[:, 1:]

            # Conditions
            i, j = np.where(d - c * (n + 1) * dt < 0)
            dij = (dx * i) ** 2 + (dx * j) ** 2
            u[i.tolist(), j.tolist(), n + 1] = g_2d((n + 1) * dt - dij / c)

        return u

    raise Exception('Invalid method specified')
