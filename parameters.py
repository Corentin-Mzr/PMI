# All physical parameters for the discretization

length: float = 1.0  # Length of the spatial domain
tf: float = 1.0  # Total time of the simulation
c: float = 4.0  # Velocity
nx: int = 100  # Number of points in the 1D space (and also 2d space)
nt: int = int(c * nx)  # Number of time iterations
dx: float = length / (nx - 1)  # Space step (for x and y axes)
dt: float = tf / nt  # Time step


def check_cfl(dim: int) -> None:
    """
    Verifies the Courant–Friedrichs–Lewy condition
    :param dim: Dimension (1 or 2)
    """
    k = c * dt / dx
    if dim == 2:
        k *= 2

    if k > 1:
        raise ValueError(f'CFL Condition not reached (value is {k:.2f} > 1)\n'
                         f'Change to values of c, dx or dt to verify the condition')
    else:
        print(f'CFL Verified (value is {k:.2f} <= 1)')
