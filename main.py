import numpy as np
import os.path

from numerical import get_numerical_solution_1d, get_numerical_solution_2d
from analytical import get_analytical_solution_1d, get_analytical_solution_2d
from display import show_as_image, show_as_animation, show_as_average

import matplotlib.pyplot as plt

from parameters import length, tf, nx, dt, nt, check_cfl
from matplotlib.animation import FuncAnimation

import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'


def update(i, x, y, u_ana, u_num, ax):
    ax[0].cla()
    ax[0].contourf(x, y, u_ana[:, :, i], levels=100, cmap='viridis', vmin=np.min(u_ana), vmax=np.max(u_ana))
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title(f"Analytical {i * dt:.2f} | mean : {np.mean(u_ana[:, :, i]):.2f}")

    ax[1].cla()
    ax[1].contourf(x, y, u_num[:, :, i], levels=100, cmap='viridis', vmin=np.min(u_ana), vmax=np.max(u_ana))
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title(f"Numerical {i * dt:.2f} | mean : {np.mean(u_num[:, :, i]):.2f}")


def main():
    # Choose dimension
    dim: int = 1
    n_test: int = 4

    # Change save_file to True if you want to save files
    # Change paths to avoid overwrite
    save_file: bool = True
    directory = f'output/{dim}d/test_{n_test}'

    path_1: str = os.path.join(directory, 'more_acc_'+'comparison.png')
    path_2: str = os.path.join(directory, 'absolute_error.png')
    path_3: str = os.path.join(directory, 'more_acc_'+'average.png')

    # Check CFL before starting the simulation
    check_cfl(dim)

    if dim == 1:
        # Numerical solution
        u_num = get_numerical_solution_1d()

        # Analytical solution
        u_ana = get_analytical_solution_1d()

        show_as_animation(u_list=[u_ana, u_num],
                          labels=['Analytical', 'Numerical'],
                          to_save=save_file,
                          ani_name=os.path.join(directory, 'more_acc_'+"animation.gif"))
        return 0


        # Display the solution as a map of position and time
        show_as_image(u_list=[u_ana, u_num],
                      title_list=["Analytical solution", "Numerical solution"],
                      to_save=save_file,
                      path=path_1)

        # Error between the two models
        #abs_error = np.abs(u_ana - u_num)
        #show_as_image(u_list=[abs_error],
        #              title_list=['Absolute error'],
        #              to_save=save_file,
        #              path=path_2)

        # Average value of concentration over time
        show_as_average(u_list=[u_num, u_ana],
                        title_list=["Numerical solution", "Analytical solution"],
                        to_save=save_file,
                        path=path_3)

    if dim == 2:
        method = 'FDM CENTER'
        u_ana = get_analytical_solution_2d()
        u_num = get_numerical_solution_2d(method)
        fig, ax = plt.subplots(ncols=2, nrows=1)
        ax = ax.flatten()

        if True:
            x = np.linspace(0, length, nx)
            y = np.linspace(0, length, nx)

            animation = FuncAnimation(fig,
                                      update,
                                      frames=range(nt),
                                      interval=dt,
                                      fargs=(x, y, u_ana, u_num, ax))

            animation.save(os.path.join(directory, f'{method}_animation_{tf}s.gif'), writer='ffmpeg', fps=30)
            print("saved")
            plt.show()


if __name__ == "__main__":
    main()
