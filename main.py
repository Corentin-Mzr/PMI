import argparse
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
    # Parser
    parser = argparse.ArgumentParser(description='Transport equation simulation')

    # Command line arguments
    parser.add_argument('--dimension',
                        '-d',
                        type=int,
                        help='Space dimension (1D or 2D, i.e. 1 or 2)',
                        required=True)

    parser.add_argument('--method',
                        '-m',
                        type=str,
                        help="Numerical method to use",
                        required=True)

    parser.add_argument('--test',
                        '-t',
                        type=int,
                        help='ID of the test case (1 to 4)',
                        required=True)

    parser.add_argument('--save',
                        '-s',
                        type=int,
                        help="True to save the graphs and animations",
                        required=True)

    # Parse command line arguments
    args = parser.parse_args()

    # Arguments
    dim: int = args.dimension
    method: str = args.method
    test_id: int = args.test
    save_file: bool = False if args.save != 1 else True

    print(f"Dimension choosen: {dim}")
    print(f"Numerical method: {method}")
    print(f"Test case: {test_id}")
    print(f"Save file: {save_file}")

    # Path where to save the graphs and animations
    directory: str = f'output/{dim}d/test_{test_id}/{method}'
    if save_file:
        print(f"Graphs will be saved in: {directory}")

    # Check CFL before starting the simulation
    check_cfl(dim)

    if dim == 1:
        # File names
        path_1: str = os.path.join(directory, f'comparison.png')
        path_2: str = os.path.join(directory, f'absolute_error.png')
        path_3: str = os.path.join(directory, f'average.png')

        # Numerical solution
        u_num = get_numerical_solution_1d(method=method)

        # Analytical solutions
        u_ana_same = get_analytical_solution_1d(size='same')
        u_ana_acc = get_analytical_solution_1d(size='accurate')

        # Display the solution as a map of position and time
        show_as_image(u_list=[u_ana_same, u_num],
                      title_list=["Analytical solution", f"Numerical solution\nMethod: {method}"],
                      to_save=save_file,
                      path=path_1)

        # Error between the two models
        abs_error = np.abs(u_ana_same - u_num)
        show_as_image(u_list=[abs_error],
                      title_list=[f'Absolute error\nMethod: {method}'],
                      to_save=save_file,
                      path=path_2)

        # Average value of concentration over time
        show_as_average(u_list=[u_num, u_ana_acc],
                        title_list=[f"Numerical solution\nMethod: {method}", "Analytical solution"],
                        to_save=save_file,
                        path=path_3)

        # Display animated graph
        #show_as_animation(u_list=[u_num, u_ana_acc],
        #                  labels=['Numerical solution', 'Analytical solution'],
        #                  to_save=save_file,
        #                  ani_name=os.path.join(directory, f"{method}_animation.gif"))

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
