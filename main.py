import argparse
import numpy as np
import os.path

from numerical import get_numerical_solution_1d, get_numerical_solution_2d
from analytical import get_analytical_solution_1d, get_analytical_solution_2d
from display import show_image, show_animation_1d, show_average, show_animation_2d, save_animation_2d
from conditions import test_case_1d, vectorize_test_case, test_case_2d

from parameters import length, tf, nx, dt, nt, check_cfl


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

        # Choosen test case
        test_case = test_case_1d(test_id)
        u0, g, f = test_case
        u0_vec, g_vec, f_vec = vectorize_test_case(test_case)

        # Numerical solution
        u_num = get_numerical_solution_1d(method=method, u0_vec=u0_vec, g_vec=g_vec, f_vec=f_vec)

        # Analytical solutions
        u_ana_same = get_analytical_solution_1d(size='same', u0=u0, g=g, f=f)
        u_ana_acc = get_analytical_solution_1d(size='accurate', u0=u0, g=g, f=f)

        # Display the solution as a map of position and time
        show_image(u_list=[u_num, u_ana_same],
                   title_list=[f"Numerical solution\nMethod: {method}", "Analytical solution"],
                   to_save=save_file,
                   path=path_1)

        # Error between the two models
        abs_error = np.abs(u_ana_same - u_num)
        show_image(u_list=[abs_error],
                   title_list=[f'Absolute error\nMethod: {method}'],
                   to_save=save_file,
                   path=path_2)

        # Average value of concentration over time
        show_average(u_list=[u_num, u_ana_acc],
                     title_list=[f"Numerical solution\nMethod: {method}", "Analytical solution"],
                     to_save=save_file,
                     path=path_3)

        # Display animated graph
        show_animation_1d(u_list=[u_num, u_ana_acc],
                          labels=[f'Numerical solution\nMethod: {method}', 'Analytical solution'],
                          to_save=save_file,
                          ani_name=os.path.join(directory, f"{method}_animation.gif"))

    if dim == 2:
        path = os.path.join(directory, f'{method}_animation_{tf}s.gif')
        u0, g, f = test_case_2d(test_id)

        u_num = get_numerical_solution_2d(method=method, u0=u0, g=g, f=f)
        u_ana = get_analytical_solution_2d(u0=u0, g=g, f=f)

        if save_file:
            save_animation_2d(u_list=[u_ana, u_num],
                              title_list=["Analytical solution", f"Numerical solution\nMethod: {method}"],
                              path=path)
        else:
            show_animation_2d(u_list=[u_ana, u_num],
                              title_list=["Analytical solution", f"Numerical solution\nMethod: {method}"])


if __name__ == "__main__":
    main()
