import argparse
import numpy as np
import os.path

from numerical import get_numerical_solution_1d, get_numerical_solution_2d, oil_in_water
from analytical import get_analytical_solution_1d, get_analytical_solution_2d
from display import show_image, show_average, show_animation_1d, save_animation_1d, show_animation_2d, save_animation_2d
from conditions import test_case_1d, vectorize_test_case, test_case_2d
from parameters import tf, check_cfl, length


def main():
    if dim == 1:
        # File names
        path_1: str = os.path.join(directory, f'comparison.png')
        path_2: str = os.path.join(directory, f'absolute_error.png')
        path_3: str = os.path.join(directory, f'average.png')

        # Choosen test case
        test_case = test_case_1d(test_id)
        u0, g, f = test_case
        u0_vec, g_vec, f_vec = vectorize_test_case(test_case)

        # Analytical solutions
        u_ana_same = get_analytical_solution_1d(size='same', u0=u0, g=g, f=f)
        u_ana_acc = get_analytical_solution_1d(size='accurate', u0=u0, g=g, f=f)
        u_oil = oil_in_water(u0_vec=u0_vec, g=g, f=f)

        show_average(u_list=[u_oil], title_list=['Oil in water'], to_save=save_file, path=os.path.join(directory, f'oil_average.png'))
        return 0

        # Numerical solution
        u_list = []
        label_list = []
        abs_error_list = []
        abs_error_label = []
        methods = ['explicit-upwind', 'implicit-upwind', 'implicit-centered', 'crank-nicholson']
        if method == 'all':
            u_list = [get_numerical_solution_1d(method=meth, u0_vec=u0_vec, g_vec=g_vec, f_vec=f_vec) for meth in methods]
            label_list = [f"Numerical solution\nMethod: {meth}" for meth in methods]
            abs_error_list = [np.abs(u_ana_same - u_num) for u_num in u_list]
            abs_error_label = [f'Absolute error\nMethod: {meth}' for meth in methods]
        else:
            u_list.append(get_numerical_solution_1d(method=method, u0_vec=u0_vec, g_vec=g_vec, f_vec=f_vec))
            label_list.append(f"Numerical Solution\nMethod: {method}")
            abs_error_list = [np.abs(u_ana_same - u_list[0])]
            abs_error_label = [f'Absolute error\nMethod: {method}']

        # Display the solution as a map of position and time
        show_image(u_list=u_list,
                   title_list=label_list,
                   to_save=save_file,
                   path=path_1)

        # Error between the two models
        show_image(u_list=abs_error_list,
                   title_list=abs_error_label,
                   to_save=save_file,
                   path=path_2)

        # Average value of concentration over time
        show_average(u_list=u_list + [u_ana_acc],
                     title_list=label_list + ["Analytical solution"],
                     to_save=save_file,
                     path=path_3)

        # Display animated graph
        if save_file:
            save_animation_1d(u_list=u_list + [u_ana_acc],
                              labels=label_list + ['Analytical solution'],
                              ani_name=os.path.join(directory, f"{method}_animation.gif"))
        else:
            show_animation_1d(u_list=u_list + [u_ana_acc],
                              labels=label_list + ['Analytical solution'])

    if dim == 2:
        path = os.path.join(directory, f'{method}_animation_{tf}s_{length}m.gif')
        u0, g, f = test_case_2d(test_id)

        # Analytical solution
        u_ana = get_analytical_solution_2d(u0=u0, g=g, f=f)

        # Numerical solution
        methods = ['fdm-upwind', 'fdm-centered', 'rk4']
        u_list = []
        if method == "all":
            u_list = [get_numerical_solution_2d(method=meth, u0=u0, g=g, f=f) for meth in methods]
            title_list = [f'Numerical solution\nMethod: {meth}' for meth in methods]
        else:
            u_list = [get_numerical_solution_2d(method=method, u0=u0, g=g, f=f)]
            title_list = [f'Numerical solution\nMethod: {method}']

        if save_file:
            save_animation_2d(u_list=[u_ana] + u_list,
                              title_list=["Analytical solution"] + title_list,
                              path=path)
        else:
            show_animation_2d(u_list=[u_ana] + u_list,
                              title_list=["Analytical solution"] + title_list)


if __name__ == "__main__":
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

    # Start simulation
    main()
