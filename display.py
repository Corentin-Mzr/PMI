import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from parameters import length, tf, dt, nt, nx
from math import ceil

matplotlib.use('TkAgg')
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'


def show_image(u_list: list[np.ndarray], title_list=None, to_save: bool = False, path: str = "") -> None:
    """
    Display the matrix
    :param u_list: Arrays containing all the values of U at each position, for each time step
    :param title_list : Titles of the plots
    :param to_save : True to save the figure as a PNG file
    :param path: path to the file
    """
    if title_list is None:
        title_list = len(u_list) * [""]
    # One array to plot
    if len(u_list) == 1:
        x = np.linspace(0, length, u_list[0].shape[1])
        t = np.linspace(0, tf, u_list[0].shape[0])
        fig, ax = plt.subplots(figsize=(16, 9))
        im = ax.contourf(x, t, u_list[0], levels=50, cmap='viridis', vmin=np.min(u_list[0]), vmax=np.max(u_list[0]))
        ax.set_title('Evolution of the solution for the transport equation in 1D (matrix form)\n'
                     f'{title_list[0]}')
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Time (t)')
        ax.set_aspect('equal', 'box')
        plt.colorbar(im, label='u(t,x)', shrink=0.5)
    # Multiple arrays to plot
    else:
        vmin = min([np.min(u) for u in u_list])
        vmax = max([np.max(u) for u in u_list])
        n = len(u_list)
        fig, ax = plt.subplots(nrows=n // 3 + 1, ncols=1 + n % 3, figsize=(16, 9))
        ax = ax.flatten()
        plt.suptitle('Evolution of the solution for the transport equation in 1D (matrix form)')
        for i in range(n):
            x = np.linspace(0, length, u_list[i].shape[1])
            t = np.linspace(0, tf, u_list[i].shape[0])
            im = ax[i].contourf(x, t, u_list[i], levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
            ax[i].set_title(f'{title_list[i]}')
            ax[i].set_xlabel('Position (x)')
            ax[i].set_ylabel('Time (t)')
            ax[i].set_aspect('equal', 'box')
            plt.colorbar(im, label='u(t,x)', ax=ax[i])

    plt.tight_layout()

    # Save image
    if to_save and path != "":
        plt.savefig(path)
    plt.show()


def show_animation_1d(u_list: list[np.ndarray],
                      labels: list[str] = None) -> None:
    """
    Create an animation by looping on each row of the matrix
    :param u_list: Arrays containing all the values of U at each position, for each time step
    :param labels: Titles of the curves
    """
    fig, ax = plt.subplots()
    vmin, vmax = min([np.min(u) for u in u_list]), max([np.max(u) for u in u_list])

    for i in range(nt):
        ax.cla()
        for k in range(len(u_list)):
            x = np.linspace(0, length, u_list[k].shape[1])
            ax.plot(x, u_list[k][i], label=labels[k])

        ax.set_xlabel('Position x')
        ax.set_ylabel('Concentration u(t,x)')
        ax.axis((0, length, vmin, vmax))

        plt.title(f"\nt={i * dt:.2f}s")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.pause(dt)
    plt.show()


def save_animation_1d(u_list: list[np.ndarray],
                      labels: list[str],
                      ani_name: str) -> None:
    fig, ax = plt.subplots()
    mins, maxs = min([np.min(u) for u in u_list]), max([np.max(u) for u in u_list])

    def update(i: int):
        ax.cla()
        for k in range(len(u_list)):
            x = np.linspace(0, length, len(u_list[k][i]))
            ax.plot(x, u_list[k][i], label=labels[k])
        ax.set_xlabel('Position x')
        ax.set_ylabel('Concentration u(t,x)')
        ax.axis((0, length, mins, maxs))
        plt.title(f"\nt={i * dt:.2f}s")
        plt.legend(loc='upper right')
        plt.tight_layout()

    print("Creating animation...")
    ani = FuncAnimation(fig, update, frames=range(nt), interval=dt)
    ani.save(ani_name, writer='ffmpeg', fps=30)
    print(f'Animation saved at {ani_name}')


def show_average(u_list: list[np.ndarray], title_list=None, to_save: bool = False, path: str = "") -> None:
    """
    Plot the average value of u over time
    :param u_list: Arrays containing all the values of U at each position, for each time step
    :param title_list: Titles of the plots
    :param to_save : True to save the figure as a PNG file
    :param path: path to the file
    """
    if title_list is None:
        title_list = len(u_list) * [""]

    t = np.linspace(0, tf, nt)
    for i in range(len(u_list)):
        u_avg = np.mean(u_list[i], axis=1)
        plt.plot(t, u_avg)
        plt.title('Evolution of the average concentration in 1D')
        plt.legend(title_list)
    plt.xlabel('Time (t)')
    plt.ylabel('Average concentration')

    plt.tight_layout()

    if to_save and path != "":
        plt.savefig(path)
    plt.show()


def show_animation_2d(u_list: list[np.ndarray], title_list: list[str]) -> None:
    """
    :param u_list:
    :param title_list:
    """
    x = np.linspace(0, length, nx)
    vmin = max([np.min(u) for u in u_list])
    vmax = min([np.max(u) for u in u_list])

    if len(u_list) == 1:
        fig, ax = plt.subplots()
        for n in range(nt):
            ax.cla()
            ax.set_title(f"{title_list[0]}\n{dt * n:.2f}s")
            ax.imshow(u_list[0][:, :, n],
                      vmin=vmin,
                      vmax=vmax,
                      extent=(0, length, 0, length),
                      origin='lower')
            plt.tight_layout()
            plt.pause(dt)
    else:
        n = len(u_list)
        if n == 2:
            fig, ax = plt.subplots(ncols=2, nrows=1)
        else:
            fig, ax = plt.subplots(ncols=n % 3 + 1, nrows=n // 3 + 1)
        num_plots = len(fig.get_axes())
        ax = ax.flatten()

        for i in range(n, num_plots):
            fig.delaxes(ax[i])

        for n in range(nt):
            if not plt.fignum_exists(fig.number):
                break

            for k in range(len(u_list)):
                ax[k].cla()
                ax[k].set_title(f"{title_list[k]}\n{dt * n:.2f}s | Mean : {np.mean(u_list[k][:, :, n]):.2f}")
                ax[k].imshow(u_list[k][:, :, n],
                             vmin=vmin,
                             vmax=vmax,
                             extent=(0, length, 0, length),
                             origin='lower')
                ax[k].set_aspect('equal', 'box')
                plt.tight_layout()

            plt.pause(0.001)
    plt.show()


def save_animation_2d(u_list: list[np.ndarray], title_list: list[str], path: str) -> None:
    """
    :param u_list:
    :param title_list:
    :param path:
    """
    n = len(u_list)
    if n == 2:
        fig, ax = plt.subplots(ncols=2, nrows=1)
    else:
        fig, ax = plt.subplots(ncols=n // 3 + 1, nrows=n % 3 + 1)
    ax = ax.flatten()

    vmin = max([np.min(u) for u in u_list])
    vmax = min([np.max(u) for u in u_list])

    x = np.linspace(0, length, nx)

    def update(i: int):
        for k in range(len(u_list)):
            ax[k].cla()
            ax[k].imshow(u_list[k][:, :, i], vmin=vmin, vmax=vmax, extent=(0, length, 0, length), origin='lower')
            ax[k].set_title(f"{title_list[k]}\nt={i * dt:.2f}s | Mean : {min(np.mean(u_list[k][:, :, i]), 999):.2f}")
            ax[k].set_aspect('equal', 'box')
            plt.tight_layout()

    print('Creating animation...')
    animation = FuncAnimation(fig,
                              update,
                              frames=range(nt),
                              interval=dt)
    animation.save(path, writer='ffmpeg', fps=30)
    print(f"Animation saved at {path}")
