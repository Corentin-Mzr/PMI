import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation


def wiener_process(mu: float, sigma2: float, nb_iter: int, x0: float = 0.0, y0: float = 0.0) -> np.ndarray[float]:
    """
    Describes a Wiener process
    :param mu: Mean
    :param sigma2: Variance
    :param nb_iter: Number of steps
    :param x0: Initial position of the particle along the x-axis
    :param y0: Initial position of the particle along the y-axis
    :return: Matrix of size (nb_iter, 2)
    """
    # First define the displacement at each step using the normal distribution
    random_dsp = np.random.normal(loc=mu, scale=sigma2, size=(nb_iter - 1, 2))

    # Matrix containing the position of the particle at each step
    p = np.vstack(([x0, y0], random_dsp))

    # We do the cumulative sum to know how the position of the particle evolves
    p = np.cumsum(p, axis=0, dtype=float)

    return p


def plot_wiener(p: np.ndarray[float], mu: float, sigma2: float) -> None:
    """
    Plot the brownian motion of a particle
    :param p: Matrix containing the brownian motion
    :param mu: Mean
    :param sigma2: Variance
    """
    cmap = matplotlib.cm.winter
    plt.plot(p[:, 0], p[:, 1], c='black', linewidth=1, linestyle='--')
    plt.scatter(p[:, 0], p[:, 1], c=np.linalg.norm(p, axis=1), cmap=cmap)
    plt.text(p[0, 0], p[0, 1], '($x_0$,$y_0$)')
    plt.text(p[-1, 0], p[-1, 1], '($x_f$,$y_f$)')
    plt.title(f"Brownian Motion\n$\mu$={mu} | $\sigma^2$={sigma2}")

    plt.tight_layout()
    plt.show()


def animate_wiener(p: np.ndarray, mu: float, sigma2: float) -> None:
    """
    Animation of the brownian movement
    :param p: Matrix containing the brownian motion
    :param mu: Mean
    :param sigma2: Variance
    """
    fig, ax = plt.subplots()
    cmap = matplotlib.cm.winter
    colors = np.linalg.norm(p, axis=1)

    def update(i):
        if i > len(p):
            ax.plot(p[:, 0], p[:, 1], c='black', linewidth=1, linestyle='--')
            ax.scatter(p[:, 0], p[:, 1], c=np.linalg.norm(p, axis=1), cmap=cmap)
            ax.text(p[0, 0], p[0, 1], '($x_0$,$y_0$)')
            ax.text(p[-1, 0], p[-1, 1], '($x_f$,$y_f$)')
        else:
            ax.plot(p[:i, 0], p[:i, 1], c='black', linewidth=1, linestyle='--')
            ax.scatter(p[:i, 0], p[:i, 1], c=colors[:i], cmap=cmap)
            ax.text(p[0, 0], p[0, 1], '($x_0$,$y_0$)')

        ax.set_title(f"Brownian Motion\n$\mu$={mu} | $\sigma^2$={sigma2}")
        plt.tight_layout()

    print("Creating animation...")
    animation = FuncAnimation(fig,
                              update,
                              frames=len(p) + 10,
                              interval=0.001)
    animation.save(filename='brownian_motion.gif', writer='ffmpeg', fps=10)
    print("Animation saved !")


def plot_normal(mu: float, sigma2: float) -> None:
    """
    Plot the normal distribution used
    :param mu: Mean
    :param sigma2: Variance
    """

    def gaussian(x):
        return 1 / np.sqrt(sigma2 * 2 * np.pi) * np.exp(-(x - mu) ** 2 / (2 * sigma2))

    xl = 5 * sigma2 ** 0.5
    n = 10_000
    x = np.linspace(-xl, xl, n)
    y = gaussian(x)

    rd = np.random.normal(loc=mu, scale=sigma2, size=(n,))
    plt.title(f"Normal distribution\n$\mu$={mu} | $\sigma^2$={sigma2}")
    plt.hist(rd, bins=100, edgecolor='black', density=True)
    plt.plot(x, y, c="red")
    plt.show()


def main():
    matplotlib.use('TkAgg')
    matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

    mu = 0.0
    sigma2 = 1.0
    nb_iter = 100
    x0, y0 = 0, 0

    p = wiener_process(mu, sigma2, nb_iter, x0, y0)

    plot_normal(mu=mu, sigma2=sigma2)
    plot_wiener(p=p, mu=mu, sigma2=sigma2)
    animate_wiener(p=p, mu=0.0, sigma2=1.0)


if __name__ == '__main__':
    main()
