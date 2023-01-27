from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_5_1_1(metrics_dict_last, metrics_dict_movie, plots_dir: Path):
    fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Average envy
    ax1.plot(
        metrics_dict_last["mean_envy"].keys(),
        metrics_dict_last["mean_envy"].values(),
        label="Last.fm",
    )
    ax1.plot(
        metrics_dict_movie["mean_envy"].keys(),
        metrics_dict_movie["mean_envy"].values(),
        label="MovieLens",
    )
    ax1.set_xlabel("number of factors")
    ax1.set_ylabel("average envy")

    # Prop
    ax2.plot(
        metrics_dict_last["prop_eps_envy"].keys(),
        metrics_dict_last["prop_eps_envy"].values(),
        label="Last.fm",
    )
    ax2.plot(
        metrics_dict_movie["prop_eps_envy"].keys(),
        metrics_dict_movie["prop_eps_envy"].values(),
        label="MovieLens",
    )
    ax2.set_xlabel("number of factors")
    ax2.set_ylabel("prop. of envious users (ε=0.05)")

    plt.savefig(plots_dir / "envy_model_mispecification.png")


def plot_bandit_ocef(results_dir, plots_dir):

    alphas = [i / 10 for i in range(1, 6)]
    problem_paths = [f for f in os.listdir(
        results_dir) if f.startswith("problem")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for path in problem_paths:
        # read tuple from path
        path = os.path.join(results_dir, path)

        problem = np.genfromtxt(path, delimiter=',')
        t = problem[:, 0]
        cost = problem[:, 1]

        ax1.plot(alphas, t, label=path[-5])
        ax2.plot(alphas, cost, label=path[-5])

    # fig.gca().ticklabel_format(useMathText=True)
    ax1.set_xlabel("α")
    ax1.set_ylabel("duration")
    ax1.ticklabel_format(style='sci', axis='y',
                         scilimits=(0, 0), useMathText=True)
    ax1.legend()

    ax2.set_xlabel("α")
    ax2.set_ylabel("average cost of exploration")
    ax2.legend()

    plt.show()


plot_bandit_ocef('results_ocef', 'plots')
