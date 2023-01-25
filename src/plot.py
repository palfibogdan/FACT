import matplotlib.pyplot as plt


def plot_5_1_1(metrics_dict_last, metrics_dict_movie):
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
