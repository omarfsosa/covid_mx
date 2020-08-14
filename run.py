import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan


def _validate_dates(dates):
    previous_date = dates[0]
    for date in dates[1:]:
        timedelta = date - previous_date
        assert timedelta.days == 1, str(timedelta)
        previous_date = date


def read_data_dict():
    filepath = "Casos_Diarios_Estado_Nacional_Confirmados_20200808.csv"
    df = pd.read_csv(filepath, index_col="nombre").drop("Nacional")
    y = df.filter(regex=(r"\d{2}-\d{2}-\d{4}")).transpose().astype(int)
    # Drop the last week of data as it is clearly incomplete
    y = y.iloc[:-7]
    training_dates = pd.to_datetime(y.index, format="%d-%m-%Y")
    _validate_dates(training_dates)
    poblacion = df.poblacion
    is_observed = (y != 0).astype(int)
    data = {
        "n_dates": len(y),
        "n_states": y.shape[1],
        "y": y.values,
        "is_observed": is_observed.values,
        "pop": poblacion.astype(int).values,
        "n_future": 250,  # Number of days to forecast
    }

    coordinates = {
        "state": list(y.columns),
        "date": training_dates,
        "forecast_date": pd.date_range(
            start=training_dates.min(), periods=len(training_dates) + 250
        ),
    }

    return data, coordinates


def tsplot(
    x, y, n=20, percentile_min=2.5, percentile_max=97.5, color="gray", ax=None, **kwargs
):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(
        y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0
    )
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / n

    if ax is None:
        ax = plt.gca()

    for p1, p2 in zip(perc1, perc2):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    return ax


def save_custom_forecast_plots(stan_data, stan_coordinates, samples):
    plt.style.use("ggplot")
    order = np.argsort(np.max(stan_data["y"], axis=0))[::-1]
    sorted_states = [stan_coordinates["state"][i] for i in order]
    first_day_of_the_month = [
        d for d in stan_coordinates["forecast_date"] if d.day == 1
    ]
    labels = [d.strftime("%b") for d in first_day_of_the_month]

    page1, page2 = sorted_states[:16], sorted_states[16:]

    for n, page in enumerate([page1, page2], 1):
        f, axes = plt.subplots(
            4, 4, figsize=(30, 4 * 6), sharey="row", sharex=True, dpi=300
        )
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        for ax, state in zip(axes.flatten(), page):
            stan_index = stan_coordinates["state"].index(state)
            state_samples = samples.extract()["y_trend"][..., stan_index]
            ax.axvline(
                x=stan_coordinates["date"][-1],
                linestyle="--",
                color="black",
                linewidth=1,
                alpha=0.8,
            )
            ax.plot(
                stan_coordinates["date"],
                data["y"][:, stan_index],
                marker=".",
                color="black",
                linestyle="none",
                alpha=0.8,
                label="datos",
            )
            tsplot(stan_coordinates["forecast_date"], state_samples[:, :], ax=ax)
            ax.set_title(state.lower().title(), fontsize=14)
            ax.set_facecolor("white")
            ax.tick_params(length=0)
            ax.set_xticks(first_day_of_the_month)
            ax.set_xticklabels(labels)

        for ax in axes[:, 0].flatten():
            plt.sca(ax)
            plt.ylabel("Casos confirmados por dia")

        timestamp = stan_coordinates["date"][-1]
        filename = f"./plots/forecast_{timestamp:%Y%m%d}_page{n}.png"
        f.savefig(filename, bbox_inches="tight")
        plt.clf()
        plt.close("all")


def save_final_estimatates_plot(stan_data, stan_coordinates, samples):
    plt.style.use("ggplot")
    median_final_count = np.quantile(
        np.sum(samples.extract()["y_pred"], axis=1), 0.50, axis=0
    )
    lower_final_count = np.quantile(
        np.sum(samples.extract()["y_pred"], axis=1), 0.025, axis=0
    )
    upper_final_count = np.quantile(
        np.sum(samples.extract()["y_pred"], axis=1), 0.975, axis=0
    )
    order = median_final_count.argsort()
    states = range(len(median_final_count))
    f = plt.figure(figsize=(12, 10), dpi=300)
    plt.plot(
        median_final_count[order],
        states,
        marker="o",
        linestyle="none",
        color="k",
        markerfacecolor="white",
        zorder=100,
        label="Estimado final",
    )
    for n, (x1, x2) in enumerate(
        zip(lower_final_count[order], upper_final_count[order])
    ):
        plt.plot([x1, x2], [n, n], color="gray", markerfacecolor="none")

    plt.plot(
        stan_data["y"][:, order].sum(axis=0),
        states,
        marker="o",
        color="black",
        linestyle="none",
        label="Situacion actual",
    )
    plt.yticks(states, [stan_coordinates["state"][i].lower() for i in order])
    xticks = [0, 20_000, 40000, 60000, 80000, 100000]
    plt.xticks(xticks, [format(n, ",") for n in xticks])
    plt.xlabel("Numero de casos confirmados")
    plt.legend(loc="lower right")
    timestamp = stan_coordinates["date"][-1]
    filename = f"./plots/final_estimates_{timestamp:%Y%m%d}.png"
    f.savefig(filename, bbox_inches="tight")
    plt.clf()
    plt.close("all")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    data, coords = read_data_dict()
    model = pystan.StanModel(file="covid_mx.stan")
    sampling_parameters = {
        "iter": 10 if args.debug else 500,
        "chains": 2 if args.debug else 4,
        "n_jobs": 2 if args.debug else 4,
        "init": 0,
        "refresh": 10,
        "control": {"adapt_delta": 0.85, "max_treedepth": 11,},
    }
    samples = model.sampling(data, **sampling_parameters)
    save_custom_forecast_plots(data, coords, samples)
    save_final_estimatates_plot(data, coords, samples)
    print("Finished!")
