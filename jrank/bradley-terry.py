"""Bradley Terry model for ranking models on the Rakuda benchmark"""

import json
import math
from datetime import datetime

import emcee
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from getdist import MCSamples, plots
from multiprocess import Pool
from pandas import DataFrame
from registry import StandingsRegistry
from samplotlib.circusboy import CircusBoy
from scipy.optimize import minimize

GLOBAL_SCALE = 400
GLOBAL_BASE = 10
GLOBAL_INIT_RATING = 1000

pd.options.display.float_format = "{:.3f}".format
plt.rcParams["text.usetex"] = False


def model_hyperlink(link: str, model_name: str):
    """Returns a hyperlink to the model page on huggingface.co"""
    return f'<a target="_blank" href="{link}" style="color:#1e50a2, textDecoration: underline,textDecorationStyle: dotted">{model_name}</a>'


def make_clickable_model(model_name: str):
    """Returns a hyperlink to the model page on huggingface.co"""
    link = f"https://huggingface.co/{model_name}"

    # Can hardcode urls and names here
    if "gpt-3.5-turbo" in model_name:
        link = "https://openai.com/"
        model_name = "openai/GPT-3.5"
    elif "gpt-4" in model_name:
        link = "https://openai.com/"
        model_name = "openai/GPT-4"
    elif "super-torin" in model_name:
        link = "https://ai-novel.com/index.php"
        model_name = "ainovelist/supertrin"
    elif "rwkv" in model_name:
        link = "https://huggingface.co/BlinkDL/rwkv-4-world"
        model_name = "blinkdl/rwkv-4-world-jp55"

    return model_hyperlink(link, model_name)


def log_prior(x):
    """
    Prior: What we knew before running the experiment
    #We choose flat priors that don't constrain the parameters much
    """
    if -1 < x[0] < 1 and all([-4 < beta < 4 for beta in x[1:]]):
        return 0.0
    return -np.inf


def log_likelihood(alpha, betas, Y_m, i_m, j_m):
    """log likelihood of the data given the parameters"""
    alpha_plus_betas_diff = alpha + betas[i_m] - betas[j_m]
    return np.sum(Y_m * alpha_plus_betas_diff - np.log1p(np.exp(alpha_plus_betas_diff)))


def log_probability(x, Y_m, i_m, j_m):
    """Log Probability = Log_Prior + Log Likelihood"""
    x[1] = -np.sum(x[2:])  # enforces sum of scores is 0
    lp = log_prior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(x[0], x[1:], Y_m, i_m, j_m)


def compute_elo_mle(
    df: DataFrame,
    SCALE: int = 1,
    BASE=np.e,
    INIT_RATING: int = 0,
    fit_home_advantage: bool = True,
):
    """Compute the maximum likelihood estimate of the elo scores for each model"""
    models = pd.concat([df["model1_id"], df["model2_id"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # Score vector
    Y_m = df["score"].values
    # 1 if i_m beat j_m in the mth game, .5 for a draw, 0 otherwise

    # match vectors: who played in the mth match
    i_m = df.apply(lambda row: models[row["model1_id"]], axis=1).values
    j_m = df.apply(lambda row: models[row["model2_id"]], axis=1).values

    # Initial guess for the parameters
    x0 = np.zeros(len(models) + 1)

    # Define the system of equations
    def F(x):
        if not fit_home_advantage:
            x[0] = 0
        x[1] = -np.sum(x[2:])  # enforces sum of scores is ~0
        return -log_likelihood(x[0], x[1:], Y_m, i_m, j_m)

    # Minimize the negative likelihood
    max_likelihood = minimize(F, x0, method="L-BFGS-B")

    # Impose the sum of scores is ~0
    max_likelihood.x[1] = -np.sum(max_likelihood.x[2:])

    # Scale the scores
    scaled_MLE_scores = (
        SCALE
        / math.log(BASE)
        * pd.Series(max_likelihood.x[1:], index=models.index).sort_values(
            ascending=False
        )
        + INIT_RATING
    )
    scaled_MLE_advantage = SCALE / math.log(BASE) * max_likelihood.x[0]

    if not fit_home_advantage:
        return scaled_MLE_scores
    else:
        return (
            scaled_MLE_scores,
            scaled_MLE_advantage,
        )


# TODO: change this to dynamic loading
def bradley_terry(tournament_file: str, generate_charts: bool = False):
    """Calculate Bradley Terry rankings for models"""
    charts_prefix = "./charts/" + tournament_file.split("/")[-1].split(".")[0]

    with open(tournament_file, "r", encoding="utf-8") as f:
        tournament = json.load(f)

    df = pd.DataFrame(tournament["matches"])
    df = df.sample(frac=1)
    df = df[["model1_id", "model2_id", "score"]]

    models = pd.concat([df["model1_id"], df["model2_id"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    matches_num, models_num = len(df), len(models)

    # For display purposes it's helpful to have a short version of the model names
    short_names = {}
    for name in models.index:
        if "gpt-3.5" in name:
            short_name = "gpt-3.5"
        elif "gpt-4" in name:
            short_name = "gpt-4"
        elif "super-torin" in name:
            short_name = "supertrin"
        elif "open-calm-7b" in name:
            short_name = "opencalm-7b"
        elif "stormy" in name:
            short_name = "opencalm-7b (stormy)"
        elif "StableBeluga2" in name:
            short_name = "llama2-70b (StableBeluga2)"
        elif "japanese-stablelm" in name:
            short_name = "ja-stablelm-7b (instruct-alpha)"
        elif "rwkv-world-jpn-55" in name:
            short_name = "rwkv-world-7b (v0.55)"
        elif "rwkv-world-jp-v1" in name:
            short_name = "rwkv-world-7b (jp-v1)"
        elif "instruction-ppo" in name:
            short_name = "rinna-3.6b (PPO)"
        elif "instruction-sft" in name:
            short_name = "rinna-3.6b (SFT)"
        elif "neox-3.6b" in name:
            short_name = "rinna-3.6b"
        else:
            short_name = name
        short_names[name] = short_name

    # total number of times model i in first position beats model j in second position
    a_ij = df.groupby(["model1_id", "model2_id"])["score"].sum().unstack(fill_value=0)

    # total number of times model i in first position loses model j in second position
    df_loss = df.copy()
    df_loss["score"] = df_loss["score"].apply(
        lambda x: 1 if x == 0 else (0.5 if x == 0.5 else 0)
    )
    b_ij = (
        df_loss.groupby(["model1_id", "model2_id"])["score"].sum().unstack(fill_value=0)
    )

    # total number of times model i plays model j, regardless of order
    all_pairs = pd.concat(
        [
            df[["model1_id", "model2_id"]],
            df[["model2_id", "model1_id"]].rename(
                columns={"model2_id": "model1_id", "model1_id": "model2_id"}
            ),
        ]
    )
    n_ij = all_pairs.groupby(["model1_id", "model2_id"]).size().unstack(fill_value=0)

    # Check that the number of matches is correct
    try:
        assert n_ij.sum().sum() == 2 * matches_num
    except AssertionError:
        print("WARNING: number of matches is incorrect")

    # overall average win rate
    w_i = (a_ij + b_ij.T).sum(axis=1) / ((a_ij + b_ij.T) + (a_ij + b_ij.T).T).sum(
        axis=1
    )

    win_rates = w_i.sort_values(ascending=True)

    # Get x values and error values
    x_values = win_rates.values
    labels = [short_names[model] for model in win_rates.index]
    y_values = range(len(x_values))

    if generate_charts:
        try:  # Plot
            cb = CircusBoy()
            BGColor = "#FFFFFF"
            plt.rcParams["axes.facecolor"] = BGColor
            plt.rcParams["figure.facecolor"] = BGColor
            plt.rcParams["savefig.facecolor"] = BGColor

            fig, ax = cb.handlers()

            ax.scatter(x_values * 100, y_values)
            ax.set_xlabel(r"Overall win rate")
            cb.set_byline(ax, "Sam Passaglia / YuzuAI", pad=4)

            cb.set_title(
                ax,
                title="Win rates among Japanese AI Assistants",
                subtitle="As measured against each other on the Rakuda benchmark",
            )
            ax.set_xlim([0, 100])

            ax.xaxis.set_major_formatter(mtick.PercentFormatter())

            ax.set_yticks(y_values, [])

            for i, txt in enumerate(labels):
                ax.annotate(
                    txt,
                    xy=(x_values[i] * 100, y_values[i]),
                    xytext=(-5, 7),
                    ha="right",
                    va="center",
                    textcoords="offset points",
                    color=cb.grey,
                )

            fig.savefig(charts_prefix + "winrate.png")
            fig.savefig(charts_prefix + "winrate.pdf")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams["text.usetex"] = True
        except ImportError:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values)
            plt.xlabel("Overall win rate")
            plt.yticks(y_values, labels)
            plt.savefig(charts_prefix + "winrate.png")

    scaled_MLE_scores, scaled_MLE_advantage = compute_elo_mle(
        df, SCALE=GLOBAL_SCALE, BASE=GLOBAL_BASE, INIT_RATING=GLOBAL_INIT_RATING
    )

    # Check that the MLE has converged as a function of num_matches
    nstep = 10
    steps = np.arange(200, len(df) + nstep, nstep)

    for step in steps:
        current_scores, advantage = compute_elo_mle(
            df.iloc[:step],
            SCALE=GLOBAL_SCALE,
            BASE=GLOBAL_BASE,
            INIT_RATING=GLOBAL_INIT_RATING,
        )
        current_scores = current_scores.rename(step)
        if step == steps[0]:
            score_history = current_scores
        else:
            score_history = pd.concat([score_history, current_scores], axis=1)

    if generate_charts:
        # Plotting the elo_history
        fig, ax = plt.subplots()

        for model in score_history.index:
            ax.plot(steps, score_history.loc[model], label=short_names[model])

        ax.set_xlabel("Total matches")
        ax.set_ylabel("Strength")
        ax.set_title("Maximum Likelihood Strength as a function of total matches")
        ax.legend(loc="upper left")

        plt.savefig(charts_prefix + "evolution.png")

    # # Bayesian confidence regions using MCMC

    # See the [emcee tutorial](https://emcee.readthedocs.io/en/stable/tutorials/line/) for a crash-course of Bayesian fitting, or [this nice paper](https://arxiv.org/pdf/1008.4686.pdf) for a more complete but still pedagogical look.

    # Parameter space dimension
    ndim = models_num + 1

    # This will launch an MCMC that takes about ~ 4 minute to run on a M1 mac cpu
    # If too slow just reduce nsamples or nwalkers, default numbers are overkill for computing confidence intervals

    # Maximum number of samples per chain
    nsamples = 20000

    # Number of independent chains
    nwalkers = 40

    # Initialize the walkers randomly around the MLE point
    MLE_scores, MLE_advantage = compute_elo_mle(df)
    p0 = np.random.normal(
        np.concatenate([[MLE_advantage], MLE_scores[models.index].values]),
        0.01,
        (nwalkers, ndim),
    )

    # Set up the location to save the mcmc
    # Don't forget to clear it in case the file already exists
    filename = "mcmc.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsamples)

    # This will be useful to testing convergence
    old_tau = np.inf

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, backend=backend, pool=pool
        )
        # We sample for up to nsamples steps
        for sample in sampler.sample(p0, iterations=nsamples, progress=True):
            # Only check convergence every 1000 steps
            if sampler.iteration % 1000:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # We'll consider things converged if the chain is longer than 50 times the autocorrelation time
            # and this estimate has changed by less than 5 percent since the last check.
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                break
            old_tau = tau

    # Should be 0.3~0.5
    print(
        "Mean acceptance fraction: {0:.2f}".format(np.mean(sampler.acceptance_fraction))
    )

    # Useful for chain burning and thinning
    print(
        "Mean autocorrelation time: {0:.0f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    sampler.get_autocorr_time()

    # Burn the few * autocorrelation time steps per chain to erase initial conditions
    flat_samples = sampler.get_chain(discard=1000, flat=True)
    flat_samples[:, 1] = -np.sum(flat_samples[:, 2:], axis=1)
    # impose the sum of strengths = 0 constraint that was imposed in the likelihood
    print(flat_samples.shape)
    # Convert chain to desired scaling

    scaled_samples = np.copy(flat_samples)

    scaled_samples[:, 0] = scaled_samples[:, 0] * GLOBAL_SCALE / math.log(GLOBAL_BASE)
    scaled_samples[:, 1:] = (
        scaled_samples[:, 1:] * GLOBAL_SCALE / math.log(GLOBAL_BASE)
        + GLOBAL_INIT_RATING
    )

    # Compute the confidence interval for all the parameters
    error_interval = [16, 50, 84]

    mcmc = np.percentile(scaled_samples[:, 0], error_interval)
    q = np.diff(mcmc)
    advantage_quantiles = {
        "median": mcmc[1],
        "one_sigma_up": q[1],
        "one_sigma_down": q[0],
    }
    print("advantage parameter", advantage_quantiles)

    strengths = pd.DataFrame(
        columns=["model_id", "median", "one_sigma_up", "one_sigma_down"]
    )

    for i in range(ndim - 1):
        mcmc = np.percentile(scaled_samples[:, i + 1], error_interval)
        q = np.diff(mcmc)

        strengths = pd.concat(
            [
                strengths,
                pd.DataFrame(
                    [
                        {
                            "model_id": list(models.index)[i],
                            "median": mcmc[1],
                            "one_sigma_up": q[1],
                            "one_sigma_down": q[0],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    strengths = strengths.sort_values("median", ascending=False).reset_index(drop=True)

    # Plot all the strengths
    if generate_charts:
        x_values = strengths["median"]
        y_values = strengths["median"].argsort().values
        errors = [strengths["one_sigma_up"], strengths["one_sigma_down"]]
        labels = strengths["model_id"].apply(lambda x: short_names[x])

        try:
            cb = CircusBoy()
            BGColor = "#FFFFFF"
            plt.rcParams["axes.facecolor"] = BGColor
            plt.rcParams["figure.facecolor"] = BGColor
            plt.rcParams["savefig.facecolor"] = BGColor

            fig, ax = cb.handlers()

            ax.errorbar(x_values, y_values, xerr=errors, fmt="o")
            ax.set_xlabel(r"Model Strength")
            cb.set_byline(ax, "Sam Passaglia / YuzuAI", pad=7)

            cb.set_title(
                ax,
                title="Strengths of Japanese AI Assistants",
                subtitle="by relative performance on Rakuda benchmark",
            )

            ax.set_yticks(y_values, [])

            for i, txt in enumerate(labels):
                ax.annotate(
                    txt,
                    xy=(x_values[i], y_values[i]),
                    xytext=(-3, 7),
                    ha="right",
                    va="center",
                    textcoords="offset points",
                    color=cb.grey,
                )

            fig.savefig(charts_prefix + "ranking.png")
            fig.savefig(charts_prefix + "ranking.pdf")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams["text.usetex"] = True
        except ImportError:
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                x_values,
                y_values,
                xerr=errors,
                fmt="o",
            )
            plt.xlabel("Model Strength")
            plt.yticks(y_values, labels)
            plt.savefig(charts_prefix + "ranking.png")

        # Plot a parameter distribution
        # Note we don't expect maximum likelihood point to necessarily agree with maximum of the marginalized 1D posterior distribution
        # See page 16 of https://arxiv.org/pdf/1008.4686.pdf

        model = "rwkv-world-jp-v1"

        scaled_MLE_scores, scaled_MLE_advantage = compute_elo_mle(
            df, SCALE=GLOBAL_SCALE, BASE=GLOBAL_BASE, INIT_RATING=GLOBAL_INIT_RATING
        )
        plt.hist(
            scaled_samples[:, models[model] + 1],
            100,
            color="k",
            histtype="step",
            label="Posterior Samples",
        )
        plt.axvline(scaled_MLE_scores[model], color="k", label="MLE")

        plt.gca().set_yticks([])
        plt.legend()
        plt.savefig(charts_prefix + "parameter.png")

    # Compute relative strength probabilities
    modelA = "stabilityai/StableBeluga2"
    modelB = "rwkv-world-jp-v1"

    diffs = (
        scaled_samples[:, models[modelA] + 1] - scaled_samples[:, models[modelB] + 1]
    )

    A_stronger_rate = np.sum(
        scaled_samples[:, models[modelA] + 1] > scaled_samples[:, models[modelB] + 1]
    ) / len(scaled_samples)

    print(
        f"{short_names[modelA]} is stronger than {short_names[modelB]} with {A_stronger_rate:.2%} confidence"
    )

    if generate_charts:
        # Plot
        try:
            cb = CircusBoy()
            BGColor = "#FFFFFF"
            plt.rcParams["axes.facecolor"] = BGColor
            plt.rcParams["figure.facecolor"] = BGColor
            plt.rcParams["savefig.facecolor"] = BGColor

            fig, ax = cb.handlers()

            hist = ax.hist(diffs, 100, histtype="step")
            ax.set_xlabel(
                rf"{short_names[modelA]} $-$ {short_names[modelB]}$",
                size=16,
            )
            arrow_height = 1.0
            ax.annotate(
                "",
                xytext=(0, arrow_height),
                xy=(75, arrow_height),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(
                    facecolor=hist[2][0]._facecolor[:3],
                    edgecolor=hist[2][0]._facecolor[:3],
                    width=0.15,
                    headlength=5,
                    headwidth=5,
                    shrink=0.1,
                ),
            )

            ax.annotate(
                f"{A_stronger_rate:.1%} of samples have \n {short_names[modelA].split('(')[0].strip()} > {short_names[modelB].split('(')[0].strip()}",
                xy=(0.0, arrow_height),
                xycoords=("data", "axes fraction"),
                xytext=(10, 10),
                textcoords="offset points",
            )
            plt.gca().set_yticks([])
            cb.set_byline(ax, "Sam Passaglia / YuzuAI", pad=15)
            ax.axvline(0, color=cb.grey, lw=1)
            cb.set_title(
                ax,
                title=f"Relative strength of {short_names[modelA].split('(')[0].strip()} and {short_names[modelB].split('(')[0].strip()}",
                subtitle="Posterior distribution",
                titlesize=16,
                subtitlesize=14,
                pad=40,
            )

            fig.savefig(charts_prefix + "diff.png")
            fig.savefig(charts_prefix + "diff.pdf")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams["text.usetex"] = True
        except ImportError:
            plt.close("all")
            plt.figure(figsize=(10, 6))
            plt.hist(
                diffs,
                100,
                color="k",
                histtype="step",
            )
            plt.axvline(0, ls="--", color="k")
            plt.xlabel(f"{modelA} elo - {modelB} elo")

            plt.gca().set_yticks([])
            plt.savefig(charts_prefix + "diff.png")

    # Compute this for all models
    strengths["stronger_than_next_confidence"] = -1

    for i in range(len(strengths) - 1):
        modelA = strengths.iloc[i]["model_id"]
        modelB = strengths.iloc[i + 1]["model_id"]

        A_stronger_rate = np.sum(
            scaled_samples[:, models[modelA] + 1]
            > scaled_samples[:, models[modelB] + 1]
        ) / len(flat_samples)
        print(
            f"{short_names[modelA]} is stronger than {short_names[modelB]} with {A_stronger_rate:.2%} confidence"
        )
        strengths.loc[
            strengths["model_id"] == modelA, "stronger_than_next_confidence"
        ] = A_stronger_rate

        # Also add win rates, for reference
        strengths["win_rate"] = strengths.apply(
            lambda row: win_rates[row["model_id"]], axis=1
        )

        [short_names[model] for model in list(models.index)]

    if generate_charts:
        # Finally we can make a triangle plot just to show off
        samples = MCSamples(
            samples=scaled_samples[::200],
            names=["alpha"] + [short_names[model] for model in list(models.index)],
        )

        # Triangle plot
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], ["alpha", "gpt-4", "gpt-3.5"], filled=True)
        g.export(charts_prefix + "corner_getdist.png")

    # Output standings and make table

    strengths["short_name"] = strengths["model_id"].apply(lambda x: short_names[x])
    strengths = strengths.sort_values("median")

    output = {
        "date": datetime.now().isoformat(),
        "model_metadata": tournament["model_metadata"],
        "metadata": tournament["metadata"],
        "ranking": strengths.to_dict(orient="records"),
    }

    output_path = f"./rankings/{tournament_file.split('/')[-1].split('.')[0]}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    registry = StandingsRegistry("./registry/registry.jsonl")
    registry.register(output_path)

    rankings = sorted(output["ranking"], key=lambda x: x["median"], reverse=True)
    table = "| Rank | Model | Strength | Stronger than the next model at confidence level  | \n| :--- | :---: | :---: | :---: |\n"
    for i, rank in enumerate(rankings):
        # assert(round(rank['one_sigma_up'],2) == round(rank['one_sigma_down'],2))
        table += f"| {i+1} | {make_clickable_model(rank['model_id'])} | {rank['median']:.3f} ± {rank['one_sigma_up']:.2f}  | { str(round(rank['stronger_than_next_confidence']*100,1))+'%' if rank['stronger_than_next_confidence']!=0 else 'N/A'}\n"

    print(table)
    print("\nDone!")

    # store markdown table in a markdown file
    with open(charts_prefix + "table.md", "w", encoding="utf-8") as f:
        f.write(table)

    return table
