"""Bradley Terry model for ranking models on the Rakuda benchmark
Usage:

python make_ranking.py --bench-name rakuda_v2 --model-list [LIST-OF-MODEL-ID|all] --judge-model --judge-model [gpt-4|gpt-3.5-turbo|claude-2]  --mode [pairwise|single] --compute [mcmc|mle|winrate] --make-charts [True|False]

Winrate:
    python make_ranking.py --bench-name rakuda_v2 --model-list all --judge-model claude-2 --mode pairwise --compute winrate --make-charts

    python make_ranking.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --judge-model gpt-4 --mode pairwise --compute winrate --make-charts

MLE:
    python make_ranking.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin llm-jp-13b-instruct stablelm-alpha-7b-v2 --judge-model gpt-4 --mode pairwise --compute mle --make-charts  --bootstrap-n 500 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct  --advanced-charts

    python make_ranking.py --bench-name rakuda_v2 --judge-model claude-2 --mode pairwise --compute mle --make-charts --bootstrap-n 500 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct  --advanced-charts

MCMC:
    python make_ranking.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --judge-model gpt-4 --mode pairwise --compute mcmc --make-charts --nsamples 15000 --nwalkers 40 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct  --advanced-charts

    python make_ranking.py --bench-name rakuda_v2 --judge-model claude-2 --mode pairwise --compute mcmc --make-charts --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct  --advanced-charts --nsamples 15000 --nwalkers 40
"""

import json
import math
from datetime import datetime
import argparse
from tqdm import tqdm
from typing import Callable, Optional
import os
from functools import partial

import emcee
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from samplotlib.circusboy import CircusBoy

import numpy as np
import pandas as pd
from getdist import MCSamples, plots
from multiprocess import Pool
from pandas import DataFrame
#from registry import StandingsRegistry
from scipy.optimize import minimize


def display_name(name: str) -> str:
    name_map = {
        "gpt-3.5": "gpt-3.5",
        "gpt-4": "gpt-4",
        "open-calm-7b": "opencalm-7b",
        "stormy": "opencalm-7b (stormy)",
        "stablebeluga2": "llama2-70b (StableBeluga2)",
        "jslm7b-instruct-alpha": "ja-stablelm-7b",
        "rwkv-world-jp-v1": "rwkv-world-7b-jp-v1",
        "rinna-3.6b-ppo": "rinna-3.6b (PPO)",
        "rinna-3.6b-sft": "rinna-3.6b (SFT)",
        "neox-3.6b": "rinna-3.6b",
        "line-3.6b-sft": "line-3.6b",
        "weblab-10b-instruction-sft": "weblab-10b",
        "elyza-7b-fast-instruct": "elyza-7b-fast",
    }
    for key, value in name_map.items():
        if key in name:
            return value

    return name


def licensing(name: str) -> str:
    # Basic licensing information ('closed', 'non-commercial', 'open')
    licensing_map = {
        "gpt-4": "closed",
        "gpt-3.5": "closed",
        "claude-2": "closed",
        "stormy": "open",
        "line-3.6b-sft": "open",
        "super-trin": "closed",
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo": "open",
        "stablebeluga2": "non-commercial",
        "rwkv-world-jp-v1": "open",
        "chatntq-7b-jpntuned": "open",
        "rinna-3.6b-ppo": "open",
        "rinna-3.6b-sft": "open",
        "jslm": "non-commercial",
        "gpt-4:20230713": "closed",
        "weblab-10b": "non-commercial",
        "elyza": "open",
        "elyza": "open",
    }

    for key, value in licensing_map.items():
        if key in name:
            return value

    return "unknown"


def log_prior(x):
    """
    Prior: What we knew before running the experiment
    We choose flat priors that don't constrain the parameters much
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


def save_ranking(
    strengths: DataFrame,
    judge: str,
    output_path: str = None,
):
    """
    Output standings to file
    """

    strengths["display_name"] = strengths["model_id"].apply(lambda x: display_name(x))
    strengths = strengths.sort_values("median")

    output = {
        "date": datetime.now().isoformat(),
        "judge": judge,
        # "model_metadata": tournament["model_metadata"],
        # "metadata": tournament["metadata"],
        "ranking": strengths.to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    # registry = StandingsRegistry("./registry/registry.jsonl")
    # registry.register(output_path)


def load_ranking(
    path: str,
) -> DataFrame:
    """
    Load standings from file
    """
    with open(path, "r", encoding="utf-8") as f:
        standings = json.load(f)

    return pd.DataFrame(standings["ranking"])


def compute_bt_mle(
    matches_df: DataFrame,
    SCALE: int = 1,
    BASE: float = np.e,
    INIT_RATING: int = 0,
    fit_home_advantage: bool = True,
) -> DataFrame:
    """Compute the maximum likelihood estimate of the Bradley-Terry scores for each model"""
    models = pd.concat([matches_df["model1_id"], matches_df["model2_id"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # Score vector
    Y_m = matches_df["score"].values
    # import collections
    # print(collections.Counter(Y_m))
    # 1 if i_m beat j_m in the mth game, .5 for a draw, 0 otherwise

    # match vectors: who played in the mth match
    i_m = matches_df.apply(lambda row: models[row["model1_id"]], axis=1).values
    j_m = matches_df.apply(lambda row: models[row["model2_id"]], axis=1).values

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


def mle_convergence(
    matches_df: DataFrame,
    n_steps: int = 100,
    chart_dir: Optional[str] = None,
    SCALE: int = 1,
    BASE: float = np.e,
    INIT_RATING: int = 0,
    fit_home_advantage: bool = True,
):
    matches_df = matches_df.sample(frac=1)
    # Check that the MLE has converged as a function of num_matches
    step_size = int(len(matches_df) / n_steps)
    steps = np.arange(200, len(matches_df) + step_size, step_size)

    for step in steps:
        current_scores, advantage = compute_bt_mle(
            matches_df.iloc[:step],
            SCALE=SCALE,
            BASE=BASE,
            INIT_RATING=INIT_RATING,
            fit_home_advantage=fit_home_advantage,
        )
        current_scores = current_scores.rename(step)
        if step == steps[0]:
            score_history = current_scores
        else:
            score_history = pd.concat([score_history, current_scores], axis=1)

    if chart_dir:
        # Plotting the bt_history
        fig, ax = plt.subplots()

        for model in score_history.index:
            ax.plot(steps, score_history.loc[model], label=display_name(model))

        ax.set_xlabel("Total matches")
        ax.set_ylabel("Strength")
        ax.set_title("Maximum Likelihood Strength as a function of total matches")
        ax.legend(loc="upper left")

        fig.savefig(chart_dir + "mle_evolution.png")


def get_bootstrap_result(
    matches_df: DataFrame,
    func_compute_bt: Callable[[DataFrame], DataFrame],
    num_round: int,
):
    rows = []
    for _ in tqdm(range(num_round), desc="bootstrap"):
        bs = func_compute_bt(matches_df.sample(frac=1.0, replace=True))
        if isinstance(bs, tuple):
            bs = bs[0]
        rows.append(bs)
    df = pd.DataFrame(rows)
    df = df[df.median().sort_values(ascending=False).index]

    bars = (
        pd.DataFrame(
            dict(
                # lower=df.quantile(0.025),
                # rating=df.quantile(0.5),
                # upper=df.quantile(0.975),
                lower=df.quantile(0.16),
                median=df.quantile(0.5),
                upper=df.quantile(0.84),
            )
        )
        .reset_index(names="model_id")
        .sort_values("median", ascending=False)
    )
    bars["one_sigma_plus"] = bars["upper"] - bars["median"]
    bars["one_sigma_minus"] = bars["median"] - bars["lower"]
    bars["median_rounded"] = np.round(bars["median"], 0)

    return bars


def plot_strengths(
    strengths: DataFrame,
    chart_dir: Optional[str] = None,
    advanced_charts: bool = False,
    filename: str = "ranking",
    color: Optional[str] = None,
    label: Optional[str] = None,
    order: pd.Series = None,
    figax=None,
    legend_title=None,
    show_licensing=True,
    title="Strengths of Japanese AI Assistants",
    subtitle="by relative performance on Rakuda benchmark"
):
    if order is None:
        order = strengths.sort_values(by="median")["model_id"]

    strengths = strengths.set_index("model_id").reindex(order).reset_index()
    x_values = strengths["median"]
    y_values = strengths["median"].index
    errors = [strengths["one_sigma_plus"], strengths["one_sigma_minus"]]
    labels = strengths["model_id"].apply(lambda x: display_name(x))

    license_colormap = {
        "closed": np.array([145, 49, 41]) / 255,
        "non-commercial": np.array([255, 183, 50]) / 255,
        "open": np.array([51, 153, 255]) / 255,
        "unknown": "black",
    }
    license_textmap = {
        "closed": "Closed source",
        "non-commercial": "No commercial use",
        "open": "Commercial OK",
        "unknown": "Unknown",
    }
    licenses = strengths["model_id"].apply(lambda x: licensing(x))
    license_colors = strengths["model_id"].apply(
        lambda x: license_colormap[licensing(x)]
    )

    if advanced_charts:
        if not figax:
            cb = CircusBoy()
            BGColor = "#FFFFFF"
            plt.rcParams["axes.facecolor"] = BGColor
            plt.rcParams["figure.facecolor"] = BGColor
            plt.rcParams["savefig.facecolor"] = BGColor

            fig, ax = cb.handlers()
            ax.set_xlabel(r"Model Strength")
            cb.set_byline(ax, "Sam Passaglia / YuzuAI", pad=7)
            cb.set_title(
                ax,
                title=title,
                subtitle=subtitle,
            )
            ax.set_yticks(y_values, [])
            for i, txt in enumerate(labels):
                ax.annotate(
                    txt,
                    xy=(x_values[i], y_values[i]),
                    xytext=(-3, 6),
                    ha="right",
                    va="center",
                    textcoords="offset points",
                    color=cb.grey,
                )
            used_licenses = np.unique(licenses.values)
            markers = [
                plt.Line2D(
                    [0, 0],
                    [0, 0],
                    color=license_colormap[license],
                    marker="o",
                    linestyle="",
                )
                for license in used_licenses
            ]
            if show_licensing:
                legend = ax.legend(
                    markers,
                    [license_textmap[k] for k in used_licenses],
                    prop=dict(weight="bold"),
                    labelcolor="linecolor",
                    frameon=True,
                    framealpha=1,
                    facecolor="white",
                    edgecolor="white",
                    numpoints=1,
                    loc="lower right",
                )

        else:
            fig, ax = figax

            for child in ax.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    if child.xy[0] > 1:
                        # print(child._text)
                        # print(child.xy)
                        if child.xy[0] > x_values[child.xy[1]]:
                            # child._text = 'changed'
                            child.xy = (x_values[child.xy[1]] - 3, child.xy[1])

        if show_licensing:
            colors = license_colors
        else:
            if color:
                colors = [color] * len(license_colors)
            else:
                colors = ["blue"] * len(license_colors)

        ax.errorbar(x_values, y_values, xerr=errors, fmt="None", ecolor=colors)
        for i in range(len(x_values)):
            ax.scatter(x_values[i], y_values[i], color=colors[i], label=label if i==0 else None)

        if chart_dir:
            if label:
                legend = ax.legend(
                    #markers,
                    #[license_textmap[k] for k in used_licenses],
                    prop=dict(weight="bold"),
                    title=legend_title,
                    title_fontproperties=dict(weight="bold"),
                    #labelcolor="linecolor",
                    frameon=True,
                    framealpha=1,
                    facecolor="white",
                    edgecolor="white",
                    numpoints=1,
                    loc="lower right",
                )
            fig.savefig(chart_dir + filename + ".png")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams["text.usetex"] = True
        else:
            return fig, ax
    else:
        if not figax:
            fig = plt.figure(figsize=(10, 6))
            ax = plt.gca()
            ax.set_xlabel("Model Strength")
            ax.set_yticks(y_values, labels)
        else:
            fig, ax = figax

        ax.errorbar(
            x_values,
            y_values,
            xerr=errors,
            fmt="o",
            color=color,
            label=label,
        )

        if chart_dir:
            plt.savefig(chart_dir + filename + ".png")
        else:
            return fig, ax


def compute_winrates(
    matches_df: DataFrame,
    chart_dir: Optional[str] = None,
    advanced_charts: bool = False,
):
    # total number of times model i in first position beats model j in second position
    a_ij = (
        matches_df.groupby(["model1_id", "model2_id"])["score"]
        .sum()
        .unstack(fill_value=0)
    )

    # total number of times model i in first position loses model j in second position
    df_loss = matches_df.copy()
    df_loss["score"] = df_loss["score"].apply(
        lambda x: 1 if x == 0 else (0.5 if x == 0.5 else 0)
    )
    b_ij = (
        df_loss.groupby(["model1_id", "model2_id"])["score"].sum().unstack(fill_value=0)
    )

    # total number of times model i plays model j, regardless of order
    all_pairs = pd.concat(
        [
            matches_df[["model1_id", "model2_id"]],
            matches_df[["model2_id", "model1_id"]].rename(
                columns={"model2_id": "model1_id", "model1_id": "model2_id"}
            ),
        ]
    )
    n_ij = all_pairs.groupby(["model1_id", "model2_id"]).size().unstack(fill_value=0)

    # Check that the number of matches is correct
    try:
        assert n_ij.sum().sum() == 2 * len(matches_df)
    except AssertionError:
        print("WARNING: number of matches is incorrect")

    # overall average win rate
    w_i = (a_ij + b_ij.T).sum(axis=1) / ((a_ij + b_ij.T) + (a_ij + b_ij.T).T).sum(
        axis=1
    )

    win_rates = w_i.sort_values(ascending=True)

    # Get x values and error values
    x_values = win_rates.values
    labels = [display_name(model) for model in win_rates.index]
    y_values = range(len(x_values))

    if chart_dir:
        if advanced_charts:
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

            fig.savefig(chart_dir + "winrate.png")
            fig.savefig(chart_dir + "winrate.pdf")
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams["text.usetex"] = True
        else:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values)
            plt.xlabel("Overall win rate")
            plt.yticks(y_values, labels)
            plt.savefig(chart_dir + "winrate.png")

    return win_rates


def compute_bt_mcmc(
    matches_df: DataFrame,
    chain_path: str,
    SCALE: int = 1,
    BASE: float = np.e,
    INIT_RATING: int = 0,
    fit_home_advantage: bool = True,
    chart_dir: Optional[str] = None,
    advanced_charts: bool = False,
    nsamples: int = 20000,
    nwalkers: int = 40,
) -> DataFrame:
    """
    Calculate the posterior probability distribution of Bradley Terry scores using MCMC.

    Bayesian confidence regions are computed using MCMC.
    Refer to the emcee tutorial for a crash corse: https://emcee.readthedocs.io/en/stable/tutorials/line/
    For a more comprehensive view: https://arxiv.org/abs/1008.4686

    Note:
        This MCMC process takes about ~4 minutes to run on an M1 mac CPU. If it's too slow,
        consider reducing the nsamples or nwalkers. The default numbers are overkill for computing confidence intervals.
    """

    models = pd.concat([matches_df["model1_id"], matches_df["model2_id"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # Score vector
    Y_m = matches_df["score"].values
    # 1 if i_m beat j_m in the mth game, .5 for a draw, 0 otherwise

    # match vectors: who played in the mth match
    i_m = matches_df.apply(lambda row: models[row["model1_id"]], axis=1).values
    j_m = matches_df.apply(lambda row: models[row["model2_id"]], axis=1).values

    # Compute the MLE which is used as the starting point for the MCMC
    scaled_MLE_scores, scaled_MLE_advantage = compute_bt_mle(
        matches_df,
        SCALE=SCALE,
        BASE=BASE,
        INIT_RATING=INIT_RATING,
        fit_home_advantage=fit_home_advantage,
    )

    # Parameter space dimension
    ndim = len(models) + 1

    # Set up the location to save the mcmc
    # Don't forget to clear it in case the file already exists
    if os.path.exists(chain_path):
        backend = emcee.backends.HDFBackend(chain_path)
        flat_samples = backend.get_chain(discard=1000, flat=True)
    else:
        backend = emcee.backends.HDFBackend(chain_path)
        backend.reset(nwalkers, ndim)

        # Initialize the walkers randomly around the MLE point
        MLE_scores, MLE_advantage = compute_bt_mle(matches_df)
        p0 = np.random.normal(
            np.concatenate([[MLE_advantage], MLE_scores[models.index].values]),
            0.01,
            (nwalkers, ndim),
        )

        # Track how the autocorrelation time estimate changes to test convergence
        index = 0
        autocorr = np.empty(nsamples)
        old_tau = np.inf

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                partial(log_probability, Y_m=Y_m, i_m=i_m, j_m=j_m),
                backend=backend,
                pool=pool,
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
            "Mean acceptance fraction: {0:.2f}".format(
                np.mean(sampler.acceptance_fraction)
            )
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

    scaled_samples[:, 0] = scaled_samples[:, 0] * SCALE / math.log(BASE)
    scaled_samples[:, 1:] = scaled_samples[:, 1:] * SCALE / math.log(BASE) + INIT_RATING

    # Compute the confidence interval for all the parameters
    error_interval = [16, 50, 84]

    mcmc = np.percentile(scaled_samples[:, 0], error_interval)
    q = np.diff(mcmc)
    advantage_quantiles = {
        "median": mcmc[1],
        "one_sigma_plus": q[1],
        "one_sigma_minus": q[0],
    }
    print("advantage parameter", advantage_quantiles)

    strengths = pd.DataFrame(
        columns=["model_id", "median", "one_sigma_plus", "one_sigma_minus"]
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
                            "one_sigma_plus": q[1],
                            "one_sigma_minus": q[0],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    strengths = strengths.sort_values("median", ascending=False).reset_index(drop=True)

    # Compute ranking confidence for all models
    strengths["stronger_than_next_confidence"] = -1

    for i in range(len(strengths) - 1):
        modelA = strengths.iloc[i]["model_id"]
        modelB = strengths.iloc[i + 1]["model_id"]

        A_stronger_rate = np.sum(
            scaled_samples[:, models[modelA] + 1]
            > scaled_samples[:, models[modelB] + 1]
        ) / len(flat_samples)
        print(
            f"{display_name(modelA)} is stronger than {display_name(modelB)} with {A_stronger_rate:.2%} confidence"
        )
        strengths.loc[
            strengths["model_id"] == modelA, "stronger_than_next_confidence"
        ] = A_stronger_rate

    return strengths

    #     # Plot a parameter distribution
    #     # Note we don't expect maximum likelihood point to necessarily agree with maximum of the marginalized 1D posterior distribution
    #     # See page 16 of https://arxiv.org/pdf/1008.4686.pdf

    #     model = "rwkv-world-jp-v1"

    #     scaled_MLE_scores, scaled_MLE_advantage = compute_bt_mle(
    #         df, SCALE=GLOBAL_SCALE, BASE=GLOBAL_BASE, INIT_RATING=GLOBAL_INIT_RATING
    #     )
    #     plt.hist(
    #         scaled_samples[:, models[model] + 1],
    #         100,
    #         color="k",
    #         histtype="step",
    #         label="Posterior Samples",
    #     )
    #     plt.axvline(scaled_MLE_scores[model], color="k", label="MLE")

    #     plt.gca().set_yticks([])
    #     plt.legend()
    #     plt.savefig(charts_prefix + "parameter.png")

    # # Compute relative strength probabilities
    # modelA = "stabilityai/StableBeluga2"
    # modelB = "rwkv-world-jp-v1"

    # diffs = (
    #     scaled_samples[:, models[modelA] + 1] - scaled_samples[:, models[modelB] + 1]
    # )

    # A_stronger_rate = np.sum(
    #     scaled_samples[:, models[modelA] + 1] > scaled_samples[:, models[modelB] + 1]
    # ) / len(scaled_samples)

    # print(
    #     f"{display_names[modelA]} is stronger than {display_names[modelB]} with {A_stronger_rate:.2%} confidence"
    # )

    # if generate_charts:
    #     # Plot
    #     try:
    #         cb = CircusBoy()
    #         BGColor = "#FFFFFF"
    #         plt.rcParams["axes.facecolor"] = BGColor
    #         plt.rcParams["figure.facecolor"] = BGColor
    #         plt.rcParams["savefig.facecolor"] = BGColor

    #         fig, ax = cb.handlers()

    #         hist = ax.hist(diffs, 100, histtype="step")
    #         ax.set_xlabel(
    #             rf"{display_names[modelA]} $-$ {display_names[modelB]}$",
    #             size=16,
    #         )
    #         arrow_height = 1.0
    #         ax.annotate(
    #             "",
    #             xytext=(0, arrow_height),
    #             xy=(75, arrow_height),
    #             xycoords=("data", "axes fraction"),
    #             arrowprops=dict(
    #                 facecolor=hist[2][0]._facecolor[:3],
    #                 edgecolor=hist[2][0]._facecolor[:3],
    #                 width=0.15,
    #                 headlength=5,
    #                 headwidth=5,
    #                 shrink=0.1,
    #             ),
    #         )

    #         ax.annotate(
    #             f"{A_stronger_rate:.1%} of samples have \n {display_names[modelA].split('(')[0].strip()} > {display_names[modelB].split('(')[0].strip()}",
    #             xy=(0.0, arrow_height),
    #             xycoords=("data", "axes fraction"),
    #             xytext=(10, 10),
    #             textcoords="offset points",
    #         )
    #         plt.gca().set_yticks([])
    #         cb.set_byline(ax, "Sam Passaglia / YuzuAI", pad=15)
    #         ax.axvline(0, color=cb.grey, lw=1)
    #         cb.set_title(
    #             ax,
    #             title=f"Relative strength of {display_names[modelA].split('(')[0].strip()} and {display_names[modelB].split('(')[0].strip()}",
    #             subtitle="Posterior distribution",
    #             titlesize=16,
    #             subtitlesize=14,
    #             pad=40,
    #         )

    #         fig.savefig(charts_prefix + "diff.png")
    #         fig.savefig(charts_prefix + "diff.pdf")
    #         matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #         plt.rcParams["text.usetex"] = True
    #     except ImportError:
    #         plt.close("all")
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(
    #             diffs,
    #             100,
    #             color="k",
    #             histtype="step",
    #         )
    #         plt.axvline(0, ls="--", color="k")
    #         plt.xlabel(f"{modelA} bt - {modelB} bt")

    #         plt.gca().set_yticks([])
    #         plt.savefig(charts_prefix + "diff.png")

    # if generate_charts:
    #     # Finally we can make a triangle plot just to show off
    #     samples = MCSamples(
    #         samples=scaled_samples[::200],
    #         names=["alpha"] + [display_names[model] for model in list(models.index)],
    #     )

    #     # Triangle plot
    #     g = plots.get_subplot_plotter()
    #     g.triangle_plot([samples], ["alpha", "gpt-4", "gpt-3.5"], filled=True)
    #     g.export(charts_prefix + "corner_getdist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="rakuda_v2",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument(
        "--compute",
        type=str,
        default="mle",
        choices=["mcmc", "mle", "winrate"],
        help="Whether the script should compute the ranking with MLE or MCMC ranking.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise` runs pairwise comparision between n pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=["all"],
        help="A list of models to be evaluated. Defaults to all models for which reviews exist",
    )
    parser.add_argument(
        "--make-charts",
        action="store_true",
        help="Whether to output charts",
    )
    parser.add_argument(
        "--advanced-charts",
        action="store_true",
        help="Whether to output charts",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=400,
        help="Bradley-Terry scale value",
    )
    parser.add_argument(
        "--base",
        type=float,
        default=10,
        help="Bradley-Terry base value",
    )
    parser.add_argument(
        "--init-rating",
        type=int,
        default=1000,
        help="Bradley-Terry initial rating value",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=500,
        help="Number of iterations for bootstrap estimator",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=20000,
        help="Number of iterations for MCMC",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=40,
        help="Number of walkers for MCMC",
    )
    parser.add_argument(
        "--chain-path",
        type=str,
        default=None,
        help="Path to MCMC chain",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the ranking",
    )
    parser.add_argument(
        "--plot-skip-list",
        type=str,
        nargs="+",
        default=[],
        help="A list of models to not show in plots",
    )
    args = parser.parse_args()

    # Paths
    reviews_file = (
        f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
    )
    chart_dir = (
        f"data/{args.bench_name}/charts/{args.judge_model}/"
        if args.make_charts
        else None
    )
    output_path = (
        args.output_path
        or f"./data/{args.bench_name}/rankings/{args.judge_model}_{args.compute}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if chart_dir:
        os.makedirs(chart_dir, exist_ok=True)

    # Load reviews
    with open(reviews_file, "r", encoding="utf-8") as f:
        matches = []
        for line in f:
            # print(line)
            data = json.loads(line)
            matches.append(data)

    # Remove all reviews that have model1_id or model2_id not in the model_list
    if args.model_list[0] != "all":
        matches = [
            match
            for match in matches
            if match["model1_id"] in args.model_list
            and match["model2_id"] in args.model_list
        ]

    matches_df = pd.DataFrame(matches)
    matches_df["score"] = matches_df["winner"].map({1: 1, 2: 0, 3: 0.5})
    matches_df = matches_df[["model1_id", "model2_id", "score"]]

    if args.compute == "mle":
        print("Computing maximum likelihood estimate of model strengths")
        mle_df, hfa_alpha = compute_bt_mle(
            matches_df,
            SCALE=args.scale,
            BASE=args.base,
            INIT_RATING=args.init_rating,
            fit_home_advantage=True,
        )
        print("MLE:")
        print(mle_df)
        print("Home field advantage:")
        print(hfa_alpha)

        if chart_dir:
            print("Generating convergence chart")
            mle_convergence(matches_df, chart_dir=chart_dir, n_steps=20)

        strengths_df = get_bootstrap_result(
            matches_df,
            lambda df: compute_bt_mle(
                df,
                SCALE=args.scale,
                BASE=args.base,
                INIT_RATING=args.init_rating,
                fit_home_advantage=True,
            ),
            args.bootstrap_n,
        )
        print(strengths_df)
        if chart_dir:
            print("Generating strength chart")
            plottable_strengths = strengths_df[
                strengths_df["model_id"].map(lambda x: x not in args.plot_skip_list)
            ].reset_index()
            plot_strengths(
                plottable_strengths,
                chart_dir=chart_dir,
                filename="mle_ranking",
                advanced_charts=args.advanced_charts,
            )

        save_ranking(strengths_df, judge=args.judge_model, output_path=output_path)

    elif args.compute == "mcmc":
        chain_path = (
            args.chain_path or f"./data/{args.bench_name}/chains/{args.judge_model}.h5"
        )
        os.makedirs(os.path.dirname(chain_path), exist_ok=True)

        strengths_df = compute_bt_mcmc(
            matches_df,
            chain_path=chain_path,
            SCALE=args.scale,
            BASE=args.base,
            INIT_RATING=args.init_rating,
            fit_home_advantage=True,
            chart_dir=chart_dir,
            advanced_charts=args.advanced_charts,
            nsamples=args.nsamples,
            nwalkers=args.nwalkers,
        )
        print(strengths_df)
        if chart_dir:
            print("Generating strength chart")
            plottable_strengths = strengths_df[
                strengths_df["model_id"].map(lambda x: x not in args.plot_skip_list)
            ].reset_index()
            plot_strengths(
                plottable_strengths,
                chart_dir=chart_dir,
                advanced_charts=args.advanced_charts,
                filename="mcmc_ranking",
            )

        save_ranking(strengths_df, judge=args.judge_model, output_path=output_path)

    elif args.compute == "winrate":
        winrates_df = compute_winrates(
            matches_df, chart_dir=chart_dir, advanced_charts=args.advanced_charts
        )
        print(winrates_df)
