from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()


def plot_statistical_test(
    df_variant: pd.DataFrame,
    df_control: pd.DataFrame,
    metric: str,
    margin: float,
    direction: str,
    diffs: np.ndarray,
    result: dict[str, tuple],
    normal: tuple,
    base_dir: Path,
    name: str = "stat",
):
    base_dir.mkdir(exist_ok=True, parents=True)

    df1 = df_control.assign(Group="Control")
    df2 = df_variant.assign(Group="Variant")
    df3 = df_variant.assign(Group="Differences")
    df3[metric] = diffs
    cdf = pd.concat([df1, df2, df3])

    _, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(cdf, x="Group", y=metric, ax=ax, color=".8")
    if direction == "greater":
        ax.hlines(
            -margin, 2 - 0.5, 2 + 0.5, color="k", ls="--", label=f"NI Margin: {-margin}"
        )
    else:
        ax.hlines(
            margin, 2 - 0.5, 2 + 0.5, color="k", ls="--", label=f"NI Margin: {margin}"
        )
    ax.set_title(
        "Not-Normal: %s with p=%.3f \nNon-inferior: %s with p=%.3f \nSuperior: %s with p=%.3f"
        % (
            normal[0],
            normal[1],
            result["non_inferior"][0],
            result["non_inferior"][1],
            result["superior"][0],
            result["superior"][1],
        )
    )
    plt.legend()
    plt.xlabel("")
    plt.savefig(base_dir / f"{name}_{metric}_box.png")
    plt.close()

    _, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=df1[metric], y=df2[metric], ax=ax, c="k")
    ax.plot(ax.get_xlim(), ax.get_xlim(), "k--")
    ax.set_title(
        "Not-Normal: %s with p=%.3f \nNon-inferior: %s with p=%.3f \nSuperior: %s with p=%.3f"
        % (
            normal[0],
            normal[1],
            result["non_inferior"][0],
            result["non_inferior"][1],
            result["superior"][0],
            result["superior"][1],
        )
    )
    plt.xlabel("Control: " + metric)
    plt.ylabel("Variant: " + metric)
    plt.savefig(base_dir / f"{name}_{metric}_scatter.png")
    plt.close()


def plot_bland_altman(
    avg: pd.Series,
    diff: pd.Series,
    name: str,
    n_good: float,
    base_dir: Path = Path("results/regression/acceptance"),
):
    """
    Creates a Bland-Altman plot to analyze the agreement between predicted and ground truth SUVmean values.
    This plot illustrates the difference against the average of the two measurements and marks the mean difference
    and the limits of agreement (±1.96 standard deviations).

    Args:
        avg (Series): A pandas Series containing the averages of predicted and ground truth SUVmean values.
        diff (Series): A pandas Series containing the differences between predicted and ground truth SUVmean values.
        name (str): A descriptive name for the test series, used as part of the filename for saving the plot.
        n_good (float): The proportion of samples that fall within the acceptable error margin, used in the plot title.

    """
    xrange = [avg.min(), avg.max()]
    diff_mean = diff.mean()
    diff_std = diff.std()

    fig, ax = plt.subplots()
    ax.set_ylabel(f"Difference: SUVmean (pred, gt)")
    ax.set_xlabel(f"Average: SUVmean (pred, gt)")
    ax.plot(
        xrange,
        [diff_mean - 1.96 * diff_std, diff_mean - 1.96 * diff_std],
        "--",
        color="r",
        alpha=0.5,
    )
    ax.text(
        x=xrange[0],
        y=diff_mean - 1.96 * diff_std,
        s="-1.96 SD: {:.2f}".format(diff_mean - 1.96 * diff_std),
    )
    ax.plot(
        xrange,
        [diff_mean + 1.96 * diff_std, diff_mean + 1.96 * diff_std],
        "--",
        color="r",
        alpha=0.5,
    )
    ax.text(
        x=xrange[0],
        y=diff_mean + 1.96 * diff_std,
        s="+1.96 SD: {:.2f}".format(diff_mean + 1.96 * diff_std),
    )
    ax.plot(xrange, [diff_mean, diff_mean], "k--", alpha=0.5)
    ax.text(x=xrange[0], y=diff_mean, s="Mean: {:.2f}".format(diff_mean))

    sns.scatterplot(x=avg, y=diff, ax=ax, alpha=0.4)
    ax.set_title(
        "%.2f %% of the samples fall within a 10%% SUV mean error margin"
        % (n_good * 100)
    )
    plt.savefig(base_dir / f"{name}_SUVmean.png")
    plt.close()


def plot_cumulative_error(
    diff: pd.Series,
    gt: pd.Series,
    name: str,
    base_dir: Path = Path("results/regression/acceptance"),
):
    """
    Generates a cumulative distribution plot of the relative error of SUVmean values,
    expressed as a percentage of the ground truth (gt), and saves the plot as a PNG file.

    This function calculates the absolute relative error as a percentage, then uses seaborn's ecdfplot
    to visualize the cumulative distribution of these errors. The plot helps in assessing the error
    distribution across the entire range of data points.

    Args:
        diff (Series): A pandas Series containing the differences between predicted and ground truth SUVmean values.
        gt (Series): A pandas Series containing the ground truth SUVmean values against which the error is calculated.
        name (str): A descriptive name for the test series, used as part of the filename for saving the plot.
    """
    rel_error = 100 * diff.abs() / gt
    fig, ax = plt.subplots()
    sns.ecdfplot(x=rel_error)
    ax.set_xlabel("SUVmean rel. error [%]")
    plt.savefig(base_dir / f"{name}_SUVmean_cum.png")
    plt.close()