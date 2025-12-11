from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon, shapiro
from pandas import DataFrame

from visualization import (
    plot_statistical_test,
    plot_bland_altman,
    plot_cumulative_error,
)

# Define metrics and their corresponding superior hypothesis
metric_directions = {
    "NRMSE": "less",
    "PSNR": "greater",
    "SSIM": "greater",
    "SUVstd_pred": "less",
    "SUVmax_diff": "less",
}
metric_ni_margins = {
    "NRMSE": 0.05,
    "PSNR": 3.38,
    "SSIM": 0.05,
    "SUVstd_pred": 0.1,
    "SUVmax_diff": 0.25,
}


def statistical_test(
    df_variant: pd.DataFrame,
    df_control: pd.DataFrame,
    metric: str,
    margin: float,
    direction="greater",
    alpha=0.05,
    base_dir: Path = Path("results/"),
    name: str = "",
    plot: bool = True,
):
    """
    Performs statistical tests to determine non-inferiority and superiority
    between two datasets based on a specified metric and direction.

    Args:
        df_variant (pd.DataFrame): Dataframe containing the variant group data.
        df_control (pd.DataFrame): Dataframe containing the control group data.
        metric (str): The metric on which the test will be performed.
        margin (float): The margin of error to apply in the direction specified.
        direction (str, optional): The direction of the test ('greater', 'less').
            Defaults to 'greater'.
        alpha (float, optional): Significance level for the test. Defaults to 0.05.
        base_dir (Path, optional): Base directory where results are saved.
            Defaults to Path("results/").
        name (str, optional): Optional name for the plot file. Defaults to empty string.
        plot (bool, optional): Flag to determine if the results should be plotted.
            Defaults to True.

    Raises:
        ValueError: If 'direction' is not 'greater' or 'less', or if the indexes
            of `df_variant` and `df_control` do not match.

    Returns:
        dict: A dictionary containing the results of the non-inferiority and
        superiority tests, with keys 'non_inferior' and 'superior'.
    """

    if all(df_variant.index != df_control.index):
        raise ValueError("The test and base dataframes have different indexes.")

    diffs = df_variant[metric].values - df_control[metric].values
    _, p_sh = shapiro(diffs)
    normal = (p_sh < alpha, p_sh)

    if direction == "greater":
        adj_diffs = diffs + margin
    elif direction == "less":
        adj_diffs = diffs - margin
    else:
        raise ValueError(f"Invalid direction {direction}")

    # Wilcoxon signed-rank test on adjusted differences for non-inferiority
    _, p_ni = wilcoxon(x=adj_diffs, alternative=direction)

    non_inferior = (p_ni < alpha / 2, p_ni)

    # Wilcoxon signed-rank test on original differences for superiority
    _, p_sp = wilcoxon(x=diffs, alternative=direction)

    superior = (p_sp < alpha / 2, p_sp)
    result = {"non_inferior": non_inferior, "superior": superior}

    if plot:
        plot_statistical_test(
            df_variant,
            df_control,
            metric,
            margin,
            direction,
            diffs,
            result,
            normal,
            base_dir,
            name,
        )

    return result


def superiority_test(
    dfs_test: list[DataFrame], dfs_base: list[DataFrame], name: str, fold_name : str
) -> DataFrame:
    """
    Performs a series of Wilcoxon signed-rank tests to determine if the test dataset is statistically superior
    to the base dataset across specified metrics.

    This function aligns each DataFrame in the `dfs_test` list with a corresponding DataFrame in the `dfs_base` list,
    and then performs a Wilcoxon signed-rank test for specified metrics. The function adjusts the hypothesis for each
    metric to test for superiority (e.g., a higher PSNR or lower NRMSE is considered superior). It generates results
    for each unique combination of 'center', 'drf', and 'tracer', as well as aggregated results across the entire dataset.

    Args:
        dfs_test (list[DataFrame]): A list of DataFrames containing the test data.
        dfs_base (list[DataFrame]): A list of DataFrames containing the base data.
        name (str): A string identifier used in plotting distributions, it can be a descriptive name of the test set.

    Returns:
        DataFrame: A DataFrame summarizing the results of the tests, with keys for each metric tested and values
                   indicating whether the test data was superior to the base data.

    Raises:
        ValueError: If the lists of test and base DataFrames are not of equal length and cannot be aligned for paired testing.
    """

    if len(dfs_base) == 1:
        dfs_base = dfs_base * len(dfs_test)
    elif len(dfs_test) == 1:
        dfs_test = dfs_test * len(dfs_base)
    else:
        pass
    if len(dfs_base) != len(dfs_test):
        raise ValueError("The list of dataframes for test and base do not correspond.")
    # Initialize results dictionary
    results = {}
    for c, (df_test, df_base) in enumerate(zip(dfs_test, dfs_base)):
        # Apply tests for each unique value in 'center', 'drf', and 'tracer'
        for column in ["center", "drf", "tracer"]:
            print(f"Testing {column} in {name} c{c}")
            for value in df_test[column].unique():
                results[value] = {}
                for metric, direction in metric_directions.items():
                    result = statistical_test(
                        df_variant=df_test[df_test[column] == value],
                        df_control=df_base[df_base[column] == value],
                        metric=metric,
                        margin=0,
                        direction=direction,
                        plot=False,
                    )
                    results[value][metric] = result["superior"][0]

        # Apply tests for the total dataset
        results[f"total_{c}"] = {}
        for metric, direction in metric_directions.items():
            result = statistical_test(
                df_variant=df_test,
                df_control=df_base,
                metric=metric,
                margin=0,
                direction=direction,
                base_dir=Path("results/regression/acceptance/" + fold_name),
                name=name + f"_c{c}",
            )
            results[f"total_{c}"][metric] = result["superior"][0]

    return pd.DataFrame(results)


def non_inferiority_test(
    dfs_test: list[DataFrame], dfs_base: list[DataFrame], name: str, fold_name: str
) -> DataFrame:
    """
    Performs a series of Wilcoxon signed-rank tests to determine if the test dataset is statistically non-inferior
    to the base dataset across specified metrics.

    This function aligns each DataFrame in the `dfs_test` list with a corresponding DataFrame in the `dfs_base` list,
    and then performs a Wilcoxon signed-rank test for specified metrics. The function adjusts the hypothesis for each
    metric to test for superiority (e.g., a higher PSNR or lower NRMSE is considered superior). It generates results
    for each unique combination of 'center', 'drf', and 'tracer', as well as aggregated results across the entire dataset.

    Args:
        dfs_test (list[DataFrame]): A list of DataFrames containing the test data.
        dfs_base (list[DataFrame]): A list of DataFrames containing the base data.
        name (str): A string identifier used in plotting distributions, it can be a descriptive name of the test set.

    Returns:
        DataFrame: A DataFrame summarizing the results of the tests, with keys for each metric tested and values
                   indicating whether the test data was superior to the base data.

    Raises:
        ValueError: If the lists of test and base DataFrames are not of equal length and cannot be aligned for paired testing.
    """

    if len(dfs_base) == 1:
        dfs_base = dfs_base * len(dfs_test)
    elif len(dfs_test) == 1:
        dfs_test = dfs_test * len(dfs_base)
    else:
        pass
    if len(dfs_base) != len(dfs_test):
        raise ValueError("The list of dataframes for test and base do not correspond.")

    # Initialize results dictionary
    results = {}
    for c, (df_test, df_base) in enumerate(zip(dfs_test, dfs_base)):
        # Apply tests for each unique value in 'center', 'drf', and 'tracer'
        for column in ["center", "drf", "tracer"]:
            for value in df_test[column].unique():
                results[value] = {}
                for metric, direction in metric_directions.items():
                    result = statistical_test(
                        df_variant=df_test[df_test[column] == value],
                        df_control=df_base[df_base[column] == value],
                        metric=metric,
                        margin=metric_ni_margins[metric],
                        direction=direction,
                        plot=False,
                    )
                    results[value][metric] = result["non_inferior"][0]

        # Apply tests for the total dataset
        results[f"total_{c}"] = {}
        for metric, direction in metric_directions.items():
            result = statistical_test(
                df_variant=df_test,
                df_control=df_base,
                metric=metric,
                margin=metric_ni_margins[metric],
                direction=direction,
                base_dir=Path("results/regression/acceptance/" + fold_name),
                name=name + f"_c{c}",
            )
            results[f"total_{c}"][metric] = result["non_inferior"][0]

    return pd.DataFrame(results)


def quantitative_acceptance(dfs_test: list[DataFrame], name: str, results: DataFrame, fold_name: str) -> DataFrame:
    """
    Performs quantitative acceptance testing on a list of test dataframes, specifically examining the accuracy
    of SUVmean predictions. The function updates the provided results DataFrame with acceptance values based on
    whether the absolute error in SUVmean predictions is within 10% of the ground truth for at least 95% of the cases.

    Args:
        dfs_test (list[DataFrame]): A list of DataFrames containing the test data. Each DataFrame should have
                                    columns for 'SUVmean_pred' (predicted SUV mean values) and 'SUVmean_gt'
                                    (ground truth SUV mean values).
        name (str): A string identifier used for labeling plots generated during analysis.
        results (DataFrame): A DataFrame that accumulates the results of various tests. This function updates this
                             DataFrame with the results of the SUVmean prediction accuracy test.

    Returns:
        DataFrame: The updated results DataFrame with a new row ('SUVmean error') indicating whether each test set
                   meets the acceptance criteria (true if it does, false otherwise).
    """
    # Initialize results dictionary
    for c, df_test in enumerate(dfs_test):
        diff = df_test["SUVmean_pred"] - df_test["SUVmean_gt"]
        avg = (df_test["SUVmean_pred"] + df_test["SUVmean_gt"]) / 2
        n_within_10 = ((diff.abs() / df_test["SUVmean_gt"]) < 0.1).sum()
        n_good = n_within_10 / len(df_test)
        results.loc["SUVmean error", f"total_{c}"] = n_good > 0.95

        plot_bland_altman(avg, diff, name + f"_c{c}", n_good, base_dir=Path("results/regression/acceptance/" + fold_name))
        plot_cumulative_error(diff, df_test["SUVmean_gt"], name + f"_c{c}", base_dir=Path("results/regression/acceptance/" + fold_name))
    return results


if __name__ == "__main__":
    candidate_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/unet_wide/results/predictions_epoch_1500.csv")
    low_count_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/unet_wide/results/lc.csv")
    
    #candidate_results = candidate_results[candidate_results['drf'] == '50pc']
    #low_count_results = low_count_results[low_count_results['drf'] == '50pc']

    #candidate_results = candidate_results[candidate_results['center'] == 'quadra']
    #low_count_results = low_count_results[low_count_results['center'] == 'quadra']

    results = superiority_test([candidate_results], [low_count_results], "new_base", "unet")
    results = quantitative_acceptance([candidate_results], "new_base", results, "unet")


    candidate_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/nuclarity/results/predictions_c0.csv")
    low_count_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/nuclarity/results/lc.csv")

    #candidate_results = candidate_results[candidate_results['drf'] == '50pc']
    #low_count_results = low_count_results[low_count_results['drf'] == '50pc']

    #candidate_results = candidate_results[candidate_results['center'] == 'quadra']
    #low_count_results = low_count_results[low_count_results['center'] == 'quadra']

    results = superiority_test([candidate_results], [low_count_results], "new_base", "nuclarity")
    results = quantitative_acceptance([candidate_results], "new_base", results, "nuclarity")

    candidate_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/unet_wide/results/predictions_epoch_1500.csv")
    base_results = pd.read_csv("/home/vicde/nucli-train/experiments/nuclarity_data/nuclarity/results/predictions_c0.csv")

    non_inferiority_test([candidate_results], [base_results], "new_base", "unet_vs_nuclarity")