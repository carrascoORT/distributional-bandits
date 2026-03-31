import pandas as pd


def aggregate_global_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the global summary by instance and algorithm.

    Returns a compact table with means and standard deviations
    of the main metrics across seeds.
    """
    group_cols = ["instance_name", "algorithm_name"]

    agg_df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean_final_utility=("final_utility", "mean"),
            std_final_utility=("final_utility", "std"),
            mean_final_gap=("final_gap", "mean"),
            std_final_gap=("final_gap", "std"),
            mean_l1_error=("l1_error_to_w_star", "mean"),
            std_l1_error=("l1_error_to_w_star", "std"),
            mean_mean_utility=("mean_utility", "mean"),
            mean_mean_gap=("mean_gap", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(["instance_name", "mean_final_gap", "algorithm_name"])
        .reset_index(drop=True)
    )

    return agg_df


def rank_algorithms_by_instance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank algorithms within each instance by mean final gap.
    Smaller final gap is better.
    """
    agg_df = aggregate_global_summary(df).copy()

    agg_df["rank_by_final_gap"] = (
        agg_df.groupby("instance_name")["mean_final_gap"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )

    return agg_df.sort_values(
        ["instance_name", "rank_by_final_gap", "algorithm_name"]
    ).reset_index(drop=True)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pivot a metric into a matrix:
        rows = instances
        columns = algorithms
    """
    return df.pivot(
        index="instance_name",
        columns="algorithm_name",
        values=metric,
    ).sort_index()