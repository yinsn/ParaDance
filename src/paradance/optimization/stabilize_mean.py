from typing import List

import pandas as pd


def stabilize_mean_with_exponents(
    dataframe: pd.DataFrame,
    keep_columns: List[str],
    boost_columns: List[str],
    boost_scale: float,
    compensation: float = 0.0,
    tolerance: float = 1e-6,
    low: float = 0.0,
    high: float = 5.0,
) -> float:
    """
    Adjusts the values in the dataframe to stabilize the mean of the product of specified columns
    to a target value by applying an exponent transformation.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        keep_columns (List[str]): List of column names whose values are to be adjusted.
        boost_columns (List[str]): List of column names whose values are boosted by raising to the power of boost_scale.
        boost_scale (float): The exponent to apply to the boost_columns.
        compensation (float, optional): Additional value added to the target mean calculation. Defaults to 0.0.
        tolerance (float, optional): The tolerance for the difference between the current and target means in the optimization process. Defaults to 1e-6.
        low (float, optional): The lower bound of the search interval for finding the optimal exponent. Defaults to 0.0.
        high (float, optional): The upper bound of the search interval for finding the optimal exponent. Defaults to 5.0.

    Returns:
        float: The deboost ratio for the keep_columns that stabilizes the mean of the product of all the columns.
    """
    dataframe_tmp = dataframe.copy()
    for column in boost_columns:
        dataframe_tmp[column] = dataframe_tmp[column] ** boost_scale
    target_mean = (
        dataframe[keep_columns + boost_columns].prod(axis=1).mean() + compensation
    )
    while low <= high:
        mid = (low + high) / 2
        for column in keep_columns:
            dataframe_tmp[column] = dataframe[column] ** mid
        transformed_mean = (
            dataframe_tmp[keep_columns + boost_columns].prod(axis=1).mean()
        )
        if abs(transformed_mean - target_mean) < tolerance:
            return mid
        elif transformed_mean < target_mean:
            low = mid + tolerance
        else:
            high = mid - tolerance
    return mid


def stabilize_mean_with_additional_factors(
    dataframe: pd.DataFrame,
    keep_columns: List[str],
    additional_columns: List[str],
    compensation: float = 0.0,
    tolerance: float = 1e-6,
    low: float = 0.0,
    high: float = 5.0,
) -> float:
    """
    Stabilizes the mean of the product of specified columns in a DataFrame with additional factors.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        keep_columns (List[str]): List of column names to keep and transform.
        additional_columns (List[str]): List of additional column names to include in the mean calculation.
        compensation (float, optional): Compensation value to adjust the target mean. Defaults to 0.0.
        tolerance (float, optional): Tolerance level for mean stabilization. Defaults to 1e-6.
        low (float, optional): Lower bound of the exponent search interval. Defaults to 0.0.
        high (float, optional): Upper bound of the exponent search interval. Defaults to 5.0.

    Returns:
        float: The exponent value that stabilizes the mean within the specified tolerance.
    """
    dataframe_tmp = dataframe.copy()
    target_mean = dataframe[keep_columns].prod(axis=1).mean() + compensation
    while low <= high:
        mid = (low + high) / 2
        for column in keep_columns:
            dataframe_tmp[column] = dataframe[column] ** mid
        transformed_mean = (
            dataframe_tmp[keep_columns + additional_columns].prod(axis=1).mean()
        )
        if abs(transformed_mean - target_mean) < tolerance:
            return mid
        elif transformed_mean < target_mean:
            low = mid + tolerance
        else:
            high = mid - tolerance
    return mid
