from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from .calculator import Calculator


def merge_and_count(
    target_queue: np.ndarray,
    temporary_queue: np.ndarray,
    start_index: int,
    end_index: int,
    weight_type: str,
) -> float:
    """
    A helper function for the merge sort algorithm. It is used to merge two sorted
    subarrays and count the inverse pairs based on the provided weights type.

    :param target_queue: The original target scores to be sorted and merged.
    :param temp_queue: A temporary array to store merged results.
    :param left: The starting index of the portion to be merged.
    :param right: The ending index of the portion to be merged.
    :param weights_type: The type of weights used in calculating inverse pairs.
    :return: The computed inverse pairs value for the merged segment.
    """

    def merge_ranges(
        target_queue: np.ndarray,
        temporary_queue: np.ndarray,
        i: int,
        j: int,
        end: int,
        weight_type: str,
    ) -> float:
        """
        Helper function to merge two ranges and count inverse pairs based on the provided weights type.
        """

        result = 0.0
        batch_size, _ = target_queue.shape

        for k in range(batch_size):
            idx_temporary, idx_left, idx_right = i, i, j
            while idx_left < j and idx_right < end:
                if target_queue[k, idx_left] >= target_queue[k, idx_right]:
                    temporary_queue[k, idx_temporary] = target_queue[k, idx_left]
                    idx_left += 1
                else:
                    temporary_queue[k, idx_temporary] = target_queue[k, idx_right]
                    idx_right += 1
                    if weight_type == "count":
                        result += 1
                    elif weight_type == "linear":
                        result += (j - idx_left) * 0.5
                    elif weight_type == "exponential":
                        result += (j - idx_left) * (0.8**k)
                idx_temporary += 1

            temporary_queue[k, idx_temporary:j] = target_queue[k, idx_left:j]
            temporary_queue[k, idx_temporary:end] = target_queue[k, idx_right:end]

        return float(result)

    length = end_index - start_index
    step = 1
    result = 0.0

    while step < length:
        for i in range(start_index, end_index, 2 * step):
            j = min(i + step, end_index)
            end = min(j + step, end_index)
            result += merge_ranges(
                target_queue, temporary_queue, i, j, end, weight_type
            )
            target_queue[:, i:end] = temporary_queue[:, i:end]
        step *= 2

    return float(result)


def calculate_inverse_pairs(
    target_scores: np.ndarray, merge_scores: np.ndarray, weights_type: str
) -> float:
    """
    Calculate the inverse pairs for the given target scores based on merge scores.

    :param target_scores: The original target scores.
    :param merge_scores: The scores used to determine the order of merging.
    :param weights_type: The type of weights used in calculating inverse pairs.
    :return: The computed inverse pairs value.
    """

    num_samples, dimension = target_scores.shape
    sorted_indices = np.argsort(-merge_scores, axis=-1)
    sample_indices = np.arange(0, num_samples).reshape([-1, 1])
    sorted_target_scores = target_scores[sample_indices, sorted_indices]
    temp_scores = np.zeros((num_samples, dimension), dtype=np.float32)
    result = merge_and_count(
        sorted_target_scores, temp_scores, 0, dimension, weights_type
    )
    return float(result / num_samples)


def calculate_inverse_pair(
    calculator: "Calculator",
    weights_for_equation: List[float],
    weights_type: str = "count",
) -> float:
    """
    Calculates the weighted sum of inverse pairs for selected columns
    using the specified weighting scheme.

    :param weights_for_equation: The list of weights to be applied to each column.
    :param weights_type: The type of weights used for inverse pair calculation.
                            Supported values are "count", "linear", and "exponential".
    :return: The computed weighted sum of inverse pairs.
    """
    calculator.get_overall_score(weights_for_equation)

    result = 0.0
    for i, weight in enumerate(weights_for_equation):
        score = calculate_inverse_pairs(
            calculator.selected_values[:, i],
            calculator.df["overall_score"],
            weights_type,
        )
        result += score * weight
    return result
