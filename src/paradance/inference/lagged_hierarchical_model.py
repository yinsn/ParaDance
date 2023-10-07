from typing import Dict, List, Optional

import numpy as np
import pymc as pm

from .base import BayesianBaseModel


class LaggedHierarchicalModel(BayesianBaseModel):
    """
    A model to incorporate side information with time lags.

    Args:
        primary_series (np.ndarray): Primary data series under investigation.
        side_information_series (np.ndarray): Series of side information data.
        max_time_lags (List[int]): List of maximum time delays for each side information.
        baseline_offset_prior (Optional[pm.Distribution], optional): Prior distribution for the baseline offset. Defaults to None.
        side_info_weights_prior (Optional[pm.Distribution], optional): Prior distribution for the side info weights. Defaults to None.
    """

    def __init__(
        self,
        primary_series: np.ndarray,
        side_information_series: np.ndarray,
        max_time_lags: List[int],
        baseline_offset_prior: Optional[pm.Distribution] = None,
        side_info_weights_prior: Optional[pm.Distribution] = None,
    ):
        super().__init__(primary_series)
        self.side_information_series = side_information_series
        self.number_of_side_series = side_information_series.shape[0]
        self.max_time_lags = max_time_lags
        self.baseline_offset_prior = baseline_offset_prior
        self.side_info_weights_prior = side_info_weights_prior

    def train(
        self, priors: Optional[Dict[str, Dict]] = None, sample_size: int = 1000
    ) -> None:
        """
        Train the model using MCMC sampling.

        Args:
            priors (Dict[str, Dict], optional): Prior distributions for the parameters. Defaults to None.
            sample_size (int, optional): Number of MCMC samples. Defaults to 1000.
        """
        if priors is None:
            priors = {
                "baseline_offset": {"mu": 0, "sigma": 10},
                "side_info_weights": {"mu": 0, "sigma": 10},
                "noise_stddev": {"alpha": 1, "beta": 1},
                "time_lags": {"lower": 1, "upper": self.max_time_lags},
            }

        max_time_lag = max(self.max_time_lags)
        n_obs = self.primary_series.shape[0]

        with pm.Model():
            baseline_offset = self.baseline_offset_prior or pm.Normal(
                "baseline_offset", mu=0, sigma=10
            )
            side_info_weights = self.side_info_weights_prior or pm.Normal(
                "side_info_weights", mu=0, sigma=10, shape=self.number_of_side_series
            )
            noise_stddev = pm.Gamma(
                "noise_stddev",
                alpha=priors["noise_stddev"]["alpha"],
                beta=priors["noise_stddev"]["beta"],
            )
            time_lags = pm.DiscreteUniform(
                "time_lags",
                lower=priors["time_lags"]["lower"],
                upper=priors["time_lags"]["upper"],
                shape=self.number_of_side_series,
            )

            mu_list = []
            for t in range(max_time_lag, n_obs):
                mu_t = baseline_offset
                for j in range(self.number_of_side_series):
                    temp_sum = pm.math.sum(
                        pm.math.switch(
                            pm.math.ge(
                                t - np.arange(self.side_information_series.shape[1]),
                                time_lags[j],
                            ),
                            self.side_information_series[j, :],
                            0,
                        )
                    )
                    mu_t += side_info_weights[j] * temp_sum
                mu_list.append(mu_t)

            mu = pm.math.stack(mu_list)

            pm.Normal(
                "observed_data",
                mu=mu,
                sigma=noise_stddev,
                observed=self.primary_series[max_time_lag:],
            )

            self.trace = pm.sample(sample_size)

    def get_trace(self) -> pm.backends.base.MultiTrace:
        """
        Retrieve the trace of the trained model.

        Returns:
            pm.backends.base.MultiTrace: Trace of the trained model.
        """
        return self.trace
