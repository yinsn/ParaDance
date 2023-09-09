import csv
import pickle

from .multiple_objective import MultipleObjective


def get_best_trials(multiple_objective: MultipleObjective) -> None:
    """
    Extracts and saves the best trials from the given MultipleObjective's log file.

    This function reads the 'paradance.log' file associated with the provided MultipleObjective object,
    and identifies the best trials based on the comparison of values. These trials, along with their
    results, targets, and weights are then saved to a CSV file named 'paradance_best_trials.csv' in
    the same directory.

    Args:
        multiple_objective (MultipleObjective): An instance of the MultipleObjective class containing the study to be saved.
    """
    ob = multiple_objective
    file_path = f"{ob.full_path}/paradance.log"
    output_path = f"{ob.full_path}/paradance_best_trials.csv"

    last_best_value = None
    last_trial_number = -1
    extracted_data = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        for idx, line in enumerate(lines):
            if "Best is trial" in line:
                current_best_value = float(line.split("with value:")[1].split(" ")[1])
                if last_best_value is None or last_best_value != current_best_value:
                    trial_number = int(line.split("Best is trial")[1].split(" ")[1])
                    if trial_number > last_trial_number:
                        results_line = lines[idx - 3]
                        targets_line = lines[idx - 2]
                        weights_line = lines[idx - 1]

                        try:
                            results = float(results_line.split("result:")[1].strip())
                            targets_str = [
                                val
                                for val in targets_line.split(":")[1]
                                .strip()
                                .strip("[]\n")
                                .split(",")
                            ]
                            weights_str = [
                                val
                                for val in weights_line.split(":")[1]
                                .strip()
                                .strip("[]\n")
                                .split(", ")
                            ]

                            if len(weights_str) != 1:
                                targets = [float(val) for val in targets_str]
                                weights = [float(val) for val in weights_str]
                                extracted_data.append(
                                    (results, trial_number, targets, weights)
                                )

                                last_trial_number = trial_number

                                last_best_value = current_best_value
                        except:
                            pass

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Result", "Trial", "Targets", "Weights"])
        for data in extracted_data:
            writer.writerow([data[0], data[1], str(data[2]), str(data[3])])


def save_study(multiple_objective: MultipleObjective) -> None:
    """Save the study results of the given multiple objective optimization.

    Args:
        multiple_objective (MultipleObjective): An instance of the MultipleObjective class containing the study to be saved.
    """
    ob = multiple_objective
    ob.study.trials_dataframe().to_csv(f"{ob.full_path}/paradance_full_trials.csv")
    with open(f"{ob.full_path}/study.pkl", "wb") as f:
        pickle.dump(ob.study, f)
    get_best_trials(ob)
