import csv
import pickle
import sys

from .multiple_objective import MultipleObjective


def get_best_trials(multiple_objective: MultipleObjective) -> None:
    """
    Extracts and saves the best trials from the provided log content.
    """
    ob = multiple_objective
    file_path = f"{ob.full_path}/paradance.log"
    output_path = f"{ob.full_path}/paradance_best_trials.csv"

    with open(file_path, "r") as file:
        lines = file.readlines()

    best_trials = set()
    extracted_data = []
    sys.stdout.write(f"\nFormula:\t{ob.formula}\n")
    sys.stdout.write(f"Evaluators:\t{ob.evaluator_flags}\n")
    sys.stdout.write(f"Features:\t{ob.calculator.selected_columns}\n")
    for idx, line in enumerate(lines):
        if "Best is trial" in line:
            trial_number = int(line.split("Best is trial")[1].split(" ")[1])

            if trial_number in best_trials:
                continue

            best_trials.add(trial_number)

            for sub_idx in range(idx, -1, -1):
                if f"Trial {trial_number} finished with result:" in lines[sub_idx]:
                    results_line = lines[sub_idx]
                    sub_idx += 1

                    while "targets:" not in lines[sub_idx]:
                        sub_idx += 1
                    targets_line = (
                        lines[sub_idx].split("targets:")[1].strip().strip("[]")
                    )

                    while "weights:" not in lines[sub_idx]:
                        sub_idx += 1
                    weights_line = (
                        lines[sub_idx].split("weights:")[1].strip().strip("[]")
                    )

                    results = float(results_line.split("result:")[1].strip())
                    targets_str = [
                        val.strip() for val in targets_line.split(",") if val.strip()
                    ]
                    weights_str = [
                        val.strip() for val in weights_line.split(",") if val.strip()
                    ]

                    try:
                        targets = [float(val) for val in targets_str]
                        weights = [float(val) for val in weights_str]
                        sys.stdout.write(f"\ntrail {trial_number}:\t{results}\n")
                        sys.stdout.write(f"targets:\t{targets}\n")
                        sys.stdout.write(f"features:\t{weights}\n")
                        extracted_data.append((results, trial_number, targets, weights))
                    except ValueError as e:
                        sys.stdout.write(
                            f"Error processing line: {sub_idx}, error: {e}\n"
                        )
                    break

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                f"{ob.formula}",
                "Trial",
                f"{ob.evaluator_flags}",
                f"{ob.calculator.selected_columns}",
            ]
        )
        for data in extracted_data:
            writer.writerow([data[0], data[1], str(data[2]), str(data[3])])


def save_multiple_objective_info(ob: MultipleObjective, filename: str) -> None:
    """Save the parameters and evaluator info of a MultipleObjective object to a txt file.

    Args:
        ob (MultipleObjective): The object whose info needs to be saved.
        filename (str): The name of the txt file.
    """

    with open(filename, "w") as file:
        file.write(f"Study Name: {ob.study_name}\n")
        file.write("-" * 50 + "\n")
        file.write(f"Formula: {ob.formula}\n")
        file.write(f"Selected Columns: {ob.calculator.selected_columns}\n")
        file.write(f"Direction: {ob.direction}\n")
        file.write(f"Weights Number: {ob.weights_num}\n")
        file.write(f"Dirichlet: {ob.dirichlet}\n")
        file.write(f"Power: {ob.power}\n")
        file.write(f"Power Lower Bound: {ob.power_lower_bound}\n")
        file.write(f"Power Upper Bound: {ob.power_upper_bound}\n")
        file.write(f"First Order: {ob.first_order}\n")
        file.write(f"First Order Lower Bound: {ob.first_order_lower_bound}\n")
        file.write(f"First Order Upper Bound: {ob.first_order_upper_bound}\n")

        file.write("\nEvaluators Info:\n")
        file.write("-" * 50 + "\n")
        for flag, target_column, hyperparameter, evaluator_property, groupby in zip(
            ob.evaluator_flags,
            ob.target_columns,
            ob.hyperparameters,
            ob.evaluator_propertys,
            ob.groupbys,
        ):
            file.write(f"Flag: {flag}\n")
            file.write(f"Target Column: {target_column}\n")
            file.write(f"Hyperparameter: {hyperparameter}\n")
            file.write(f"Evaluator Property: {evaluator_property}\n")
            file.write(f"Groupby: {groupby}\n")
            file.write("\n")


def save_study(multiple_objective: MultipleObjective) -> None:
    """Save the study results of the given multiple objective optimization.

    Args:
        multiple_objective (MultipleObjective): An instance of the MultipleObjective class containing the study to be saved.
    """
    ob = multiple_objective
    save_multiple_objective_info(ob, f"{ob.full_path}/objective_info.txt")
    ob.study.trials_dataframe().to_csv(f"{ob.full_path}/paradance_full_trials.csv")
    with open(f"{ob.full_path}/study.pkl", "wb") as f:
        pickle.dump(ob.study, f)
    get_best_trials(ob)
