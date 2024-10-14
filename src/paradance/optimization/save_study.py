import pickle

from paradance.evaluation.calculator import Calculator

from .multiple_objective import MultipleObjective


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
        file.write(f"equation_type: {ob.calculator.equation_type}\n")

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

    if isinstance(ob.calculator, Calculator) and hasattr(
        ob.calculator, "equation_json"
    ):
        ob.export_completed_formulas()

    save_multiple_objective_info(ob, f"{ob.full_path}/objective_info.txt")
    ob.study.trials_dataframe().to_csv(f"{ob.full_path}/paradance_full_trials.csv")
    with open(f"{ob.full_path}/study.pkl", "wb") as f:
        pickle.dump(ob.study, f)
