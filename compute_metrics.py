import json
import datasets
from fire import Fire
from loguru import logger
import os
import pandas as pd
import glob




from utils import (
    compute_accuracy,
    compute_persuaded,
    compute_recovery_rate,
    compute_deception_corruption_rate,
    DEBUG

)





def load_references(reference_path):
    with open(reference_path) as f:
        reference_responses = json.load(f)
    
    # annotated_references = [item["references"] for item in reference_responses]
    return reference_responses #, annotated_references


def compute_metrics(
    name: str,
    output_path: str,
    output_type: str,
    categorical_output: bool = True,
    num_samples: int = None,
    results_dir = None,
    baseline_paths = None,
    baseline_output_types: str = None,
):
    """
    Compute metrics for the model output.
    Args:
        model: aggregator model name
        name: name to be used when saving results
        output_path: path to save the output
        output_type: output type of the model
        num_samples: number of samples to evaluate
        results_dir: directory to save results
        baseline_paths: paths to the baseline outputs
        baseline_output_types: len(baseline_paths) output types for the baseline outputs to be used ("output", "robust_output")
    Returns:
        None

    """
    if "json" not in output_path:
        output_path = output_path + ".json"

    with open(output_path) as f:
        eval_set = json.load(f)
    eval_set = datasets.Dataset.from_list(eval_set)
    assert "robust_output" or "output" in eval_set[0], "robust_output not found in the input file"

    logger.info(f"Loaded {len(eval_set)} samples from {output_path}")
    if len(eval_set) < 500:
        logger.warning(f"Number of samples is less than 500: {len(eval_set)}, aborting")
        return
    
    if num_samples is not None:
        # Subsample num_samples items after random shuffle
        eval_set = eval_set.select(range(num_samples))

    logger.info(f"Start. {output_path}")

    # metrics only for categorical output type, otherwise run alpaca eval!
    if categorical_output:
        # extracts the answer if necessary
        # compute eval metrics, accuracy and persuaded percentage
        accuracy, ac_se = compute_accuracy(eval_set, output_type)
        if "incorrect_answer" in eval_set[0]:
            persuaded, p_se = compute_persuaded(eval_set, output_type)
        else:
            persuaded, p_se = 0, 0
            logger.info("Persuaded not found in the input file")

        if baseline_paths is not None:
            baseline_paths = baseline_paths.split(",")
            # baseline_output_types = baseline_output_types.split(",")
            rr_list = []
            dcr_list = []
            for i, baseline_path in enumerate(baseline_paths):
                with open(baseline_path) as f:
                        baseline_set = json.load(f)
                rr, rr_se = compute_recovery_rate(eval_set, baseline_set, (baseline_output_types[i], output_type))
                if "incorrect_answer" in eval_set[0]:
                    dcr, dcr_se = compute_deception_corruption_rate(eval_set, baseline_set, (baseline_output_types[i], output_type))
                else:
                    dcr, dcr_se = 0, 0
                    logger.info("Deception not found in the input file")

                if DEBUG:
                    logger.info(f"RR w.r.t. {baseline_path}+{baseline_output_types[i]}: {rr} +/- {rr_se}")
                    logger.info(f"DCR w.r.t. {baseline_path}+{baseline_output_types[i]}: {dcr} +/- {dcr_se}")
                rr_list.append((rr, rr_se))
                dcr_list.append((dcr, dcr_se))
        if DEBUG:
            logger.info(f"Accuracy: {accuracy} +/- {ac_se}")
            logger.info(f"Persuaded: {persuaded} +/- {p_se}")
        if results_dir is not None:
            results_file = os.path.join(results_dir, "results_all_metrics.csv")
            df = {"accuracy": accuracy, "accuracy_se": ac_se, "persuaded": persuaded, "persuaded_se": p_se}
            if baseline_paths is not None:
                for i, baseline_path in enumerate(baseline_paths):
                    baseline_name = baseline_path.split("/")[-1].replace(".json", "")
                    df[f"dcr {baseline_name}"] = dcr_list[i][0]
                    df[f"dcr_se {baseline_name}"] = dcr_list[i][1]
                    df[f"rr {baseline_name}"] = rr_list[i][0]
                    df[f"rr_se {baseline_name}"] = rr_list[i][1]

            # for saving results later
            df_new = pd.DataFrame([df], index=[name])

            if os.path.exists(results_file):
                df_existing = pd.read_csv(results_file, index_col=0)
                if name in df_existing.index:
                    df_new.index = [name + "_v2"]

                df_existing = pd.concat([df_existing, df_new])
                df_existing.to_csv(results_file)
            else:
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                df_new.to_csv(results_file)
    

if __name__ == "__main__":
    Fire(compute_metrics)
