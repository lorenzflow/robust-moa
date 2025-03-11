import json
import datasets
from typing import List
from loguru import logger
import os
import random
import pandas as pd


from moa_class import MoA_Alpaca, MoA_Quality
from utils import (
    load_model_dict,
    process_file,
    compute_accuracy,
    compute_persuaded,
    DEBUG,
)

import yaml
from dataclasses import asdict
# from config import Config

## Hydra version!!!!

# main.py
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING, DictConfig

@dataclass
class MoAConfig:
    aggregator: str
    reference_models: List[str]
    temperature: float = 0.7
    max_tokens: int = 2048
    rounds: int = 1

@dataclass
class DeceptiveConfig:
    deceptive_model_dict_name: Optional[str] = None
    deceptive_ignore_refs: bool = False

@dataclass
class ExperimentConfig:
    task: str
    output_path: str
    num_samples: Optional[int] = None
    save_references: bool = False
    use_subpassages: bool = True
    hard_only: bool = False
    num_proc: int = 16
    seed: int = 42
    results_dir: Optional[str] = None

@dataclass
class PromptsConfig:
    deceptive_proposer_system_prompt: str
    deceptive_aggregating_proposer_system_prompt: str
    deceptive_aggregating_proposer_system_prompt_end: str = ""
    deceptive_proposer_user_prompt_end: str = ""
    deceptive_aggregating_proposer_user_prompt_end: str = ""

@dataclass
class Config:
    moa: MoAConfig
    experiment: ExperimentConfig
    deceptive: DeceptiveConfig
    prompts: PromptsConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run the MoA on a given task.

    Config: 

    task: str, the task to run the MoA on. Options: quality, alpaca
    aggregator: str, the aggregator model to use.
    output_path: str, the path to save the output to.
    reference_models: str, comma-separated list of reference models to use. Default is None.
    temperature: float, the temperature to use for sampling. Default is 0.7.
    max_tokens: int, the maximum number of tokens to generate. Default is 2048.
    rounds: int, the number of rounds to run the MoA for. Default is 1.
    num_proc: int, the number of processes to use for multiprocessing. Default is 16.
    deceptive_model_dict_name: str, the name of the deceptive model dictionary to use. Default is None which results in all truthful models.
    deceptive_ignore_refs: bool, whether to ignore the reference models when generating deceptive responses. Default is False.
    num_samples: int, the number of samples to generate. Default is None.
    save_references: bool, whether to save the references in the output. Default is False.

    Alpaca-specific arguments:

    deceptive_proposer_system_prompt: str, the prompt to use for the deceptive proposer system. Default is the provided prompt.
    deceptive_aggregating_proposer_system_prompt: str, the prompt to use for the deceptive aggregating proposer system. Default is the provided prompt.
    deceptive_aggregating_proposer_system_prompt_end: str, the end prompt to use for the deceptive aggregating proposer system. Default is the provided prompt.
    deceptive_proposer_user_prompt_end: str, the end prompt to use for the deceptive proposer user. Default is the provided prompt.
    deceptive_aggregating_proposer_user_prompt_end: str, the end prompt to use for the deceptive aggregating proposer user. Default is the provided prompt.
    
    Quality-specific arguments:
    use_subpassages: bool, whether to use subpassages for the quality task. Default is True.
    seed: int, the seed to use for the quality task. Default is 42.
    hard_only: bool, whether to use only hard examples for the quality task. Default is False.
    results_dir: str, the directory to save the metrics to. Default is None.

    Example of deceptive_model_dict_name usage options:
    deceptive_model_dict_name: determines which models are deceptive and with which level. Default is None, which means all models are truthful.
    Example: - if sample_k_1 randomly sample k of the models to be deceptive with status 1 (subtle lying)
             - if sample_k_2 randomly sample k of the models to be deceptive with status 2 (arguing for incorrect answer)
             - if sample_k1_1_k2_2 randomly sample k1 of the models to be deceptive with status 1 (subtle lying) and k2 of the models to be deceptive with status 2 (arguing for incorrect answer)            
             - otherwise specify as follows: for the 3-3-1 architecture the input 110_021 layers are separated by _, and deceptive status is as follows:
                0: truthful
                1: deceptive
    """
    print("Config:", cfg)
    reference_models = cfg.moa.reference_models
    deceptive_model_dict_name = cfg.deceptive.deceptive_model_dict_name
    
    if deceptive_model_dict_name is None:
        output_path = cfg.experiment.output_path + "/" + cfg.moa.aggregator + "/" + "truthful"
    else:
        output_path = cfg.experiment.output_path + "/" + cfg.moa.aggregator + "/" + cfg.deceptive.deceptive_model_dict_name

    # Your existing function logic here
    print(f"Running with model: {cfg.moa.aggregator}")
    print(f"Reference models: {reference_models}")
    print(f"Output path: {output_path}")

    logger.info(f"Generating {cfg.experiment.num_samples} samples.")
    
    if reference_models is None:
        reference_models = []
    
    if len(reference_models) > 0:
        if deceptive_model_dict_name is None:
            generator_name = cfg.moa.aggregator + "-together"
            deceptive_model_dict = {}
            for i in range(cfg.moa.rounds):
                deceptive_model_dict[i] = {model: 0 for model in reference_models} # new model dict
                # deceptive_model_dict[i] = {model: False for model in reference_models}
        else:
            if isinstance(deceptive_model_dict_name, int): # hack since sometimes Fire processes as int instead of str
                deceptive_model_dict_name = str(deceptive_model_dict_name)
                deceptive_model_dict_name = deceptive_model_dict_name[:len(reference_models)] + "_" + deceptive_model_dict_name[len(reference_models):]
            elif "status" in deceptive_model_dict_name: # hack since sometimes Fire processes as int instead of str
                deceptive_model_dict_name = deceptive_model_dict_name.split(".")[-1]
            
            if DEBUG:
                print(deceptive_model_dict_name)
            # generator_name = cfg.experiment.output_path.split("/")[-1]
            generator_name = cfg.moa.aggregator + "-" + deceptive_model_dict_name
            # generator_name = model + deceptive_model_dict_name
            deceptive_model_dict = load_model_dict(deceptive_model_dict_name, reference_models, cfg.moa.rounds) # now have deceptive status instead of true or false
            # deceptive_model_dict = load_model_dict(deceptive_model_dict_name, reference_models, rounds)
        
        print(deceptive_model_dict)

    if cfg.experiment.task == "quality":
        data_path = "QuALITY/QuALITY.v1.0.1.htmlstripped.train"
        eval_set = process_file(data_path, hard_only=cfg.experiment.hard_only)
        eval_set = datasets.Dataset.from_list(eval_set)
        
        rng = random.Random(cfg.experiment.seed) # for incorrect answers

        # instantiate MoA
        moa = MoA_Quality(
            aggregator=cfg.moa.aggregator,
            reference_models=reference_models,
            temperature=cfg.moa.temperature,
            max_tokens=cfg.moa.max_tokens,
            rounds=cfg.moa.rounds,
            generator_name=generator_name,
            deceptive_model_dict=deceptive_model_dict,
            deceptive_ignore_refs = cfg.deceptive.deceptive_ignore_refs,
            use_subpassages=cfg.experiment.use_subpassages,
            rng=rng,
        )
            
    elif cfg.experiment.task == "alpaca":
        eval_set = datasets.load_dataset(
                                        "tatsu-lab/alpaca_eval", 
                                        "alpaca_eval_gpt4_baseline", 
                                        trust_remote_code=True
                                        )["eval"]

        eval_set = eval_set.remove_columns(["output", "generator"])
        # instantiate MoA
        moa = MoA_Alpaca(
            aggregator=cfg.moa.aggregator,
            reference_models=reference_models,
            temperature=cfg.moa.temperature,
            max_tokens=cfg.moa.max_tokens,
            rounds=cfg.moa.rounds,
            generator_name=generator_name,
            deceptive_model_dict=deceptive_model_dict,
            deceptive_ignore_refs = cfg.deceptive.deceptive_ignore_refs,
            deceptive_proposer_system_prompt = cfg.prompts.deceptive_proposer_system_prompt,
            deceptive_aggregating_proposer_system_prompt = cfg.prompts.deceptive_aggregating_proposer_system_prompt,
            deceptive_aggregating_proposer_system_prompt_end = cfg.prompts.deceptive_aggregating_proposer_system_prompt_end,
            deceptive_proposer_user_prompt_end = cfg.prompts.deceptive_proposer_user_prompt_end,
            deceptive_aggregating_proposer_user_prompt_end = cfg.prompts.deceptive_aggregating_proposer_user_prompt_end
        )

    else:
        raise ValueError(f"Task {cfg.experiment.task} not recognized.")
    
    # subsample if specified
    if cfg.experiment.num_samples is not None:
        random.seed(cfg.experiment.seed)
        # Subsample num_samples items after random shuffle
        eval_set = eval_set.shuffle(seed=cfg.experiment.seed).select(range(cfg.experiment.num_samples))
        # eval_set = eval_set.select(range(num_samples))
    
    if len(reference_models):
        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )
    logger.info(f"Start.")

    # generate with MoA
    eval_set = eval_set.map(
        moa.full_synthesise,
        batched=False,
        num_proc=cfg.experiment.num_proc,
    )

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if cfg.experiment.save_references:
        logger.info(f"Saving outputs with annotated references to {output_path}_with_ref.json.")
        with open(output_path + "_with_ref" + ".json", "w") as f:

            json.dump(list(eval_set), f, indent=2)

    logger.info(f"Saving outputs to {output_path}.json.")
    try:
        eval_set = eval_set.remove_columns(f"references")
    except Exception as e:
        pass
    try:
        eval_set = eval_set.remove_columns(f"annotated references")
    except Exception as e:
        pass
    
    with open(output_path + ".json", "w") as f:
        json.dump(list(eval_set), f, indent=2)

    if cfg.experiment.task == "quality": # compute quality specific metrics, for alpaca eval call eval separately
        accuracy, ac_se = compute_accuracy(eval_set)
        persuaded, p_se = compute_persuaded(eval_set)

        logger.info(f"Accuracy: {accuracy} +/- {ac_se}")
        logger.info(f"Persuaded: {persuaded} +/- {p_se}")
        
        if cfg.experiment.results_dir is not None:
            results_file = os.path.join(cfg.experiment.results_dir, "results.csv")
            df = {"accuracy": accuracy, "accuracy_se": ac_se, "persuaded": persuaded, "persuaded_se": p_se}
            # for saving results later
            if deceptive_model_dict_name is None:
                if "baseline" in output_dir:
                    method_name = "baseline"
                elif "truthful" in output_dir:
                    method_name = "truthful"
                else:
                    method_name = output_path.split("/")[-1]
            else:
                method_name = deceptive_model_dict_name

            df_new = pd.DataFrame([df], index=[method_name])

            if os.path.exists(results_file):
                df_existing = pd.read_csv(results_file, index_col=0)
                if method_name in df_existing.index:
                    df_new.index = [method_name + "_v2"]

                df_existing = pd.concat([df_existing, df_new])
                df_existing.to_csv(results_file)
            else:
                if not os.path.exists(cfg.experiment.results_dir):
                    os.makedirs(cfg.experiment.results_dir)
                df_new.to_csv(results_file)

if __name__ == "__main__":
    main()