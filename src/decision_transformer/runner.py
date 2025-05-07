import json
import os
import time
import warnings
from typing import Callable

import torch as t

import wandb
from src.config import (
    ConfigJsonEncoder,
    EnvironmentConfig,
    OfflineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from src.environments.registration import register_envs
from src.models.trajectory_transformer import (
    DecisionTransformer,
)

# from .model import DecisionTransformer
from .offline_dataset import (
    TrajectoryDataset,
    TrajectoryVisualizer,
    one_hot_encode_observation,
)
from .train import train
from .utils import get_max_len_from_model_type


def run_decision_transformer(
    run_config: RunConfig,
    transformer_config: TransformerModelConfig,
    offline_config: OfflineTrainConfig,
    make_env: Callable,
):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    device = set_device(run_config)

    if offline_config.trajectory_path is None:
        raise ValueError("Must specify a trajectory path.")

    max_len = get_max_len_from_model_type(
        offline_config.model_type, transformer_config.n_ctx
    )

    preprocess_observations = (
        None
        if not offline_config.convert_to_one_hot
        else one_hot_encode_observation
    )

    trajectory_paths = offline_config.trajectory_path
    task_datasets = {}
    for i, path in enumerate(trajectory_paths):
        dataset = TrajectoryDataset(
            trajectory_path=path,
            max_len=max_len,
            pct_traj=offline_config.pct_traj,
            prob_go_from_end=offline_config.prob_go_from_end,
            device=device,
            preprocess_observations=preprocess_observations,
        )
        dataset.task_id = i  # 🟢 각 task에 고유 id 부여
        task_datasets[f"task_{i}"] = dataset

    # ensure all the environments we need are registered
    register_envs()

    # 🟢 대표 task 하나 선택 (예: task_0)
    first_task_key = list(task_datasets.keys())[0]
    first_dataset = task_datasets[first_task_key]
    # 환경 ID 및 설정
    env_id = first_dataset.metadata["args"]["env_id"]
    print(first_dataset.metadata)

    if "view_size" not in first_dataset.metadata["args"]:
        first_dataset.metadata["args"]["view_size"] = 7

    environment_config = EnvironmentConfig(
        env_id=env_id,
        one_hot_obs=first_dataset.observation_type == "one_hot",
        view_size=first_dataset.metadata["args"]["view_size"],
        fully_observed=False,
        capture_video=False,
        render_mode="rgb_array",
    )

    env = make_env(environment_config, seed=0, idx=0, run_name="dev")()

    wandb_args = (
        run_config.__dict__
        | transformer_config.__dict__
        | offline_config.__dict__
    )

    if run_config.track:
        run_name = f"{env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
        wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            name=run_name,
            config=wandb_args,
        )
        trajectory_visualizer = TrajectoryVisualizer(first_dataset)
        fig = trajectory_visualizer.plot_reward_over_time()
        wandb.log({"dataset/reward_over_time": wandb.Plotly(fig)})
        fig = trajectory_visualizer.plot_base_action_frequencies()
        wandb.log({"dataset/base_action_frequencies": wandb.Plotly(fig)})
        wandb.log({"dataset/num_trajectories": first_dataset.num_trajectories})


    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
        num_tasks = 3
    )
   

    model = train(
        model=model,
        trajectory_data_set=task_datasets,
        env=env,
        make_env=make_env,
        device=device,
        offline_config=offline_config,
    )

    if run_config.track:
        # save the model with pickle, then upload it
        # as an artifact, then delete it.
        # name it after the run name.
        if not os.path.exists("models"):
            os.mkdir("models")

        model_path = f"models/{run_name}.pt"

        store_transformer_model(
            path=model_path,
            model=model,
            offline_config=offline_config,
        )

        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        os.remove(model_path)

        wandb.finish()


def store_transformer_model(path, model, offline_config):
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "offline_config": json.dumps(
                offline_config, cls=ConfigJsonEncoder
            ),
            "environment_config": json.dumps(
                model.environment_config, cls=ConfigJsonEncoder
            ),
            "model_config": json.dumps(
                model.transformer_config, cls=ConfigJsonEncoder
            ),
        },
        path,
    )


def set_device(run_config):
    if run_config.device == t.device("cuda"):
        if t.cuda.is_available():
            device = t.device("cuda")
        else:
            print("CUDA not available, using CPU instead.")
            device = t.device("cpu")
    elif run_config.device == t.device("cpu"):
        device = t.device("cpu")
    elif run_config.device == t.device("mps"):
        if t.mps.is_available():
            device = t.device("mps")
        else:
            print("MPS not available, using CPU instead.")
            device = t.device("cpu")
    else:
        print("Invalid device, using CPU instead.")
        device = t.device("cpu")

    return device
