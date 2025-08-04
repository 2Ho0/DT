import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from dataclasses import asdict
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from collections import deque
import sys
import os

import wandb
import ruamel.yaml as yaml
from src.config import EnvironmentConfig, OfflineTrainConfig
from src.models.trajectory_transformer import (
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .eval import evaluate_dt_agent
from .utils import configure_optimizers, get_scheduler
from torch.utils.data import ConcatDataset


# Import DreamerV3 modules from dedicated folder
from .dreamerv3 import PERBuffer, DreamerV3Wrapper

# For DreamerV3 Space objects
import elements


def train(
    model: TrajectoryTransformer,
    trajectory_data_set: TrajectoryDataset,
    env,
    make_env,
    offline_config: OfflineTrainConfig,
    device="cpu",
    
):
    
    import torch.nn as nn
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # 전체 학습 데이터의 task_label 수집 필요
    # modify task weight
    task_labels_list = [0] * 9506 + [1] * 11024 + [2] * 8562
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1,2]), y=task_labels_list)
    print("Class Weights:", class_weights)

    weight_tensor = t.tensor(class_weights, dtype=t.float32).to(device)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    mode = offline_config.mode

    try:
        # Create proper spaces for DreamerV3 using elements.Space
        obs_space = {
            'observation': elements.Space(np.uint8, (64, 64, 3)),  # Standard DreamerV3 size
            'reward': elements.Space(np.float32, ()),
            'is_first': elements.Space(bool, ()),
            'is_last': elements.Space(bool, ()),
            'is_terminal': elements.Space(bool, ()),
        }
        act_space = {
            'action': elements.Space(np.int32, (), 0, 6)
        }
        
        dreamer_wrapper = DreamerV3Wrapper(obs_space, act_space)
        per_buffer = PERBuffer(capacity=10000)
        
    except Exception as e:
        print(f"⚠️ Failed to initialize DreamerV3: {e}")
        import traceback
        traceback.print_exc()
        dreamer_wrapper = None
        per_buffer = PERBuffer(capacity=10000)

    # 각 태스크별 데이터셋 수 출력
    print("\n===== 태스크별 데이터셋 크기 =====")
    for task_id, dataset in trajectory_data_set.items():
        print(f"Task {task_id}: {len(dataset)} 샘플")

    train_dataloader, test_dataloader = get_dataloaders(
        trajectory_data_set, offline_config
    )
    
    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]

    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )
    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(train_dataloader):

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n  # dummy action for padding

            optimizer.zero_grad()

           
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            state_preds, action_preds, reward_preds= model(
                states=s,
                # remove last action
                actions=action,
                rtgs=rtg,  # remove last rtg
                timesteps=ti.unsqueeze(-1),
                mlp_learn=False,
            )

            if mode == 'state':
                state_preds = state_preds[:, :-1] # choose all of first index and choose index from start to before end
                state_preds = rearrange(state_preds, "b t s -> (b t) s") # 128, 4, 7, 7, 20
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)
                loss = nn.MSELoss()(state_preds, s_exp)

            elif mode == 'action':
                action_preds = action_preds[:, :-1]
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)
                mask = a_exp != env.action_space.n
                loss = loss_fn(action_preds[mask], a_exp[mask])

            elif mode == 'rtg':
                reward_preds = reward_preds[:, :-1]
                r = r[:, 1:] # 128, 100, 1
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s") # 12800, 1
                r_exp = rearrange(r.squeeze(-1), "b t -> (b t)").to(t.float32)
                loss = nn.MSELoss()(reward_preds.squeeze(-1), r_exp)
           
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if offline_config.track:
                wandb.log({
                    "train/loss": loss.item(),
                })

        # at test frequency
            
        representative_dataset = list(trajectory_data_set.values())[0]
        eval_env_config = EnvironmentConfig(
            env_id=env.spec.id,
            capture_video=True,
            max_steps=min(
                model.environment_config.max_steps,
                offline_config.eval_max_time_steps,
            ),
            fully_observed=False,
            one_hot_obs=(representative_dataset.observation_type == "one_hot"),
            view_size=env.observation_space["image"].shape[0]
            if "image" in list(env.observation_space.keys())
            else 7,
        )

        eval_env_func = make_env(
            config=eval_env_config,
            seed=epoch,
            idx=0,
            run_name=f"dt_eval_videos_{epoch}",
        )

        if epoch % offline_config.eval_frequency == 0:
            for rtg in offline_config.initial_rtg:
                evaluate_dt_agent(
                    env_id=env.spec.id,
                    model=model,
                    env_func=eval_env_func,
                    trajectories=offline_config.eval_episodes,
                    track=offline_config.track,
                    # batch_number=batch_number,
                    initial_rtg=float(rtg),
                    device=device,
                    num_envs=offline_config.eval_num_envs,
                )

    # MLP training start
    # Step 2: Freeze all except MLP layers (penultimate_layer, output_layer)
    print("\n🔒 Freezing all layers except MLP (penultimate_layer, output_layer)")
    for name, param in model.named_parameters():
        if "penultimate_layer" in name or "output_layer" in name:
            param.requires_grad = True
            print(f"✅ {name} will be updated.")
        else:
            param.requires_grad = False
            print(f"❌ {name} is frozen.")

    # 새로운 옵티마이저와 스케줄러 설정
    optimizer = configure_optimizers(model, offline_config)
    scheduler = get_scheduler(
        offline_config.scheduler,
        optimizer,
        training_steps=offline_config.mlp_train_epochs * len(train_dataloader)
    )

    # MLP만을 위한 추가 학습 루프
    pbar_mlp = tqdm(range(offline_config.mlp_train_epochs), desc="MLP Fine-Tuning")
    for epoch in pbar_mlp:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(train_dataloader):
            model.train()
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None

            state_preds, action_preds, reward_preds, task_preds, _ = model(
                states=s,
                actions=action,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                mlp_learn=True,
            )

            # task classification loss만 사용
            task_labels = task_id.to(task_preds.device)
            task_loss = model.label_smoothing_loss(task_preds, task_labels, class_weights=weight_tensor)

            task_pred = t.argmax(task_preds, dim=-1)
            n_correct = (task_pred == task_labels).sum().item()
            n_total = task_labels.shape[0]
            task_accuracy = n_correct / n_total

            # task accuracy
            task_correct = {}
            task_total = {}
            for pred, true in zip(task_pred.cpu(), task_labels.cpu()):
                true = int(true)
                if true not in task_correct:
                    task_correct[true] = 0
                    task_total[true] = 0
                task_correct[true] += int(pred == true)
                task_total[true] += 1

            if offline_config.track:
                wandb.log({
                    "train/MLP_loss": task_loss.item(),
                    "train/MLP_accuracy": task_accuracy,
                })
                for tid in sorted(task_correct.keys()):
                    acc = task_correct[tid] / task_total[tid]
                    wandb.log({f"train/MLP_task{tid}_accuracy": acc})

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar_mlp.set_description(f"MLP Fine-Tuning: task_loss = {task_loss.item():.4f}")

    result = test(
        model=model,
        dataloader=test_dataloader,
        env=env,
        epochs=offline_config.test_epochs,
        track=offline_config.track,
        # batch_number=batch_number,
        mode = mode,
        class_weights=weight_tensor,
        device=device
    )
    return result


@pytest.mark.skip(reason="This is not a test")
def test(
    model: TrajectoryTransformer,
    dataloader: DataLoader,
    env,
    epochs=10,
    track=False,
    batch_number=0,
    mode="rtg",
    class_weights=None,
    device="cpu",
):

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    # Initialize DreamerV3 and PER buffer for test function
    try:
        # Create proper spaces for DreamerV3 using elements.Space
        obs_space = {
            'observation': elements.Space(np.uint8, (64, 64, 3)),  # Standard DreamerV3 size
            'reward': elements.Space(np.float32, ()),
            'is_first': elements.Space(bool, ()),
            'is_last': elements.Space(bool, ()),
            'is_terminal': elements.Space(bool, ()),
        }
        act_space = {
            'action': elements.Space(np.int32, (), 0, 6)
        }
        
        dreamer_wrapper = DreamerV3Wrapper(obs_space, act_space)
        per_buffer = PERBuffer(capacity=10000)
        
    except Exception as e:
        print(f"⚠️ Failed to initialize DreamerV3 for testing: {e}")
        import traceback
        traceback.print_exc()
        dreamer_wrapper = None
        per_buffer = PERBuffer(capacity=10000)

    main_loss = 0
    main_total = 0
    main_correct = 0

    task_loss = 0
    n_task_correct = 0
    n_task_total = 0
    has_task_labels = False
    pre_task = 0

    all_task_preds = []
    all_task_labels = []
    all_embeddings = []
    task_correct = {}
    task_total = {}

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(dataloader):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            with t.no_grad():
                state_preds, action_preds, reward_preds, task_preds, penultimate_out = model(
                    states=s,
                    actions=action,
                    rtgs=rtg,
                    timesteps=ti.unsqueeze(-1),
                    mlp_learn=True,
                )

            if mode == "state":
                state_preds = state_preds[:, :-1]
                state_preds = rearrange(state_preds, "b t s -> (b t) s")
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)

                main_loss += nn.MSELoss()(state_preds, s_exp).item()
                main_total += s_exp.shape[0]

            elif mode == "action":
                action_preds = action_preds[:, :-1]
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)

                mask = a_exp != env.action_space.n
                action_preds = action_preds[mask]
                a_exp = a_exp[mask]
                a_hat = t.argmax(action_preds, dim=-1)

                main_loss += loss_fn(action_preds, a_exp).item()
                main_total += a_exp.shape[0]
                main_correct += (a_hat == a_exp).sum().item()

            elif mode == "rtg":
                reward_preds = reward_preds[:, :-1]
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s")
                r_exp = rearrange(r[:, 1:].squeeze(-1), "b t -> (b t)").to(t.float32)

                main_loss += nn.MSELoss()(reward_preds.squeeze(-1), r_exp).item()
                main_total += r_exp.shape[0]

            embeddings = penultimate_out  # shape: (B, D)
            all_embeddings.append(embeddings.cpu())

            task_ids_expanded = match_task_ids(task_id, embeddings)
            all_task_labels.extend(task_ids_expanded.cpu().tolist())

            # ✅ Task classification evaluation
            if task_id is not None:
                has_task_labels = True
                task_shift_detected = False
                task_id = task_id.to(task_preds.device)
                task_loss += model.label_smoothing_loss(task_preds, task_id, class_weights=class_weights).item()
                
                task_pred = t.argmax(task_preds, dim=-1)

                # Check Task Changing
                recent_k = s.shape[0] // 16 # recent_k = 8

                # (1) recent task predictions
                recent_task_preds = task_preds[-recent_k:]  # (recent_k, num_tasks)
                # softmax over task logits
                task_probs = t.softmax(recent_task_preds, dim=-1)  # (B, num_tasks)
                mean_task_probs = task_probs.mean(dim=0)  # (num_tasks,) = e.g., (3,)

                max_prob, current_task = t.max(mean_task_probs, dim=-1)  # current_task: int (0~2)

                if pre_task == current_task:
                    task_shift_detected = False
                    
                    # Task가 shifted되지 않았으면 imagine trajectory로 학습
                    if dreamer_wrapper is not None:
                        print("📚 Using imagine trajectory for learning (no task shift)")
                        # Imagine trajectory learning 구현
                        # 여기서는 기존 DT 학습을 계속 수행
                        
                else:
                    task_shift_detected = True
                    print("🚨 Task shift likely detected: previous =", pre_task, ", current =", current_task)
                    
                    # Task가 shifted되면 DreamerV3로 dynamics + behavior learning
                    if dreamer_wrapper is not None:                       
                        # 새로운 배치 데이터 준비
                        current_batch = {
                            'states': s,
                            'actions': a,
                            'rewards': r if 'r' in locals() else None,
                            'timesteps': t if 't' in locals() else None,
                            'task_id': task_id
                        }
                        
                        try:
                            # 1. Dynamics learning
                            carry, outputs, metrics = dreamer_wrapper.dynamics_learning(current_batch)
                            
                            # 2. Behavior learning  
                            carry = dreamer_wrapper.behavior_learning(carry)
                            
                        except Exception as e:
                            print(f"⚠️ DreamerV3 learning failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Debug: Check config structure
                            if hasattr(dreamer_wrapper, 'agent') and hasattr(dreamer_wrapper.agent, 'config'):
                                config = dreamer_wrapper.agent.config
                                print(f"🔍 Config type: {type(config)}")
                                print(f"🔍 Config has seed: {hasattr(config, 'seed')}")
                                if hasattr(config, 'seed'):
                                    print(f"🔍 Seed value: {config.seed}")
                                print(f"🔍 Config keys: {dir(config) if hasattr(config, '__dict__') else 'No __dict__'}")
                                if hasattr(config, '__dict__'):
                                    print(f"🔍 Config dict: {config.__dict__}")
                                elif hasattr(config, '_data'):
                                    print(f"🔍 Config _data: {config._data}")
                            else:
                                print("🔍 No config found in dreamer_wrapper.agent")

                    # PER buffer에 experience와 embedding 저장
                    if per_buffer is not None and 'embeddings' in locals():
                        # embedding similarity 기반 priority 계산
                        priority = per_buffer.compute_similarity_priority(embeddings)
                        
                        # experience와 embedding을 버퍼에 저장
                        experience = {
                            'states': s.cpu(),
                            'actions': a.cpu() if a is not None else None,
                            'rewards': r.cpu() if 'r' in locals() else None,
                            'task_id': task_id.cpu(),
                            'task_preds': task_preds.cpu(),
                            'embeddings': embeddings.cpu()
                        }
                        
                        per_buffer.add(experience, embeddings.cpu(), priority)

                pre_task = current_task

                # evaluate real changing
                task_id_list = task_id.cpu().tolist()
                real_task_change = any(t != task_id_list[0] for t in task_id_list[1:])
                correct_detection = int(task_shift_detected == real_task_change)

                wandb.log({
                    "eval/task_shift_detected": int(task_shift_detected),
                    "eval/current_task": current_task,
                    "eval/task_changed_actual": int(real_task_change),
                    "eval/task_shift_correct": correct_detection,
                    "eval/task_id_distribution": wandb.Histogram(task_id.cpu().numpy())
                })

                n_task_correct += (task_pred == task_id).sum().item()
                n_task_total += task_id.shape[0]

                # 분포 기록
                all_task_preds.extend(task_pred.cpu().tolist())

                # 개별 task 정확도 기록
                for pred, true in zip(task_pred.cpu(), task_id.cpu()):
                    true = int(true)
                    if true not in task_correct:
                        task_correct[true] = 0
                        task_total[true] = 0
                    task_correct[true] += int(pred == true)
                    task_total[true] += 1
                    
    mean_main_loss = main_loss / (epochs * test_batches_per_epoch)
    main_accuracy = main_correct / main_total if mode == "action" else None

    mean_task_loss = task_loss / (epochs * test_batches_per_epoch) if has_task_labels else None
    task_accuracy = n_task_correct / n_task_total if has_task_labels else None

    # ✅ Print logs
    print(f"\n==== [Test Summary] ====")
    print(f"{mode} loss: {mean_main_loss:.4f}")
    if mode == "action":
        print(f"{mode} accuracy: {main_accuracy:.4f}")
    if has_task_labels:
        print(f"Task loss: {mean_task_loss:.4f}")
        print(f"Task accuracy: {task_accuracy:.4f}")
        for tid in sorted(task_correct.keys()):
            acc = task_correct[tid] / task_total[tid]
            print(f"- Task {tid}: {acc:.4f} ({task_correct[tid]}/{task_total[tid]})")

    # wandb 로그
    wandb.log({f"test/{mode}_loss": mean_main_loss}, step=batch_number)
    if mode == "action":
        wandb.log({f"test/{mode}_accuracy": main_accuracy}, step=batch_number)

    if has_task_labels:
        wandb.log({
            "test/task_loss": mean_task_loss,
            "test/task_accuracy": task_accuracy,
            "test/task_pred_distribution": wandb.Histogram(all_task_preds),
            "test/task_true_distribution": wandb.Histogram(all_task_labels),
        }, step=batch_number)

        for task_id in task_total:
            acc = task_correct[task_id] / task_total[task_id]
            wandb.log({f"test/task{task_id}_accuracy": acc}, step=batch_number)



    all_embeddings_tensor = t.cat(all_embeddings, dim=0)  # ¡æ (N, D)
    all_task_labels_tensor = t.tensor(all_task_labels)
    print("all_task_labels: ", len(all_task_labels))
    print("all_task_labels_tensor: ", all_task_labels_tensor.shape)

    return {
        "model": model,
        "embeddings": all_embeddings_tensor,  # (NEW)
        "task_ids": all_task_labels_tensor,
    }


import torch as t
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import random

class MultiTaskBatchDataset(t.utils.data.IterableDataset):
    def __init__(self, task_datasets: dict, batch_size: int):
        super().__init__()
        self.task_datasets = {
            int(k.replace("task_", "")) if isinstance(k, str) and "task_" in k else int(k): v
            for k, v in task_datasets.items()
        }
        self.batch_size = batch_size
        self.task_ids = list(self.task_datasets.keys())
        self.num_batches = sum(len(ds) // self.batch_size for ds in self.task_datasets.values())

    def __iter__(self):
        loaders = {
            task_id: iter(DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            ))
            for task_id, ds in self.task_datasets.items()
        }

        for _ in range(self.num_batches):
            task_id = random.choice(self.task_ids)
            try:
                batch = next(loaders[task_id])
            except StopIteration:

                loaders[task_id] = iter(DataLoader(
                    self.task_datasets[task_id],
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True
                ))
                batch = next(loaders[task_id])
            yield (*batch, t.full((self.batch_size,), task_id))

    def __len__(self):
        return self.num_batches



def get_dataloaders(trajectory_data_set, offline_config):
    """
    trajectory_data_set: Dict[int, Dataset]
    """

    if isinstance(trajectory_data_set, ConcatDataset):
        raise ValueError("ConcatDataset Dict[int, Dataset] ")
    else:
        dataset_list = list(trajectory_data_set.values())

    task_counts = {}
    total = 0
    for task_id, dataset in trajectory_data_set.items():
        count = len(dataset)
        task_counts[task_id] = count
        total += count

    # train/test
    train_datasets = {}
    test_datasets = {}
    for task_id, dataset in trajectory_data_set.items():
        count = len(dataset)
        train_size = int(0.7 * count)
        test_size = count - train_size

        seed = 42 + (hash(task_id) % 1000)

        train_subset, test_subset = random_split(
            dataset,
            [train_size, test_size],
            generator=t.Generator().manual_seed(seed)
        )
        train_datasets[task_id] = train_subset
        test_datasets[task_id] = test_subset

    print(f"total train_datasets: {sum(len(v) for v in train_datasets.values())}")
    print(f"total test_datasets: {sum(len(v) for v in test_datasets.values())}")

    # DataLoader
    train_dataset = MultiTaskBatchDataset(train_datasets, batch_size=offline_config.batch_size)
    test_dataset = MultiTaskBatchDataset(test_datasets, batch_size=offline_config.batch_size)

    train_dataloader = DataLoader(train_dataset, batch_size=None)
    test_dataloader = DataLoader(test_dataset, batch_size=None)

    return train_dataloader, test_dataloader

def match_task_ids(task_id, embedding_tensor):
    N = embedding_tensor.shape[0]
    B = task_id.shape[0]
    repeats = N // B
    expanded = task_id.repeat_interleave(repeats)

    if expanded.shape[0] > N:
        expanded = expanded[:N]
    elif expanded.shape[0] < N:
        print(f"?? Padding task_ids with last label. ({expanded.shape[0]} ¡æ {N})")
        pad = t.full((N - expanded.shape[0],), expanded[-1], dtype=expanded.dtype)
        expanded = t.cat([expanded, pad])
    return expanded