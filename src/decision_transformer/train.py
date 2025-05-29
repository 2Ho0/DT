import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from dataclasses import asdict
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import numpy as np

import wandb
from src.config import EnvironmentConfig, OfflineTrainConfig
from src.models.trajectory_transformer import (

    DecisionTransformer,
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .eval import evaluate_dt_agent
from .utils import configure_optimizers, get_scheduler
from torch.utils.data import ConcatDataset

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

    # Check gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Will update: {name}")


    # 각 태스크별 데이터셋 수 출력
    print("\n===== 태스크별 데이터셋 크기 =====")
    for task_id, dataset in trajectory_data_set.items():
        print(f"Task {task_id}: {len(dataset)} 샘플")

    # 🟢 모든 task의 dataset을 하나의 ConcatDataset으로 결합
    combined_dataset = ConcatDataset(list(trajectory_data_set.values()))
    train_dataloader, test_dataloader = get_dataloaders(
        combined_dataset, offline_config
    )

    # 첫 번째 배치를 추출하여 각 태스크의 비율 확인
    print("\n===== 첫 배치에서의 태스크 분포 확인 =====")
    first_batch = next(iter(train_dataloader))
    task_ids = first_batch[7].numpy()  # task_id는 8번째 항목
    unique_tasks, counts = np.unique(task_ids, return_counts=True)
    for task, count in zip(unique_tasks, counts):
        print(f"Task {task}: {count} 샘플 ({count/len(task_ids)*100:.2f}%)")
    
    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]

    # get total number of training steps.
    train_batches_per_epoch = len(train_dataloader)
    scheduler_config["training_steps"] = (
        offline_config.train_epochs * train_batches_per_epoch
    )
    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )
    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _) in enumerate(train_dataloader):
            total_batches = epoch * train_batches_per_epoch + batch

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
                print('s.shape:', s.shape)
                print('state_preds.shape:', state_preds.shape)
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)
                loss = nn.MSELoss()(state_preds, s_exp)

            elif mode == 'action':
                action_preds = action_preds[:, :-1]
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                print('a.shape:', a.shape)
                print('action_preds.shape:', action_preds.shape)
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)
                mask = a_exp != env.action_space.n
                loss = loss_fn(action_preds[mask], a_exp[mask])

            elif mode == 'rtg':
                reward_preds = reward_preds[:, :-1]
                r = r[:, 1:]
                print('r.shape:', r.shape) # 128, 100, 1
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s")
                print('reward_preds.shape:', reward_preds.shape) # 12800, 1
                r_exp = rearrange(r.squeeze(-1), "b t -> (b t)").to(t.float32)
                loss = nn.MSELoss()(reward_preds.squeeze(-1), r_exp)
            
            print("s[1:].shape (GT):", s[:, 1:].shape) # 128, 100, 7, 7, 20
            print("state_preds.shape (pred):", state_preds.shape) # 128, 101, 980

            print("r[1:].shape (GT):", r[:, 1:].shape) # 128, 99, 1
            print("reward_preds.shape (pred):", reward_preds.shape) # 12800, 1
           
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            #
            if offline_config.track:
                tokens_seen = (
                    (total_batches + 1)
                    * offline_config.batch_size
                    * model.transformer_config.n_ctx
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "train/loss": loss.item(),
                })

                wandb.log(
                    {"metrics/tokens_seen": tokens_seen}, step=total_batches
                )
                wandb.log(
                    {"metrics/learning_rate": learning_rate},
                    step=total_batches,
                )

        batch_number = epoch * train_batches_per_epoch
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
                    batch_number=batch_number,
                    initial_rtg=float(rtg),
                    device=device,
                    num_envs=offline_config.eval_num_envs,
                )

    # MLP training start!!
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
        for batch, (s, a, r, d, rtg, ti, m, task_id) in enumerate(train_dataloader):
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
            print('task_labels: ', task_labels, ", task_preds:", task_preds)
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

            # 🔍 여기서 gradient 확인 penultimate_0,1,3,5,6,8, output_0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"{name}: grad mean = {param.grad.abs().mean().item():.6f}")

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
        batch_number=batch_number,
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

    main_loss = 0
    main_total = 0
    main_correct = 0

    task_loss = 0
    n_task_correct = 0
    n_task_total = 0
    has_task_labels = False

    all_task_preds = []
    all_task_labels = []
    all_embeddings = []
    task_correct = {}
    task_total = {}

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, task_id) in enumerate(dataloader):
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

            if mode == "state":
                state_preds = state_preds[:, :-1]
                state_preds = rearrange(state_preds, "b t s -> (b t) s")
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)

                main_loss += nn.MSELoss()(state_preds, s_exp).item()
                main_total += s_exp.shape[0]
                # all_embeddings.append(state_preds.reshape(-1, state_preds.shape[-1]).cpu())
                #
                # task_ids_expanded = match_task_ids(task_id, state_preds)
                # all_task_labels.extend(task_ids_expanded.cpu().tolist())

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
                # reshaped = action_preds.reshape(-1, action_preds.shape[-1]).cpu()
                # all_embeddings.append(reshaped)
                #
                # task_ids_expanded = match_task_ids(task_id, reshaped)
                # all_task_labels.extend(task_ids_expanded.cpu().tolist())



            elif mode == "rtg":
                reward_preds = reward_preds[:, :-1]
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s")
                r_exp = rearrange(r[:, 1:].squeeze(-1), "b t -> (b t)").to(t.float32)

                main_loss += nn.MSELoss()(reward_preds.squeeze(-1), r_exp).item()
                main_total += r_exp.shape[0]
                # all_embeddings.append(reward_preds.reshape(-1, reward_preds.shape[-1]).cpu())
                #
                # task_ids_expanded = match_task_ids(task_id, reward_preds)
                # all_task_labels.extend(task_ids_expanded.cpu().tolist())

            embeddings = penultimate_out  # shape: (B, D)
            all_embeddings.append(embeddings.cpu())

            task_ids_expanded = match_task_ids(task_id, embeddings)
            all_task_labels.extend(task_ids_expanded.cpu().tolist())

            print("all_task_labels: ", len(all_task_labels))
            # ✅ Task classification evaluation
            if task_id is not None:
                has_task_labels = True
                task_id = task_id.to(task_preds.device)
                task_loss += model.label_smoothing_loss(task_preds, task_id, class_weights=class_weights).item()

                print('task_preds: ', task_preds)
                task_pred = t.argmax(task_preds, dim=-1)
                print(f"task_pred: {task_pred}, task_id: {task_id}")
                n_task_correct += (task_pred == task_id).sum().item()
                n_task_total += task_id.shape[0]

                # 분포 기록
                all_task_preds.extend(task_pred.cpu().tolist())
                # all_task_labels.extend(task_id.cpu().tolist())

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

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    if has_task_labels:
        true_labels = np.array(all_task_labels)
        pred_labels = np.array(all_task_preds)

        # (1) Raw confusion matrix
        cm_raw = confusion_matrix(true_labels, pred_labels, normalize=None)
        disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw)

        print("\n[Raw Confusion Matrix] (Counts)")
        print(cm_raw)

        plt.figure(figsize=(8, 6))
        disp_raw.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Task Confusion Matrix (Raw Count)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("confusion_matrix_raw.png")
        print("Saved raw confusion matrix to: confusion_matrix_raw.png")
        plt.close()

        # (2) Normalized confusion matrix
        cm_norm = confusion_matrix(true_labels, pred_labels, normalize='true')
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

        print("\n[Normalized Confusion Matrix] (Per-Row Proportions)")
        print(np.round(cm_norm, 2))  # ¼Ò¼öÁ¡ 2ÀÚ¸®·Î º¸±â ÁÁ°Ô

        plt.figure(figsize=(8, 6))
        disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.title("Task Confusion Matrix (Normalized)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("confusion_matrix_normalized.png")
        print("Saved normalized confusion matrix to: confusion_matrix_normalized.png")
        plt.close()

    all_embeddings_tensor = t.cat(all_embeddings, dim=0)  # ¡æ (N, D)
    all_task_labels_tensor = t.tensor(all_task_labels)
    print("all_task_labels: ", len(all_task_labels))
    print("all_task_labels_tensor: ", all_task_labels_tensor.shape)

    return {
        "model": model,
        "embeddings": all_embeddings_tensor,  # (NEW)
        "task_ids": all_task_labels_tensor,
    }


def get_dataloaders(trajectory_data_set, offline_config):
    """
    trajectory_data_set: torch.utils.data.ConcatDataset 또는 Dict[int, Dataset]
    task별 비율을 유지하며 train/test를 나눕니다.
    """

    # ✅ 각 태스크별 데이터셋 확인
    if isinstance(trajectory_data_set, ConcatDataset):
        dataset_list = trajectory_data_set.datasets
    else:
        dataset_list = list(trajectory_data_set.values())

    print("\n===== 원본 데이터셋 태스크별 분포 =====")
    task_counts = {}
    total = 0
    for task_id, dataset in enumerate(dataset_list):
        count = len(dataset)
        task_counts[task_id] = count
        total += count

    print(f"전체 데이터 수: {total}")
    for task_id, count in task_counts.items():
        percentage = (count / total) * 100
        print(f"Task {task_id}: {count} 샘플 ({percentage:.2f}%)")

    # ✅ task별로 split
    train_subsets = []
    test_subsets = []
    for task_id, dataset in enumerate(dataset_list):
        count = len(dataset)
        train_size = int(0.7 * count)
        test_size = count - train_size

        train_subset, test_subset = random_split(
            dataset,
            [train_size, test_size],
            generator=t.Generator().manual_seed(42 + task_id)
        )
        train_subsets.append(train_subset)
        test_subsets.append(test_subset)

    # ✅ task별 subset들을 합치기
    train_dataset = ConcatDataset(train_subsets)
    test_dataset = ConcatDataset(test_subsets)

    print(f"\n===== 학습/테스트 데이터 분할 (태스크 균형 유지) =====")
    print(f"학습 데이터: {len(train_dataset)} 샘플")
    print(f"테스트 데이터: {len(test_dataset)} 샘플")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=offline_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=offline_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


# def get_dataloaders(trajectory_data_set, offline_config):
#     """
#     trajectory_data_set: torch.utils.data.ConcatDataset 형태로 task별 dataset이 결합되어 있음.
#     WeightedRandomSampler는 사용하지 않고, shuffle=True 기반 학습 구조 사용.
#     """
#     # 원본 데이터셋에서 각 태스크별 데이터 수 출력
#     if isinstance(trajectory_data_set, ConcatDataset):
#         dataset_list = trajectory_data_set.datasets  # 내부에 합쳐진 데이터셋들 (task별)
        
#         task_counts = {}
#         total = 0
#         for task_id, dataset in enumerate(dataset_list):
#             count = len(dataset)
#             task_counts[task_id] = count
#             total += count

#         print(f"전체 데이터 수: {total}")
#         for task_id, count in task_counts.items():
#             percentage = (count / total) * 100
#             print(f"Task {task_id}: {count} 샘플 ({percentage:.2f}%)")
#     else:
#         print("\n===== 원본 데이터셋 태스크별 분포 =====")
#         task_counts = {task_id: len(dataset) for task_id, dataset in trajectory_data_set.items()}
#         total = sum(task_counts.values())
#         print(f"전체 데이터 수: {total}")
#         for task_id, count in task_counts.items():
#             percentage = (count / total) * 100
#             print(f"Task {task_id}: {count} 샘플 ({percentage:.2f}%)")
        
#         # ConcatDataset으로 변환
#         trajectory_data_set = ConcatDataset(list(trajectory_data_set.values()))

#     total = len(trajectory_data_set)
#     train_size = int(0.7 * total)
#     test_size = total - train_size

#     print(f"\n===== 학습/테스트 데이터 분할 =====")
#     print(f"학습 데이터: {train_size} 샘플 ({train_size/total*100:.2f}%)")
#     print(f"테스트 데이터: {test_size} 샘플 ({test_size/total*100:.2f}%)")

#     train_dataset, test_dataset = random_split(
#         trajectory_data_set,
#         [train_size, test_size],
#         generator=t.Generator().manual_seed(42)  # reproducibility
#     )

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=offline_config.batch_size,
#         shuffle=True,  # ✅ 학습용은 반드시 섞기
#         drop_last=True,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=offline_config.batch_size,
#         shuffle=True,  # ✅ 테스트 시에도 섞어서 다양한 task 조합 확인 가능
#         drop_last=False,
#     )

#     return train_dataloader, test_dataloader

