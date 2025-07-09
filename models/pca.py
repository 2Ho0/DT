import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load checkpoint
checkpoint = torch.load("rtg.pt", map_location='cpu')
embeddings = checkpoint["eval_embeddings"].numpy()
task_ids = checkpoint["eval_task_ids"].numpy().astype(int)

print("embeddings: ", embeddings.shape)
assert embeddings.shape[0] == task_ids.shape[0]

# Count task samples
unique, counts = np.unique(task_ids, return_counts=True)
min_count = min(counts)  # °¡Àå ÀûÀº task¿¡ ¸ÂÃç ±Õµî »ùÇÃ¸µ
print("Task sample counts:", dict(zip(unique, counts)))
print(f"Sampling {min_count} samples per task...")

# Sample equal number per task
sampled_embeddings = []
sampled_task_ids = []
for tid in unique:
    mask = task_ids == tid
    indices = np.where(mask)[0]
    sampled_idx = np.random.choice(indices, size=min_count, replace=False)
    sampled_embeddings.append(embeddings[sampled_idx])
    sampled_task_ids.append(np.full(min_count, tid))

# Stack sampled data
embeddings_sampled = np.concatenate(sampled_embeddings, axis=0)
task_ids_sampled = np.concatenate(sampled_task_ids, axis=0)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings_sampled)

# Define colors
color_map = {0: "red", 1: "green", 2: "blue"}
label_map = {0: "DoorKey", 1: "LavaCrossing", 2: "SimpleCrossing"}

# Plot
plt.figure(figsize=(20,14))
for tid in unique:
    mask = task_ids_sampled == tid
    plt.scatter(
        pca_result[mask, 0],
        pca_result[mask, 1],
        c=color_map[tid],
        label=label_map[tid],
        alpha=0.2,
        s=10
    )

plt.title("PCA of Rtg", fontsize=30)
plt.legend(markerscale=5, fontsize=30)

plt.grid(True)
plt.tight_layout()
plt.savefig("poster_rtg.png")
plt.show()
