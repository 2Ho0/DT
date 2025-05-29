import torch
import matplotlib.pyplot as plt

checkpoint = torch.load("test_state.pt", map_location="cpu")
embeddings = checkpoint["eval_embeddings"].numpy()     # (N, D)
task_ids = checkpoint["eval_task_ids"].numpy().astype(int)  # (N,)

print("Embeddings shape:", embeddings.shape)
print("Task IDs shape:", task_ids.shape)

assert embeddings.shape[0] == task_ids.shape[0], "Mismatch!"

# Use the mean of each embedding vector for 1D scatter
embedding_values = embeddings.mean(axis=1)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    x=range(len(embedding_values)),
    y=embedding_values,
    c=task_ids,
    cmap="tab10",
    s=1,
    alpha=0.6
)
plt.title("1D Scatter of Embeddings (mean across dimensions)")
plt.xlabel("Sample Index")
plt.ylabel("Mean Embedding Value")
plt.colorbar(scatter, label="Task ID")
plt.grid(True)
plt.tight_layout()
plt.savefig("embedding_scatter_by_task.png")
plt.show()
