import gzip
import pickle

def inspect_trajectory_metadata(path):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    print("task_id in metadata:", data.get("metadata", {}).get("task_id", "MISSING"))
    return data.get("metadata", {})

# 사용 예시
metadata = inspect_trajectory_metadata("./trajectories/b5b9802e-5aeb-46e5-aa16-9e24828c352c.gz")
