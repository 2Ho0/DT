import gzip
import pickle
import numpy as np
import os

import gzip
import pickle
import numpy as np
import os

import gzip
import pickle
import os
import numpy as np
from typing import List

def merge_trajectory_gz_files(input_paths: List[str], output_path: str):
    merged_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "truncated": [],
        "rtgs": [],
        "infos": [],
    }
    merged_metadata = {
        "args": None,
        "time": [],
    }

    for idx, path in enumerate(input_paths):
        with gzip.open(path, "rb") as f:
            content = pickle.load(f)
            data = content["data"]
            metadata = content["metadata"]

            for key in merged_data:
                items = data.get(key, [])
                if isinstance(items, np.ndarray):
                    items = items.tolist()
                merged_data[key].extend(items)

            if idx == 0:
                merged_metadata["args"] = metadata.get("args", None)
            merged_metadata["time"].append(metadata.get("time", 0.0))

    def to_array(name, data):
        try:
            if name == "actions":
                return np.array([int(a) if not isinstance(a, (list, np.ndarray)) else int(a[0]) for a in data],
                                dtype=np.int64)
            elif name in ["rewards", "rtgs"]:
                return np.array(data, dtype=np.float32)
            elif name in ["dones", "truncated"]:
                return np.array(data, dtype=bool)
            elif name == "observations":
                arr = np.array(data)
                if isinstance(arr[0], np.ndarray) and arr[0].dtype != object:
                    return np.stack(arr).astype(np.float32)
                return np.array(data, dtype=np.float32)
            elif name == "infos":
                return np.array(data, dtype=object)
            else:
                return np.array(data)
        except Exception as e:
            print(f"[Error converting {name}]: {e}")
            return np.array(data, dtype=object)

    final_data = {
        key: to_array(key, merged_data[key]) for key in merged_data
    }

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with gzip.open(output_path, "wb") as f:
        pickle.dump({"data": final_data, "metadata": merged_metadata}, f)

    print(f"? Merged {len(input_paths)} trajectory files into: {output_path}")

gz_paths = [
    "data/DoorKey.gz",
    "data/DoorKey1.gz",
    "data/DoorKey2.gz",
    # "data/DoorKey2734.gz"
]
output = "data/merge/door_merged.gz"
merge_trajectory_gz_files(gz_paths, output)