import torch

data = torch.load("rtg.pt", map_location="cpu")

print(type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    for k in list(data.keys())[:3]:  # ÃÖ´ë 3°³ ¹Ì¸®º¸±â
        print(f"{k}: {type(data[k])}, shape/info: {getattr(data[k], 'shape', type(data[k]))}")
elif isinstance(data, list):
    print("First item type:", type(data[0]))
    print("First item content preview:", data[0])
