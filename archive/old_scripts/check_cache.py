import pickle
from pathlib import Path

cache_dir = Path("data/cache/tuab_preprocessed")
cache_files = list(cache_dir.glob("*.pkl"))
print(f"Found {len(cache_files)} cache files")

if cache_files:
    # Check first file
    with open(cache_files[0], 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nFirst file: {cache_files[0].name}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, tuple):
        print(f"Tuple length: {len(data)}")
        for i, item in enumerate(data):
            print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
    elif isinstance(data, dict):
        print(f"Dict keys: {list(data.keys())}")
    elif hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")