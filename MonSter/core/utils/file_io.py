import os
import yaml
import shutil
import pickle
import numpy as np
from collections import namedtuple
from lz4.frame import decompress as lzdecompress

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    # save cfg into train_log
    ensure_dir(args.log_dir)
    dst_file = os.path.join(args.log_dir, file_path.split('/')[-1])
    shutil.copy2(file_path, dst_file)
    return args

def pklload(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def decompress(data, dtype):
    data = lzdecompress(data)
    data = np.frombuffer(data, dtype=dtype)
    return data

def load_pkl(pkl_path):
    data = pklload(pkl_path)
    return_data = {}
    for k, v in data.items():
        if k not in ["shape", "dtype"]:
            v = decompress(v, data["dtype"])
            v = v.reshape(data["shape"])
        return_data[k] = v
    return return_data