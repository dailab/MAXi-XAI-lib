from typing import Union

import glob
import re
import subprocess
from pathlib import Path


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def check_requirements(requirements="requirements.txt", exclude=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    import pkg_resources as pkg

    prefix = colorstr("red", "bold", "requirements:")
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            n += 1
            print(f"{prefix} {r} not found and is required by pancake, attempting auto-update...")
            print(subprocess.check_output(f"pip install '{r}'", shell=True).decode())

    if n:  # if packages updated
        source = file.resolve() if "file" in locals() else requirements
        s = (
            f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n"
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )


def increment_path(
    path: Union[str, Path] = Path("/xai/run"), exist_ok: bool = False, sep: str = "_", mkdir: bool = True
) -> Path:
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def check_epoch(iter_count: int, freq: int, max_iter: int) -> bool:
    return iter_count % freq == 0 or iter_count == max_iter or iter_count == 1


import numpy as np
import tensorflow as tf
import torch


def to_numpy(data: Union[np.ndarray, tf.Tensor, torch.Tensor]) -> np.ndarray:
    if type(data) in [np.ndarray, np.float32, np.float64, np.int32, np.int64, float, int]:
        return data
    elif type(data) is tf.Tensor:
        return data.numpy()
    elif type(data) is torch.Tensor:
        return data.detach().numpy()
    else:
        try:
            return data.numpy()
        except Exception as e:
            raise ValueError(f"Cannot convert data of type '{type(data)}' to a numpy array.") from e
