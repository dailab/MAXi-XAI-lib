import numpy as np
import matplotlib.pyplot as plt

from maxi import *
from maxi.utils.result_processing.res2graydenoised import (
    transform_res2graydenoised,
)


def main():
    result = np.load("/home/tuananhroman/DAI/xai/run_22/2021-10-20 13:30:47.922852/iter_100/raw_result.npy")
    proc_result = transform_res2graydenoised(
        result,
        in_range=(-0.5, 0.5),
        out_range=(0, 1),
        threshold_factor=1.75,
        gaussian_filter_sigma=0.75,
    )
    plt.imshow(proc_result, cmap="gray")

    proc_result


if __name__ == "__main__":
    main()
