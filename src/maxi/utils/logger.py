import logging
import numpy as np
from scipy.optimize import OptimizeResult


def setup_logger(name: str) -> logging.Logger:
    """Logger setup.

    If you want to adjust the level, use
    > logger.setLevel(logging.DEBUG)
    where level is one of DEBUG, INFO, WARNING, ERROR or CRITICAL.
    """
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(filename="CEM.log")
    stderr_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)

    return logger


def setup_logging(log: logging.Logger, log_level: str):
    try:
        log_level = getattr(logging, log_level)
        log.setLevel(log_level)
    except:
        log.setLevel(logging.INFO)
    log.debug(f"Log level set to {log_level}.")


def _callback(res: OptimizeResult) -> None:
    # print(
    #     "iteration: ",
    #     res.nit,
    #     " overall loss: ",
    #     res.func,
    #     " attack loss: ",
    #     res.loss,
    #     " l1: ",
    #     res.l1,
    #     " l2: ",
    #     res.l2,
    # )
    print(
        "iteration: {:6} || overall_loss: {:10.4f} | attack_loss: {:6.4f} | l1: {:8.6f} | l2: {:8.6f}".format(
            res.nit, res.func, res.loss, res.l1, res.l2
        )
    )
