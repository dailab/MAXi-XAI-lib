from typing import Tuple
import numpy as np
import tensorflow as tf
import torch

from .general import to_numpy


def extract_prob(P: np.ndarray, t: int, non_target: bool) -> float:
    """Method to extract the targets predicted value or the next highest value

    Args:
        P (np.ndarray): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array
        non_target (bool): If the target value should be disregarded

    Raises:
        ValueError: Raised when logits matrix is of dimension larger than 2

    Returns:
        float: Predicted value for the non-/target
    """
    P = to_numpy(P)
    mask = np.ones(P.shape, dtype=bool)

    if P.ndim == 1:
        mask[t] = False
        if non_target:
            return np.max(P[mask])
        return P[t]
    elif P.ndim == 2:
        mask[0][t] = False
        if non_target:
            return np.max(P[0][mask[0]], axis=0)
        return P[:, t]
    else:
        raise ValueError("Got logits matrix of dimension larger than 2")


def np_extract_target_proba(P: np.ndarray, t: int) -> np.ndarray:
    """Numpy method to extract the target's probability

    Args:
        P (np.ndarray): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        np.ndarray: Prediction score for target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    P = to_numpy(P)

    if P.ndim == 1:
        return P[t]

    target_vector = np.zeros(len(P), dtype=np.float32)
    for i in range(len(P)):
        target_vector[i] = np_extract_target_proba(P[i], t)
    return target_vector


def np_extract_nontarget_proba(P: np.ndarray, t: int) -> np.ndarray:
    """Numpy method to extract the highest non-target's probability

    Args:
        P (np.ndarray): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        np.ndarray: Prediction score for hightest non-target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    P = to_numpy(P)

    if P.ndim == 1:
        mask = np.ones(P.shape, dtype=bool)
        mask[t] = False
        return np.max(P[mask])

    prob_vector = np.zeros(len(P), dtype=np.float32)
    for i in range(len(P)):
        prob_vector[i] = np_extract_nontarget_proba(P[i], t)
    return prob_vector


def tf_extract_target_proba(P: tf.Tensor, t: int) -> tf.Tensor:
    """Tensorflow method to extract the target's probability

    Args:
        P (tf.Tensor): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        tf.Tensor: Prediction score for target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    if P.ndim == 1:
        return P[t]

    return tf.squeeze(tf.gather(P, t, axis=1), axis=1)


def tf_extract_nontarget_proba(P: tf.Tensor, t: int) -> tf.Tensor:
    """Tensorflow method to extract the highest non-target's probability

    Args:
        P (tf.Tensor): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        tf.Tensor: Prediction score for hightest non-target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    if P.ndim == 1:
        for index in tf.math.top_k(P, 2).indices:
            if index != t:
                return P[index]

    mask = [True] * len(P[0])
    mask[int(t)] = False

    P_wo_target = tf.boolean_mask(P, mask, axis=1)
    return tf.squeeze(tf.math.top_k(P_wo_target, 1)[0], axis=1)


def torch_extract_prob(P: Tuple[torch.Tensor, np.ndarray], t: int, non_target: bool) -> float:
    """PyTorch method to extract the targets predicted value or the next highest value

    Args:
        P (Tuple[torch.Tensor, np.ndarray]): Classification result from model prediction
            in [num classes] or [bs, num classes]
        t (int): Target index in result array
        non_target (bool): If the target value should be disregarded

    Raises:
        ValueError: Raised when logits matrix is of dimension larger than 2

    Returns:
        float: Predicted value for the non-/target
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    if not non_target:
        return P[t] if P.ndim == 1 else P[0][t]

    for index in torch.topk(P, 2, dim=-1).indices:
        if index != t:
            return P[index] if P.ndim == 1 else P[0][index]


def torch_extract_target_proba(P: torch.Tensor, t: int) -> torch.Tensor:
    """PyTorch method to extract the target's probability

    Args:
        P (tf.Tensor): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        tf.Tensor: Prediction score for target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    if P.ndim == 1:
        return P[t]

    target_vector = torch.zeros(len(P), dtype=torch.float32)
    for i in range(len(P)):
        target_vector[i] = torch_extract_target_proba(P[i], t)
    return target_vector


def torch_extract_nontarget_proba(P: torch.Tensor, t: int) -> torch.Tensor:
    """PyTorch method to extract the highest non-target's probability

    Args:
        P (tf.Tensor): Classification result from model prediction in [num classes] or [bs, num classes]
        t (int): Target index in result array

    Raises:
        ValueError: Raised when prob matrix is of dimension larger than 2

    Returns:
        tf.Tensor: Prediction score for hightest non-target class
    """
    if P.ndim not in [1, 2]:
        raise ValueError("Got logits matrix of dimension larger than 2")

    if P.ndim == 1:
        for index in torch.topk(P, 2, dim=-1).indices:
            if index != t:
                return P[index]

    prob_vector = torch.zeros(len(P), dtype=torch.float32)
    for i in range(len(P)):
        prob_vector[i] = torch_extract_nontarget_proba(P[i], t)
    return prob_vector
