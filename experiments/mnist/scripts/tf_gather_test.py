import tensorflow as tf
import maxi
import numpy as np


def test_tf():
    test_prediction = tf.ones((100, 28, 28, 3))

    norm = tf.norm(tf.norm(test_prediction, axis=-1), axis=(-2, -1))
    norm


if __name__ == "__main__":
    test_tf()
