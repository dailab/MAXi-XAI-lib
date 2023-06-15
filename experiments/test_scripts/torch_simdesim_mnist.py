import sys

import matplotlib.pyplot as plt
import maxi
import torch
import numpy as np

sys.path.append("MAXi-XAI-lib/experiments/mnist/src")
from official_mnist_tf.train_mnist_model import init_model, load_mnist


def main():
    x_train, y_train, x_test, y_test = load_mnist()

    encoder = torch.load(
        "/home/tuananhroman/dai/MAXi-XAI-lib/experiments/mnist/models/encoder_latent_dim_8.pt"
    )
    decoder = torch.load(
        "/home/tuananhroman/dai/MAXi-XAI-lib/experiments/mnist/models/decoder_latent_dim_8.pt"
    )

    inputs = x_train[5].reshape(1, 28, 28)
    inputs = np.expand_dims(inputs, axis=0)

    inference = (
        lambda x: encoder(x)
        if torch.is_tensor(x)
        else encoder(torch.tensor(x, dtype=torch.float32, device="cuda"))
    )

    loss_class = maxi.loss.Torch_SimDesimLoss
    optimizer_class = maxi.optimizer.AdaExpGradOptimizer
    gradient_class = maxi.gradient.Torch_Gradient

    loss_kwargs = {"target_index": 1, "device": "cuda"}
    optimizer_kwargs = {"l1": 0.05, "l2": 0.005, "channels_first": False}
    gradient_kwargs = {"device": "cuda"}

    expl_gen = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        num_iter=100,  # number of optimization iterations
        save_freq=25,  # at which iterations the result should be saved
        verbose=True,  # print optimization metrics
    )

    results, _ = expl_gen.run(image=inputs, inference_call=inference)


if __name__ == "__main__":
    main()
