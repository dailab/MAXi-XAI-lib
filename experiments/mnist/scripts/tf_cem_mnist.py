import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import maxi
from official_mnist_tf.train_mnist_model import load_mnist, init_model


def main():
    # load the mnist data, model
    x_train, y_train, x_test, y_test = load_mnist()
    model = init_model()
    model.load_weights("./experiments/mnist/models/tf/tf")
    
    _input = x_train[0].reshape(1, 28, 28)
    plt.imshow(_input.squeeze(axis=0), cmap="gray", vmin=0.0, vmax=1.0)
    
    # chose desired component classes for the loss, optimizer and gradient
    loss_class = maxi.loss.TF_CEMLoss
    optimizer_class = maxi.optimizer.AdaExpGradOptimizer
    gradient_class = maxi.gradient.TF_Gradient

    # specify the configuration for the components
    loss_kwargs = {"mode": "PN", "gamma": 0.0, "K": 2}
    optimizer_kwargs = {
        "l1": 0.005,
        "l2": 0.0005,
        "channels_first": False,
    }
    gradient_kwargs = {}

    # instantiate the "ExplanationGenerator" with our settings
    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=500,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=150,
        verbose=False,
    )
    
    # start the explanation procedure and retrieve the results
    results, _ = cem.run(image=_input, inference_call=model)

    # visualize the savepoints
    f, axarr = plt.subplots(1, len(results) + 1)
    for i, (iter_, result) in enumerate(results.items()):
        axarr[i].title.set_text("Iteration: " + iter_)
        axarr[i].imshow(
            result.squeeze(axis=0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        
        pred = model(_input + result)
        print(f"Iter: {iter_} || Prediction: {np.argmax(pred)} || Prediction Score: {np.max(pred, axis=1)}")

    axarr[-1].title.set_text("Orig + Perturbation")
    axarr[-1].imshow(
        (_input + result).squeeze(axis=0),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

if __name__ == '__main__':
    main()