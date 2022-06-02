from matplotlib import pyplot as plt
import torch
import maxi
import numpy as np


def preprocess(data):
    return torch.tensor(data, dtype=torch.float32) if type(data) is not torch.Tensor else data


# def preprocess(data):
#     if type(data) is not np.ndarray:
#         return torch.tensor(data.numpy(), dtype=torch.float32) if type(data.numpy()) is not torch.Tensor else data
#     else:
#         return torch.tensor(data, dtype=torch.float32) if type(data) is not torch.Tensor else data


def explain(image, inference_call):
    loss_class = maxi.CEMLoss
    optimizer_class = maxi.AoExpGradOptimizer
    gradient_class = maxi.URVGradientEstimator

    loss_kwargs = {"mode": "PP", "c": 1, "gamma": 3, "K": 20}
    optimizer_kwargs = {"l1": 0.25, "l2": 0.0000025, "eta": 1.0, "channels_first": True}
    gradient_kwargs = {
        "mu": 1 / np.sqrt(500 * 28 * 28),
        "sample_num": 500,
        "batch_num": 1,
        "channels_first": True,
        "batch_mode": True,
    }

    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=500,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=250,
        verbose=True,
    )

    async_cem = maxi.AsyncExplanationWrapper(cem, n_workers=4)

    inputs = [image]
    meta_data = {inputs[0].tobytes(): {"data_index": 2953}}
    inference_calls = {img.tobytes(): inference_call for img in inputs}

    return async_cem.run(inputs, inference_calls, meta_data)


def visualize_from_async(results):
    for k, (savepoints, meta_data) in enumerate(results):
        data_idx = meta_data["data_index"]
        f, axarr = plt.subplots(1, len(savepoints))

        f.tight_layout()
        f.suptitle(f"Data with index: {data_idx}", fontsize=14, fontweight="bold")
        for i, (iter, result) in enumerate(savepoints.items()):
            axarr[i].title.set_text(f"Iteration: {iter}")
            # axarr[i].imshow((result + inputs[k]).squeeze(axis=-1).squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)
            axarr[i].imshow((result).squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)


def main():
    from torch.autograd import Variable
    from utee import selector

    from mnist_tf.utils import setup_mnist

    model_raw, ds_fetcher, is_imagenet = selector.select("mnist", cuda=False)
    ds_val = ds_fetcher(batch_size=10, train=False, val=True)

    data, model, AE = setup_mnist(
        model_path="/home/tuananhroman/dai/constrastive-explaination-prototype/experiments/mnist/models/mnist",
        ae_path="/home/tuananhroman/dai/constrastive-explaination-prototype/experiments/mnist/models/AE_codec",
    )

    _input = np.expand_dims(data.test_data[2953], axis=0)
    plt.imshow(_input.squeeze(axis=0).squeeze(axis=-1), cmap="gray", vmin=-0.5, vmax=0.5)
    output = model_raw(Variable(torch.FloatTensor(_input)))

    results = explain(
        _input,
        maxi.InferenceWrapper(
            inference_model=model_raw,
            # quantizer=lambda x: x.squeeze(axis=0),
            preprocess=preprocess,
        ),
    )

    visualize_from_async(results)

    # for idx, (data, target) in enumerate(ds_val):

    #     for d, t in zip(data[3:], target[3:]):
    #         data = Variable(torch.FloatTensor(d))
    #         output = model_raw(data)
    #         plt.imshow(data.numpy().squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)

    #         results = explain(
    #             data.numpy(),
    #             maxi.InferenceWrapper(
    #                 inference_model=model_raw,
    #                 quantizer=lambda x: x.squeeze(axis=0),
    #                 preprocess=preprocess,
    #             ),
    #         )

    #         visualize_from_async(results)


if __name__ == "__main__":
    main()
