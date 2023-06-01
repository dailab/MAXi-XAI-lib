from matplotlib import pyplot as plt
import torch
import maxi
import numpy as np
import cv2


def preprocess(data):
    return torch.tensor(data, dtype=torch.float32) if type(data) is not torch.Tensor else data


def explain(image, inference_call):
    loss_class = maxi.Torch_CEMLoss
    optimizer_class = maxi.AdaExpGradOptimizer
    gradient_class = maxi.lib.computation_components.Torch_Gradient

    # loss_class = maxi.CEMLoss
    # optimizer_class = maxi.SpectralAoExpGradOptimizer
    # gradient_class = maxi.URVGradientEstimator

    loss_kwargs = {"mode": "PP", "c": 1, "gamma": 500, "K": 10}
    optimizer_kwargs = {"l1": 100, "l2": 0.0001, "eta": 1, "channels_first": True}
    gradient_kwargs = {"mu": None}

    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=500,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=100,
        verbose=True,
    )

    async_cem = maxi.AsyncExplanationWrapper(cem, n_workers=4)

    inputs = [image]
    meta_data = {inputs[0].tobytes(): {"data_index": 10}}
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

    model_raw, ds_fetcher, is_imagenet = selector.select("resnet50", cuda=False)
    ds_val = ds_fetcher(batch_size=10, train=False, val=True)

    for idx, (data, target) in enumerate(ds_val):

        for d, t in zip(data, target):
            data = Variable(torch.FloatTensor(d)).unsqueeze(dim=0)
            n_data = data.squeeze(axis=0).transpose(0, 2).transpose(0, 1).numpy()
            n_data = cv2.normalize(n_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            plt.imshow(np.array(n_data, dtype=np.int32), vmin=0, vmax=255)
            output = model_raw(data)

            results = explain(
                data.numpy(),
                maxi.InferenceWrapper(
                    inference_model=model_raw,
                    quantizer=lambda x: x.squeeze(axis=0),
                    preprocess=preprocess,
                ),
            )

            visualize_from_async(results)


if __name__ == "__main__":
    main()
