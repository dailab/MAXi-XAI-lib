from matplotlib import pyplot as plt
import torch
import maxi
import numpy as np
import cv2
import PIL

from PIL import Image
from urllib.request import urlopen
from keras.preprocessing import image
from keras.applications.resnet import decode_predictions
from torchvision import transforms

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(data):
    return torch.tensor(data, dtype=torch.float32) if type(data) is not torch.Tensor else data


def explain(image, inference_call):
    loss_class = maxi.Torch_CEMLoss
    optimizer_class = maxi.SpectralAoExpGradOptimizer
    gradient_class = maxi.Torch_Gradient

    # loss_class = maxi.CEMLoss
    # optimizer_class = maxi.SpectralAoExpGradOptimizer
    # gradient_class = maxi.URVGradientEstimator

    loss_kwargs = {"mode": "PP", "c": 1, "gamma": 1, "K": 10}
    optimizer_kwargs = {"l1": 0.25, "l2": 0.025, "eta": 1, "channels_first": True}
    gradient_kwargs = {"mu": 2.5, "num_iter": 250, "batch_mode": True, "channels_first": True}

    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=1000,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=500,
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

            if i == len(savepoints) - 1:
                np.save("torch_elephant.npy", result.squeeze(axis=0))

            tensor = torch.squeeze(torch.from_numpy(result), dim=0)
            # tensor = unnormalize(tensor)

            scaled = cv2.normalize(
                tensor.detach().numpy(),
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

            # axarr[i].imshow(tensor.permute(1, 2, 0), vmin=0, vmax=1)

            axarr[i].imshow(np.moveaxis(scaled, 0, -1), vmin=0, vmax=1)


unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def main():
    from torchvision import datasets, models

    def get_data() -> np.ndarray:
        # img_path = "https://repository-images.githubusercontent.com/296744635/39ba6700-082d-11eb-98b8-cb29fb7369c0"
        img_path = "https://raw.githubusercontent.com/larq/zoo/master/tests/fixtures/elephant.jpg"
        with urlopen(img_path) as f:
            return Image.open(f)

    def tf(img: PIL.Image) -> torch.Tensor:

        transform = transforms.Compose(
            [  # [1]
                transforms.Resize(256),  # [2]
                transforms.CenterCrop(224),  # [3]
                transforms.ToTensor(),  # [4]
                transforms.Normalize(mean=mean, std=std),  # [5]  # [6]  # [7]
            ]
        )

        return transform(img)

    resnet50 = models.quantization.resnet50(pretrained=True)
    # googlenet = models.quantization.googlenet(pretrained=True)

    # o = get_data()
    # t_norm = tf(o)
    # np_t_norm = t_norm.numpy()
    # t_unnorm = unnormalize(t_norm)
    # np_t_unnorm = t_unnorm.numpy()
    # plt.imshow(t_unnorm.permute(1, 2, 0), vmin=0, vmax=1)

    img = tf(get_data()).unsqueeze(axis=0)

    resnet50.eval()
    # googlenet.eval()
    out = resnet50(img)
    # out2 = googlenet(img)
    print("Predicted:", decode_predictions(out.detach().numpy(), top=5)[0])
    # print("Predicted:", decode_predictions(out2.detach().numpy(), top=5)[0])

    results = explain(
        img.detach().numpy(),
        maxi.InferenceWrapper(inference_model=resnet50, preprocess=preprocess),
    )

    visualize_from_async(results)


if __name__ == "__main__":
    main()
