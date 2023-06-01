import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from urllib.request import urlopen
from PIL import Image
from torchvision import datasets, models

import maxi


def unnormalize(img):
    x = img.copy()
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    x = x[:, :, ::-1]
    return x


def visualize_from_async(results):
    for k, (savepoints, meta_data) in enumerate(results):
        data_idx = meta_data["data_index"]
        f, axarr = plt.subplots(1, len(savepoints))

        f.tight_layout()
        f.suptitle(f"Data with index: {data_idx}", fontsize=14, fontweight="bold")
        for i, (iter, result) in enumerate(savepoints.items()):
            axarr[i].title.set_text(f"Iteration: {iter}")
            # axarr[i].imshow((result + inputs[k]).squeeze(axis=-1).squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)
            # axarr[i].imshow((result).squeeze(axis=0), cmap="gray", vmin=-0.5, vmax=0.5)
            res = np.array(unnormalize(result.squeeze(axis=0)), dtype=np.int32)
            axarr[i].imshow(res, vmin=0, vmax=255)


def explain(image, inference_call):
    loss_class = maxi.TF_CEMLoss
    optimizer_class = maxi.AdaExpGradOptimizer
    gradient_class = maxi.TF_Gradient

    loss_kwargs = {"mode": "PP", "c": 10, "gamma": 3, "K": 20}
    optimizer_kwargs = {"l1": 0.025, "l2": 0.00025, "eta": 1.0, "channels_first": False}
    gradient_kwargs = {"mu": None}

    cem = maxi.ExplanationGenerator(
        loss=loss_class,
        optimizer=optimizer_class,
        gradient=gradient_class,
        num_iter=1000,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        gradient_kwargs=gradient_kwargs,
        save_freq=250,
        verbose=True,
    )

    async_cem = maxi.AsyncExplanationWrapper(cem, n_workers=4)

    inputs = [image]
    meta_data = {inputs[0].tobytes(): {"data_index": 10}}
    inference_calls = {img.tobytes(): inference_call for img in inputs}

    return async_cem.run(inputs, inference_calls, meta_data)


def main():
    model = ResNet50(weights="imagenet")

    # img_path = "https://raw.githubusercontent.com/larq/zoo/master/tests/fixtures/elephant.jpg"
    img_path = "https://repository-images.githubusercontent.com/296744635/39ba6700-082d-11eb-98b8-cb29fb7369c0"
    with urlopen(img_path) as f:
        img = Image.open(f)
        img = img.resize((224, 224))

    x = image.img_to_array(img)
    x = preprocess_input(x, data_format="channels_last")
    # plt.imshow(np.array(un_x, np.int32))
    x = np.expand_dims(x, axis=0)

    # plt.imshow(x.squeeze(axis=0), vmin=np.min(x), vmax=np.max(x), interpolation="nearest")

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("Predicted:", decode_predictions(preds, top=3)[0])

    results = explain(
        x,
        maxi.InferenceWrapper(
            inference_model=model,
        ),
    )

    visualize_from_async(results)


if __name__ == "__main__":
    main()
