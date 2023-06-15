import sys

# sys.path.append("../../../mnist/src")
sys.path.append("/home/tuananhroman/dai/MAXi-XAI-lib/experiments/mnist/src")


import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from official_mnist_tf.encoder_model import Encoder, Decoder

LATENT_DIM = 8


# Data Preprocessing
def get_data_loaders():
    data_dir = "dataset"

    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m = len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, valid_loader, test_loader, train_dataset, test_dataset


def init_loss_optimizer():
    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    lr = 0.001

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    d = LATENT_DIM

    # model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)
    params_to_optimize = [
        {"params": encoder.parameters()},
        {"params": decoder.parameters()},
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}")

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    return loss_fn, optim, encoder, decoder, device


# Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for (
        image_batch,
        _,
    ) in (
        dataloader
    ):  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print("\t partial train loss (single batch): %f" % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, device, dataloader: DataLoader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch: torch.Tensor = image_batch.unsqueeze(1).to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(encoder, decoder, test_dataset, device, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Original images")
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title("Reconstructed images")
    plt.show()


def main():
    (
        train_loader,
        valid_loader,
        test_loader,
        train_dataset,
        test_dataset,
    ) = get_data_loaders()
    loss_fn, optim, encoder, decoder, device = init_loss_optimizer()

    num_epochs = 40
    diz_loss = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(encoder, decoder, device, test_dataset, loss_fn)
        print(
            f"\n EPOCH {epoch + 1}/{num_epochs} \t train loss {train_loss} \t val loss {val_loss}"
        )
        diz_loss["train_loss"].append(train_loss)
        diz_loss["val_loss"].append(val_loss)
        plot_ae_outputs(encoder, decoder, test_dataset, device, n=10)

    ENC_PATH = f"encoder_latent_dim_{LATENT_DIM}.pt"
    DEC_PATH = f"decoder_latent_dim_{LATENT_DIM}.pt"
    torch.save(encoder, ENC_PATH)
    torch.save(decoder, DEC_PATH)


if __name__ == "__main__":
    main()
