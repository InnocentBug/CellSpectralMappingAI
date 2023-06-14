import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from csmai import VAE, TiffDataset


def plot_projected_data(data):
    fig, ax = plt.subplots()

    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")

    ax.plot(data[:, 0], data[:, 1], ".")

    fig.savefig("project.pdf")
    plt.close(fig)


def main(argv):
    if len(argv) != 2:
        print("Data-Directory Model-path")
        return

    directory = argv[0]
    model_path = argv[1]

    tiff_data = TiffDataset(directory)
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(tiff_data[0].shape[0], tiff_data[0].shape[1], 2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    compressed_data = []
    for _i, data in enumerate(tqdm(tiff_data)):
        data = data.reshape((1,) + data.shape)
        mu, logvar = model.encode(data)
        z = mu  # model.reparameterize(mu, logvar)
        compressed_data.append(z[0].detach().numpy())

    compressed_data = np.asarray(compressed_data)
    z_projected = TSNE(2).fit_transform(compressed_data)
    plot_projected_data(z_projected)


if __name__ == "__main__":
    main(sys.argv[1:])
