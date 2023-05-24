import sys

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from csmai import VAE, TiffDataset, vae_loss


def main(argv):
    if len(argv) != 1:
        print("Directory")
        return

    directory = argv[0]

    tiff_data = TiffDataset(directory)

    train_dataloader = DataLoader(tiff_data, batch_size=32, shuffle=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = 1e-4
    num_epochs = 10

    # Define transformations
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create VAE model
    model = VAE(tiff_data[0].shape[0], tiff_data[0].shape[1], 20).to(device)
    print(model)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Progress bar
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )

        for _batch_idx, images in enumerate(progress_bar):
            images = images.to(device)
            # Forward pass
            recon_images, mu, logvar = model(images)

            # Compute loss
            loss = vae_loss(recon_images, images, mu, logvar)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar description
            progress_bar.set_postfix({"Loss": loss.item()})

            # Update total loss
            total_loss += loss.item()

        # Compute average loss
        avg_loss = total_loss / len(tiff_data)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.4f}")

        # # Save reconstructed images
        # if (epoch + 1) % 5 == 0:
        #     with torch.no_grad():
        #         sample = torch.randn(64, model.latent_dim).to(device)
        #         reconstructed_images = model.decode(sample).cpu()
        #         save_image(reconstructed_images, f"reconstructed_images_{epoch+1}.png", nrow=8, normalize=True)

    # Save the trained model
    torch.save(model.state_dict(), "vae_model.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
