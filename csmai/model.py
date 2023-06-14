import torch
import torch.nn as nn


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_channels, image_size, latent_dim):
        super(VAE, self).__init__()

        self.input_channels = input_channels
        self.image_size = image_size

        # Encoder layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the flattened size after convolutions
        conv_output_size = self.calculate_conv_output_size()

        # Latent space layers
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(conv_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, self.latent_dim)

        # Decoder layers
        self.decoder_fc = nn.Linear(self.latent_dim, conv_output_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, self.input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

    def calculate_conv_output_size(self):
        x = torch.randn(1, self.input_channels, self.image_size, self.image_size)
        x = self.encoder_conv(x)
        return x.view(x.size(0), -1).size(1)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 256, int(self.image_size / 8), int(self.image_size / 8))
        # x = x.view(x.size(0), 128, int(self.image_size / 4), int(self.image_size / 4))
        x = self.decoder_conv(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction="sum")(x_recon, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence
