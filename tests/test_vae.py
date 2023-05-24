import torch

import csmai


def test_with_random_data():
    input_channels = 6
    image_size = 128
    latent_dim = 10
    model = csmai.VAE(
        input_channels=input_channels, image_size=image_size, latent_dim=latent_dim
    )

    # Example usage
    batch_size = 2
    input_tensor = torch.randn(batch_size, input_channels, image_size, image_size)
    output_tensor, mu, logvar = model(input_tensor)
    assert output_tensor.shape == torch.Size(
        (batch_size, input_channels, image_size, image_size)
    )
    assert mu.shape == torch.Size((batch_size, latent_dim))
    assert logvar.shape == torch.Size((batch_size, latent_dim))
