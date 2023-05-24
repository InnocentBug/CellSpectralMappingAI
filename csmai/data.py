import os

import numpy as np
import tifffile
import torch
import torch.utils.data as data
from scipy import ndimage


class TiffDataset(data.Dataset):
    def __init__(self, directory=".", target_resolution=128):
        self.filenames = []
        self.target_resolution = target_resolution
        for filename in os.listdir(directory):
            if filename.endswith(".tif"):
                self.filenames.append(os.path.join(directory, filename))

        self.data = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if index >= len(self.filenames):
            raise StopIteration
        while index >= len(self.data):
            image = tifffile.imread(self.filenames[len(self.data)])
            image = image.astype(np.float32)
            scaled_image = ndimage.zoom(
                image,
                (
                    1,
                    self.target_resolution / image.shape[1],
                    self.target_resolution / image.shape[2],
                ),
                order=1,
            )
            self.data.append(torch.from_numpy(scaled_image))
        return self.data[index]

    @property
    def channel_size(self):
        return self[0].shape[0]

    @property
    def image_size(self):
        return self[0].shape[1]
