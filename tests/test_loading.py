from __future__ import annotations

from nucli_train.data_management.dataset import Blosc2Dataset

from torch.utils.data import DataLoader
import yaml

import numpy as np

import matplotlib.pyplot as plt


dataset = Blosc2Dataset('/home/vicky/projects/nuclivision/DATA/save/')


dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

for i in range(10):
    batch = next(iter(dataloader))
    print(f"Batch {i+1}:")
    print("Input shape:", batch['input'].shape)
    if 'target' in batch:
        print("Target shape:", batch['target'].shape)
    print()  # Print a newline for better readability
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(np.max(batch['input'][0, 0].numpy(), 2), cmap='gray')
    axs[1].imshow(np.max(batch['target'][0, 0].numpy(), 2), cmap='gray')
    axs[2].imshow(np.max(np.abs(batch['input'][0, 0].numpy() - batch['target'][0, 0].numpy()), 2), cmap='gray')
    plt.show()
    plt.close()
