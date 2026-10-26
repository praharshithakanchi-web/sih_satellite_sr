import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PairedSatelliteDataset(Dataset):
    """PyTorch Dataset loading paired LR (Sentinel-2 10m) and HR target patches."""

    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, "r") as hf:
            self.dataset_length = len(hf["lr_patches"])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_file_path, "r") as hf:
            lr_patch = hf["lr_patches"][idx]  # Shape: [H, W, C]
            hr_patch = hf["hr_patches"][idx]  # Shape: [H, W, C]

        # Convert [H, W, C] NumPy arrays to PyTorch tensor format: [C, H, W]
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float()

        return lr_tensor, hr_tensor


def get_paired_dataloader(
    h5_file_path: str, batch_size: int = 16, shuffle: bool = True
) -> DataLoader:
    """Returns a PyTorch DataLoader yielding (LR_Input, HR_Target) mini-batches."""
    dataset = PairedSatelliteDataset(h5_file_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return loader


if __name__ == "__main__":
    print("Testing Paired Dataloader with dummy sample data...")

    os.makedirs("data/sample", exist_ok=True)
    sample_h5 = "data/sample/worldstrat_paired_sample.h5"

    # Create a small dummy HDF5 file for local testing
    with h5py.File(sample_h5, "w") as hf:
        dummy_lr = np.random.rand(20, 128, 128, 4).astype(np.float32)
        dummy_hr = np.random.rand(20, 512, 512, 4).astype(np.float32)
        hf.create_dataset("lr_patches", data=dummy_lr)
        hf.create_dataset("hr_patches", data=dummy_hr)

    # Test the dataloader iteration
    loader = get_paired_dataloader(sample_h5, batch_size=4)
    for lr_batch, hr_batch in loader:
        print("✅ Dataloader Test Successful!")
        print(f"   LR Input Batch Shape : {lr_batch.shape} (Expected: [4, 4, 128, 128])")
        print(f"   HR Target Batch Shape: {hr_batch.shape} (Expected: [4, 4, 512, 512])")
        break