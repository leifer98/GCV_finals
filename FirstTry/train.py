import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from unet import UnetModel
from utils import compute_sdf_loss, dummy_inputs
from data import SlideTileDataset

def train_model(
    dataset_dir="./intermediate_data",
    batch_size=4,
    num_epochs=10,
    learning_rate=1e-3,
    tile_size=(256, 256),
    threshold=0.5,
    num_channels=32,
    num_pool_layers=4,
    drop_prob=0.1,
    device=None,
    checkpoint_dir="./checkpoints"
):
    # Set device
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    grid_dir = os.path.join(dataset_dir, "grids")
    thumb_dir = os.path.join(dataset_dir, "thumbs")
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=tile_size,
        threshold=threshold,
        use_thumbnails=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = UnetModel(
        in_chans=3,  # Assuming RGB input
        out_chans=1,  # Single-channel height map
        chans=num_channels,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob
    ).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            resized_tile = batch["resized_tile"].clone().detach().permute(0, 3, 1, 2).float().to(device)  # Convert to NCHW
            thresholded_heightmap = batch["thresholded_heightmap"].to(device)

            # Generate dummy normals and mask for loss calculation
            pred, normals, mask = dummy_inputs(
                batch_size=resized_tile.size(0),
                height=tile_size[0],
                width=tile_size[1]
            )
            pred = model(resized_tile)

            # Compute loss
            loss = compute_sdf_loss(pred, normals.to(device), mask.to(device))
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    train_model(
        dataset_dir="./intermediate_data",
        batch_size=4,
        num_epochs=10,
        learning_rate=1e-3,
        tile_size=(256, 256),
        threshold=0.5,
        num_channels=32,
        num_pool_layers=4,
        drop_prob=0.1,
        checkpoint_dir="./checkpoints"
    )

if __name__ == "__main__":
    main()
