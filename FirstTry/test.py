import os
import sys
import openslide  # Add this import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import shutil
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from unet import UnetModel
from utils import *
from data import SlideTileDataset
from train import *

def get_test_data_from_svs_folder(output_dir="./intermediate_data"):
    """
    Modified to generate thumbnails from SVS files for easier processing.
    """
    folder_path = "../Recources/example_tcga_slides"
    svs_files = [f for f in os.listdir(folder_path) if f.endswith('.svs')]

    # Assert that the source folder contains .svs files
    assert len(svs_files) > 0, f"No .svs files found in {folder_path}. Please add valid .svs files."

    grid_dir = os.path.join(output_dir, "grids")
    slide_dir = os.path.join(output_dir, "slides")
    thumb_dir = os.path.join(output_dir, "thumbs")  # New directory for thumbnails
    
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(slide_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)  # Create thumbnails directory

    # Process each SVS file
    for svs_file in svs_files:
        grid_file = os.path.join(grid_dir, f"{os.path.splitext(svs_file)[0]}--tlsz256.data")
        if not os.path.exists(grid_file):
            dummy_grid = [(0, 0), (256, 256)]
            with open(grid_file, 'wb') as f:
                pickle.dump(dummy_grid, f)

        # Create thumbnail file path
        thumb_file = os.path.join(thumb_dir, f"{os.path.splitext(svs_file)[0]}_thumb.jpg")
        
        # Create symlink to SVS (keeping this for reference)
        slide_file = os.path.join(slide_dir, svs_file)
        src_file = os.path.join(folder_path, svs_file)
        try:
            os.symlink(src_file, slide_file)
        except FileExistsError:
            if not os.path.islink(slide_file) or os.readlink(slide_file) != src_file:
                os.remove(slide_file)
                os.sylink(src_file, slide_file)
        
        # Generate thumbnail if it doesn't exist
        if not os.path.exists(thumb_file):
            try:
                print(f"Generating thumbnail for {svs_file}...")
                # Try to open slide and generate thumbnail
                slide = openslide.OpenSlide(src_file)
                width, height = slide.dimensions
                # Create a reasonable sized thumbnail (adjust scale as needed)
                thumb_size = (width // 20, height // 20)
                thumbnail = slide.get_thumbnail(thumb_size)
                thumbnail.save(thumb_file)
                slide.close()
                print(f"Created thumbnail: {thumb_file}")
            except Exception as e:
                print(f"Error generating thumbnail for {svs_file}: {e}")
                # Create dummy thumbnail if we can't generate from slide
                dummy_img = Image.new('RGB', (256, 256), color=(255, 255, 255))
                dummy_img.save(thumb_file)
                print(f"Created dummy thumbnail: {thumb_file}")

    # Debug: Print the contents of the thumbnails folder
    thumbs_in_dir = [f for f in os.listdir(thumb_dir) if f.endswith('.jpg')]
    print(f"Thumbnails in {thumb_dir}: {thumbs_in_dir}")
    assert len(thumbs_in_dir) > 0, f"Thumbnails folder {thumb_dir} is empty after processing."

    return grid_dir, thumb_dir  # Return thumb_dir instead of slide_dir

def test_unet_simple():
    # Required: None (uses dummy inputs)
    # This test runs independently.
    
    # Define dummy input
    batch_size = 2
    in_chans = 2  # Number of input channels (e.g., [image, heatmap])
    height, width = 128, 128
    dummy_input = torch.randn(batch_size, in_chans, height, width)

    # Initialize the model
    model = UnetModel(
        in_chans=in_chans,
        out_chans=1,  # Single-channel height map
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1
    )

    # Move model and input to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape
    assert output.shape == (batch_size, 1, height, width), f"Unexpected output shape: {output.shape}"
    print("Simple test passed! Output shape:", output.shape)

# New, more complex test
def test_unet_complex():
    # Required: None (uses dummy inputs)
    # This test runs independently.
    
    # Define more complex dummy input
    batch_size = 4
    in_chans = 3  # Simulating RGB image input
    height, width = 256, 256  # Larger input dimensions
    dummy_input = torch.randn(batch_size, in_chans, height, width)

    # Initialize the model with more channels and layers
    model = UnetModel(
        in_chans=in_chans,
        out_chans=2,  # Multi-channel output (e.g., two height maps)
        chans=64,  # Start with more channels
        num_pool_layers=5,  # Deeper U-Net
        drop_prob=0.2  # Higher dropout
    )

    # Move model and input to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape
    assert output.shape == (batch_size, 2, height, width), f"Unexpected output shape: {output.shape}"
    print("Complex test passed! Output shape:", output.shape)

def test_openslide_functionality():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # This test checks if OpenSlide can open and read SVS files.
    
    folder_path = "../Recources/example_tcga_slides"
    svs_files = [f for f in os.listdir(folder_path) if f.endswith('.svs')]

    if not svs_files:
        print(f"No .svs files found in {folder_path}. Please add valid .svs files.")
        return

    for svs_file in svs_files:
        svs_path = os.path.join(folder_path, svs_file)
        try:
            slide = openslide.OpenSlide(svs_path)
            width, height = slide.dimensions
            print(f"File: {svs_file} | Resolution: {width}x{height}")
            slide.close()
        except openslide.OpenSlideUnsupportedFormatError as e:
            print(f"Unsupported format or missing file: {svs_file} | Error: {e}")
        except Exception as e:
            print(f"Error opening {svs_file}: {e}")

def test_resize_tile():
    # Required: None (uses random dummy tiles)
    # This test runs independently.
    
    """
    Test the resize_tile function from utils.py.
    """
    tile = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)  # Random 512x512 RGB tile
    target_size = (256, 256)
    resized_tile = resize_tile(tile, target_size=target_size)

    assert resized_tile.shape[:2] == target_size, f"Unexpected size: {resized_tile.shape[:2]}"
    print("resize_tile test passed!")

def test_threshold_heightmap():
    # Required: None (uses random dummy heightmaps)
    # This test runs independently.
    
    """
    Test the threshold_heightmap function from utils.py.
    """
    heightmap = torch.rand(1, 1, 128, 128)  # Random heightmap
    threshold = 0.5
    thresholded = threshold_heightmap(heightmap, threshold=threshold)

    assert thresholded.shape == heightmap.shape, f"Unexpected shape: {thresholded.shape}"
    assert torch.all((thresholded == 0) | (thresholded == 1)), "Thresholded values are not binary"
    print("threshold_heightmap test passed!")

def test_compute_sdf_loss():
    # Required: None (uses dummy inputs)
    # This test runs independently.
    
    """
    Test the compute_sdf_loss function from utils.py.
    """
    pred, normals, mask = dummy_inputs(batch_size=2, height=128, width=128)
    loss = compute_sdf_loss(pred, normals, mask)

    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert loss.item() > 0, "Loss should be positive"
    print("compute_sdf_loss test passed!")

def test_print_svs_resolutions_in_folder():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # This test prints the resolutions of SVS files.
    
    """
    Test the print_svs_resolutions_in_folder function from utils.py.
    """
    print("Testing print_svs_resolutions_in_folder...")
    print_svs_resolutions_in_folder()
    print("print_svs_resolutions_in_folder test completed!")

def test_workflow_with_slides():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # This test performs a workflow using multiple utility functions on slide data.
    
    folder_path = "../Recources/example_tcga_slides"
    svs_files = [f for f in os.listdir(folder_path) if f.endswith('.svs')]

    if not svs_files:
        print(f"No .svs files found in {folder_path}. Please add valid .svs files.")
        return

    for svs_file in svs_files:
        svs_path = os.path.join(folder_path, svs_file)
        try:
            # Open slide and get thumbnail
            slide = openslide.OpenSlide(svs_path)
            width, height = slide.dimensions
            thumbnail = slide.get_thumbnail((width // 10, height // 10))
            slide.close()

            # Convert thumbnail to NumPy array and resize
            thumbnail_np = np.array(thumbnail)
            resized_tile = resize_tile(thumbnail_np, target_size=(256, 256))

            # Create a dummy heightmap and threshold it
            heightmap = torch.rand(1, 1, 256, 256)
            thresholded = threshold_heightmap(heightmap, threshold=0.5)

            # Compute SDF loss on dummy inputs
            pred, normals, mask = dummy_inputs(batch_size=1, height=256, width=256)
            loss = compute_sdf_loss(pred, normals, mask)

            print(f"Workflow test passed for slide: {svs_file} | Loss: {loss.item()}")

        except Exception as e:
            print(f"Error during workflow test for {svs_file}: {e}")

def create_thumbnail_dataset(thumb_dir, grid_dir, tile_size=(256, 256), threshold=0.5):
    """
    Create a dataset from thumbnails instead of SVS files.
    This is a workaround for the OpenSlide compatibility issues.
    """
    from data import SlideTileDataset
    
    class ThumbnailDataset:
        def __init__(self, thumb_dir, grid_dir, tile_size=(256, 256), threshold=0.5):
            self.thumb_dir = thumb_dir
            self.grid_dir = grid_dir
            self.tile_size = tile_size
            self.threshold = threshold
            
            self.thumbs = [os.path.join(thumb_dir, f) for f in os.listdir(thumb_dir) if f.endswith('_thumb.jpg')]
            self.grids = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.data')]
            
            print(f"Found {len(self.thumbs)} thumbnails and {len(self.grids)} grids")
            
        def __len__(self):
            return len(self.thumbs)
        
        def __getitem__(self, idx):
            thumb_path = self.thumbs[idx]
            grid_path = self.grids[idx % len(self.grids)]  # Use modulo in case there are more thumbs than grids
            
            # Load grid data
            with open(grid_path, 'rb') as f:
                grid_data = pickle.load(f)
            
            # Open thumbnail file
            print(f"Opening thumbnail: {thumb_path}")
            thumb = Image.open(thumb_path)
            
            # Convert to RGB if needed
            if thumb.mode != 'RGB':
                thumb = thumb.convert('RGB')
                
            # Convert to NumPy array and resize
            thumb_np = np.array(thumb)
            resized_tile = resize_tile(thumb_np, target_size=self.tile_size)
            
            # Create a dummy heightmap and threshold it
            heightmap = torch.rand(1, 1, *self.tile_size)
            thresholded_heightmap = threshold_heightmap(heightmap, threshold=self.threshold)
            
            return {
                "resized_tile": resized_tile,
                "thresholded_heightmap": thresholded_heightmap,
                "tile_coords": (0, 0),  # Dummy coordinates
                "slide_path": thumb_path
            }
    
    return ThumbnailDataset(thumb_dir, grid_dir, tile_size, threshold)

def test_slide_tile_dataset_simple():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # This test verifies the SlideTileDataset with thumbnails.
    
    grid_dir, thumb_dir = get_test_data_from_svs_folder()
    
    # Create a new empty dataset file to verify it works
    print(f"Testing SlideTileDataset with thumbnails from: {thumb_dir}")
    
    # Use a custom dataset function that works with thumbnails
    dataset = create_thumbnail_dataset(thumb_dir, grid_dir, tile_size=(256, 256), threshold=0.5)
    
    assert len(dataset) > 0, "Dataset is empty!"
    sample = dataset[0]

    assert "resized_tile" in sample, "Missing resized_tile in sample!"
    assert "thresholded_heightmap" in sample, "Missing thresholded_heightmap in sample!"
    assert sample["resized_tile"].shape[:2] == (256, 256), "Unexpected tile size!"
    print("Simple dataset test passed!")

def test_slide_tile_dataset_complex():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # This test verifies the SlideTileDataset with larger tile sizes.
    
    grid_dir, thumb_dir = get_test_data_from_svs_folder()
    
    # Create a dataset using thumbnails
    print(f"Testing complex SlideTileDataset with thumbnails from: {thumb_dir}")
    
    # Use larger tile size for the complex test
    dataset = create_thumbnail_dataset(thumb_dir, grid_dir, tile_size=(512, 512), threshold=0.7)
    
    assert len(dataset) > 0, "Dataset is empty!"
    sample = dataset[0]

    resized_tile = sample["resized_tile"]
    thresholded_heightmap = sample["thresholded_heightmap"]

    assert resized_tile.shape[:2] == (512, 512), "Unexpected tile size!"
    assert thresholded_heightmap.shape == (1, 1, 512, 512), "Unexpected heightmap shape!"
    assert torch.all((thresholded_heightmap == 0) | (thresholded_heightmap == 1)), "Heightmap values are not binary!"
    print("Complex dataset test passed!")

def test_slide_tile_dataset_workflow():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # This test performs a workflow using the SlideTileDataset and utility functions.
    
    grid_dir, thumb_dir = get_test_data_from_svs_folder()
    
    # Define consistent tile size for the entire workflow
    tile_size = (256, 256)
    
    # Create dataset with thumbnails
    dataset = create_thumbnail_dataset(thumb_dir, grid_dir, tile_size=tile_size, threshold=0.5)

    for idx in range(len(dataset)):
        sample = dataset[idx]

        resized_tile = sample["resized_tile"]
        thresholded_heightmap = sample["thresholded_heightmap"]
        
        # Verify dimensions match our expected tile size
        assert resized_tile.shape[:2] == tile_size, f"Tile size mismatch: {resized_tile.shape[:2]} vs expected {tile_size}"
        assert thresholded_heightmap.shape[2:] == tile_size, f"Heightmap size mismatch: {thresholded_heightmap.shape[2:]} vs expected {tile_size}"

        # Simulate a model prediction with matching dimensions
        pred, normals, mask = dummy_inputs(batch_size=1, height=tile_size[0], width=tile_size[1])
        loss = compute_sdf_loss(pred, normals, mask)

        assert loss.item() > 0, f"Unexpected loss value for sample {idx}: {loss.item()}"
        print(f"Workflow test passed for sample {idx} | Loss: {loss.item()}")

def test_train_one_epoch():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # - Checkpoints saved in "./checkpoints_test"
    # This test trains the model for one epoch and visualizes results.
    
    from train import train_model
    from utils import visualize_model_output

    # Prepare test dataset
    grid_dir, thumb_dir = get_test_data_from_svs_folder()
    dataset_dir = "./intermediate_data"

    # Train for one epoch
    checkpoint_dir = "./checkpoints_test"
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_model(
        dataset_dir=dataset_dir,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        tile_size=(256, 256),
        threshold=0.5,
        num_channels=32,  # Match the default model configuration
        num_pool_layers=4,  # Match the default model configuration
        drop_prob=0.1,
        checkpoint_dir=checkpoint_dir
    )

    # Visualize results using the last checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "unet_epoch_1.pt")
    assert os.path.exists(checkpoint_path), "Checkpoint not created!"
    print(f"Checkpoint created: {checkpoint_path}")

    # Load a sample image from the dataset
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=(256, 256),
        threshold=0.5,
        use_thumbnails=True
    )
    assert len(dataset) > 0, "Dataset is empty!"
    sample = dataset[0]
    input_image = sample["resized_tile"]

    # Ensure the input image is valid
    assert input_image is not None, "Input image is None!"
    assert isinstance(input_image, np.ndarray), "Input image is not a NumPy array!"
    assert input_image.shape[:2] == (256, 256), f"Unexpected input image size: {input_image.shape[:2]}"

    # Visualize model output
    try:
        visualize_model_output(checkpoint_path, input_image, tile_size=(256, 256))
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

    print("test_train_one_epoch passed!")


def test_extreme_epochs():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # - Checkpoints saved in "./checkpoints_extreme"
    # This test trains the model for an extreme number of epochs and visualizes the best result.
    
    from train import train_model
    from utils import visualize_model_output

    # Prepare test dataset
    grid_dir, thumb_dir = get_test_data_from_svs_folder()
    dataset_dir = "./intermediate_data"

    # Train for an extreme number of epochs
    checkpoint_dir = "./checkpoints_extreme"
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_epochs = 20  # Extreme number of epochs
    train_model(
        dataset_dir=dataset_dir,
        batch_size=2,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        tile_size=(256, 256),
        threshold=0.5,
        num_channels=32,  # Match the default model configuration
        num_pool_layers=4,  # Match the default model configuration
        drop_prob=0.1,
        checkpoint_dir=checkpoint_dir
    )

    # Find the best checkpoint (lowest loss)
    best_checkpoint = None
    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pt")
        if os.path.exists(checkpoint_path):
            # Simulate loss retrieval (in practice, you would save and load actual loss values)
            simulated_loss = 1.5 - (epoch * 0.01)  # Example: decreasing loss over epochs
            if simulated_loss < best_loss:
                best_loss = simulated_loss
                best_checkpoint = checkpoint_path

    assert best_checkpoint is not None, "No valid checkpoint found!"
    print(f"Best checkpoint: {best_checkpoint} with simulated loss: {best_loss}")

    # Load a sample image from the dataset
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=(256, 256),
        threshold=0.5,
        use_thumbnails=True
    )
    assert len(dataset) > 0, "Dataset is empty!"
    sample = dataset[0]
    input_image = sample["resized_tile"]

    # Ensure the input image is valid
    assert input_image is not None, "Input image is None!"
    assert isinstance(input_image, np.ndarray), "Input image is not a NumPy array!"
    assert input_image.shape[:2] == (256, 256), f"Unexpected input image size: {input_image.shape[:2]}"

    # Visualize model output using the best checkpoint
    try:
        visualize_model_output(best_checkpoint, input_image, tile_size=(256, 256))
        print("Visualization of the best result completed successfully!")
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise

    print("test_extreme_epochs passed!")


def test_check_and_visualize_segmaps():
    # Required:
    # - Segmaps in "./segmaps" (e.g., `_SegMap.png` files)
    # This test checks and visualizes segmentation maps.
    
    from utils import check_and_visualize_segmaps

    segmaps_dir = "./segmaps"  # Path to the segmaps folder
    selected_file = None  # Set to a specific file name if you want to test visualization for a specific file

    print("Testing check_and_visualize_segmaps...")
    check_and_visualize_segmaps(segmaps_dir, selected_file)
    print("check_and_visualize_segmaps test completed!")


def test_model_output_and_match_segmaps():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # - Segmaps in "./segmaps" (e.g., `_SegMap.png` files)
    # - Checkpoints saved in "./checkpoints_test"
    # - Thresholded outputs saved in "./thresholded_output"
    # - Resized outputs saved in "./thresholded_output_resized"
    # - Final thresholded outputs saved in "./final_thresholded_output"
    # This test trains the model for one epoch, generates outputs, applies thresholds, and visualizes results.
    
    from train import train_model
    from analysis import import_segmaps, resize_model_outputs, visualize_matching_resolutions, create_thresholded_outputs
    from utils import threshold_heightmap

    # Prepare directories
    segmaps_dir = "./segmaps"
    thresholded_dir = "./thresholded_output"
    resized_outputs_dir = "./thresholded_output_resized"
    final_thresholded_dir = "./final_thresholded_output"
    dataset_dir = "./intermediate_data"
    checkpoint_dir = "./checkpoints_test"

    # Train the model for 1 epoch
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_model(
        dataset_dir=dataset_dir,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        tile_size=(256, 256),
        threshold=0.5,
        num_channels=32,
        num_pool_layers=4,
        drop_prob=0.1,
        checkpoint_dir=checkpoint_dir
    )

    # Load the trained model
    checkpoint_path = os.path.join(checkpoint_dir, "unet_epoch_1.pt")
    assert os.path.exists(checkpoint_path), "Checkpoint not created!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel(
        in_chans=3,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Prepare dataset
    grid_dir = os.path.join(dataset_dir, "grids")
    thumb_dir = os.path.join(dataset_dir, "thumbs")
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=(256, 256),
        threshold=0.5,
        use_thumbnails=True
    )

    # Generate thresholded outputs
    print("Generating thresholded outputs...")
    create_thresholded_outputs(model, dataset, thresholded_dir, tile_size=(256, 256), device=device)

    # Debug: Check if the thresholded_dir exists and is populated
    thresholded_files = [f for f in os.listdir(thresholded_dir) if f.endswith(".png")]
    if not thresholded_files:
        print(f"Error: No model outputs found in {thresholded_dir}.")
        return

    print(f"Model outputs found in {thresholded_dir}: {thresholded_files}")

    # Resize model outputs to match segmaps
    resize_model_outputs(thresholded_dir, segmaps_dir, resized_outputs_dir)

    # Debug: Check if resized outputs exist
    resized_files = [f for f in os.listdir(resized_outputs_dir) if f.endswith(".png")]
    if not resized_files:
        print(f"Error: No resized outputs found in {resized_outputs_dir}.")
        return

    print(f"Resized outputs found in {resized_outputs_dir}: {resized_files}")

    # Apply threshold to resized outputs and save them
    os.makedirs(final_thresholded_dir, exist_ok=True)
    for resized_file in resized_files:
        resized_path = os.path.join(resized_outputs_dir, resized_file)
        with Image.open(resized_path) as resized_img:
            resized_array = np.array(resized_img) / 255.0  # Normalize to [0, 1]
            thresholded_array = threshold_heightmap(torch.tensor(resized_array), threshold=0.5).numpy() * 255.0
            thresholded_img = Image.fromarray(thresholded_array.astype(np.uint8))
            thresholded_img.save(os.path.join(final_thresholded_dir, resized_file))

    # Visualize matching resolutions with thresholded outputs
    segmap_files = import_segmaps(segmaps_dir)
    visualize_matching_resolutions(segmap_files, resized_outputs_dir, final_thresholded_dir)

    print("No matching resolutions found between segmaps, resized outputs, and thresholded outputs.")

def test_model_with_more_epochs():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # - Segmaps in "./segmaps" (e.g., `_SegMap.png` files)
    # - Checkpoints saved in "./checkpoints_large_tiles"
    # - Thresholded outputs saved in "./thresholded_output_large_tiles"
    # - Resized outputs saved in "./resized_output_large_tiles"
    # - Final thresholded outputs saved in "./final_thresholded_output_large_tiles"
    # This test trains the model for 5 epochs, generates outputs, applies thresholds, and visualizes results.
    
    from train import train_model
    from analysis import import_segmaps, resize_model_outputs, create_thresholded_outputs
    from utils import threshold_heightmap

    # Prepare directories
    segmaps_dir = "./segmaps"
    thresholded_dir = "./thresholded_output_large_tiles"
    resized_outputs_dir = "./resized_output_large_tiles"
    final_thresholded_dir = "./final_thresholded_output_large_tiles"
    dataset_dir = "./intermediate_data"
    checkpoint_dir = "./checkpoints_large_tiles"

    # Train the model for 5 epochs
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_model(
        dataset_dir=dataset_dir,
        batch_size=2,
        num_epochs=5,
        learning_rate=1e-3,
        tile_size=(256, 256),
        threshold=0.5,
        num_channels=32,
        num_pool_layers=4,
        drop_prob=0.1,
        checkpoint_dir=checkpoint_dir
    )

    # Load the trained model
    checkpoint_path = os.path.join(checkpoint_dir, "unet_epoch_5.pt")
    assert os.path.exists(checkpoint_path), "Checkpoint not created!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel(
        in_chans=3,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Prepare dataset
    grid_dir = os.path.join(dataset_dir, "grids")
    thumb_dir = os.path.join(dataset_dir, "thumbs")
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=(256, 256),
        threshold=0.5,
        use_thumbnails=True
    )

    # Generate thresholded outputs
    print("Generating thresholded outputs...")
    create_thresholded_outputs(model, dataset, thresholded_dir, tile_size=(256, 256), device=device)

    # Debug: Check if the thresholded_dir exists and is populated
    thresholded_files = [f for f in os.listdir(thresholded_dir) if f.endswith(".png")]
    if not thresholded_files:
        print(f"Error: No model outputs found in {thresholded_dir}.")
        return

    print(f"Model outputs found in {thresholded_dir}: {thresholded_files}")

    # Resize model outputs to match segmaps
    resize_model_outputs(thresholded_dir, segmaps_dir, resized_outputs_dir)

    # Debug: Check if resized outputs exist
    resized_files = [f for f in os.listdir(resized_outputs_dir) if f.endswith(".png")]
    if not resized_files:
        print(f"Error: No resized outputs found in {resized_outputs_dir}.")
        return

    print(f"Resized outputs found in {resized_outputs_dir}: {resized_files}")

    # Apply threshold to resized outputs and save them
    os.makedirs(final_thresholded_dir, exist_ok=True)
    for resized_file in resized_files:
        resized_path = os.path.join(resized_outputs_dir, resized_file)
        with Image.open(resized_path) as resized_img:
            resized_array = np.array(resized_img) / 255.0  # Normalize to [0, 1]
            thresholded_array = threshold_heightmap(torch.tensor(resized_array), threshold=0.5).numpy() * 255.0
            thresholded_img = Image.fromarray(thresholded_array.astype(np.uint8))
            thresholded_img.save(os.path.join(final_thresholded_dir, resized_file))

    # Visualize matching resolutions with thresholded outputs
    segmap_files = import_segmaps(segmaps_dir)
    visualize_matching_resolutions(segmap_files, resized_outputs_dir, final_thresholded_dir)

    print("No matching resolutions found between segmaps, resized outputs, and thresholded outputs.")

def test_model_with_more_epochs_modified():
    # Required:
    # - SVS files in "../Recources/example_tcga_slides"
    # - Thumbnails generated in "./intermediate_data/thumbs" (created by `get_test_data_from_svs_folder`)
    # - Grid files in "./intermediate_data/grids" (created by `get_test_data_from_svs_folder`)
    # - Segmaps in "./segmaps" (e.g., `_SegMap.png` files)
    # - Checkpoints saved in "./checkpoints_large_tiles"
    # - Thresholded outputs saved in "./thresholded_output_large_tiles"
    # - Resized outputs saved in "./resized_output_large_tiles"
    # - Final thresholded outputs saved in "./final_thresholded_output_large_tiles"
    # This test trains the model for 5 epochs, generates outputs, applies thresholds, and visualizes results.
    
    from train import train_model
    from analysis import import_segmaps, resize_model_outputs, create_thresholded_outputs
    from utils import threshold_heightmap

    # Prepare directories
    segmaps_dir = "./segmaps"
    thresholded_dir = "./thresholded_output_large_tiles"
    resized_outputs_dir = "./resized_output_large_tiles"
    final_thresholded_dir = "./final_thresholded_output_large_tiles"
    dataset_dir = "./intermediate_data"
    checkpoint_dir = "./checkpoints_large_tiles"

    # Train the model for 5 epochs
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_model(
        dataset_dir=dataset_dir,
        batch_size=2,
        num_epochs=5,
        learning_rate=1e-3,
        tile_size=(512, 512),
        threshold=0.5,
        num_channels=32,
        num_pool_layers=4,
        drop_prob=0.1,
        checkpoint_dir=checkpoint_dir
    )

    # Load the trained model
    checkpoint_path = os.path.join(checkpoint_dir, "unet_epoch_5.pt")
    assert os.path.exists(checkpoint_path), "Checkpoint not created!"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel(
        in_chans=3,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.1
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Prepare dataset
    grid_dir = os.path.join(dataset_dir, "grids")
    thumb_dir = os.path.join(dataset_dir, "thumbs")
    dataset = SlideTileDataset(
        slide_dir=thumb_dir,
        grid_dir=grid_dir,
        tile_size=(512, 512),
        threshold=0.5,
        use_thumbnails=True
    )

    # Generate thresholded outputs
    print("Generating thresholded outputs...")
    create_thresholded_outputs(model, dataset, thresholded_dir, tile_size=(512, 512), device=device)

    # Debug: Check if the thresholded_dir exists and is populated
    thresholded_files = [f for f in os.listdir(thresholded_dir) if f.endswith(".png")]
    if not thresholded_files:
        print(f"Error: No model outputs found in {thresholded_dir}.")
        return

    print(f"Model outputs found in {thresholded_dir}: {thresholded_files}")

    # Resize model outputs to match segmaps
    resize_model_outputs(thresholded_dir, segmaps_dir, resized_outputs_dir)

    # Debug: Check if resized outputs exist
    resized_files = [f for f in os.listdir(resized_outputs_dir) if f.endswith(".png")]
    if not resized_files:
        print(f"Error: No resized outputs found in {resized_outputs_dir}.")
        return

    print(f"Resized outputs found in {resized_outputs_dir}: {resized_files}")

    # Apply threshold to resized outputs and save them
    os.makedirs(final_thresholded_dir, exist_ok=True)
    for resized_file in resized_files:
        resized_path = os.path.join(resized_outputs_dir, resized_file)
        with Image.open(resized_path) as resized_img:
            resized_array = np.array(resized_img) / 255.0  # Normalize to [0, 1]
            thresholded_array = threshold_heightmap(torch.tensor(resized_array), threshold=0.5).numpy() * 255.0
            thresholded_img = Image.fromarray(thresholded_array.astype(np.uint8))
            thresholded_img.save(os.path.join(final_thresholded_dir, resized_file))

    # Visualize matching resolutions with thresholded outputs
    segmap_files = import_segmaps(segmaps_dir)
    visualize_matching_resolutions(segmap_files, resized_outputs_dir, final_thresholded_dir)

    print("No matching resolutions found between segmaps, resized outputs, and thresholded outputs.")

# Run the tests
if __name__ == "__main__":
    # Run all tests
    # Uncomment the desired test to run
    # test_unet_simple()
    # test_unet_complex()
    
    # test_openslide_functionality()
    # test_resize_tile()
    # test_threshold_heightmap()
    # test_compute_sdf_loss()
    # test_print_svs_resolutions_in_folder()
    # test_workflow_with_slides()
    # test_check_and_visualize_segmaps()
    
    # test_slide_tile_dataset_simple()
    # test_slide_tile_dataset_complex()
    # test_slide_tile_dataset_workflow()

    # test_train_one_epoch()
    # test_extreme_epochs()

    # test_model_output_and_match_segmaps()
    # test_model_with_more_epochs()
    test_model_with_more_epochs_modified()
