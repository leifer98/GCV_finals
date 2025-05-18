import os
import torch  # Add this import
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import resize_tile

def import_segmaps(segmaps_dir):
    """
    Import all segmaps from the specified directory.

    Args:
        segmaps_dir (str): Path to the segmaps folder.

    Returns:
        list: List of segmap file paths.
    """
    return [os.path.join(segmaps_dir, f) for f in os.listdir(segmaps_dir) if f.endswith(('.jpg', '.png'))]

def resize_model_outputs(thresholded_dir, segmaps_dir, output_dir):
    """
    Resize thresholded model outputs to match the resolution of segmaps.

    Args:
        thresholded_dir (str): Directory containing thresholded model outputs.
        segmaps_dir (str): Directory containing segmaps.
        output_dir (str): Directory to save resized outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    segmap_files = import_segmaps(segmaps_dir)

    for segmap_file in segmap_files:
        segmap_path = segmap_file
        with Image.open(segmap_path) as segmap_img:
            segmap_res = segmap_img.size  # (width, height)

            # Resize corresponding thresholded output
            thresholded_file = os.path.basename(segmap_file).replace("_SegMap", "")
            thresholded_path = os.path.join(thresholded_dir, thresholded_file)
            if os.path.exists(thresholded_path):
                with Image.open(thresholded_path) as thresholded_img:
                    # Ensure resizing back to segmap resolution
                    resized_thresholded = thresholded_img.resize(segmap_res, Image.NEAREST)
                    resized_thresholded.save(os.path.join(output_dir, thresholded_file))

def visualize_matching_resolutions(segmaps_dir, resized_outputs_dir):
    """
    Visualize segmaps and resized model outputs side by side if their resolutions match.

    Args:
        segmaps_dir (str): Directory containing segmaps.
        resized_outputs_dir (str): Directory containing resized model outputs.
    """
    segmap_files = import_segmaps(segmaps_dir)
    resized_files = [f for f in os.listdir(resized_outputs_dir) if f.endswith(".png")]

    for segmap_file in segmap_files:
        segmap_path = segmap_file
        with Image.open(segmap_path) as segmap_img:
            segmap_res = segmap_img.size  # (width, height)

            # Find a matching resized output
            resized_file = os.path.basename(segmap_file).replace("_SegMap", "")
            resized_path = os.path.join(resized_outputs_dir, resized_file)
            if os.path.exists(resized_path):
                with Image.open(resized_path) as resized_img:
                    resized_res = resized_img.size  # (width, height)

                    if segmap_res == resized_res:
                        # Truncate names until the first dot
                        segmap_title = f"Segmap: {os.path.basename(segmap_file).split('.')[0]}"
                        model_output_title = f"Model Output: {os.path.basename(resized_file).split('.')[0]}"

                        # Visualize the matching example
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        axes[0].imshow(segmap_img, cmap="gray")
                        axes[0].set_title(segmap_title)
                        axes[0].axis("off")
                        axes[1].imshow(resized_img, cmap="gray")
                        axes[1].set_title(model_output_title)
                        axes[1].axis("off")
                        plt.show()
                        return
    print("No matching resolutions found between segmaps and resized outputs.")

def create_thresholded_outputs(model, dataset, output_dir, tile_size=(256, 256), device=None):
    """
    Generate thresholded outputs from the model and save them to the specified directory.

    Args:
        model (torch.nn.Module): The trained model.
        dataset (SlideTileDataset): The dataset to process.
        output_dir (str): Directory to save the thresholded outputs.
        tile_size (tuple): Tile size for resizing the input image.
        device (torch.device): Device to run the model on (default: auto-detect).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        input_image = sample["resized_tile"]
        slide_path = sample["slide_path"]

        # Preprocess the input image
        input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to NCHW

        # Get the model output
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze().cpu().numpy()  # Convert to NumPy array

        # Save the output
        output_filename = os.path.basename(slide_path).replace("_thumb.jpg", ".png")
        output_path = os.path.join(output_dir, output_filename)
        Image.fromarray((output * 255).astype(np.uint8)).save(output_path)
        print(f"Saved thresholded output: {output_path}")
