import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_backend
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_fill_holes  # Import the new method
import os

# Directory containing the input images
input_dir = Path(__file__).resolve().parent.parent / "starting_data" / "segmaps"

# Directory to save the output images
output_dir = Path(__file__).resolve().parent.parent / "intermediate_data" / "segmaps_filled_holes"

# Ensure the output directory exists
try:
    print(f"Checking if output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ensured: {output_dir}")
except Exception as e:
    print(f"Failed to create output directory: {output_dir}. Error: {e}")
    raise

# Ensure the input directory exists and is valid
print(f"Checking if input directory exists: {input_dir}")
assert input_dir.exists(), f"Input directory does not exist: {input_dir}"
assert input_dir.is_dir(), f"Input path is not a directory: {input_dir}"
print(f"Input directory is valid: {input_dir}")

def fill_black_holes(binary_img, debug=False):
    """
    Fill black holes (0) surrounded by white (255) in a binary image using scipy.ndimage.binary_fill_holes.
    
    Args:
        binary_img (np.ndarray): Binary image with white=foreground (255), black=background (0).
        debug (bool): If True, print debug information.
    
    Returns:
        np.ndarray: Binary image with holes filled.
    """
    if debug:
        print("Starting to fill black holes in the binary image...")
    # Ensure the input is binary (0 or 255)
    binary_img = (binary_img > 0).astype(np.uint8)
    if debug:
        print("Converted image to binary format.")

    # Use scipy's binary_fill_holes to fill holes
    filled = binary_fill_holes(binary_img).astype(np.uint8)
    if debug:
        print("Applied binary_fill_holes to the image.")

    # Convert back to 0 and 255 format
    filled_image = filled * 255
    if debug:
        print("Converted filled image back to 0 and 255 format.")
    
    return filled_image

# Load and process an image
def process_image_black_holes(path, debug=False):
    """
    Process a single image to fill black holes.
    
    Args:
        path (Path): Path to the input image.
        debug (bool): If True, print debug information.
    
    Returns:
        tuple: Binary image and filled image.
    """
    if debug:
        print(f"Processing image: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    filled = fill_black_holes(binary, debug=debug)
    return binary, filled

def save_filled_images(debug=False):
    """
    Process all images in the input directory, fill black holes, and save the results
    directly to the intermediate_data directory.
    """
    if debug:
        print("Collecting all .png files from the input directory...")
    image_paths = list(input_dir.glob("*.png"))
    if debug:
        print(f"Found {len(image_paths)} .png files in {input_dir}.")
    assert len(image_paths) > 0, f"No .png files found in {input_dir}"

    for image_path in image_paths:
        if debug:
            print(f"Saving: {image_path}")
        try:
            _, filled = process_image_black_holes(image_path, debug=debug)
            if debug:
                print(f"Processed: {image_path}")

            # Save the filled image directly to the output directory
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), filled)
            if debug:
                print(f"Saved filled image to: {output_path}")
        except Exception as e:
            print(f"Failed to process or save image: {image_path}. Error: {e}")

def visualize_filled_images(debug=False):
    """
    Visualize the original and filled images from the input and output directories.
    
    Args:
        debug (bool): If True, print debug information.
    """
    if debug:
        print("Starting visualization of filled images...")
    # Collect all .png files from the output directory
    image_paths = list(output_dir.glob("*.png"))
    if debug:
        print(f"Found {len(image_paths)} .png files in {output_dir}.")
    assert len(image_paths) > 0, f"No .png files found in {output_dir}"

    results = []
    for image_path in image_paths:
        if debug:
            print(f"Loading: {image_path}")
        try:
            # Load the original and filled images from the output directory
            filled = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            original_path = input_dir / image_path.name
            original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)

            # Extract the title as the part before the first dot
            title = image_path.name.split('.')[0]
            results.append((title, original, filled))
        except Exception as e:
            print(f"Failed to load image for visualization: {image_path}. Error: {e}")

    # Split results into two groups
    mid_index = len(results) // 2
    groups = [results[:mid_index], results[mid_index:]]

    for group_index, group in enumerate(groups):
        if debug:
            print(f"Visualizing group {group_index + 1} with {len(group)} images...")
        num_images = len(group)
        num_cols = 2  # Original and filled images side by side
        num_rows = num_images

        # Adjust figure size to fit all images in one window
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows))
        for i, (title, original, filled) in enumerate(group):
            axs[i, 0].imshow(original, cmap='gray')
            axs[i, 0].set_title(f"Original: {title}", fontsize=5)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(filled, cmap='gray')
            axs[i, 1].set_title(f"Filled: {title}", fontsize=5)
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Group {group_index + 1}", fontsize=8)
        plt.show()

def test_fill_black_holes(debug=False):
    """
    Test the process of filling black holes and saving the results.
    
    Args:
        debug (bool): If True, print debug information.
    """
    if debug:
        print("Starting test_fill_black_holes...")
    # Collect all .png files from the input directory
    image_paths = list(input_dir.glob("*.png"))
    if debug:
        print(f"Found {len(image_paths)} .png files in {input_dir}.")
    assert len(image_paths) > 0, f"No .png files found in {input_dir}"

    for image_path in image_paths:
        if debug:
            print(f"Processing: {image_path}")
        try:
            _, filled = process_image_black_holes(image_path, debug=debug)
            if debug:
                print(f"Processed: {image_path}")

            # Save the filled image
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), filled)
            if debug:
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process or save image: {image_path}. Error: {e}")

def transfer_filled_images(source_dir, target_dir, debug=False):
    """
    Transfers filled images from the source directory to the target directory with the same names.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if os.path.isfile(source_path):
            if debug:
                print(f"Transferring {source_path} to {target_path}")
            os.rename(source_path, target_path)

# Run the test function and visualize filled images
if __name__ == "__main__":
    print("Starting the script...")
    save_filled_images(debug=False)
    visualize_filled_images(debug=False)
    print("Script completed.")
