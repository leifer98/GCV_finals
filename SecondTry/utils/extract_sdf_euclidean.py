import os
import numpy as np
from skimage import io
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2  # Importing cv2 for image resizing
from scipy.ndimage import gaussian_filter

def compute_sdf(binary_img, debug=False):
    """
    Computes the Signed Distance Function (SDF) for a binary image.
    Positive values are outside the object, negative values are inside.
    """
    if debug:
        print("Computing SDF for binary image...")
    binary_img = (binary_img > 0).astype(np.uint8)  # Ensure binary format
    outside_dist = distance_transform_edt(1 - binary_img)  # Distance to background
    inside_dist = distance_transform_edt(binary_img)      # Distance to foreground
    sdf = outside_dist - inside_dist                      # Signed distance
    return sdf

def save_sdf(input_dir, output_dir, debug=False):
    """
    Processes all binary images in the input directory, computes the SDF,
    and saves the results as .npy files in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Adjust file extension as needed
            path = os.path.join(input_dir, filename)
            binary_img = io.imread(path, as_gray=True)
            sdf = compute_sdf(binary_img, debug=debug)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
            np.save(output_path, sdf)
            if debug:
                print(f"Saved SDF to {output_path}")

def save_visualize_sdf(input_dir, sdf_dir, output_dir, debug=False):
    """
    Saves visualizations of the SDF by overlaying it on the original binary images
    and saves the visualizations as .png files in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(sdf_dir):
        if filename.endswith(".npy"):
            sdf_path = os.path.join(sdf_dir, filename)
            sdf = np.load(sdf_path)
            binary_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.png")
            binary_img = io.imread(binary_path, as_gray=True)

            plt.figure(figsize=(4, 4))
            plt.imshow(binary_img, cmap='gray', alpha=0.5)
            plt.contour(sdf, levels=20, cmap='coolwarm', linewidths=1)
            # plt.colorbar(label="Signed Distance")
            # plt.title(f"SDF Visualization: {filename}")
            plt.axis('off')

            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if debug:
                print(f"Saved SDF visualization to {output_path}")

def visualize_sdf_comparison(input_dir, visualization_dir, debug=False):
    """
    Visualizes the original segmentation maps and the SDF visualizations side by side.
    """
    if debug:
        print("Starting visualization of SDF comparison...")
    visualization_paths = [f for f in os.listdir(visualization_dir) if f.endswith(".png")]
    if debug:
        print(f"Found {len(visualization_paths)} visualization images in {visualization_dir}.")
    assert len(visualization_paths) > 0, f"No .png files found in {visualization_dir}"

    results = []
    for vis_filename in visualization_paths:
        if debug:
            print(f"Loading visualization: {vis_filename}")
        try:
            vis_path = os.path.join(visualization_dir, vis_filename)
            original_path = os.path.join(input_dir, vis_filename)
            visualization = io.imread(vis_path)
            original = io.imread(original_path, as_gray=True)

            # Resize images to the same dimensions for comparison
            height, width = visualization.shape[:2]
            original_resized = cv2.resize(original, (width, height))

            # Extract the title as the part before the first dot
            title = vis_filename.split('.')[0]
            results.append((title, original_resized, visualization))
        except Exception as e:
            print(f"Failed to load image for visualization: {vis_filename}. Error: {e}")

    # Split results into two groups for better display
    mid_index = len(results) // 2
    groups = [results[:mid_index], results[mid_index:]]

    for group_index, group in enumerate(groups):
        if debug:
            print(f"Visualizing group {group_index + 1} with {len(group)} images...")
        num_images = len(group)
        num_cols = 2  # Original and visualization side by side
        num_rows = num_images

        # Adjust figure size to fit all images in one window
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 2))
        for i, (title, original, visualization) in enumerate(group):
            axs[i, 0].imshow(original, cmap='gray')
            axs[i, 0].set_title(f"Original: {title}", fontsize=10)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(visualization)
            axs[i, 1].set_title(f"Visualization: {title}", fontsize=10)
            axs[i, 1].axis('off')

        # Hide any unused subplots
        for j in range(len(group), len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Group {group_index + 1}", fontsize=14)
        plt.show()

def normalize_and_smooth_segmap(segmap, sigma=2):
    """
    Normalize and smooth a segmentation map to behave similarly to an SDF.
    """
    segmap = segmap.astype(np.float32)
    segmap = (segmap - segmap.min()) / (segmap.max() - segmap.min())  # Normalize to [0, 1]
    segmap = 2 * segmap - 1  # Scale to [-1, 1]
    segmap = gaussian_filter(segmap, sigma=sigma)  # Apply Gaussian smoothing
    return segmap

if __name__ == "__main__":
    input_dir = os.path.join("..", "intermediate_data", "segmaps_filled_holes")
    sdf_dir = os.path.join("..", "intermediate_data", "sdf_euclidean")
    visualization_dir = os.path.join("..", "intermediate_data", "sdf_euclidean_visualization")

    debug = True
    save_sdf(input_dir, sdf_dir, debug=debug)
    save_visualize_sdf(input_dir, sdf_dir, visualization_dir, debug=debug)
    visualize_sdf_comparison(input_dir, visualization_dir, debug=debug)

    print("SDF computation and visualization complete.")
