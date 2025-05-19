import os
import numpy as np
from skimage import measure, io
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label
from analyze_curves import curve_area, smallest_curve_area

def preprocess_and_count_areas(binary_img, debug=False):
    """
    Preprocess the binary image to count the main connected areas.
    This step helps guide the curve extraction process.
    """
    if debug:
        print("Preprocessing binary image to count main areas...")
    # Ensure binary format
    binary_img = (binary_img > 0).astype(np.uint8)
    # Label connected components
    labeled_array, num_features = label(binary_img)
    if debug:
        print(f"Found {num_features} connected areas.")
    return labeled_array, num_features

def extract_curves_from_binary(binary_img, max_curves=None, debug=False):
    """
    Extracts contours (curves) from a binary segmentation map, limiting the number of curves if needed.
    """
    if debug:
        print("Extracting contours from binary image...")
    normalized_image = binary_img / 255.0  # Normalize to [0, 1]
    contours = measure.find_contours(normalized_image, level=0.5)
    if max_curves:
        contours = sorted(contours, key=lambda c: len(c), reverse=True)[:max_curves]  # Keep largest curves
    formatted_contours = [np.fliplr(contour) for contour in contours]  # Convert (row, col) to (x, y)
    return formatted_contours

def process_image_extract_curves(path, debug=False):
    """
    Processes a single image to extract contours from its binary representation.
    """
    if debug:
        print(f"Processing image: {path}")
    binary_img = io.imread(path, as_gray=True)
    labeled_array, num_areas = preprocess_and_count_areas(binary_img, debug=debug)
    contours = extract_curves_from_binary(binary_img, max_curves=num_areas, debug=debug)
    return binary_img, contours

def format_curves(contours, original_image, debug=False):
    """
    Formats the extracted contours to ensure they are closed shapes and include image size.
    """
    if debug:
        print("Formatting contours...")
    if not contours:
        return None
    image_size = original_image.shape[::-1]  # (width, height)
    formatted = {"contours": contours, "image_size": image_size}
    return formatted

def save_formatted_curves(input_dir, output_dir, debug=False):
    """
    Processes all images in the input directory, extracts and formats contours,
    and saves the formatted curves directly to the intermediate_data directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Adjust file extension as needed
            path = os.path.join(input_dir, filename)
            binary_img, contours = process_image_extract_curves(path, debug=debug)
            formatted = format_curves(contours, binary_img, debug=debug)
            if formatted:
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
                np.save(output_path, formatted)
                if debug:
                    print(f"Saved formatted curves to {output_path}")

def save_formatted_curves_modified(input_dir, output_dir, threshold_factor=0.01, debug=False):
    """
    Processes all images in the input directory, extracts and formats contours,
    filters out small-area curves, and saves the modified formatted curves.

    Args:
        input_dir (str): Path to the directory containing binary segmentation maps.
        output_dir (str): Path to the directory to save the formatted curves (overrides existing files).
        threshold_factor (float): Threshold factor for the smallest curve area (e.g., 0.01 for 1/100).
        debug (bool): If True, print debug information.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Adjust file extension as needed
            path = os.path.join(input_dir, filename)
            binary_img, contours = process_image_extract_curves(path, debug=debug)

            # Calculate area threshold based on image size
            width, height = binary_img.shape[::-1]
            area_threshold = (threshold_factor * width) * (threshold_factor * height)

            if debug:
                print(f"Processing {filename}: width={width}, height={height}, area_threshold={area_threshold:.2f}")

            # Filter out small-area curves
            filtered_contours = [c for c in contours if curve_area(c) >= area_threshold]

            # Format the filtered contours
            formatted = format_curves(filtered_contours, binary_img, debug=debug)
            if formatted:
                # Verify the smallest curve area meets the threshold
                smallest_area = smallest_curve_area(filtered_contours)
                if debug:
                    print(f"After filtering {filename}: smallest_area={smallest_area:.2f}")
                assert smallest_area >= area_threshold, (
                    f"Smallest curve area {smallest_area:.2f} is below the threshold {area_threshold:.2f} in {filename}"
                )

                # Save the formatted curves (overwriting existing files)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
                np.save(output_path, formatted)
                if debug:
                    print(f"Saved modified formatted curves to {output_path}")

def save_visualizations(input_dir, formatted_dir, output_dir, debug=False):
    """
    Creates visualizations by overlaying extracted contours on the original segmentation maps
    and saves them directly to the intermediate_data directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(formatted_dir):
        if filename.endswith(".npy"):
            formatted_path = os.path.join(formatted_dir, filename)
            formatted = np.load(formatted_path, allow_pickle=True).item()
            original_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.png")
            original_image = io.imread(original_path, as_gray=True)
            plt.imshow(original_image, cmap='gray')
            for contour in formatted["contours"]:
                plt.plot(contour[:, 0], contour[:, 1], linewidth=2)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            if debug:
                print(f"Saved visualization to {output_path}")

def visualize_curves_comparison(starting_data_dir, visualization_dir, debug=False):
    """
    Visualize the original segmentation maps and the extracted curves side by side.
    
    Args:
        starting_data_dir (str): Path to the starting data segmentation maps.
        visualization_dir (str): Path to the directory containing visualizations of extracted curves.
        debug (bool): If True, print debug information.
    """
    if debug:
        print("Starting visualization of curves comparison...")
    # Collect all .png files from the visualization directory
    visualization_paths = [f for f in os.listdir(visualization_dir) if f.endswith(".png")]
    if debug:
        print(f"Found {len(visualization_paths)} .png files in {visualization_dir}.")
    assert len(visualization_paths) > 0, f"No .png files found in {visualization_dir}"

    results = []
    for vis_filename in visualization_paths:
        if debug:
            print(f"Loading visualization: {vis_filename}")
        try:
            # Load the visualization and the corresponding original segmentation map
            vis_path = os.path.join(visualization_dir, vis_filename)
            original_path = os.path.join(starting_data_dir, vis_filename)
            visualization = io.imread(vis_path)
            original = io.imread(original_path, as_gray=True)

            # Extract the title as the part before the first dot
            title = vis_filename.split('.')[0]
            results.append((title, original, visualization))
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
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 2))
        for i, (title, original, visualization) in enumerate(group):
            axs[i, 0].imshow(original, cmap='gray')
            axs[i, 0].set_title(f"Original: {title}", fontsize=8)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(visualization)
            axs[i, 1].set_title(f"Visualization: {title}", fontsize=8)
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Group {group_index + 1}", fontsize=12)
        plt.show()

if __name__ == "__main__":
    input_dir = os.path.join("..", "intermediate_data", "segmaps_filled_holes")
    formatted_dir = os.path.join("..", "intermediate_data", "curves_formatted")  # Overwrite this directory
    visualization_dir = os.path.join("..", "intermediate_data", "curves_visualization")
    starting_data_dir = os.path.join("..", "starting_data", "segmaps")

    # Use the modified function to save filtered and formatted curves
    save_formatted_curves_modified(input_dir, formatted_dir, threshold_factor=0.01, debug=False)
    save_visualizations(input_dir, formatted_dir, visualization_dir, debug=False)
    visualize_curves_comparison(starting_data_dir, visualization_dir, debug=False)

    print("Processing complete.")
