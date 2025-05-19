import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def is_closed_curve(contour, tol=1e-3):
    """Check if the curve is closed (first and last points are the same within tolerance)."""
    return np.linalg.norm(contour[0] - contour[-1]) < tol

def curve_area(contour):
    """Calculate the area inside a closed curve using the shoelace formula."""
    if len(contour) < 3:  # not a polygon
        return 0
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def smallest_curve_area(contours):
    """Calculate the smallest area contained by a single closed curve."""
    closed_areas = [curve_area(c) for c in contours if is_closed_curve(c)]
    return min(closed_areas) if closed_areas else 0

def analyze_curves(formatted_dir, segmap_dir, debug=False):
    """
    Analyze all formatted curves in the directory and their corresponding binary images.
    """
    results = []
    for filename in os.listdir(formatted_dir):
        if filename.endswith(".npy"):
            formatted_path = os.path.join(formatted_dir, filename)
            segmap_path = os.path.join(segmap_dir, f"{os.path.splitext(filename)[0]}.png")
            
            if not os.path.exists(segmap_path):
                if debug:
                    print(f"Segmentation map not found for {filename}. Skipping.")
                continue
            
            formatted = np.load(formatted_path, allow_pickle=True).item()
            contours = formatted["contours"]

            total_curves = len(contours)
            closed_curves = [c for c in contours if is_closed_curve(c)]
            num_closed = len(closed_curves)
            total_area_closed = sum(curve_area(c) for c in closed_curves)
            smallest_area = smallest_curve_area(contours)
            binary_img = io.imread(segmap_path, as_gray=True)
            binary_img = (binary_img > 0.5).astype(np.uint8)
            white_area = np.sum(binary_img)

            results.append({
                "filename": filename,
                "total_curves": total_curves,
                "num_closed": num_closed,
                "smallest_area": smallest_area,
                "white_area": white_area,
            })

            if debug:
                print(f"Analyzed {filename}: {results[-1]}")

    return results

def save_visualize_analysis(results, formatted_dir, segmap_dir, output_dir, debug=False):
    """
    Save visualizations by overlaying curves on the original segmentation maps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for result in results:
        filename = result["filename"]
        formatted_path = os.path.join(formatted_dir, filename)
        segmap_path = os.path.join(segmap_dir, f"{os.path.splitext(filename)[0]}.png")
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_analysis.png")

        formatted = np.load(formatted_path, allow_pickle=True).item()
        contours = formatted["contours"]
        binary_img = io.imread(segmap_path, as_gray=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(binary_img, cmap='gray')
        for contour in contours:
            plt.plot(contour[:, 0], contour[:, 1], linewidth=2)

        # Truncate the title at the first dot
        truncated_filename = filename.split('.')[0]
        plt.title(f"File: {truncated_filename}\n"
                  f"Total Curves: {result['total_curves']}, Closed Curves: {result['num_closed']}\n"
                  f"Smallest Area (Closed): {result['smallest_area']:.1f}, White Area: {result['white_area']}")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if debug:
            print(f"Saved {output_path}")

def visualize_analysis(output_dir, debug=False):
    """
    Visualize the saved analysis results by displaying the images in two windows.
    """
    if debug:
        print("Starting visualization of saved analysis results...")
    image_paths = [f for f in os.listdir(output_dir) if f.endswith("_analysis.png")]
    if debug:
        print(f"Found {len(image_paths)} analysis images in {output_dir}.")
    assert len(image_paths) > 0, f"No _analysis.png files found in {output_dir}"

    results = []
    for image_path in image_paths:
        try:
            img = io.imread(os.path.join(output_dir, image_path))
            title = image_path.split('.')[0]  # Extract title before the first dot
            results.append((title, img))
        except Exception as e:
            print(f"Failed to load image for visualization: {image_path}. Error: {e}")

    # Split results into two groups for two windows
    mid_index = len(results) // 2
    groups = [results[:mid_index], results[mid_index:]]

    for group_index, group in enumerate(groups):
        if debug:
            print(f"Visualizing group {group_index + 1} with {len(group)} images...")
        num_images = len(group)
        num_cols = 2  # Two images per row
        num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows dynamically

        # Adjust figure size to fit all images in one window
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axs = axs.flatten()

        for i, (title, img) in enumerate(group):
            axs[i].imshow(img)
            truncated_title = title.split('_')[0]  # Ensure the curve visualization title is truncated
            axs[i].set_title(truncated_title, fontsize=10)
            axs[i].axis('off')

        # Hide any unused subplots
        for j in range(len(group), len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Group {group_index + 1}", fontsize=16)
        plt.show()

if __name__ == "__main__":
    formatted_dir = os.path.join("..", "intermediate_data", "curves_formatted")
    segmap_dir = os.path.join("..", "intermediate_data", "segmaps_filled_holes")
    output_dir = os.path.join("..", "intermediate_data", "curves_analysis")

    debug = True
    results = analyze_curves(formatted_dir, segmap_dir, debug=False)
    save_visualize_analysis(results, formatted_dir, segmap_dir, output_dir, debug=False)
    visualize_analysis(output_dir, debug=debug)

    print("Analysis and visualization complete.")
