import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import io
import torch.nn.functional as F
import matplotlib.pyplot as plt  # Add this import for visualization

def soft_normalize_sdf(sdf, shift=0.1, scale=0.9):
    """
    Apply soft normalization to the SDF to shift and scale values slightly.
    This ensures the loss is not trivially zero.
    """
    sdf = sdf.astype(np.float32)
    sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-6)  # Normalize to [0, 1]
    sdf = sdf * scale + shift  # Scale and shift values
    sdf = 2 * sdf - 1  # Scale to [-1, 1]
    return sdf

def compute_sdf_loss(model_output, gt_sdf, alpha=100.0, epsilon=1e-3, debug=False, filename="output"):
    """
    Implementation of SDF loss.

    Parameters:
    - model_output: Predicted SDF values from the model.
    - gt_sdf: Ground truth SDF values.
    - alpha: Weighting factor for the off-surface penalty term.
    - epsilon: Threshold for surface mask.
    - debug: If True, print the first, second, and third term results.
    - filename: Name of the file being processed, used for saving outputs.
    """
    # Compute surface mask
    surface_mask = (torch.abs(gt_sdf) < epsilon).float()
    
    print("surface mask shape", surface_mask.shape)
    print(f"surface mask max {surface_mask.max()}, surface mask min {surface_mask.min()}, surface mask mean {surface_mask.mean()}")
    print(f"number of zeros in gt_sdf: {torch.numel(gt_sdf)-torch.count_nonzero(gt_sdf)}")
    print(f"abs gt_sdf min: {torch.abs(gt_sdf).min()}, abs gt_sdf max: {torch.abs(gt_sdf).max()}")

    # Save the surface mask
    surface_masks_dir = os.path.join("..", "intermediate_data", "loss_surface_masks")
    save_surface_mask(surface_mask, surface_masks_dir, filename=f"{filename}.png")

    # Compute spatial gradients of the predicted SDF
    grad_x = torch.gradient(model_output, dim=(2, 3))
    grad = torch.cat(grad_x, dim=1)
    grad_norm = torch.norm(grad, dim=1, keepdim=True)
    normal_dir = os.path.join("..", "intermediate_data", "loss_normals")
    save_model_output_norm(grad_norm, normal_dir, filename=f"{filename}.png")

    # Eikonal term: Enforce ||∇Φ(x)| - 1|| ≈ 0
    eikonal_term = F.l1_loss(grad_norm, torch.ones_like(grad_norm), reduction='none').mean()

    # Absolute SDF values for surface points
    abs_phi = torch.abs(model_output)

    # Compute gradients of gt_sdf to get ground truth normals
    grad_gt_x = torch.gradient(gt_sdf, dim=(2, 3))
    grad_gt = torch.cat(grad_gt_x, dim=1)
    normals_gt = F.normalize(grad_gt, dim=1)

    # Normalize predicted gradients to compute predicted normals
    normals_pred = F.normalize(grad, dim=1)

    # Angle term: Enforce alignment between predicted and ground truth normals
    angle_term = 1 - F.cosine_similarity(normals_pred, normals_gt, dim=1, eps=1e-6).unsqueeze(1)
    surface_loss = (abs_phi * surface_mask + angle_term * surface_mask).mean()

    # Off-surface penalty: Exponential decay based on SDF magnitude
    psi = torch.exp(-alpha * abs_phi)
    off_surface_loss = (psi * (1.0 - surface_mask)).mean()

    # Combine all terms into the final loss
    total_loss = eikonal_term + surface_loss + off_surface_loss

    if debug:
        print(f"First Term (Eikonal Loss): {eikonal_term.item()}")
        print(f"Second Term (Surface Loss): {surface_loss.item()}")
        print(f"Third Term (Off-Surface Loss): {off_surface_loss.item()}")
        print(f"Total Loss: {total_loss.item()}")

    return total_loss

def normalize_and_smooth_segmap(segmap, sigma=2):
    """
    Normalize and smooth a segmentation map to behave similarly to an SDF.
    Ensures the range of values is appropriate for loss computation.
    """
    segmap = segmap.astype(np.float32)
    segmap = (segmap - segmap.min()) / (segmap.max() - segmap.min() + 1e-6)  # Normalize to [0, 1], avoid division by zero
    segmap = 2 * segmap - 1  # Scale to [-1, 1]
    segmap = gaussian_filter(segmap, sigma=sigma)  # Apply Gaussian smoothing
    segmap = np.clip(segmap, -1, 1)  # Ensure values remain in the range [-1, 1]
    return segmap

def preprocess_segmaps(input_dir, output_dir, sigma=2, debug=False):
    """
    Preprocess segmentation maps by normalizing and smoothing them.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            segmap = io.imread(input_path, as_gray=True)
            processed_segmap = normalize_and_smooth_segmap(segmap, sigma=sigma)
            np.save(output_path.replace(".png", ".npy"), processed_segmap)

            if debug:
                print(f"Processed and saved: {output_path}")

def analyze_test(model_output_dir, gt_sdf_dir, sigma=2, alpha=100.0, debug=False):
    """
    Compute the loss between model output and ground truth SDFs and print results.

    Parameters:
    - model_output_dir: Directory containing the model output SDFs.
    - gt_sdf_dir: Directory containing the ground truth SDFs.
    - sigma: Smoothing factor for preprocessing (if needed).
    - alpha: Weighting factor for the off-surface penalty term.
    - debug: If True, print debug information.
    """
    total_loss = 0
    count = 0

    for filename in os.listdir(gt_sdf_dir):
        if filename.endswith(".npy"):
            gt_sdf_path = os.path.join(gt_sdf_dir, filename)
            model_output_path = os.path.join(model_output_dir, filename)

            if not os.path.exists(model_output_path):
                if debug:
                    print(f"Model output not found for {filename}. Skipping.")
                continue

            try:  # Add the missing try block
                gt_sdf = np.load(gt_sdf_path)
                model_output = np.load(model_output_path)

                # Apply soft normalization to the model output
                model_output = soft_normalize_sdf(model_output)


                # Convert to PyTorch tensors and ensure they require gradients
                gt_sdf_tensor = torch.tensor(gt_sdf, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)
                model_output_tensor = torch.tensor(model_output, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)

                if debug:
                    print(f"Loaded GT SDF: {gt_sdf_path}, Model Output: {model_output_path}")
                    print(f"GT SDF shape: {gt_sdf_tensor.shape}, Model Output shape: {model_output_tensor.shape}")
                    print(f"GT SDF min ; max: {gt_sdf_tensor.min()}, {gt_sdf_tensor.max()}, Model Output min ; max: {model_output_tensor.min()}, {model_output_tensor.max()}")
                
                # Compute loss
                loss = compute_sdf_loss(model_output_tensor, gt_sdf_tensor, alpha=alpha, debug=debug, filename=os.path.splitext(filename)[0])
                if loss.item() == 0:
                    if debug:
                        print(f"Warning: Loss is zero for {filename}.")
                total_loss += loss.item()
                count += 1

                if debug:
                    print(f"Loss for {filename}: {loss.item()}")
            except Exception as e:  # Correctly pair the except block
                print(f"Error computing loss for {filename}: {e}")
                raise e

    average_loss = total_loss / count if count > 0 else 0
    print(f"Average Loss: {average_loss}")
    return average_loss

def save_surface_mask(surface_mask, output_dir, filename="surface_mask.png"):
    """
    Save the surface mask as a PNG file in the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    surface_mask_np = surface_mask.cpu().numpy().squeeze()  # Convert to NumPy array
    plt.imsave(output_path, surface_mask_np, cmap='gray')
    print(f"Surface mask saved to {output_path}")

def save_model_output_norm(grad_norm, output_dir, filename="grad_norm.png"):
    """
    Save the gradient norm of the model output as a PNG file in the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure grad_norm is a valid tensor
    if not isinstance(grad_norm, torch.Tensor):
        raise ValueError("grad_norm must be a torch.Tensor")
    
    # Convert to NumPy array and ensure it is squeezed to 2D
    grad_norm_np = grad_norm.detach().cpu().numpy().squeeze()
    if grad_norm_np.ndim != 2:
        raise ValueError("grad_norm must be squeezable to a 2D array for saving as an image")
    
    # Save the gradient norm as a PNG file
    output_path = os.path.join(output_dir, filename)
    plt.imsave(output_path, grad_norm_np, cmap='viridis')
    print(f"Gradient norm saved to {output_path}")

def visualize_comparisons(model_output_dir, gt_sdf_dir, surface_masks_dir, normal_masks_dir, debug=False):
    """
    Visualize comparisons of model output, ground truth SDF, surface masks, and normal masks.
    Split the 11 comparisons into 3 separate windows.
    """
    filenames = [f for f in os.listdir(model_output_dir) if f.endswith(".npy")]
    filenames = sorted(filenames)[:11]  # Ensure we only process up to 11 files

    if debug:
        print(f"Visualizing comparisons for {len(filenames)} files...")

    # Split filenames into 3 groups for separate windows
    groups = [filenames[:4], filenames[4:8], filenames[8:11]]

    for group_index, group in enumerate(groups):
        if debug:
            print(f"Visualizing group {group_index + 1} with {len(group)} images...")

        num_images = len(group)
        fig, axs = plt.subplots(num_images, 4, figsize=(16, num_images * 4))

        for i, filename in enumerate(group):
            base_name = os.path.splitext(filename)[0]

            # Load images
            model_output = np.load(os.path.join(model_output_dir, filename))
            gt_sdf = np.load(os.path.join(gt_sdf_dir, filename))
            surface_mask_path = os.path.join(surface_masks_dir, f"{base_name}.png")
            normal_mask_path = os.path.join(normal_masks_dir, f"{base_name}.png")

            surface_mask = plt.imread(surface_mask_path) if os.path.exists(surface_mask_path) else None
            normal_mask = plt.imread(normal_mask_path) if os.path.exists(normal_mask_path) else None

            # Plot model output
            axs[i, 0].imshow(model_output, cmap='viridis')
            axs[i, 0].set_title(f"Model Output: {base_name.split('.')[0]}", fontsize=10)
            axs[i, 0].axis('off')

            # Plot ground truth SDF
            axs[i, 1].imshow(gt_sdf, cmap='viridis')
            axs[i, 1].set_title("Ground Truth SDF", fontsize=10)
            axs[i, 1].axis('off')

            # Plot surface mask
            if surface_mask is not None:
                axs[i, 2].imshow(surface_mask, cmap='gray')
                axs[i, 2].set_title("Surface Mask", fontsize=10)
            else:
                axs[i, 2].text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=12)
            axs[i, 2].axis('off')

            # Plot normal mask
            if normal_mask is not None:
                axs[i, 3].imshow(normal_mask, cmap='viridis')
                axs[i, 3].set_title("Normal", fontsize=10)
            else:
                axs[i, 3].text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=12)
            axs[i, 3].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Paths
    model_output = os.path.join("..", "intermediate_data", "sdf_euclidean")
    gt_sdf = os.path.join("..", "intermediate_data", "sdf_euclidean")
    surface_masks_dir = os.path.join("..", "intermediate_data", "loss_surface_masks")
    normal_dir = os.path.join("..", "intermediate_data", "loss_normals")


    # Run the analyze test with the old SDF loss implementation
    alpha = 100.0  # Pass alpha as a parameter
    analyze_test(model_output, gt_sdf, sigma=2, alpha=alpha, debug=True)

    # Visualize comparisons after the test is finished
    visualize_comparisons(model_output, gt_sdf, surface_masks_dir, normal_dir, debug=True)



