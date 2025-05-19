import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import io
import torch.nn.functional as F

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

def firstTerm(sdf_pred, debug=False):
    """
    Compute the first term: ||∇Φ(x)| - 1|| over the domain Ω.
    """
    if debug:
        print("Computing First Term (Eikonal)...")
    # Generate spatial coordinates
    batch_size, _, height, width = sdf_pred.shape
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height, device=sdf_pred.device),
        torch.linspace(-1, 1, width, device=sdf_pred.device),
        indexing="ij"
    )
    coords = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # Shape: [B, H, W, 2]

    # Compute gradients of sdf_pred w.r.t. spatial coordinates
    try:
        gradients = torch.autograd.grad(
            outputs=sdf_pred,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # Shape: [B, H, W, 2]
    except Exception as e:
        if debug:
            print(f"Error in gradient computation for First Term: {e}")
        return torch.tensor(0.0, device=sdf_pred.device)

    grad_norm = torch.norm(gradients, dim=-1)  # Compute gradient norm along spatial dimensions
    eikonal_loss = F.l1_loss(grad_norm, torch.ones_like(grad_norm))  # L1 loss to enforce norm = 1

    if debug:
        print(f"First Term (Eikonal Loss): {eikonal_loss.item()}")

    return eikonal_loss

def secondTerm(sdf_pred, sdf_gt, epsilon=1e-3, debug=False):
    """
    Compute the second term: |Φ(x)| + (1 - ⟨∇Φ(x), n(x)⟩) over the zero-level set Ω₀.
    """
    if debug:
        print("Computing Second Term (Surface)...")

    # Identify surface points where |sdf_gt| < epsilon
    surface_mask = torch.abs(sdf_gt) < epsilon

    # Enforce |sdf_pred| ≈ 0 on surface points
    sdf_loss = torch.abs(sdf_pred[surface_mask]).mean()

    # Compute gradients of sdf_pred and sdf_gt
    gradients_pred = torch.autograd.grad(
        outputs=sdf_pred,
        inputs=sdf_pred,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients_gt = torch.autograd.grad(
        outputs=sdf_gt,
        inputs=sdf_gt,
        grad_outputs=torch.ones_like(sdf_gt),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Normalize gradients to compute normals
    normals_gt = F.normalize(gradients_gt[surface_mask], dim=1)
    normals_pred = F.normalize(gradients_pred[surface_mask], dim=1)

    # Compute cosine similarity loss
    cosine_loss = 1 - (normals_pred * normals_gt).sum(dim=1).mean()

    # Combine losses
    surface_loss = sdf_loss + cosine_loss

    if debug:
        print(f"Second Term (Surface Loss): {surface_loss.item()}")

    return surface_loss

def thirdTerm(sdf_pred, sdf_gt, alpha=100.0, epsilon=1e-3, debug=False):
    """
    Compute the third term: ψ(Φ(x)) = exp(-α * |Φ(x)|) over the off-surface region Ω \ Ω₀.
    """
    if debug:
        print("Computing Third Term (Off-Surface)...")

    # Identify off-surface points where |sdf_gt| >= epsilon
    off_surface_mask = torch.abs(sdf_gt) >= epsilon

    # Compute exponential penalty for off-surface points
    psi = torch.exp(-alpha * torch.abs(sdf_pred[off_surface_mask]))
    off_surface_loss = psi.mean()

    if debug:
        print(f"Third Term (Off-Surface Loss): {off_surface_loss.item()}")

    return off_surface_loss

def compute_sdf_loss_old(pred, normals, mask, alpha=100.0):
    """
    Old implementation of SDF loss.
    """
    grad_x = torch.gradient(pred, dim=(2, 3))
    grad = torch.cat(grad_x, dim=1)
    grad_norm = torch.norm(grad, dim=1, keepdim=True)
    eikonal_term = F.l1_loss(grad_norm, torch.ones_like(grad_norm), reduction='none')
    abs_phi = torch.abs(pred)
    angle_term = 1 - F.cosine_similarity(grad, normals, dim=1, eps=1e-6).unsqueeze(1)
    psi = torch.exp(-alpha * abs_phi)
    on_surface = mask
    off_surface = 1.0 - mask
    loss_on = abs_phi * on_surface + angle_term * on_surface
    loss_off = psi * off_surface
    return eikonal_term.mean() + loss_on.mean() + loss_off.mean()

def sdf_loss(sdf_pred, sdf_gt, alpha=100.0, epsilon=1e-3, debug=False, use_old=False):
    """
    Compute the total SDF loss.
    If `use_old` is True, use the old implementation of SDF loss.
    """
    if use_old:
        if debug:
            print("Using old SDF loss implementation...")
        # Prepare inputs for the old function
        mask = (torch.abs(sdf_gt) < epsilon).float()  # Surface mask
        normals = torch.autograd.grad(
            outputs=sdf_gt,
            inputs=sdf_gt,
            grad_outputs=torch.ones_like(sdf_gt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return compute_sdf_loss_old(sdf_pred, normals, mask, alpha)

    if debug:
        print("Using new SDF loss implementation...")
    first = firstTerm(sdf_pred, debug=debug)
    second = secondTerm(sdf_pred, sdf_gt, epsilon, debug=debug)
    third = thirdTerm(sdf_pred, sdf_gt, alpha, epsilon, debug=debug)
    total_loss = first + second + third
    if debug:
        print(f"Total SDF Loss: {total_loss.item()}")
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

def analyze_test(sdf_dir, processed_segmaps_dir, sigma=2, alpha=100.0, debug=False, use_old=False):
    """
    Preprocess segmentation maps, compute the loss between them and SDFs, and print results.
    """
    # Compute loss between processed segmentation maps and SDFs
    total_loss = 0
    count = 0

    for filename in os.listdir(sdf_dir):
        if filename.endswith(".npy"):
            sdf_path = os.path.join(sdf_dir, filename)
            segmap_path = os.path.join(sdf_dir, filename)  # Use the same SDF directory for model output

            if not os.path.exists(segmap_path):
                if debug:
                    print(f"Segmentation map not found for {filename}. Skipping.")
                continue

            sdf_gt = np.load(sdf_path)
            sdf_pred = np.load(segmap_path)

            # Apply soft normalization to the predicted SDF
            sdf_pred = soft_normalize_sdf(sdf_pred)

            if debug:
                print(f"Loaded SDF GT: {sdf_path}, SDF Pred: {segmap_path}")
                print(f"SDF GT shape: {sdf_gt.shape}, SDF Pred shape: {sdf_pred.shape}")

            # Convert to PyTorch tensors and ensure they require gradients
            sdf_gt_tensor = torch.tensor(sdf_gt, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)
            sdf_pred_tensor = torch.tensor(sdf_pred, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0)

            if debug:
                print(f"sdf_gt_tensor shape: {sdf_gt_tensor.shape}, requires_grad: {sdf_gt_tensor.requires_grad}")
                print(f"sdf_pred_tensor shape: {sdf_pred_tensor.shape}, requires_grad: {sdf_pred_tensor.requires_grad}")

            # Compute loss
            try:
                loss = sdf_loss(sdf_pred_tensor, sdf_gt_tensor, alpha=alpha, debug=debug, use_old=use_old)
                if loss.item() == 0:
                    if debug:
                        print(f"Warning: Loss is zero for {filename}.")
                total_loss += loss.item()
                count += 1

                if debug:
                    print(f"Loss for {filename}: {loss.item()}")
            except Exception as e:
                print(f"Error computing loss for {filename}: {e}")

    average_loss = total_loss / count if count > 0 else 0
    print(f"Average Loss: {average_loss}")
    return average_loss

if __name__ == "__main__":
    # Paths
    sdf_dir = os.path.join("..", "intermediate_data", "sdf_euclidean")
    processed_segmaps_dir = os.path.join("..", "intermediate_data", "segmaps_normalized")

    # Run the analyze test with the old SDF loss implementation
    alpha = 100.0  # Pass alpha as a parameter
    analyze_test(sdf_dir, processed_segmaps_dir, sigma=2, alpha=alpha, debug=False, use_old=True)
