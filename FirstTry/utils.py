import os
import sys
import glob
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from enum import Enum
from tqdm import tqdm
from shutil import copy2, copyfile
from functools import partial
from typing import List, Tuple
from matplotlib.collections import PatchCollection
from colorsys import rgb_to_hsv

# ============
# GLOBAL DEFAULTS (Modify as Needed)
# ============
DEFAULT_ROOT_DIR = './TCGA'
DEFAULT_MAGNIFICATION = 10
DEFAULT_TILE_SIZE = 256
DEFAULT_TISSUE_COVERAGE = 0.5
DEFAULT_NUM_WORKERS = 1

# ============
# ENUMS
# ============
class OTSU_METHOD(Enum):
    OTSU3_FLEXIBLE_THRESH = 0
    OTSU3_LOWER_THRESH = 1
    OTSU3_UPPER_THRESH = 2
    OTSU_REGULAR = 3

# ============
# SDF Loss
# ============
def compute_sdf_loss(pred, normals, mask, alpha=100.0):
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

# ============
# DUMMY UTILS
# ============
def dummy_inputs(batch_size=2, height=128, width=128):
    pred = torch.randn(batch_size, 1, height, width)
    normals = F.normalize(torch.randn(batch_size, 2, height, width), dim=1)
    mask = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    return pred, normals, mask

def threshold_heightmap(heightmap, threshold=0.5):
    return (heightmap > threshold).float()

# ============
# Otsu Threshold
# ============
def otsu3(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1, -1
    for i in range(1, 256):
        for j in range(i + 1, 256):
            p1, p2, p3 = np.hsplit(hist_norm, [i, j])
            q1, q2, q3 = Q[i], Q[j] - Q[i], Q[255] - Q[j]
            if q1 < 1.e-6 or q2 < 1.e-6 or q3 < 1.e-6:
                continue
            b1, b2, b3 = np.hsplit(bins, [i, j])
            m1 = np.sum(p1 * b1) / q1
            m2 = np.sum(p2 * b2) / q2
            m3 = np.sum(p3 * b3) / q3
            v1 = np.sum(((b1 - m1) ** 2) * p1) / q1
            v2 = np.sum(((b2 - m2) ** 2) * p2) / q2
            v3 = np.sum(((b3 - m3) ** 2) * p3) / q3
            fn = v1 * q1 + v2 * q2 + v3 * q3
            if fn < fn_min:
                fn_min = fn
                thresh = i, j
    return thresh

# ============
# WSI SEGMENTATION UTILITIES
# ============
# Note: These require `openslide` and a custom `get_slide_magnification()` function

def _make_segmentation_for_image(file, data_path, slides_meta_data_DF, magnification, output_dir="./intermediate_data"):
    """
    Modified to save intermediate files in a separate directory instead of overwriting resource files.
    """
    from utils import get_slide_magnification
    import openslide
    os.makedirs(output_dir, exist_ok=True)
    fn, data_format = os.path.splitext(os.path.basename(file))
    thumb_path = os.path.join(output_dir, fn + '_thumb.jpg')
    segmap_path = os.path.join(output_dir, fn + '_SegMap.jpg')

    if os.path.exists(thumb_path) and os.path.exists(segmap_path):
        return []

    try:
        slide = openslide.open_slide(file)
    except:
        print(f'Cannot open slide at {file}')
        return {'File': file, 'Error': 'OpenSlide failure'}

    slide_mag = get_slide_magnification(slide, data_format)
    width, height = slide.dimensions
    try:
        thumb = slide.get_thumbnail((width / (slide_mag / magnification), height / (slide_mag / magnification)))
    except openslide.lowlevel.OpenSlideError as err:
        e = sys.exc_info()
        return {'File': file, 'Error': err, 'Details': (e[0], e[1])}

    seg_map, edge_img = _calc_segmentation_for_image(thumb, magnification, OTSU_METHOD.OTSU_REGULAR)
    slide.close()
    thumb.save(thumb_path)
    seg_map.save(segmap_path)
    return []


def _calc_segmentation_for_image(image: Image, magnification: int, otsu_method: OTSU_METHOD):
    image_array = np.array(image.convert('CMYK'))[:, :, 1]
    image_array_rgb = np.array(image)
    image_is_black = np.prod(image_array_rgb, axis=2) < 20**3
    image_array[image_is_black] = 0

    if otsu_method in [OTSU_METHOD.OTSU3_FLEXIBLE_THRESH, OTSU_METHOD.OTSU3_LOWER_THRESH]:
        thresh = otsu3(image_array)
        _, seg_map = cv.threshold(image_array, thresh[0], 255, cv.THRESH_BINARY)
    elif otsu_method == OTSU_METHOD.OTSU3_UPPER_THRESH:
        thresh = otsu3(image_array)
        _, seg_map = cv.threshold(image_array, thresh[1], 255, cv.THRESH_BINARY)
    else:
        _, seg_map = cv.threshold(image_array, 0, 255, cv.THRESH_OTSU)

    kernel = np.ones((10 * magnification, 10 * magnification), dtype=np.float32) / (10 * magnification)**2
    seg_map_filt = cv.filter2D(seg_map, -1, kernel)
    seg_map_filt[seg_map_filt > 5] = 255
    seg_map_filt[seg_map_filt <= 5] = 0

    seg_map *= (seg_map_filt > 0)
    seg_map_PIL = Image.fromarray(seg_map)
    edge_img = cv.Canny(seg_map, 1, 254)
    kernel_dilate = np.ones((3, 3))
    edge_img = cv.dilate(edge_img, kernel_dilate, iterations=magnification * 2)
    return seg_map_PIL, Image.fromarray(edge_img).convert('RGB')

# ============
# ⚠️ MISSING / REQUIRED FOR FULL FUNCTIONALITY
# ============
# 1. `get_slide_magnification()` must exist in utils or another importable module.
# 2. `openslide` must be installed and configured correctly.
# 3. Data folder structure expected at `ROOT_DIR/SegData/Thumbs/` and `SegMaps/`.
# 4. Excel input: `slides_data.xlsx` must exist and be properly formatted.
# 5. You may need to set up file permissions if working on shared systems.

# Optional future additions: _make_grid_for_image, _legit_grid, make_grid, make_segmentations
# ... those require deeper integration and are omitted here for clarity

def custom_radon(input_tensor, theta):
    """
    Perform the Radon transform on the input tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        theta (torch.Tensor): Angles (in degrees) at which to compute the Radon transform.
    
    Returns:
        torch.Tensor: Radon transform of the input tensor.
    """
    batch_size, channels, height, width = input_tensor.shape
    radon_output = []
    for i in range(batch_size):
        img = input_tensor[i, 0].cpu().numpy()  # Assuming single-channel input
        radon_img = np.array([np.sum(img, axis=0) for _ in theta])  # Simplified Radon transform
        radon_output.append(torch.from_numpy(radon_img).unsqueeze(0))
    return torch.stack(radon_output).to(input_tensor.device)

def custom_iradon(input_tensor, theta, filter_name="ramp"):
    """
    Perform the inverse Radon transform on the input tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [batch_size, channels, projections, width].
        theta (torch.Tensor): Angles (in degrees) used for the Radon transform.
        filter_name (str): Filter to use for reconstruction (default: "ramp").
    
    Returns:
        torch.Tensor: Reconstructed image from the inverse Radon transform.
    """
    batch_size, channels, projections, width = input_tensor.shape
    iradon_output = []
    for i in range(batch_size):
        radon_img = input_tensor[i, 0].cpu().numpy()  # Assuming single-channel input
        reconstructed_img = np.sum(radon_img, axis=0).reshape(width, width)  # Simplified inverse Radon
        iradon_output.append(torch.from_numpy(reconstructed_img).unsqueeze(0))
    return torch.stack(iradon_output).to(input_tensor.device)

def to_np(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        tensor (torch.Tensor): Input PyTorch tensor.
    
    Returns:
        np.ndarray: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy()

def print_svs_resolutions_in_folder():
    """
    Print the resolutions of all .svs files in a predefined relative folder.

    The folder path is defined as a relative path within the function.
    """
    import openslide

    # Define the relative path to the folder containing .svs files
    folder_path = "../Recources/example_tcga_slides"
    
    svs_files = glob.glob(os.path.join(folder_path, "*.svs"))
    if not svs_files:
        print(f"No .svs files found in {folder_path}")
        return

    for svs_file in svs_files:
        try:
            slide = openslide.OpenSlide(svs_file)
            width, height = slide.dimensions
            print(f"File: {os.path.basename(svs_file)} | Resolution: {width}x{height}")
            slide.close()
        except Exception as e:
            print(f"Error reading {svs_file}: {e}")

def make_grid(ROOT_DIR: str = './TCGA',
              tile_sz: int = 256,
              tissue_coverage: float = 0.5,
              desired_magnification: int = 10,
              num_workers: int = 1,
              output_dir="./intermediate_data"):
    """
    Modified to save grid data in a separate directory instead of overwriting resource files.
    """
    os.makedirs(output_dir, exist_ok=True)
    grids_dir = os.path.join(output_dir, f'Grids_{desired_magnification}')
    grid_images_dir = os.path.join(output_dir, f'GridImages_{desired_magnification}_{tissue_coverage}')
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(grid_images_dir, exist_ok=True)

    slides_data_file = os.path.join(ROOT_DIR, 'slides_data.xlsx')
    slides_meta_data_DF = pd.read_excel(slides_data_file)
    files = slides_meta_data_DF['file'].tolist()

    meta_data_DF = pd.DataFrame(files, columns=['file'])
    slides_meta_data_DF.set_index('file', inplace=True)
    meta_data_DF.set_index('file', inplace=True)

    tile_nums = []
    total_tiles = []

    print('Starting Grid production...')

    with multiprocessing.Pool(num_workers) as pool:
        for tile_nums1, total_tiles1 in tqdm(pool.imap(partial(_make_grid_for_image,
                                                               meta_data_DF=slides_meta_data_DF,
                                                               ROOT_DIR=ROOT_DIR,
                                                               tissue_coverage=tissue_coverage,
                                                               tile_sz=tile_sz,
                                                               desired_magnification=desired_magnification,
                                                               grids_dir=grids_dir,
                                                               grid_images_dir=grid_images_dir),
                                                       files), total=len(files)):
            tile_nums.append(tile_nums1)
            total_tiles.append(total_tiles1)

    slide_usage = ((np.array(tile_nums) / np.array(total_tiles)) * 100).astype(int)

    meta_data_DF[f'Legitimate tiles - {tile_sz} compatible @ X{desired_magnification}'] = tile_nums
    meta_data_DF[f'Total tiles - {tile_sz} compatible @ X{desired_magnification}'] = total_tiles
    meta_data_DF[f'Slide tile usage [%] (for {tile_sz}^2 Pix/Tile) @ X{desired_magnification}'] = slide_usage
    meta_data_DF['bad segmentation'] = ''

    meta_data_DF.to_excel(os.path.join(grids_dir, 'Grid_data.xlsx'))


def _make_grid_for_image(file, meta_data_DF, ROOT_DIR,
                         tissue_coverage, tile_sz, desired_magnification, grids_dir, grid_images_dir):
    """
    Modified to save grid data in a separate directory instead of overwriting resource files.
    """
    filename = '.'.join(os.path.basename(file).split('.')[:-1])
    grid_file = os.path.join(grids_dir, f'{filename}--tlsz{tile_sz}.data')
    segmap_file = os.path.join(ROOT_DIR, 'SegData', 'SegMaps', f'{filename}_SegMap.jpg')

    if os.path.isfile(os.path.join(ROOT_DIR, file)) and os.path.isfile(segmap_file):
        slide = openslide.open_slide(os.path.join(ROOT_DIR, file))
        height, width = slide.dimensions
        data_format = file.split('.')[-1]
        magnification = get_slide_magnification(slide, data_format)

        adjusted_tile_size_at_level = tile_sz * desired_magnification // magnification

        if os.path.isfile(grid_file):
            with open(grid_file, 'rb') as f:
                grid_data = pickle.load(f)
                if (len(grid_data['tiles']) > 0) and (grid_data['tile_size'] == adjusted_tile_size_at_level):
                    slide.close()
                    return len(grid_data['tiles']), len(grid_data['all_tiles'])

    else:
        return 0, 0

    print(f'Creating grid for {file}')

    slide_image = np.array(slide.get_thumbnail((width / magnification, height / magnification)).convert('RGB'))
    slide_mask = np.zeros(slide_image.shape[:2], dtype=np.uint8)

    seg_map = cv.imread(segmap_file, cv.IMREAD_GRAYSCALE)
    seg_map = cv.resize(seg_map, (slide_image.shape[1], slide_image.shape[0]), interpolation=cv.INTER_NEAREST)
    slide_mask[seg_map > 0] = 255

    grid_mask = cv.createGridMask(slide_mask.shape, tileSize=tile_sz, stride=tile_sz, borderBits=0)
    valid_tiles_mask = cv.bitwise_and(slide_mask, grid_mask)

    contours, _ = cv.findContours(valid_tiles_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    tiles = []
    all_tiles = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        if w < adjusted_tile_size_at_level or h < adjusted_tile_size_at_level:
            continue

        tile = slide_image[y:y + h, x:x + w]
        tile_mask = slide_mask[y:y + h, x:x + w]

        tile = cv.resize(tile, (adjusted_tile_size_at_level, adjusted_tile_size_at_level), interpolation=cv.INTER_AREA)
        tile_mask = cv.resize(tile_mask, (adjusted_tile_size_at_level, adjusted_tile_size_at_level), interpolation=cv.INTER_NEAREST)

        if np.mean(tile_mask) < (tissue_coverage * 255):
            continue

        tiles.append((tile, tile_mask))
        all_tiles.append((tile, tile_mask))

    grid_data = {
        'tiles': tiles,
        'all_tiles': all_tiles,
        'tile_size': adjusted_tile_size_at_level
    }
    with open(grid_file, 'wb') as f:
        pickle.dump(grid_data, f)

    slide.close()
    return len(tiles), len(all_tiles)

def resize_tile(tile: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """
    Resize a tile to the target size using OpenCV.

    Args:
        tile (np.ndarray): Input tile as a NumPy array.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Resized tile.
    """
    return cv.resize(tile, target_size, interpolation=cv.INTER_AREA)

def visualize_model_output(checkpoint_path, input_image, tile_size=(256, 256), device=None):
    """
    Visualize the model's output for a given input image using a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        input_image (np.ndarray): Input image as a NumPy array (H, W, C).
        tile_size (tuple): Tile size for resizing the input image.
        device (torch.device): Device to run the model on (default: auto-detect).
    """
    from unet import UnetModel

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = UnetModel(in_chans=3, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Preprocess the input image
    input_image_resized = resize_tile(input_image, target_size=tile_size)
    input_tensor = torch.tensor(input_image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Convert to NCHW

    # Get the model output
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze().cpu().numpy()  # Convert to NumPy array

    # Visualize the input and output
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_image_resized)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Model Output")
    plt.imshow(output, cmap="viridis")
    plt.axis("off")

    plt.show()

def check_and_visualize_segmaps(segmaps_dir, selected_file=None):
    """
    Check the resolution of all images in the segmaps folder and visualize a selected one.

    Args:
        segmaps_dir (str): Path to the segmaps folder.
        selected_file (str): Name of the file to visualize (optional). If None, the first file is visualized.
    """
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    # List all files in the segmaps directory
    segmap_files = [f for f in os.listdir(segmaps_dir) if f.endswith(('.jpg', '.png'))]
    if not segmap_files:
        print(f"No image files found in {segmaps_dir}.")
        return

    print(f"Found {len(segmap_files)} image files in {segmaps_dir}.")

    # Print the resolution of each image
    for file in segmap_files:
        file_path = os.path.join(segmaps_dir, file)
        with Image.open(file_path) as img:
            print(f"File: {file} | Resolution: {img.size}")

    # Visualize the selected file
    if selected_file is None:
        selected_file = segmap_files[0]  # Default to the first file
    if selected_file not in segmap_files:
        print(f"Selected file '{selected_file}' not found in {segmaps_dir}.")
        return

    selected_file_path = os.path.join(segmaps_dir, selected_file)
    with Image.open(selected_file_path) as img:
        plt.figure(figsize=(8, 8))
        plt.title(f"Visualizing: {selected_file}")
        plt.imshow(img)
        plt.axis("off")
        plt.show()



