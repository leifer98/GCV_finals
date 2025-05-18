import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
import openslide
from utils import *

class SlideTileDataset(Dataset):
    def __init__(self, slide_dir, grid_dir, tile_size=(256, 256), threshold=0.5, use_thumbnails=False):
        """
        Initialize the dataset with slide and grid directories.

        Args:
            slide_dir (str): Directory containing slide files or thumbnails.
            grid_dir (str): Directory containing grid files.
            tile_size (tuple): Desired tile size (width, height).
            threshold (float): Threshold for heightmap processing.
            use_thumbnails (bool): Whether to use thumbnails instead of full SVS files
        """
        self.slide_dir = slide_dir
        self.grid_dir = grid_dir
        self.tile_size = tile_size
        self.threshold = threshold
        self.use_thumbnails = use_thumbnails

        # Check if we're using thumbnails or SVS files
        if use_thumbnails:
            self.slides = [os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith('_thumb.jpg')]
        else:
            self.slides = [os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith('.svs')]
            
        self.grids = [os.path.join(grid_dir, f) for f in os.listdir(grid_dir) if f.endswith('.data')]

        # Make sure we have at least some data
        assert len(self.slides) > 0, "No slides or thumbnails found!"
        assert len(self.grids) > 0, "No grid files found!"
        
        # It's okay if the counts don't match exactly when using thumbnails
        if not use_thumbnails:
            assert len(self.slides) == len(self.grids), "Mismatch between slides and grids!"

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the resized tile, thresholded heightmap, and other metadata.
        """
        slide_path = self.slides[idx]
        # Use idx % len(self.grids) in case there are more slides than grids
        grid_path = self.grids[idx % len(self.grids)]

        # Debug: Print the slide path being accessed
        print(f"Accessing slide: {slide_path}")

        # Load grid data
        with open(grid_path, 'rb') as f:
            grid_data = pickle.load(f)

        if self.use_thumbnails or slide_path.endswith('.jpg'):
            # Using thumbnail images instead of OpenSlide
            image = Image.open(slide_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tile_np = np.array(image)
            resized_tile = resize_tile(tile_np, target_size=self.tile_size)
            tile_coords = (0, 0)  # Dummy coordinates for thumbnails
        else:
            # Using OpenSlide for SVS files
            slide = openslide.OpenSlide(slide_path)
            tile_coords = grid_data[0] if isinstance(grid_data, list) else grid_data['tiles'][0][0]  # Handle both formats
            tile = slide.read_region(tile_coords, 0, self.tile_size).convert('RGB')
            slide.close()
            tile_np = np.array(tile)
            resized_tile = resize_tile(tile_np, target_size=self.tile_size)

        # Create a dummy heightmap and threshold it
        heightmap = torch.rand(1, 1, *self.tile_size)
        thresholded_heightmap = threshold_heightmap(heightmap, threshold=self.threshold)

        return {
            "resized_tile": resized_tile,
            "thresholded_heightmap": thresholded_heightmap,
            "tile_coords": tile_coords,
            "slide_path": slide_path
        }
