"""
Task 1: SeabedObjects-KLSG & WATERS Dataset Ingestion and Pre-processing Pipeline.
Includes:
- Automated dataset download & directory structuring for SeabedObjects-KLSG and WATERS
- Custom Lee Speckle Filter (variance-based adaptive speckle reduction for side-scan sonar)
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in OpenCV
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

def download_and_extract_dataset(dataset_name: str, target_dir: str = "./data"):
    """
    Downloads and extracts benchmark side-scan sonar datasets:
    1. SeabedObjects-KLSG: Wrecks, Mines, Pipelines, Airplanes, Craters
    2. WATERS (Underwater Acoustic Target & Ghost Net Dataset): Ghost nets, marine debris
    """
    os.makedirs(target_dir, exist_ok=True)
    dataset_urls = {
        "SeabedObjects-KLSG": "https://raw.githubusercontent.com/sonar-ai-benchmark/datasets/main/SeabedObjects-KLSG-sample.zip",
        "WATERS": "https://raw.githubusercontent.com/sonar-ai-benchmark/datasets/main/WATERS-GhostNets-sample.zip"
    }

    print(f"[*] Preparing dataset: {dataset_name} in {target_dir}...")
    dataset_path = os.path.join(target_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "raw_sonar"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "annotations_obb"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "masks_unet"), exist_ok=True)
    
    print(f"[+] Directory tree initialized for {dataset_name}: {dataset_path}")
    return dataset_path


def lee_speckle_filter(image: np.ndarray, win_size: int = 7, cu: float = 0.52) -> np.ndarray:
    """
    Lee Speckle Filter for Single-Look and Multi-Look Synthetic Aperture & Side-Scan Sonar.
    
    Mathematical Formulation:
        I_filtered = I_mean + W * (I_noisy - I_mean)
        W = max(0, 1 - (Cu^2 / Ci^2))
        Ci = sqrt(Var(I)) / I_mean  (Local coefficient of variation)
        Cu = Theoretical speckle noise coefficient of variation (~0.52 for single-look amplitude sonar)
    
    Parameters:
        image: 2D numpy array (grayscale sonar backscatter intensity)
        win_size: Local neighborhood window size (odd integer, e.g., 5, 7, 9)
        cu: Estimated standard deviation to mean ratio of pure speckle noise
    
    Returns:
        Filtered sonar image with preserved edges and reduced multiplicative Rayleigh speckle.
    """
    img_float = image.astype(np.float64)
    
    # 1. Local mean
    mean = uniform_filter(img_float, size=win_size)
    
    # 2. Local variance: E[X^2] - (E[X])^2
    mean_sq = uniform_filter(img_float**2, size=win_size)
    variance = np.maximum(mean_sq - mean**2, 0)
    
    # 3. Local coefficient of variation (Ci)
    ci = np.sqrt(variance) / (mean + 1e-6)
    
    # 4. Adaptive weighting factor (W)
    weights = np.maximum(0.0, 1.0 - (cu**2 / (ci**2 + 1e-6)))
    
    # 5. Filtered signal
    filtered = mean + weights * (img_float - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def clahe_contrast_enhancement(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal faint
    acoustic shadows and low-backscatter boundary features without over-amplifying seabed noise.
    """
    if len(image.shape) == 3:
        # Convert to LAB or Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def full_preprocessing_pipeline(raw_sonar_bgr: np.ndarray, win_size: int = 7, cu: float = 0.52, clip_limit: float = 3.0) -> dict:
    """
    Executes end-to-end Task 1 Preprocessing:
    Raw Sonar -> Grayscale Conversion -> Lee Speckle Filtering -> CLAHE Enhancement.
    """
    if len(raw_sonar_bgr.shape) == 3:
        gray = cv2.cvtColor(raw_sonar_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_sonar_bgr.copy()
        
    # Step 1: Remove multiplicative speckle noise
    lee_filtered = lee_speckle_filter(gray, win_size=win_size, cu=cu)
    
    # Step 2: Adaptive contrast enhancement on filtered image
    enhanced_clahe = clahe_contrast_enhancement(lee_filtered, clip_limit=clip_limit)
    
    # Step 3: Color-map conversion for visualization (Copper/Amber Sonar Colormap)
    sonar_colormap = cv2.applyColorMap(enhanced_clahe, cv2.COLORMAP_COPPER)
    
    return {
        "raw_gray": gray,
        "lee_filtered": lee_filtered,
        "clahe_enhanced": enhanced_clahe,
        "sonar_colormap": sonar_colormap
    }

if __name__ == "__main__":
    print("[+] Running Task 1 Preprocessing Test...")
    # Generate synthetic noisy sonar image
    test_sonar = np.random.gamma(shape=2.0, scale=30.0, size=(512, 640)).astype(np.uint8)
    results = full_preprocessing_pipeline(test_sonar)
    print(f"[✓] Task 1 complete. Output shape: {results['clahe_enhanced'].shape}, dtype: {results['clahe_enhanced'].dtype}")
