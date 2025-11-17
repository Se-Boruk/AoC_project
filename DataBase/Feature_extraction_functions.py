import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import hog




def symmetry_scores_ssim(img):
    """
    Computes horizontal and vertical symmetry using SSIM.
    Returns values in [0,1], where 1 = perfectly symmetric.
    """

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)
    H, W = gray.shape

    # ---- Vertical symmetry: left vs mirrored right ----
    mid_v = W // 2
    left = gray[:, :mid_v]
    right = gray[:, W - mid_v:]
    right_flip = np.flip(right, axis=1)

    vert_sym = ssim(left, right_flip, data_range=left.max() - left.min())

    # ---- Horizontal symmetry: top vs mirrored bottom ----
    mid_h = H // 2
    top = gray[:mid_h, :]
    bottom = gray[H - mid_h:, :]
    bottom_flip = np.flip(bottom, axis=0)

    horiz_sym = ssim(top, bottom_flip, data_range=top.max() - top.min())

    return float(vert_sym), float(horiz_sym)




def img_entropy(img, n_bins=256):
    """
    Compute the Shannon entropy of an image.

    Args:
        img: np.array of shape (H,W,C) or (H,W)
        n_bins: number of histogram bins (default 256)
    
    Returns:
        entropy: float, higher = more complex / more information
    """
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Ensure 0-255 uint8
    if gray.dtype != np.uint8:
        gray = (gray*255).astype(np.uint8)

    # Compute histogram
    hist, _ = np.histogram(gray, bins=n_bins, range=(0, 256), density=True)

    # Remove zero entries (log(0) undefined)
    hist_nonzero = hist[hist > 0]

    # Shannon entropy
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return entropy


def radial_power_spectrum(img, n_bins=100):
    """
    Compute radially averaged power spectrum of an image.

    Args:
        img: np.array, grayscale or RGB (use grayscale)
        n_bins: number of radial bins

    Returns:
        radial_prof: np.array of length n_bins
    """
    # Convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)

    # 2D Fourier transform
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    power = np.abs(Fshift)**2

    H, W = gray.shape
    y, x = np.indices((H, W))
    cx, cy = W // 2, H // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Radial binning
    r_flat = r.flatten()
    power_flat = power.flatten()
    radial_bins = np.linspace(0, r.max(), n_bins+1)
    radial_prof = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        mask = (r_flat >= radial_bins[i]) & (r_flat < radial_bins[i+1])
        if np.any(mask):
            radial_prof[i] = power_flat[mask].mean()
        else:
            radial_prof[i] = 0.0

    # Optional: normalize
    radial_prof /= radial_prof.sum() + 1e-8

    return radial_prof




def color_moments(img):
    """
    Compute first three color moments (mean, variance, skewness) per channel
    and return as a dictionary with descriptive keys.

    Args:
        img: np.array, shape (H,W,3) RGB

    Returns:
        moments_dict: dict with keys like 'R_mean', 'R_var', 'R_skew', etc.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be RGB with 3 channels")

    channels = ['R', 'G', 'B']
    moments_dict = {}

    for i, c in enumerate(channels):
        channel = img[:, :, i].astype(np.float32)
        moments_dict[f'{c}_mean'] = np.mean(channel)
        moments_dict[f'{c}_var']  = np.var(channel)
        moments_dict[f'{c}_skew'] = skew(channel.flatten())

    return moments_dict



def colorfulness_saturation(img):
    """
    Compute Hasler-Süsstrunk colorfulness and mean saturation.

    Args:
        img: np.array, shape (H,W,3) RGB

    Returns:
        dict: {'colorfulness': ..., 'mean_saturation': ...}
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be RGB with 3 channels")

    # Convert to float32
    img = img.astype(np.float32)

    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    # Hasler–Süsstrunk colorfulness
    rg = R - G
    yb = 0.5 * (R + G) - B

    sigma_rg = np.std(rg)
    sigma_yb = np.std(yb)
    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    colorfulness = np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2)

    # Mean saturation
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mean_saturation = np.mean(hsv[:,:,1])  # 0-255

    return {'colorfulness': colorfulness, 'mean_saturation': mean_saturation}



def color_harmony_contrast(img, n_bins=32):
    """
    Compute color harmony / contrast features:
    - Inter-channel contrasts (std of differences)
    - Histogram overlaps between channels

    Args:
        img: np.array RGB
        n_bins: histogram bins

    Returns:
        dict with features
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be RGB")

    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    features = {}

    # Inter-channel contrasts
    features['RG_contrast'] = np.std(R - G)
    features['RB_contrast'] = np.std(R - B)
    features['GB_contrast'] = np.std(G - B)

    # Histogram overlaps (normalized cross-correlation)
    hist_R, _ = np.histogram(R, bins=n_bins, range=(0,255), density=True)
    hist_G, _ = np.histogram(G, bins=n_bins, range=(0,255), density=True)
    hist_B, _ = np.histogram(B, bins=n_bins, range=(0,255), density=True)

    def hist_overlap(h1, h2):
        return np.sum(np.minimum(h1, h2))  # sum of min values

    features['hist_overlap_RG'] = hist_overlap(hist_R, hist_G)
    features['hist_overlap_RB'] = hist_overlap(hist_R, hist_B)
    features['hist_overlap_GB'] = hist_overlap(hist_G, hist_B)

    return features





def lbp_histogram(img, P=8, R=1):
    """
    Compute uniform Local Binary Pattern (LBP) histogram for an image.

    Args:
        img (np.array): Input image, either grayscale or RGB.
        P (int): Number of circular neighbors (default=8)
        R (int): Radius for LBP (default=1)

    Returns:
        hist (np.array): Normalized histogram of LBP codes (fixed length)
    """
    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Compute LBP using uniform patterns
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')

    # Number of output bins for uniform LBP: P*(P-1) + 3
    n_bins = int(lbp.max() + 1)

    # Compute histogram and normalize
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist  # sum(hist) == 1



def contour_statistics(img):
    """
    Compute contour-based features:
    - Number of contours
    - Mean contour length
    - Mean contour area
    - Mean aspect ratio of bounding rectangles
    
    Args:
        img (np.array): RGB or grayscale image
    
    Returns:
        dict: feature dictionary
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Apply simple thresholding to get binary image
    # You can tune thresholding method depending on dataset
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_contours = len(contours)
    
    # Avoid division by zero
    if n_contours == 0:
        return {
            'n_contours': 0,
            'mean_contour_length': 0,
            'mean_contour_area': 0,
            'mean_contour_aspect': 0
        }

    # Compute contour lengths, areas, aspect ratios
    lengths = np.array([cv2.arcLength(c, closed=True) for c in contours])
    areas = np.array([cv2.contourArea(c) for c in contours])
    aspect_ratios = np.array([
        (cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3]) if cv2.boundingRect(c)[3] != 0 else 0
        for c in contours
    ])

    features = {
        'n_contours': n_contours,
        'mean_contour_length': lengths.mean(),
        'mean_contour_area': areas.mean(),
        'mean_contour_aspect': aspect_ratios.mean()
    }

    return features



def haralick_features(img, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Computes Haralick texture features (GLCM-based).
    Input: RGB uint8 image (0-255)
    Returns: dict of mean and std of 6 Haralick properties
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feats = {}
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        arr = graycoprops(glcm, prop)
        feats[f'glcm_{prop}_mean'] = arr.mean()
        feats[f'glcm_{prop}_std'] = arr.std()
    return feats


def gabor_features(img, frequencies=[0.1,0.2,0.4], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Multi-scale, multi-orientation Gabor filter responses.
    Input: RGB uint8 image (0-255)
    Returns: dict of mean/std per frequency/theta
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = {}
    for f in frequencies:
        for t in thetas:
            k = cv2.getGaborKernel(ksize=(31,31), sigma=4.0, theta=t, lambd=1.0/f, gamma=0.5)
            r = cv2.filter2D(gray, -1, k)
            feats[f'gabor_f{f:.3f}_t{t:.2f}_mean'] = r.mean()
            feats[f'gabor_f{f:.3f}_t{t:.2f}_std'] = r.std()
    return feats


def hog_features(img, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
    """
    Compute HOG feature histogram for RGB image (converted to grayscale).
    Returns normalized 1D array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = hog(gray,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   feature_vector=True)
    return features  # Already normalized


def radial_spectral_summary(img, n_bins=100):
    """
    Extract spectral slope and low/high frequency energy ratio.
    Input: RGB uint8
    Returns: dict with 'spectral_slope' and 'low_high_ratio'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    power = np.abs(Fshift)**2
    H, W = gray.shape
    y, x = np.indices((H, W))
    cx, cy = W//2, H//2
    r = np.sqrt((x-cx)**2 + (y-cy)**2).flatten()
    p = power.flatten()
    radial_bins = np.linspace(0, r.max(), n_bins+1)
    radial_prof = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
        if np.any(mask):
            radial_prof[i] = p[mask].mean()
    radial_prof /= radial_prof.sum() + 1e-8

    # spectral slope (log-log)
    mask = radial_prof > 0
    x_log = np.log(np.linspace(1,n_bins,n_bins)[mask])
    y_log = np.log(radial_prof[mask])
    slope = np.polyfit(x_log, y_log, 1)[0]
    low_energy = radial_prof[:n_bins//10].sum()
    high_energy = radial_prof[n_bins//2:].sum()
    return {'spectral_slope': slope, 'low_high_ratio': low_energy/(high_energy+1e-8)}


def fractal_dimension(img, threshold=128):
    """
    Estimate fractal dimension using box-counting method.
    Input: RGB uint8 (converted to grayscale)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Z = (gray > threshold).astype(int)
    sizes = 2**np.arange(int(np.log2(min(Z.shape))), 1, -1)
    counts = []
    for size in sizes:
        S = (Z.reshape(Z.shape[0]//size, size, -1, size).sum(axis=(1,3))>0).sum()
        counts.append(S)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]








