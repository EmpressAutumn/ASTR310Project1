#%% Authors -  Autumn Hoffensetz, Evelynn Chara McNeil

import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
from tqdm import tqdm

def imshift(im, nr, nc, rotate=False):
    """
    Shift image by (nr, nc) with NaN padding for proper stack summation.
    Positive nr shifts DOWN.
    Positive nc shifts RIGHT.
    """

    # Force float for NaN safety & stacking
    im = np.asarray(im, dtype=np.float64)

    # Rotate 180 if needed
    if rotate:
        im = np.rot90(im, 2)

    a, b = im.shape
    out = np.full_like(im, np.nan)

    # Source region
    src_r1 = max(0, -nr)
    src_r2 = min(a, a - nr)
    src_c1 = max(0, -nc)
    src_c2 = min(b, b - nc)

    # Destination region
    dst_r1 = max(0, nr)
    dst_r2 = min(a, nr + a)
    dst_c1 = max(0, nc)
    dst_c2 = min(b, nc + b)

    # Copy overlapping region
    if src_r1 < src_r2 and src_c1 < src_c2:
        out[dst_r1:dst_r2, dst_c1:dst_c2] = \
            im[src_r1:src_r2, src_c1:src_c2]

    return out

#%% Define functions

def load_images(path, num_images, filter_name, file_prefix):
    images = []  # this creates an unfilled list
    exptime = 0

    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        
        try:
            hdu = fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue

        images.append([np.array(hdu.data)])
        exptime = hdu.header["EXPTIME"]
    print("Created a list containing each image")

    print("Stacked each image matrix")
    return np.vstack(images), exptime

def median_combine(image_array):
    # Transpose the matrix: image[row[column[]]] -> row[column[image[]]]
    array_image = np.transpose(image_array, (1, 2, 0))
    print("Transposed the matrix")

    # Take the median of each pixel value
    for i in tqdm(range(len(array_image)), desc=f"{len(image_array)} Images Median Combination", unit="row"):
        for j in range(len(array_image[i])):
            array_image[i, j] = np.median(array_image[i, j])

    # Remove the duplicate median values (Thank you NumPy for being awful!)
    print("Removed duplicate median values")
    return array_image[:, :, 0]

def autostrip(imshifts):
    for key in imshifts.keys():
        key.strip()
        for i in range(len(imshifts[key])):
            imshifts[key][i] = imshifts[key][i].strip()
    return imshifts

#%% Creating the master biases

# Median combine the bias images
def create_master_bias(image_folder, num_images, filter_name, file_prefix=""):
    # Load the images
    bias, exptime = load_images(f"/users/kushpatel/{image_folder}/BIAS", num_images, filter_name, file_prefix)

    # Median combine the biases
    master_bias = median_combine(bias)
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(master_bias)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits", overwrite = True)
    print('Saved the .fits image')

# Create master biases
# create_master_bias("20250908_07in_NGC6946", 12, "g'")
create_master_bias("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
#create_master_bias("20251003_07in_NGC6946", 7, "i'", "BIAS_NGC 6946_")
#create_master_bias("20251009_07in_NGC6946", 7, "ha", "BIAS_NGC6946_")
#create_master_bias("20251015_07in_NGC6946", 7, "g'", "BIAS_NGC6946_")

#%% Creating the master darks

def create_master_dark(image_folder, num_images, filter_name, file_prefix=""):
    # Load the images
    dark, exptime = load_images(f"/users/kushpatel/{image_folder}/DARK", num_images, filter_name, file_prefix)

    bias_hdu = fits.open(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits")[0]
    
    # Load the master bias and subtract it
    bias = np.asarray(bias_hdu.data, dtype=np.float64)

    bias_subtracted = [
    np.asarray(d, dtype=np.float64) - bias
    for d in dark
    ]

    master_dark = np.nanmedian(np.stack(bias_subtracted), axis=0)


    # Save the combined FITS dark image
    hdu = fits.PrimaryHDU(master_dark)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits", overwrite = True)
    print('Saved the .fits image')

create_master_dark("20250908_07in_NGC6946", 7, "g'")
# Don't create dark for 9/28
create_master_dark("20251003_07in_NGC6946", 7, "i'", "DARK_NGC 6946_")
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")

#%% Reuse the 10/03 master dark for 9/28

dark_1003 = fits.open("/users/kushpatel/20251003_07in_NGC6946/DARK/master_dark.fits")[0]

dark_1003.writeto("/users/kushpatel/20250928_07in_NGC6946/DARK/master_dark.fits", overwrite = True)
print('Saved the .fits image')

#%% Creating the master flats

def create_master_flat(image_folder, num_images, filter_name, file_prefix="", kind=""):
    # Load the images
    images = []  # this creates an unfilled list
    exptime = 0

    bias_hdu = fits.open(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits")[0]
    bias = np.asarray(bias_hdu.data, dtype=np.float64)

    dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
    dark_master = np.asarray(dark_hdu.data, dtype=np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]

    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        try:
            hdu = fits.open(f"/users/kushpatel/{image_folder}/FLAT/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue

        exptime = hdu.header["EXPTIME"]
        img = np.asarray(hdu.data, dtype=np.float64)

        # Scale dark to exposure time
        dark = exptime / dark_exptime * dark_master

        # Calibrate flat
        img = img - bias - dark

        # Normalize by its own median (illumination correction)
        divider = np.nanmedian(img)
        img /= divider
        images.append(img)

    if len(images) == 0:
        raise RuntimeError("No flat images were successfully loaded.")

    # TRUE PIXEL-WISE MEDIAN COMBINE
    flat_stack = np.stack(images, axis=0)
    master_flat = np.nanmedian(flat_stack, axis=0)

    # Save the combined FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    hdu.writeto(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}{kind}.fits", overwrite = True)
    print('Saved the .fits image')

# create_master_flat("20250908_07in_NGC6946", 12, "g'")

create_master_flat("20250928_07in_NGC6946", 10, "g'", "NGC6946_")
create_master_flat("20250928_07in_NGC6946", 9, "ha", "NGC6946_")
"""
create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_NGC 6946_","dome")
create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_SKYFLAT_",'sky')

create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_skyflats_","sky")

create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_SKYFLAT_","sky")


create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_SKYFLAT_","sky")
"""

#%%

def combine_master_flat(image_folder, filter_name, kind):
    mastflats = []
    for i in kind:
        mastflat_hdu = fits.open(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}{i}.fits")[0]
        mastflats.append(np.array(mastflat_hdu.data))

    comb_mastflats = np.stack(mastflats, axis = 0) #only 2 dimensions
    comb_mastflat = median_combine(comb_mastflats)
    print("combined master flats for night and filter")
    
    hdu = fits.PrimaryHDU(comb_mastflat)
    hdu.writeto(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    print("saved the .fits image")


combine_master_flat("20251003_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251009_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251015_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251015_07in_NGC6946", "g'", ["sky", "dome"])




#%%

import numpy as np
from astropy.io import fits
import os

def safe_median(arr):
    return np.nanmedian(arr)

def stats_string(arr):
    return f"shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):.3g}, med={np.nanmedian(arr):.3g}, max={np.nanmax(arr):.3g}, std={np.nanstd(arr):.3g}"

def calibrate_single_science_debug(
    science_path,
    master_bias_path,
    master_dark_path,
    master_flat_path,
    out_path=None,
    xshift=0,
    yshift=0,
    rotate_180=False,
    save_intermediates=False,
    flat_min_threshold=1e-3
):
    """
    Calibrate ONE science image with many safety checks and diagnostics.
    Returns calibrated image (float64) and prints step-by-step stats.

    flat_min_threshold: any flat pixel <= this will be treated as bad (masked).
                        Increase if your flats have extremely small values.
    save_intermediates: saves bias-subtracted, dark-subtracted, flat-normalized FITS for inspection.
    """

    # -------- load science --------
    with fits.open(science_path) as hdul:
        sci = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
        exptime_sci = header.get("EXPTIME", None)

    print("SCIENCE:", science_path)
    print(" science stats (raw):", stats_string(sci))

    # -------- load bias --------
    with fits.open(master_bias_path) as hdul:
        bias = np.asarray(hdul[0].data, dtype=np.float64)
    print("BIAS:", master_bias_path)
    print(" bias stats:", stats_string(bias))

    # -------- load dark --------
    with fits.open(master_dark_path) as hdul:
        dark_master = np.asarray(hdul[0].data, dtype=np.float64)
        exptime_dark = hdul[0].header.get("EXPTIME", None)
    print("DARK:", master_dark_path)
    print(" dark master stats:", stats_string(dark_master))
    print(" dark exptime:", exptime_dark, "science exptime:", exptime_sci)

    # -------- load flat --------
    with fits.open(master_flat_path) as hdul:
        flat = np.asarray(hdul[0].data, dtype=np.float64)
    print("FLAT:", master_flat_path)
    print(" flat stats (raw):", stats_string(flat))

    # ---- shape checks ----
    if sci.shape != bias.shape or sci.shape != dark_master.shape or sci.shape != flat.shape:
        raise RuntimeError(f"Shape mismatch: sci {sci.shape}, bias {bias.shape}, dark {dark_master.shape}, flat {flat.shape}")

    # -------- dark scaling --------
    if exptime_sci is None or exptime_dark is None:
        raise RuntimeError("Missing EXPTIME in headers; cannot scale dark.")
    scale = exptime_sci / exptime_dark
    dark_scaled = dark_master * scale
    print(" dark_scaled stats:", stats_string(dark_scaled))

    # -------- bias + dark subtraction --------
    step1 = sci - bias
    print(" after bias subtraction:", stats_string(step1))

    step2 = step1 - dark_scaled
    print(" after dark subtraction:", stats_string(step2))

    if save_intermediates and out_path is not None:
        base = os.path.splitext(out_path)[0]
        os.makedirs(os.path.dirname(base), exist_ok=True)
        fits.PrimaryHDU(step2.astype(np.float32), header=header).writeto(base + "_bias_dark_sub.fits", overwrite=True)

    # -------- handle flat: normalization + mask bad pixels --------
    # Prevent division by zeros/small numbers: mask flat <= flat_min_threshold
    flat_median = safe_median(flat)
    if flat_median == 0 or np.isnan(flat_median):
        raise RuntimeError("Flat median is zero or NaN; check master_flat file.")

    flat_norm = flat / flat_median
    print(" flat normalized stats:", stats_string(flat_norm))

    # Identify bad flat pixels (too small or NaN)
    bad_flat_mask = (~np.isfinite(flat_norm)) | (flat_norm <= flat_min_threshold)
    n_bad = np.count_nonzero(bad_flat_mask)
    print(f" bad flat pixels: {n_bad} / {flat_norm.size} ({100 * n_bad/flat_norm.size:.3f}%)")

    if n_bad > 0:
        # Replace bad flat pixels by 1.0 (neutral) or interpolated value.
        # Here we set to median of neighbors via a simple nearest-neighbor fill:
        flat_filled = flat_norm.copy()
        flat_filled[bad_flat_mask] = np.nan
        # simple inpainting: replace NaN with median of whole flat_filled (safe fallback)
        global_med = np.nanmedian(flat_filled)
        flat_filled[bad_flat_mask] = global_med if np.isfinite(global_med) else 1.0
        flat_norm = flat_filled
        print(" filled bad flat pixels with global median:", global_med)
        print(" flat_norm stats after fill:", stats_string(flat_norm))

    # -------- divide by flat-field --------
    calibrated = step2 / flat_norm
    print(" after flat division:", stats_string(calibrated))

    if save_intermediates and out_path is not None:
        base = os.path.splitext(out_path)[0]
        fits.PrimaryHDU(calibrated.astype(np.float32), header=header).writeto(base + "_flatdiv.fits", overwrite=True)

    # -------- rotation + shift safety --------
    if rotate_180:
        calibrated = np.rot90(calibrated, 2)
        print(" rotated 180")

    if xshift != 0 or yshift != 0:
        # imshift should handle NaNs and return same shape
        calibrated = imshift(calibrated, yshift, xshift, rotate=False)  # note: imshift signature (nr, nc)
        print(f" shifted by (y,x)=({yshift},{xshift})")

    print(" calibrated final stats:", stats_string(calibrated))

    # -------- final save if requested --------
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # save as float32 to reduce size but preserve precision
        fits.PrimaryHDU(calibrated.astype(np.float32), header=header).writeto(out_path, overwrite=True)
        print("Saved calibrated image to:", out_path)

    return calibrated

paths = "/users/kushpatel/"
sciencepath = paths + "20251015_07in_NGC6946/LIGHT/LIGHT_NGC6946_0000-g'.fits"
biaspath = paths + "20251015_07in_NGC6946/BIAS/master_bias.fits"
darkpath = paths + "20251015_07in_NGC6946/DARK/master_dark.fits"
flatpath = paths + "20251015_07in_NGC6946/FLAT/master_flat-g'.fits"
outputpath = paths + "20251015_07in_NGC6946/LIGHT/LIGHT_NGC6946_0000-g_calibrated'.fits"

calibrate_single_science_debug(sciencepath,biaspath,darkpath,flatpath,outputpath, xshift=0,
    yshift=0, rotate_180=False, save_intermediates=True, flat_min_threshold=1e-3)


#%%

from scipy.ndimage import median_filter, generic_filter

def fix_hot_pixels(image, threshold_sigma=10, box_size=5):
    """
    Replace extreme high-value pixels with the local median.
    """
    img = image.astype(np.float64)

    med = np.median(img)
    std = np.std(img)

    # Detect extreme pixels only
    hot_mask = img > (med + threshold_sigma * std)

    # Local median image
    local_median = median_filter(img, size=box_size)

    # Replace only hot pixels
    img[hot_mask] = local_median[hot_mask]

    print(f"Replaced {hot_mask.sum()} hot pixels")

def fix_hot_pixels_8conn(image, sigma=10):
    """
    Replace extreme hot pixels using ONLY the 8 surrounding neighbors.
    """
    img = image.astype(np.float64).copy()

    med = np.nanmedian(img)
    std = np.nanstd(img)

    # Detect only extreme outliers
    hot_mask = img > med + sigma * std
    hot_y, hot_x = np.where(hot_mask)

    ny, nx = img.shape
    replaced = 0

    for y, x in zip(hot_y, hot_x):
        y0, y1 = max(0, y-1), min(ny, y+2)
        x0, x1 = max(0, x-1), min(nx, x+2)

        neighborhood = img[y0:y1, x0:x1].flatten()
        neighborhood = neighborhood[neighborhood < img[y, x]]  # exclude the hot pixel itself

        if len(neighborhood) > 0:
            img[y, x] = np.median(neighborhood)
            replaced += 1

    print(f"8-neighbor hot pixels corrected: {replaced}")
    return img

import numpy as np
from scipy.ndimage import median_filter, generic_filter

def fix_hot_pixels_fast(image, sigma=5, size=3, max_iter=4):
    img = image.astype(np.float64).copy()
    total_replaced = 0

    for _ in range(max_iter):
        local_med = median_filter(img, size=size)
        local_std = generic_filter(img, np.nanstd, size=size)

        # Prevent zero-noise failure
        local_std = np.maximum(local_std, 1.0)

        mask = img > local_med + sigma * local_std

        n = np.sum(mask)
        if n == 0:
            break

        img[mask] = local_med[mask]
        total_replaced += n

    print(f"Total replaced pixels: {total_replaced}")
    return img

def clip_and_replace(image, max_adu=64000, size=3):
    bad = image > max_adu
    med = median_filter(image, size=size)
    fixed = image.copy()
    fixed[bad] = med[bad]
    return fixed, np.sum(bad)

import os
from astropy.stats import sigma_clip


def clean_hot_pixels_fast(frame, sigma=8, box=5, max_frac=0.001):
    """
    frame : 2D calibrated image
    sigma : detection threshold above local background
    box   : median filter size (odd)
    max_frac : safety stop (never replace more than this fraction of pixels)
    """
    frame = frame.astype(np.float64)

    # local median background
    med = median_filter(frame, size=box)

    # local residual
    resid = frame - med
    std = np.nanstd(resid)

    bad = resid > sigma * std

    # safety: never touch large star cores
    nbad = np.sum(bad)
    if nbad > max_frac * frame.size:
        print("WARNING: too many pixels flagged, skipping frame")
        return frame, 0

    cleaned = frame.copy()
    cleaned[bad] = med[bad]

    return cleaned, int(nbad)

import numpy as np
from scipy.ndimage import median_filter

def replace_outliers_local(image, contrast_sigma=8):
    """
    Replace extreme pixels using LOCAL contrast, not global sigma.
    Photometrically safe. Preserves NaNs.
    """

    img = image.copy().astype(np.float32)

    # Preserve NaNs exactly
    nan_mask = np.isnan(img)

    # Local median (3x3)
    local_med = median_filter(img, size=3, mode="mirror")

    # Local absolute deviation
    local_dev = np.abs(img - local_med)

    # Robust local noise estimate using MAD
    mad = median_filter(local_dev, size=3, mode="mirror")
    local_sigma = 1.4826 * mad  # Convert MAD → sigma

    # Avoid division by zero
    local_sigma[local_sigma == 0] = np.nanmedian(local_sigma)

    # Outlier detection via LOCAL contrast
    outlier_mask = (img - local_med) > contrast_sigma * local_sigma

    # Replace only spikes
    img[outlier_mask] = local_med[outlier_mask]

    # Restore NaNs
    img[nan_mask] = np.nan

    return img

def replace_spikes_photometry_safe(
    image,
    adu_max=10000,     # HARD physical limit for your system
    contrast_sigma=8  # Local contrast sensitivity
):
    img = image.copy().astype(np.float32)

    # Preserve NaNs
    nan_mask = np.isnan(img)

    # Absolute barrier (ONLY pixels above this even get tested)
    barrier_mask = img > adu_max

    if not np.any(barrier_mask):
        return img  # Fast exit if nothing is extreme

    # Local median
    local_med = median_filter(img, size=3, mode="mirror")

    # Local deviation
    local_dev = np.abs(img - local_med)
    mad = median_filter(local_dev, size=3, mode="mirror")
    local_sigma = 1.4826 * mad

    # Prevent divide-by-zero
    local_sigma[local_sigma == 0] = np.nanmedian(local_sigma)

    # Final outlier test (only above ADU barrier)
    spike_mask = barrier_mask & ((img - local_med) > contrast_sigma * local_sigma)

    # Replace only confirmed spikes
    img[spike_mask] = local_med[spike_mask]

    # Restore NaNs
    img[nan_mask] = np.nan

    print("Pixels replaced:", np.sum(spike_mask))
    return img



#%% Calibrating the science images

from astropy.stats import sigma_clip

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix=""):
    science = []
    exptime = 0
    shifts = np.loadtxt("/users/kushpatel/downloads/Imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)
    
    for i in tqdm(range(num_images)):
        # Load the image
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        try:
            hdu = fits.open(f"/users/kushpatel/{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue
        exptime = hdu.header["EXPTIME"]

        # Load the master bias and subtract it
        bias_hdu = fits.open(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits")[0]
        image = np.array(hdu.data) - np.array(bias_hdu.data)

        # Load the master dark and subtract it, accounting for different exposure times
        dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
        dark = exptime / dark_hdu.header["EXPTIME"] * np.array(dark_hdu.data)
        image = image - dark

        # Load the master flat and divide by it
        flat_hdu = fits.open(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}.fits")[0]
        flat = np.array(flat_hdu.data, dtype=np.float64)
        calibrated_image = image / flat

        x = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][2])
        y = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][1])
        rotate_180 = "Rotate 180" in shifts[f"{file_prefix}{number}-{filter_name}.fits"][4]

        #  ROTATE FIRST
        if rotate_180:
            calibrated_image = np.rot90(calibrated_image, 2)
        
        #  SHIFT SECOND
        calibrated_image = imshift(calibrated_image, x, y, False)

        image_new = replace_spikes_photometry_safe(calibrated_image,adu_max=10000,contrast_sigma=8)  # Local contrast sensitivity
        science.append(image_new)
        # fix_hot_pixels_8conn(calibrated_image, sigma=10)
        # clipped = sigma_clip(calibrated_image, sigma=4, axis=0)
        # calibrated_image = np.asarray(clipped)
        print(f"Calibrated image {number}")

    hot_pixels = 0
    master = np.nansum(science, axis=0)
    # fix_hot_pixels_8conn(master_science, sigma=10)
    #print("MAX BEFORE:", np.nanmax(master_science))
    cleaned = master
    # cleaned = fix_hot_pixels_fast(master_science, sigma=5, size=3, max_iter=4)
    print("MAX AFTER :", np.nanmax(cleaned))
    print(cleaned.shape)
    print(f"Total hot pixels fixed in master science: {hot_pixels}")


    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(cleaned)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"/users/kushpatel/{image_folder}/LIGHT/master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")


# calibrate_science_images("20250908_07in_NGC6946", 10, "g'")

#calibrate_science_images("20250928_07in_NGC6946", 10, "g'","NGC6946_")
#calibrate_science_images("20250928_07in_NGC6946", 10, "ha","NGC6946_")

"""
calibrate_science_images("20251003_07in_NGC6946", 14, "ha", "LIGHT_NGC 6946_")

calibrate_science_images("20251009_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 23, "g'", "LIGHT_NGC6946_")
"""
calibrate_science_images("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")


#%%

def calibrate_science_images_safe(image_folder, num_images, filter_name, file_prefix=""):

    science = []
    exptime = None

    shifts = np.loadtxt(
        "/users/kushpatel/downloads/Imshifts.txt",
        delimiter=",", skiprows=1, dtype=str
    )
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    for i in tqdm(range(num_images)):
        number = str(i).zfill(4)

        try:
            hdu = fits.open(
                f"/users/kushpatel/{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits"
            )[0]
        except FileNotFoundError:
            continue

        exptime = hdu.header["EXPTIME"]
        image = hdu.data.astype(np.float64)

        # --- Calibration ---
        bias = fits.getdata(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits")
        image -= bias

        dark = fits.getdata(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")
        dark_exptime = fits.getheader(
            f"/users/kushpatel/{image_folder}/DARK/master_dark.fits"
        )["EXPTIME"]
        image -= dark * (exptime / dark_exptime)

        flat = fits.getdata(
            f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}.fits"
        ).astype(np.float64)

        image /= flat

        # --- Rotation then shift ---
        x = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][2])
        y = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][1])
        rotate_180 = "Rotate 180" in shifts[f"{file_prefix}{number}-{filter_name}.fits"][4]

        if rotate_180:
            image = np.rot90(image, 2)

        image = imshift(image, x, y, False)

        # --- CLEAN BEFORE STACK ---
        image, nfix = clean_hot_pixels_fast(image, sigma=8, box=5)
        print(f"Frame {number}: replaced {nfix} pixels")
        calibrated_image = sigma_clip(image,sigma=4,maxiters=2,masked=False)
        science.append(calibrated_image.astype(np.float32))

    # --- STACK SAFELY ---
    stack = np.stack(science, axis=0)
    master = np.nansum(stack, axis=0)

    hdu = fits.PrimaryHDU(master.astype(np.float32))
    hdu.header["EXPTIME"] = exptime * stack.shape[0]
    hdu.writeto(
        f"/users/kushpatel/{image_folder}/LIGHT/master_science-{filter_name}.fits",
        overwrite=True
    )
    print("MASTER MIN:", np.nanmin(master))
    print("MASTER MED:", np.nanmedian(master))
    print("MASTER MAX:", np.nanmax(master))
    print("Pixels > 60000:", np.sum(master > 60000))
    print("Saved final photometric-safe master image")

calibrate_science_images_safe("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")

#%% Merge the images into one image

def final_shift(image_folders, filter_name):
    science = []
    shifts = np.loadtxt("/users/kushpatel/desktop/Imshifts.txt", delimiter = ',' , skiprows = 1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    autostrip(shifts)

    exptime = 0
    
    for i in range(len(image_folders)):
        # Load the image
        image_hdu = fits.open(f"{image_folders[i]}/LIGHT/master_science-{filter_name}.fits")[0]
        exptime += image_hdu.header["EXPTIME"]

        # Rotate the image
        print(f"Rotating by {float(shifts[f'{image_folders[i]}-{filter_name}'][3])} degrees.")
        #image = rotate(image_hdu.data, float(shifts[f"{image_folders[i]}-{filter_name}"][3]), reshape=False)

        # Shift the image
        image = imshift(image_hdu.data, int(shifts[f"{image_folders[i]}-{filter_name}"][2]), int(shifts[f"{image_folders[i]}-{filter_name}"][1]))
        science.append(np.array([image])[:image_hdu.data.shape[0], :image_hdu.data.shape[1]])

        output_hdu = fits.PrimaryHDU(np.array(image))
        output_hdu.writeto(f"output/master_science-{filter_name}-{i}.fits", overwrite=True)

    master_science = np.transpose(np.vstack(science), (1, 2, 0))
    print("Transposed the matrix")

    # Take the median of each pixel value
    for i in tqdm(range(len(master_science)), desc=f"{len(master_science)} Master Science Sum", unit="row"):
        for j in range(len(master_science[i])):
            master_science[i, j] = np.sum(master_science[i, j])

    # Remove the duplicate median values (Thank you NumPy for being awful!)
    print("Removed duplicate median values")
    master_science = master_science[:, :, 0]

    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(master_science)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")

#final_shift(["20250908_07in_NGC6946", "20251015_07in_NGC6946"], "g'")
final_shift(["20251003_07in_NGC6946", "20251015_07in_NGC6946"], "ha")

# %%
