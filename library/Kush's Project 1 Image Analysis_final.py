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

    # Overlapping region
    if src_r1 < src_r2 and src_c1 < src_c2:
        out[dst_r1:dst_r2, dst_c1:dst_c2] = \
            im[src_r1:src_r2, src_c1:src_c2]

    return out

#%% Define functions

def load_images(path, num_images, filter_name, file_prefix=""):
    images = []
    exptime = None

    for i in range(num_images):
        number = str(i).zfill(4)

        try:
            hdu = fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue

        images.append(hdu.data.astype(np.float64))

        if exptime is None:
            exptime = hdu.header["EXPTIME"]

    images = np.stack(images)  # shape (N, Y, X)
    print("Loaded image stack:", images.shape)
    return images, exptime

def median_combine(image_stack):
    # image_stack shape should be (N, Y, X)
    return np.nanmedian(image_stack, axis=0)

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
create_master_bias("20250908_07in_NGC6946", 12, "g'")
create_master_bias("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_bias("20251003_07in_NGC6946", 7, "i'", "BIAS_NGC 6946_")
create_master_bias("20251009_07in_NGC6946", 7, "ha", "BIAS_NGC6946_")
create_master_bias("20251015_07in_NGC6946", 7, "g'", "BIAS_NGC6946_")

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


def create_master_flat(image_folder, num_images, filter_name, kind= [""]):
    images = []
    exptime = 0

    bias_hdu = fits.open(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits")[0]
    bias = np.asarray(bias_hdu.data, dtype=np.float64)

    dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
    dark_master = np.asarray(dark_hdu.data, dtype=np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]

    for j in kind:
        for i in range(num_images):
            number = str(i)
            while len(number) < 4:
                number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
            try:
                hdu = fits.open(f"/users/kushpatel/{image_folder}/FLAT/{j}{number}-{filter_name}.fits")[0]
            except FileNotFoundError:
                continue

            exptime = hdu.header["EXPTIME"]
            img = np.asarray(hdu.data, dtype=np.float64)

            # Scale dark to exposure time
            dark = exptime / dark_exptime * dark_master

            # Calibrate flat
            img = img - bias - dark

            img[img < 0] = 0  # Clip negative values to zero

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
    hdu.writeto(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}-2.fits", overwrite = True)
    print('Saved the .fits image')


create_master_flat("20250908_07in_NGC6946", 12, "g'")

create_master_flat("20250928_07in_NGC6946", 10, "g'", ["NGC6946_"])
create_master_flat("20250928_07in_NGC6946", 9, "ha", ["NGC6946_"])

create_master_flat("20251003_07in_NGC6946", 12, "ha", ["FLAT_NGC 6946_","FLAT_SKYFLAT_"])

create_master_flat("20251009_07in_NGC6946", 13, "ha", ["FLAT_NGC6946_","FLAT_skyflats_"])

create_master_flat("20251015_07in_NGC6946", 13, "g'", ["FLAT_NGC6946_","FLAT_SKYFLAT_"])

create_master_flat("20251015_07in_NGC6946", 13, "ha", ["FLAT_NGC6946_","FLAT_SKYFLAT_"])


#%%

from scipy.ndimage import median_filter, generic_filter
import os
from astropy.stats import sigma_clip

import numpy as np
import matplotlib.pyplot as plt

def plot_adu_distribution(
    frames,
    bins=1000,
    use_log=True,
    clip_percentile=None,
    title="ADU Distribution"
):
    """
    frames : list of 2D numpy arrays
    bins   : number of histogram bins
    use_log: log-scale on y-axis (recommended)
    clip_percentile : e.g. 99.99 to ignore extreme cosmic rays
    """

    # gather all values safely
    all_vals = []
    for frame in frames:
        v = frame[np.isfinite(frame)]
        all_vals.append(v)

    all_vals = np.concatenate(all_vals)

    if clip_percentile is not None:
        hi = np.percentile(all_vals, clip_percentile)
        all_vals = all_vals[all_vals <= hi]

    adu_min = float(all_vals.min())
    adu_max = float(all_vals.max())

    # create bins explicitly from min to max
    bins_edges = np.linspace(adu_min, adu_max, bins + 1)

    # histogram
    plt.figure()
    plt.hist(all_vals, bins=bins_edges)
    if use_log:
        plt.yscale("log")

    plt.xlabel("ADU")
    plt.ylabel("Pixel Count")
    plt.title(f"{title}\nMin={adu_min:.1f}  Max={adu_max:.1f}")
    plt.grid(True)
    plt.show()

    return adu_min, adu_max

#%%

from scipy.ndimage import shift, rotate

def rotate_about_point(img, angle_deg, center):
    """
    Rotate image about a specific (y,x) point.
    """
    cy, cx = center
    # shift so rotation point is at array center
    shifted = shift(img, shift=[img.shape[0]/2 - cy, img.shape[1]/2 - cx],
                     order=3, mode='constant', cval=np.nan)
    # rotate
    rotated = rotate(shifted, angle_deg, reshape=False, order=3,
                      mode='constant', cval=np.nan)
    # shift back
    unshifted = shift(rotated, shift=[cy - img.shape[0]/2, cx - img.shape[1]/2],
                       order=3, mode='constant', cval=np.nan)
    return unshifted



#%% Calibrating the science images

from astropy.stats import sigma_clip

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix="",label=''):
    science = []
    exptime = 0
    shifts = np.loadtxt("/users/kushpatel/downloads/Imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    bias = fits.getdata(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits").astype(np.float64)
    dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
    dark_master = dark_hdu.data.astype(np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]
    flat = fits.getdata(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}-2.fits").astype(np.float64)
    
    for i in tqdm(range(num_images)):
        # Load the image
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        try:
            hdu = fits.open(f"/users/kushpatel/{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
            print(number)
        except FileNotFoundError:
            continue
        exptime = hdu.header["EXPTIME"]

        # Load the master bias and subtract it
        image = np.array(hdu.data) - bias

        # Load the master dark and subtract it, accounting for different exposure times
        dark = exptime / dark_exptime * np.array(dark_master)
        image = image - dark

        # Load the master flat and divide by it
        flat_safe = flat.copy()
        flat_safe[flat_safe <= 0] = np.nan
        calibrated_image = image / flat_safe

        x = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][2])
        y = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][1])
        rotate_180 = "Rotate 180" in shifts[f"{file_prefix}{number}-{filter_name}.fits"][4]

        #  ROTATE FIRST
        if rotate_180:
            calibrated_image = np.rot90(calibrated_image, 2)
        
        #  SHIFT SECOND
        calibrated_image = imshift(calibrated_image, x, y, False)
        science.append(calibrated_image.astype(np.float32))
        print(f"Calibrated image {number}")
    plot_adu_distribution(
    science,
    bins=2000,
    use_log=True,
    clip_percentile=99.999,
    title=f"{filter_name} ADU Distribution — Night {image_folder}")
    plt.savefig(f"/users/kushpatel/{image_folder}/LIGHT/{filter_name}_ADU_Distribution.png")

    hot_pixels = 0
    master = np.nansum(science, axis=0)
    cleaned = master
    print("MAX AFTER :", np.nanmax(cleaned))
    print(cleaned.shape)
    print(f"Total hot pixels fixed in master science: {hot_pixels}")


    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(cleaned)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"/users/kushpatel/{image_folder}/LIGHT/master_science-{filter_name}-2({image_folder}){label}.fits", overwrite=True)
    print("Saved combined and calibrated image")


calibrate_science_images("20250908_07in_NGC6946", 10, "g'")

calibrate_science_images("20250928_07in_NGC6946", 10, "g'","NGC6946_")
calibrate_science_images("20250928_07in_NGC6946", 10, "ha","NGC6946_")

# Cant do the following without accounting for rotation
# calibrate_science_images("20251003_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")

calibrate_science_images("20251009_07in_NGC6946", 18, "ha", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 23, "g'", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")

"""
for i in range(18):
    calibrate_science_images("20251009_07in_NGC6946", i+1, "ha", "LIGHT_NGC6946_",f"{i+1}")
"""


#%%

def calibrate_science_images_1003(image_folder, num_images, filter_name, file_prefix=""):
    science = []
    exptime = 0
    shifts = np.loadtxt("/users/kushpatel/downloads/Imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    bias = fits.getdata(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits").astype(np.float64)
    dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
    dark_master = dark_hdu.data.astype(np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]
    flat = fits.getdata(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}-2.fits").astype(np.float64)
    
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
        image = np.array(hdu.data) - bias

        # Load the master dark and subtract it, accounting for different exposure times
        dark = exptime / dark_exptime * np.array(dark_master)
        image = image - dark

        # Load the master flat and divide by it
        flat_safe = flat.copy()
        flat_safe[flat_safe <= 0] = np.nan
        calibrated_image = image / flat_safe

        x = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][2])
        y = int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][1])
        rotate_180 = "Rotate 180" in shifts[f"{file_prefix}{number}-{filter_name}.fits"][4]

        #  ROTATE FIRST
        if rotate_180:
            calibrated_image = np.rot90(calibrated_image, 2)
        
        #  SHIFT SECOND
        calibrated_image = imshift(calibrated_image, x, y, False)
        science.append(calibrated_image.astype(np.float32))
        print(f"Calibrated image {number}")
    
    science_rotated = []
    for i in range(len(science)):
        if i < 6:
            science_rotated.append(science[i])
        else:
            deg_rotate = -9.79
            coordinates = (2940,2989) # x,y of star reference point
            image_rotate = rotate_about_point(science[i], deg_rotate, coordinates)
            science_rotated.append(image_rotate)

    plot_adu_distribution(
    science_rotated,
    bins=2000,
    use_log=True,
    clip_percentile=99.999,
    title=f"{filter_name} ADU Distribution — Night {image_folder}")
    plt.savefig(f"/users/kushpatel/{image_folder}/LIGHT/{filter_name}_ADU_Distribution.png")

    hot_pixels = 0
    master = np.nansum(science_rotated, axis=0)
    cleaned = master
    print("MAX AFTER :", np.nanmax(cleaned))
    print(cleaned.shape)
    print(f"Total hot pixels fixed in master science: {hot_pixels}")


    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(cleaned)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"/users/kushpatel/{image_folder}/LIGHT/master_science-{filter_name}-2({image_folder}).fits", overwrite=True)
    print("Saved combined and calibrated image")

calibrate_science_images_1003("20251003_07in_NGC6946", 15, "ha", "LIGHT_NGC 6946_")


#%%

def stat_report_fast(arr, name="array", sample=2_000_000):
    a = np.asarray(arr, dtype=np.float64)
    vals = a[np.isfinite(a)]
    
    if vals.size > sample:
        idx = np.random.choice(vals.size, sample, replace=False)
        vals = vals[idx]
    
    p = np.nanpercentile(vals, [0,1,5,25,50,75,95,99,100])
    print(f"\n{name}")
    print(" shape:", a.shape)
    print(" min,1,5,25,50,75,95,99,max:")
    print(p)

# load and inspect
bias = fits.getdata("/users/kushpatel/20251015_07in_NGC6946/BIAS/master_bias.fits")
dark = fits.getdata("/users/kushpatel/20251015_07in_NGC6946/DARK/master_dark.fits")
flat = fits.getdata("/users/kushpatel/20251015_07in_NGC6946/FLAT/master_flat-ha.fits")

stat_report_fast(bias, "master_bias")
stat_report_fast(dark, "master_dark")
stat_report_fast(flat, "master_flat (after normalization?)")

# inspect one calibrated science frame quickly
img = fits.getdata("/users/kushpatel/20251015_07in_NGC6946/LIGHT/LIGHT_NGC6946_0000-ha.fits").astype(np.float64)
dark_header_exptime = fits.open("/users/kushpatel/20251015_07in_NGC6946/DARK/master_dark.fits")[0].header["EXPTIME"]
cal = (img - bias) - (300/ dark_header_exptime * dark)  # adapt exptime accordingly
cal /= flat
stat_report_fast(cal, "single_calibrated_frame")


#%% Merge the images into one image

import pandas as pd

def load_shifts_table(path):
    shifts = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str)
    shift_dict = {row[0].strip(): row[1:] for row in shifts}

    # strip all values
    for k in shift_dict:
        shift_dict[k] = [v.strip() for v in shift_dict[k]]

    return shift_dict


def rotate_about_point_with_imshift(image, angle_deg, center_rc):
    r0, c0 = center_rc
    a, b = image.shape

    # image center
    center_r = a // 2
    center_c = b // 2

    # compute shift to move (r0,c0) to (center_r, center_c)
    row_shift = center_r - r0   # down/up
    col_shift = center_c - c0   # right/left

    # move rotation center to middle
    im1 = imshift(image, row_shift, col_shift)

    # rotate around middle
    im2 = rotate(im1, angle_deg, reshape=False,
                 order=1, mode='constant', cval=np.nan)

    # shift back
    im3 = imshift(im2, -row_shift, -col_shift)

    return im3


# DONT CHANGE THE ORDER OF THESE IMAGES
final_construct = []
final_construct.append(fits.getdata("/users/kushpatel/20250908_07in_NGC6946/LIGHT/master_science-g'-2(20250908_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20250928_07in_NGC6946/LIGHT/master_science-g'-2(20250928_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20251015_07in_NGC6946/LIGHT/master_science-g'-2(20251015_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20250928_07in_NGC6946/LIGHT/master_science-ha-2(20250928_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20251003_07in_NGC6946/LIGHT/master_science-ha-2(20251003_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20251009_07in_NGC6946/LIGHT/master_science-ha-2(20251009_07in_NGC6946).fits"))
final_construct.append(fits.getdata("/users/kushpatel/20251015_07in_NGC6946/LIGHT/master_science-ha-2(20251015_07in_NGC6946).fits"))

calibrating_set = []

shifting = load_shifts_table("/users/kushpatel/downloads/imshiftfinal.txt")


# Had to manually modify for alignment after rotation code was fixed
x_list = [0,1116,158,1122,299,161,155]
y_list = [0,74,-83,81,-67,90,-78]
Rotate_list = [0,-0.83,2.60,-0.82,-4.83,2.65,2.57]
ref_center = (3212, 2890)  # row, col in the reference frame

calibrating_set = []

for i in range(len(final_construct)):
    x = x_list[i]
    y = y_list[i]
    rotate_deg = -Rotate_list[i]

    img0 = final_construct[i]

    # --- SHIFT FIRST ---
    im_shifted = imshift(img0, y, x, False)

    # --- ROTATE SECOND ---
    im_rotated = rotate_about_point_with_imshift(im_shifted, rotate_deg, ref_center)

    calibrating_set.append(im_rotated)


g_calibrated = np.nansum(calibrating_set[0:3],axis=0)

Ha_calibrated  = np.nansum(calibrating_set[3:], axis=0)


final_calibrated = Ha_calibrated + g_calibrated


hdu_Ha = fits.PrimaryHDU(Ha_calibrated)
hdu_Ha.writeto("/users/kushpatel/downloads/final_Ha_image.fits", overwrite = True)


hdu_g = fits.PrimaryHDU(g_calibrated)
hdu_g.writeto("/users/kushpatel/downloads/final_g_image.fits", overwrite = True)


hdu = fits.PrimaryHDU(final_calibrated)
hdu.writeto("/users/kushpatel/downloads/final_combined_image.fits", overwrite = True)


print("Final finite pixels:", np.isfinite(final_calibrated).sum())
print("Final min/max:", np.nanmin(final_calibrated), np.nanmax(final_calibrated))


# %%
