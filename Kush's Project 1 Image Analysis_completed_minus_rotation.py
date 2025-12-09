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
    hdu.writeto(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}{kind}.fits", overwrite = True)
    print('Saved the .fits image')

create_master_flat("20250908_07in_NGC6946", 12, "g'")

create_master_flat("20250928_07in_NGC6946", 10, "g'", "NGC6946_")
create_master_flat("20250928_07in_NGC6946", 9, "ha", "NGC6946_")

create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_NGC 6946_","dome")
create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_SKYFLAT_",'sky')

create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_skyflats_","sky")

create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_SKYFLAT_","sky")


create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_SKYFLAT_","sky")


#%%

def combine_master_flat(image_folder, filter_name, kind):
    mastflats = []
    for i in kind:
        mastflat_hdu = fits.open(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}{i}.fits")[0]
        mastflats.append(np.array(mastflat_hdu.data))

    comb_mastflats = np.stack(mastflats, axis = 0) #only 2 dimensions
    comb_mastflat = median_combine(comb_mastflats)
    comb_mastflat /= np.nanmedian(comb_mastflat) # Renormalize
    print("combined master flats for night and filter")
    
    hdu = fits.PrimaryHDU(comb_mastflat)
    hdu.writeto(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    print("saved the .fits image")


combine_master_flat("20251003_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251009_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251015_07in_NGC6946", "ha", ["sky", "dome"])
combine_master_flat("20251015_07in_NGC6946", "g'", ["sky", "dome"])


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

    # --- gather all values safely ---
    all_vals = []
    for frame in frames:
        v = frame[np.isfinite(frame)]
        all_vals.append(v)

    all_vals = np.concatenate(all_vals)

    # --- optional extreme clipping for visualization ---
    if clip_percentile is not None:
        hi = np.percentile(all_vals, clip_percentile)
        all_vals = all_vals[all_vals <= hi]

    adu_min = float(all_vals.min())
    adu_max = float(all_vals.max())

    # --- create bins explicitly from min to max ---
    bins_edges = np.linspace(adu_min, adu_max, bins + 1)

    # --- histogram ---
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



#%% Calibrating the science images

from astropy.stats import sigma_clip

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix=""):
    science = []
    exptime = 0
    shifts = np.loadtxt("/users/kushpatel/downloads/Imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    bias = fits.getdata(f"/users/kushpatel/{image_folder}/BIAS/master_bias.fits").astype(np.float64)
    dark_hdu = fits.open(f"/users/kushpatel/{image_folder}/DARK/master_dark.fits")[0]
    dark_master = dark_hdu.data.astype(np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]
    flat = fits.getdata(f"/users/kushpatel/{image_folder}/FLAT/master_flat-{filter_name}.fits").astype(np.float64)
    
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
        # fix_hot_pixels_8conn(calibrated_image, sigma=10)
        # clipped = sigma_clip(calibrated_image, sigma=4, axis=0)
        # calibrated_image = np.asarray(clipped)
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
    hdu.writeto(f"/users/kushpatel/{image_folder}/LIGHT/master_science-{filter_name}-2.fits", overwrite=True)
    print("Saved combined and calibrated image")


calibrate_science_images("20250908_07in_NGC6946", 10, "g'")

calibrate_science_images("20250928_07in_NGC6946", 10, "g'","NGC6946_")
calibrate_science_images("20250928_07in_NGC6946", 10, "ha","NGC6946_")


calibrate_science_images("20251003_07in_NGC6946", 14, "ha", "LIGHT_NGC 6946_")

calibrate_science_images("20251009_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 23, "g'", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")

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

# show large-scale background map to inspect pedestal/amp structure
bkg = median_filter(cal, size=201, mode="mirror")
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(cal, origin='lower', cmap='gray', vmin=np.nanpercentile(cal,1), vmax=np.nanpercentile(cal,99)); plt.title("calibrated")
plt.subplot(1,2,2); plt.imshow(bkg, origin='lower', cmap='inferno', vmin=np.nanpercentile(bkg,1), vmax=np.nanpercentile(bkg,99)); plt.title("large-scale background")
plt.colorbar(fraction=0.046)
plt.show()



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
