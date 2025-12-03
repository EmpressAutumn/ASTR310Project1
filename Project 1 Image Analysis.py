#%% Authors -  Autumn Hoffensetz, Evelynn Chara McNeil

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from library.imshift import imshift

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
    bias, exptime = load_images(f"{image_folder}/BIAS", num_images, filter_name, file_prefix)

    # Median combine the biases
    master_bias = median_combine(bias)
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(master_bias)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"{image_folder}/BIAS/master_bias.fits", overwrite = True)
    print('Saved the .fits image')

# Create master biases
create_master_bias("20250908_07in_NGC6946", 12, "g'")
create_master_bias("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_bias("20251009_07in_NGC6946", 7, "ha", "BIAS_NGC6946_")
create_master_bias("20251015_07in_NGC6946", 7, "g'", "BIAS_NGC6946_")

#%% Creating the master darks

def create_master_dark(image_folder, num_images, filter_name, file_prefix=""):
    # Load the images
    dark, exptime = load_images(f"{image_folder}/DARK", num_images, filter_name, file_prefix)

    # Load the master bias and subtract it
    bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias.fits")[0]
    master_dark = median_combine(dark) - np.array(bias_hdu.data)

    # Save the combined FITS dark image
    hdu = fits.PrimaryHDU(master_dark)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"{image_folder}/DARK/master_dark.fits", overwrite = True)
    print('Saved the .fits image')

create_master_dark("20250908_07in_NGC6946", 7, "g'")
#Don't create dark for 9/28
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")

#%% Creating a master dark for 9/28

# Load the images
dark_0908 = fits.open("20250908_07in_NGC6946/BIAS/master_bias.fits")[0]
dark_1009 = fits.open("20251009_07in_NGC6946/BIAS/master_bias.fits")[0].data
dark_1015 = fits.open("20251015_07in_NGC6946/BIAS/master_bias.fits")[0].data

# Median combine the master dark images from the other nights
master_dark_0928 = median_combine(np.array([dark_0908.data, dark_1009, dark_1015]))

# Save the combined FITS dark image
hdu_dark_0928 = fits.PrimaryHDU(master_dark_0928)
hdu_dark_0928.header["EXPTIME"] = dark_0908.header["EXPTIME"]
hdu_dark_0928.writeto("20250928_07in_NGC6946/DARK/master_dark.fits", overwrite = True)
print('Saved the .fits image')

#%% Creating the master flats

def create_master_flat(image_folder, num_images, filter_name, file_prefixes=None):
    flat_list = []
    exptime = 0

    # Load the images
    if file_prefixes is None:
        file_prefixes = [""]
    for prefix in file_prefixes:
        this_flats, exptime = load_images(f"{image_folder}/FLAT", num_images, filter_name, prefix)
        for this_flat in this_flats:
            flat_list.append(np.array([this_flat]))
    flat = np.vstack(flat_list)

    # Load the master bias and subtract it
    bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias.fits")[0]
    flat = median_combine(flat) - np.array(bias_hdu.data)

    # Load the master dark and subtract it, accounting for different exposure times
    dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
    dark = exptime / dark_hdu.header["EXPTIME"] * np.array(dark_hdu.data)
    flat -= dark.astype(np.uint16)

    # Normalize the flat
    master_flat = flat / np.median(flat)

    # Save the combined FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    print('Saved the .fits image')

create_master_flat("20250908_07in_NGC6946", 12, "g'")

create_master_flat("20250928_07in_NGC6946", 8, "ha", ["NGC6946_"])

create_master_flat("20251009_07in_NGC6946", 13, "ha", ["FLAT_NGC6946_", "FLAT_skyflats_"])

create_master_flat("20251015_07in_NGC6946", 13, "g'", ["FLAT_NGC6946_", "FLAT_SKYFLAT_"])
create_master_flat("20251015_07in_NGC6946", 13, "ha", ["FLAT_NGC6946_", "FLAT_SKYFLAT_"])

#%% Calibrating the science images

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix=""):
    science = []
    exptime = 0
    shifts = np.loadtxt("Imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)
    
    for i in tqdm(range(num_images)):
        # Load the image
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        try:
            hdu = fits.open(f"{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue
        exptime = hdu.header["EXPTIME"]

        # Load the master bias and subtract it
        bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias.fits")[0]
        image = np.array(hdu.data) - np.array(bias_hdu.data)

        # Load the master dark and subtract it, accounting for different exposure times
        dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
        dark = exptime / dark_hdu.header["EXPTIME"] * np.array(dark_hdu.data)
        image -= dark.astype(np.uint16)

        # Load the master flat and divide by it
        flat_hdu = fits.open(f"{image_folder}/FLAT/master_flat-{filter_name}.fits")[0]

        calibrated_image = image / np.array(flat_hdu.data)
        
        # Shift the image
        calibrated_image = imshift(calibrated_image, int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][2]),
                                   int(shifts[f"{file_prefix}{number}-{filter_name}.fits"][1]),
                                   "Rotate 180" in shifts[f"{file_prefix}{number}-{filter_name}.fits"][3])
        
        science.append(calibrated_image)

    master_science = np.sum(np.stack(science), axis = 0)

    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(master_science)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"{image_folder}/LIGHT/master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")

calibrate_science_images("20250908_07in_NGC6946", 10, "g'")

calibrate_science_images("20250928_07in_NGC6946", 12, "ha","NGC6946_")

calibrate_science_images("20251009_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")

calibrate_science_images("20251015_07in_NGC6946", 20, "g'", "LIGHT_NGC6946_")
calibrate_science_images("20251015_07in_NGC6946", 19, "ha", "LIGHT_NGC6946_")

#%% Merge the images into one image

def final_shift(image_folders, filter_name):
    science = []
    shifts = np.loadtxt("Imshifts.txt", delimiter = ',' , skiprows = 1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    autostrip(shifts)

    exptime = 0
    
    for i in range(len(image_folders)):
        # Load the image
        image_hdu = fits.open(f"{image_folders[i]}/LIGHT/master_science-{filter_name}.fits")[0]
        exptime = image_hdu.header["EXPTIME"]
        
        image = imshift(image_hdu.data, int(shifts[f"{image_folders[i]}-{filter_name}"][2]), int(shifts[f"{image_folders[i]}-{filter_name}"][1]))
        science.append(np.array([image]))

    master_science = np.sum(np.vstack(science), axis = 0)

    # Save the calibrated FITS science image
    hdu = fits.PrimaryHDU(master_science)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")
    
final_shift(["20250908_07in_NGC6946", "20251015_07in_NGC6946"], "g'")
final_shift(["20250928_07in_NGC6946", "20251009_07in_NGC6946", "20251015_07in_NGC6946"], "ha")
