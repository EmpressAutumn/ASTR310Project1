# Authors -  Autumn Hoffensetz, Evelynn Chara McNeil

import numpy as np
from astropy.io import fits
from tqdm import tqdm

def load_images(path, num_images, filter_name, file_prefix):
    images = []  # this creates an unfilled list
    exptime = 0

    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        hdu = fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0]
        exptime = hdu.header["EXPTIME"]
        images.append([np.array(hdu.data)])
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

#%% Creating the master bias

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
    hdu.writeto(f"{image_folder}/BIAS/master_bias-{filter_name}.fits", overwrite = True)
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
    bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias-{filter_name}.fits")[0]
    master_dark = median_combine(dark) - np.array(bias_hdu.data)

    # Save the combined FITS dark image
    hdu = fits.PrimaryHDU(master_dark)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"{image_folder}/DARK/master_dark-{filter_name}.fits", overwrite = True)
    print('Saved the .fits image')

create_master_dark("20250908_07in_NGC6946", 7, "g'")
create_master_dark("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")

#%% Creating the master flats

def create_master_flat(image_folder, num_images, filter_name, file_prefix="", kind=""):
    # Load the images
    flat, exptime = load_images(f"{image_folder}/FLAT", num_images, filter_name, file_prefix)

    # Load the master bias and subtract it
    bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias-{filter_name}.fits")[0]
    flat = median_combine(flat) - np.array(bias_hdu.data)

    # Load the master dark and subtract it, accounting for different exposure times
    dark_hdu = fits.open(f"{image_folder}/DARK/master_bias-{filter_name}.fits")[0]
    dark = exptime / dark_hdu.header["EXPTIME"] * np.array(dark_hdu.data)
    flat -= dark

    # Normalize the flat
    master_flat = flat / np.median(flat)

    # Save the combined FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    if kind == "":
        hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    else:
        hdu.writeto(f"{image_folder}/FLAT/{kind}-master_flat-{filter_name}.fits", overwrite = True)
    print('Saved the .fits image')

create_master_flat("20250908_07in_NGC6946", 12, "g'")
create_master_flat("20250928_07in_NGC6946", 9, "ha", "NGC6946_")
create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_NGC6946_")
create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_skyflats_")
create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_NGC6946_", "skyflat")
create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_SKYFLAT_", "skyflat")

#%% Calibrating the science images

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix="", flat_kind=""):
    for i in range(num_images):
        # Load the image
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        hdu = fits.open(f"{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
        exptime = hdu.header["EXPTIME"]

        # Load the master bias and subtract it
        bias_hdu = fits.open(f"{image_folder}/BIAS/master_bias-{filter_name}.fits")[0]
        image = np.array(hdu.data) - np.array(bias_hdu.data)

        # Load the master dark and subtract it, accounting for different exposure times
        dark_hdu = fits.open(f"{image_folder}/DARK/master_bias-{filter_name}.fits")[0]
        dark = exptime / dark_hdu.header["EXPTIME"] * np.array(dark_hdu.data)
        image -= dark

        # Load the master flat and divide by it
        if flat_kind == "":
            flat_hdu = fits.open(f"{image_folder}/FLAT/master_flat-{filter_name}.fits")[0]
        else:
            flat_hdu = fits.open(f"{image_folder}/FLAT/{flat_kind}-master_flat-{filter_name}.fits")[0]

        calibrated_image = image / np.array(flat_hdu.data)

        # Shift the image
        """Evelynn, shift the image here"""

        # Save the calibrated FITS science image
        hdu = fits.PrimaryHDU(calibrated_image)
        hdu.header["EXPTIME"] = exptime
        hdu.writeto(f"{image_folder}/CALIBRATED/science{number}-{filter_name}.fits", overwrite=True)
        print(f"Saved calibrated image {number}")
