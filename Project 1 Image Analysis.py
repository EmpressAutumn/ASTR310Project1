# Authors -  Autumn Hoffensetz, Evelynn Chara McNeil

import numpy as np
from astropy.io import fits
from tqdm import tqdm

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
    path = f"{image_folder}/BIAS"
    bias = [] # this creates an unfilled list
    
    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}" # this is creating an index for numbers 0000 through num_images to call
        bias.append([np.array(fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0].data)])
    print('Created a list containing each image')

    bias = np.vstack(bias)
    print('Stacked each image matrix')

    # Shift the images
    """MAKE SURE THE IMAGES ARE PROPERLY SHIFTED"""

    # Median combine the biases
    master_bias = median_combine(bias)
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(master_bias)
    hdu.writeto(f"{path}/master_bias-{filter_name}.fits", overwrite = True)
    print('Saved the .fits image')

# Create master biases
create_master_bias("20250908_07in_NGC6946", 12, "g'")
create_master_bias("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_bias("20251009_07in_NGC6946", 7, "ha", "BIAS_NGC6946_")
create_master_bias("20251015_07in_NGC6946", 7, "g'", "BIAS_NGC6946_")

#%% Creating the master darks

def create_master_dark(image_folder, num_images, filter_name, file_prefix=""):
    # Load the images
    path = f"{image_folder}/DARK"
    dark = [] # this creates an unfilled list

    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}" # this is creating an index for numbers 0000 through num_images to call
        dark.append([np.array(fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0].data)])
    print('Created a list containing each image')

    dark = np.vstack(dark)
    print('Stacked each image matrix')

    # Shift the images
    """MAKE SURE THE IMAGES ARE PROPERLY SHIFTED"""

    # Load the master dark and subtract it
    bias = np.array(fits.open(f"{image_folder}/BIAS/master_bias-g'.fits"))
    master_dark = median_combine(dark) - bias

    # Save the combined FITS dark image
    hdu = fits.PrimaryHDU(master_dark)
    hdu.writeto(f"{path}/master_dark-{filter_name}.fits", overwrite = True)
    print('Saved the .fits image')

create_master_dark("20250908_07in_NGC6946", 7, "g'")
create_master_dark("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")
