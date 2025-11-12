# Author - Evelynn Chara McNeil - with significant advising from Autumn Hoffensetz

import numpy as np
from astropy.io import fits
from tqdm import tqdm

#%% Creating the master bias for 9/08

# Median combine the bias images
def create_master_bias():
    # Load the images
    path = "20250908_07in_NGC6946/BIAS"
    g_bias_908 = [] #this creates an unfilled list
    
    for i in range(12):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}" #this is creating an index for numbers 0000 through 0007 to call
        g_bias_908.append([np.array(fits.open(f"{path}/{number}-g'.fits")[0].data)])
    print('Created a list containing each image')

    g_bias_908 = np.vstack(g_bias_908)
    print('Stacked each image matrix')

    # Transpose the matrix: image[row[column[]]] -> row[column[image[]]]
    g_bias_908 = np.transpose(g_bias_908, (1, 2, 0))
    print('Transposed the matrix')
    
    # Take the median of each pixel value
    for i in tqdm(range(len(g_bias_908)), desc ='Rows', unit = 'row'):
        for j in range(len(g_bias_908[i])):
            g_bias_908[i, j] = np.median(g_bias_908[i, j])

    # Remove the duplicate median values (Thank you NumPy for being awful!)
    mast_bias_908 = g_bias_908[:, :, 0]
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(mast_bias_908)
    hdu.writeto(f"{path}/mast_bias_908.fits", overwrite = True)
    print('Saved the .fits image')

create_master_bias()

#%% Creating the Biases for 9/28

# Median combine the bias images
def create_master_bias():
    # Load the images
    path = "20250928_07in_NGC6946/BIAS"
    Ha_bias_928 = [] #this creates an unfilled list
    
    for i in range(7):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}" #this is creating an index for numbers 0000 through 0007 to call
        Ha_bias_928.append([np.array(fits.open(f"{path}/NGC6946_{number}-ha.fits")[0].data)])
    print('Created a list containing each image')

    Ha_bias_928 = np.vstack(Ha_bias_928)
    print('Stacked each image matrix')

    # Transpose the matrix: image[row[column[]]] -> row[column[image[]]]
    Ha_bias_928 = np.transpose(Ha_bias_928, (1, 2, 0))
    print('Transposed the matrix')
    
    # Take the median of each pixel value
    for i in tqdm(range(len(Ha_bias_928)), desc ='Rows', unit = 'row'):
        for j in range(len(Ha_bias_928[i])):
            Ha_bias_928[i, j] = np.median(Ha_bias_928[i, j])

    # Remove the duplicate median values (Thank you NumPy for being awful!)
    mast_bias_928 = Ha_bias_928[:, :, 0]
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(mast_bias_928)
    hdu.writeto(f"{path}/mast_bias_928.fits", overwrite = True)
    print('Saved the .fits image')

create_master_bias()

