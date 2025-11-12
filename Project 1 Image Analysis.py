# Authors -  Autumn Hoffensetz, Evelynn Chara McNeil

import numpy as np
from astropy.io import fits
from tqdm import tqdm

#%% Creating the master bias

# ISSUES:
#   Different file name formats
#   Different filters

# Median combine the bias images
def create_master_bias(image_folder, num_images):
    # Load the images
    path = f"{image_folder}/BIAS"
    g_bias = [] # this creates an unfilled list
    
    for i in range(num_images):
        number = str(i)
        while len(number) < 4:
            number = f"0{number}" # this is creating an index for numbers 0000 through num_images to call
        g_bias.append([np.array(fits.open(f"{path}/{number}-g'.fits")[0].data)])
    print('Created a list containing each image')

    g_bias = np.vstack(g_bias)
    print('Stacked each image matrix')

    # Transpose the matrix: image[row[column[]]] -> row[column[image[]]]
    g_bias = np.transpose(g_bias, (1, 2, 0))
    print('Transposed the matrix')
    
    # Take the median of each pixel value
    for i in tqdm(range(len(g_bias)), desc ='Rows', unit = 'row'):
        for j in range(len(g_bias[i])):
            g_bias[i, j] = np.median(g_bias[i, j])

    # Remove the duplicate median values (Thank you NumPy for being awful!)
    mast_bias = g_bias[:, :, 0]
    print('Removed duplicate median values')

    # Save the combined FITS bias image
    hdu = fits.PrimaryHDU(mast_bias)
    hdu.writeto(f"{path}/mast_bias.fits", overwrite = True)
    print('Saved the .fits image')

# Create master bias for 9/08
create_master_bias("20250908_07in_NGC6946")

# Create master bias for 9/28
create_master_bias("20250928_07in_NGC6946")
