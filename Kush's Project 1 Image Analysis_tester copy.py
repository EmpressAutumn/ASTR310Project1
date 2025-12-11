#%% Authors -  Autumn Hoffensetz, Evelynn Chara McNeil, Kush Patel

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate
from tqdm import tqdm

# imshift.py from UMD, edited to allow for 180 degree rotation
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

def autostrip(imshifts):
    for key in imshifts.keys():
        key.strip()
        for i in range(len(imshifts[key])):
            imshifts[key][i] = imshifts[key][i].strip()
    return imshifts

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

#%% Creating the master biases

def create_master_bias(image_folder, num_images, filter_name, file_prefix=""):
    # Load the raw biases
    biases, exptime = load_images(f"{image_folder}/BIAS", num_images, filter_name, file_prefix)

    # Median combine the biases
    master_bias = np.nanmedian(biases, axis=0)
    print("Removed duplicate median values")

    # Save the master FITS bias image
    hdu = fits.PrimaryHDU(master_bias)
    hdu.writeto(f"{image_folder}/BIAS/master_bias.fits", overwrite=True)
    print("Saved the .fits image")

# Create master biases
create_master_bias("20250908_07in_NGC6946", 12, "g'")
create_master_bias("20250928_07in_NGC6946", 7, "ha", "NGC6946_")
create_master_bias("20251003_07in_NGC6946", 7, "i'", "BIAS_NGC 6946_")
create_master_bias("20251009_07in_NGC6946", 7, "ha", "BIAS_NGC6946_")
create_master_bias("20251015_07in_NGC6946", 7, "g'", "BIAS_NGC6946_")

#%% Creating the master darks

def create_master_dark(image_folder, num_images, filter_name, file_prefix=""):
    # Load the darks
    darks, exptime = load_images(f"{image_folder}/DARK", num_images, filter_name, file_prefix)

    # Load the master bias and subtract it
    master_bias = np.asarray(fits.open(f"{image_folder}/BIAS/master_bias.fits")[0].data, dtype=np.float64)
    subtracted_darks = [ np.asarray(dark, dtype=np.float64) - master_bias for dark in darks ]

    # Median combine the darks
    master_dark = np.nanmedian(np.stack(subtracted_darks), axis=0)

    # Save the master FITS dark image
    hdu = fits.PrimaryHDU(master_dark)
    hdu.header["EXPTIME"] = exptime
    hdu.writeto(f"{image_folder}/DARK/master_dark.fits", overwrite=True)
    print('Saved the .fits image')

create_master_dark("20250908_07in_NGC6946", 7, "g'")
# Don't create dark for 9/28
create_master_dark("20251003_07in_NGC6946", 7, "i'", "DARK_NGC 6946_")
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")

#%% Reuse the 10/03 master dark for 9/28

dark_1003 = fits.open("20251003_07in_NGC6946/DARK/master_dark.fits")[0]
dark_1003.writeto("20250928_07in_NGC6946/DARK/master_dark.fits", overwrite=True)
print("Saved the .fits image")

#%% Creating the master sky and dome flats

def create_master_flat(image_folder, num_images, filter_name, file_prefix="", kind=""):
    # Load the flats
    flats, exptime = load_images(f"{image_folder}/FLAT", num_images, filter_name, file_prefix)

    # Load the master bias and subtract it
    master_bias = np.asarray(fits.open(f"{image_folder}/BIAS/master_bias.fits")[0].data, dtype=np.float64)
    bias_subtracted_flats = [ np.asarray(flat, dtype=np.float64) - master_bias for flat in flats ]

    # Load the master dark, adjust it for exposure time, and subtract it
    master_dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
    adjusted_master_dark = exptime * master_dark_hdu.header["EXPTIME"] * np.asarray(master_dark_hdu.data, dtype=np.float64)
    fully_subtracted_flats = [ np.asarray(flat, dtype=np.float64) - adjusted_master_dark for flat in bias_subtracted_flats ]

    # Normalize the flats
    normalized_flats = [ flat / np.nanmedian(flat) for flat in fully_subtracted_flats ]

    # Median combine the flats
    master_flat = np.nanmedian(np.stack(normalized_flats), axis=0)

    # Save the master FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    if kind in ["", "sky", "dome"]:
        hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite=True)
        if kind:
            print("Warning: 'kind' must be either 'sky' or 'dome'.")
    else:
        hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}-{kind}.fits", overwrite=True)
    print("Saved the .fits image")

create_master_flat("20250908_07in_NGC6946", 12, "g'")

create_master_flat("20250928_07in_NGC6946", 10, "g'", "NGC6946_")
create_master_flat("20250928_07in_NGC6946", 9, "ha", "NGC6946_")

create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_NGC 6946_","dome")
create_master_flat("20251003_07in_NGC6946", 12, "ha", "FLAT_SKYFLAT_","sky")

create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251009_07in_NGC6946", 13, "ha", "FLAT_skyflats_","sky")

create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "g'", "FLAT_SKYFLAT_","sky")

create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_NGC6946_","dome")
create_master_flat("20251015_07in_NGC6946", 13, "ha", "FLAT_SKYFLAT_","sky")

#%% Merging the sky and dome flats

def combine_master_flat(image_folder, filter_name):
    # Load the master sky and dome flats
    try:
        master_sky_flat = np.asarray(fits.open(f"{image_folder}/FLAT/master_flat-{filter_name}-sky.fits")[0].data, dtype=np.float64)
        master_dome_flat = np.asarray(fits.open(f"{image_folder}/FLAT/master_flat-{filter_name}-dome.fits")[0].data, dtype=np.float64)
    except FileNotFoundError:
        print(f"Sky and/or dome flats not found in {image_folder}, are you sure they exist?")
        return

    # Divide the sky flat by the dome flat
    combined_master_flat = master_sky_flat / master_dome_flat

    # Normalize the master_flat
    master_flat = combined_master_flat / np.nanmedian(combined_master_flat)

    # Save the master FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    print("saved the .fits image")

combine_master_flat("20251003_07in_NGC6946", "ha")
combine_master_flat("20251009_07in_NGC6946", "ha")
combine_master_flat("20251015_07in_NGC6946", "ha")
combine_master_flat("20251015_07in_NGC6946", "g'")

#%%

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

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix="", label=''):
    science = []
    exptime = 0
    shifts = np.loadtxt("imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    bias = fits.getdata(f"{image_folder}/BIAS/master_bias.fits").astype(np.float64)
    dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
    dark_master = dark_hdu.data.astype(np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]
    flat = fits.getdata(f"{image_folder}/FLAT/master_flat-{filter_name}.fits").astype(np.float64)
    
    for i in tqdm(range(num_images)):
        # Load the image
        number = str(i)
        while len(number) < 4:
            number = f"0{number}"  # this is creating an index for numbers 0000 through num_images to call
        try:
            hdu = fits.open(f"{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
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
    plt.savefig(f"{image_folder}/LIGHT/{filter_name}_ADU_Distribution.png")

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
    hdu.writeto(f"{image_folder}/LIGHT/master_science-{filter_name}-2({image_folder}){label}.fits", overwrite=True)
    print("Saved combined and calibrated image")

calibrate_science_images("20250908_07in_NGC6946", 10, "g'")
calibrate_science_images("20250928_07in_NGC6946", 10, "g'","NGC6946_")
calibrate_science_images("20251015_07in_NGC6946", 23, "g'", "LIGHT_NGC6946_")

calibrate_science_images("20250928_07in_NGC6946", 10, "ha","NGC6946_")
calibrate_science_images("20251003_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")
calibrate_science_images("20251009_07in_NGC6946", 18, "ha", "LIGHT_NGC6946_")
calibrate_science_images("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")

#%%

def calibrate_science_images_1003(image_folder, num_images, filter_name, file_prefix=""):
    science = []
    exptime = 0
    shifts = np.loadtxt("downloads/imshifts.txt", delimiter = ",", skiprows=1, dtype=str)
    shifts = {row[0]: row[1:] for row in shifts}
    shifts = autostrip(shifts)

    bias = fits.getdata(f"{image_folder}/BIAS/master_bias.fits").astype(np.float64)
    dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
    dark_master = dark_hdu.data.astype(np.float64)
    dark_exptime = dark_hdu.header["EXPTIME"]
    flat = fits.getdata(f"{image_folder}/FLAT/master_flat-{filter_name}.fits").astype(np.float64)
    
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
    plt.savefig(f"{image_folder}/LIGHT/{filter_name}_ADU_Distribution.png")

    hot_pixels = 0
    master = np.nansum(science_rotated, axis=0)
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
    hdu.writeto(f"{image_folder}/LIGHT/master_science-{filter_name}-2({image_folder}).fits", overwrite=True)
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
bias = fits.getdata("20251015_07in_NGC6946/BIAS/master_bias.fits")
dark = fits.getdata("20251015_07in_NGC6946/DARK/master_dark.fits")
flat = fits.getdata("20251015_07in_NGC6946/FLAT/master_flat-ha.fits")

stat_report_fast(bias, "master_bias")
stat_report_fast(dark, "master_dark")
stat_report_fast(flat, "master_flat (after normalization?)")

# inspect one calibrated science frame quickly
img = fits.getdata("20251015_07in_NGC6946/LIGHT/LIGHT_NGC6946_0000-ha.fits").astype(np.float64)
dark_header_exptime = fits.open("20251015_07in_NGC6946/DARK/master_dark.fits")[0].header["EXPTIME"]
cal = (img - bias) - (300/ dark_header_exptime * dark)  # adapt exptime accordingly
cal /= flat
stat_report_fast(cal, "single_calibrated_frame")


#%% Merge the images into one image

def load_shifts_table(path):
    shifts = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str)
    shift_dict = {row[0].strip(): row[1:] for row in shifts}

    # strip all values
    for k in shift_dict:
        shift_dict[k] = [v.strip() for v in shift_dict[k]]

    return shift_dict


def rotate_about_point_safe(image, angle_deg, center_rc):
    r0, c0 = center_rc

    # Shift center to image center
    shift_to_center = (image.shape[0]/2 - r0,
                       image.shape[1]/2 - c0)

    im1 = shift(image, shift_to_center, order=1, mode='constant', cval=0.0)

    im2 = rotate(im1, angle_deg, reshape=False, order=1, mode='constant', cval=0.0)

    im3 = shift(im2, (-shift_to_center[0], -shift_to_center[1]),
                order=1, mode='constant', cval=0.0)

    return im3

# DONT CHANGE THE ORDER OF THESE IMAGES
final_construct = []
final_construct.append(fits.getdata("20250908_07in_NGC6946/LIGHT/master_science-g'-2(20250908_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20250928_07in_NGC6946/LIGHT/master_science-g'-2(20250928_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20251015_07in_NGC6946/LIGHT/master_science-g'-2(20251015_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20250928_07in_NGC6946/LIGHT/master_science-ha-2(20250928_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20251003_07in_NGC6946/LIGHT/master_science-ha-2(20251003_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20251009_07in_NGC6946/LIGHT/master_science-ha-2(20251009_07in_NGC6946).fits"))
final_construct.append(fits.getdata("20251015_07in_NGC6946/LIGHT/master_science-ha-2(20251015_07in_NGC6946).fits"))

calibrating_set = []

shifting = load_shifts_table("downloads/imshiftfinal.txt")

x_list = [0,1116,170,1116,271,183,169]
y_list = [0,74,-67,75,-198,-93,-66]
Rotate_list = [0,-0.83,2.60,-0.82,-4.83,2.65,2.57]
ref_center = (3212, 2890)
ref_center = (3212, 2890)  # row, col in the reference frame

calibrating_set = []

for i in range(len(final_construct)):
    x = x_list[i]
    y = y_list[i]
    rotate_deg = Rotate_list[i]

    img0 = final_construct[i].astype(np.float32)

    if np.isfinite(img0).sum() < 1000:
        print("INPUT DEAD:", i)
        continue

    # --- SHIFT FIRST ---
    im_shifted = imshift(img0, x, y, False)

    if np.isfinite(im_shifted).sum() < 1000:
        print("SHIFT KILLED IMAGE:", i)
        continue

    # --- UPDATE ROTATION CENTER ---
    new_center = (ref_center[0] - y, ref_center[1] - x)

    # --- ROTATE ABOUT THE CORRECT SKY POINT ---
    im_rotated = rotate_about_point_safe(im_shifted, rotate_deg, new_center)

    if np.isfinite(im_rotated).sum() < 1000:
        print("ROTATION KILLED IMAGE:", i)
        continue

    calibrating_set.append(im_rotated)

calibrating_set = np.array(calibrating_set)
Ha_calibrated = np.nansum(calibrating_set[3:], axis=0)
g_calibrated  = np.nansum(calibrating_set[:3], axis=0)
final_calibrated = Ha_calibrated + g_calibrated

print("Final finite pixels:", np.isfinite(final_calibrated).sum())
print("Final min/max:", np.nanmin(final_calibrated), np.nanmax(final_calibrated))

hdu = fits.PrimaryHDU(final_calibrated)
hdu.writeto("downloads/final_combined_image.fits", overwrite = True)
hdu_Ha = fits.PrimaryHDU(Ha_calibrated)
hdu_Ha.writeto("downloads/final_Ha_image.fits", overwrite = True)
hdu_g = fits.PrimaryHDU(g_calibrated)
hdu_g.writeto("downloads/final_g_image.fits", overwrite = True)


#%%

def final_shift(image_folders, filter_name):
    science = []
    shifts = np.loadtxt("desktop/imshifts.txt", delimiter = ',' , skiprows = 1, dtype=str)
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
