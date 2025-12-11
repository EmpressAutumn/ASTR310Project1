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

# Altered version of AperE, removing negative sqrt errors and without the ability to plot
def aperE(im, col, row, rad1, rad2, ir1, ir2, or1, or2, Kccd):
    # Aperture Photometry
    a, b = im.shape

    xx, yy = np.meshgrid(range(b), range(a))

    ixsrc = ((xx - col) / rad1) ** 2 + ((
                                                    yy - row) / rad2) ** 2 <= 1  # returns a boolean array same size as the image where True is part of the target aperture

    ixsky = np.logical_and(
        (((xx - col) / or1) ** 2) + (((yy - row) / or2) ** 2) <= 1,
        (((xx - col) / ir1) ** 2) + (((yy - row) / ir2) ** 2) >= 1,
    )  # returns a boolean array same size as the image where True is part of the sky annulus and False is not.

    src_pixels = im[ixsrc]  # returns a 1D array of pixel values in ADUs of the target aperture pixels
    num_src = len(src_pixels)
    src_err = np.sqrt(
        np.clip(src_pixels / Kccd, 0, None))  # Removes negative sqrt errors to approximate Poisson read noise.
    # Dividing by the Kccd converts ADU to electrons.

    sky_pixels = im[ixsky]  # returns a 1D array of pixel values of the sky annulus
    num_sky = len(sky_pixels)
    sky = np.median(
        sky_pixels)  # Median value of the sky annulus is taken as 'the' sky brightness. To be subtracted from source pixels.

    sky_err = np.sqrt(np.clip(sky_pixels * num_src / num_sky / Kccd, 0, None))  # Removes negative sqrt errors.
    # Sky error normalized and scaled by the number of sky pixels in the target aperture, and then converted to electrons

    net_pixels = (src_pixels - sky)  # each src pixel is corrected by subtracting the median sky signal

    flx = np.sum(net_pixels) / Kccd  # final flux value of the source in electrons
    total_err = (
            np.sqrt(np.sum(src_err ** 2) + np.sum(sky_err ** 2)) / Kccd
    )  # final error through error propagation on the flx term.

    return flx, total_err

def load_images(path, num_images, filter_name, file_prefix=""):
    images = []
    exptime = None

    for i in range(num_images):
        # Create a 4 digit string of the index number
        number = str(i).zfill(4)

        # Try loading the image
        try:
            hdu = fits.open(f"{path}/{file_prefix}{number}-{filter_name}.fits")[0]
        except FileNotFoundError:
            continue

        # Add the image to the list
        images.append(hdu.data.astype(np.float64))
        if exptime is None:
            exptime = hdu.header["EXPTIME"]

    images = np.stack(images)  # shape (N, Y, X)
    print("Loaded image stack:", images.shape)

    return images, exptime

# Remove any trailing or leading spaces from any elements in the imshifts map
def autostrip(imshifts):
    for key in imshifts.keys():
        key.strip()
        for i in range(len(imshifts[key])):
            imshifts[key][i] = imshifts[key][i].strip()
    return imshifts

def rotate_about_point(image, angle, center):
    """
    Rotate image about a specific (y,x) point.
    """

    center_offset = (image.shape[0]/2 - center[0], image.shape[1]/2 - center[1])

    # Shift the center of the image to (0, 0)
    shifted_image = shift(image, center_offset, order=1, mode='constant', cval=0.0)

    # Rotate the image
    rotated_image = rotate(shifted_image, angle, reshape=False, order=1, mode='constant', cval=0.0)

    # Shift (0, 0) back to the original image center
    final_image = shift(rotated_image, (- center_offset[0], - center_offset[1]), order=1, mode='constant', cval=0.0)

    return final_image

#%% Creating the master biases

def create_master_bias(image_folder, num_images, filter_name, file_prefix=""):
    # Load the raw biases
    biases, exptime = load_images(f"{image_folder}/BIAS", num_images, filter_name, file_prefix)

    # Median combine the biases
    master_bias = np.nanmedian(biases, axis=0)
    print("Removed duplicate median values")

    # Save the master FITS bias image
    bias_hdu = fits.PrimaryHDU(master_bias)
    bias_hdu.writeto(f"{image_folder}/BIAS/master_bias.fits", overwrite=True)
    print("Saved the master bias")

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
    dark_hdu = fits.PrimaryHDU(master_dark)
    dark_hdu.header["EXPTIME"] = exptime
    dark_hdu.writeto(f"{image_folder}/DARK/master_dark.fits", overwrite=True)
    print("Saved the master dark")

create_master_dark("20250908_07in_NGC6946", 7, "g'")
# Don't create dark for 9/28
create_master_dark("20251003_07in_NGC6946", 7, "i'", "DARK_NGC 6946_")
create_master_dark("20251009_07in_NGC6946", 7, "ha", "DARK_NGC6946_")
create_master_dark("20251015_07in_NGC6946", 7, "g'", "DARK_NGC6946_")

#%% Reuse the 10/03 master dark for 9/28

dark_1003_hdu = fits.open("20251003_07in_NGC6946/DARK/master_dark.fits")[0]
dark_1003_hdu.writeto("20250928_07in_NGC6946/DARK/master_dark.fits", overwrite=True)
print("Saved the duplicated master dark")

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
    flat_hdu = fits.PrimaryHDU(master_flat)
    if kind in ["", "sky", "dome"]:
        flat_hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite=True)
        if kind:
            print("Warning: 'kind' must be either 'sky' or 'dome'.")
    else:
        flat_hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}-{kind}.fits", overwrite=True)
    print("Saved the master flat")

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

    # Save the combined master FITS flat image
    hdu = fits.PrimaryHDU(master_flat)
    hdu.writeto(f"{image_folder}/FLAT/master_flat-{filter_name}.fits", overwrite = True)
    print("Saved the combined master flat")

combine_master_flat("20251003_07in_NGC6946", "ha")
combine_master_flat("20251009_07in_NGC6946", "ha")
combine_master_flat("20251015_07in_NGC6946", "ha")
combine_master_flat("20251015_07in_NGC6946", "g'")

#%% Calibrating the science images

# Draws a histogram of noise, which is useful for checking for errors
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

def calibrate_science_images(image_folder, num_images, filter_name, file_prefix=""):
    # Load imshifts.txt, the file containing image shifting information
    imshifts = autostrip(
        { row[0]: row[1:] for row in np.loadtxt("imshifts.txt", delimiter = ",", skiprows=1, dtype=str) }
    )

    # Load, calibrate, and shift the science images
    sciences = []
    total_exptime = 0

    for i in range(num_images):
        # Create a 4 digit string of the index number
        number = str(i).zfill(4)

        # Try loading the image
        try:
            science_hdu = fits.open(f"{image_folder}/LIGHT/{file_prefix}{number}-{filter_name}.fits")[0]
            exptime = science_hdu.header["EXPTIME"]
        except FileNotFoundError:
            continue

        # Load the master bias and subtract it
        master_bias = np.asarray(fits.open(f"{image_folder}/BIAS/master_bias.fits")[0].data, dtype=np.float64)
        bias_subtracted_science = np.asarray(science_hdu.data, dtype=np.float64) - master_bias

        # Load the master dark, adjust it for exposure time, and subtract it
        master_dark_hdu = fits.open(f"{image_folder}/DARK/master_dark.fits")[0]
        adjusted_master_dark = exptime * master_dark_hdu.header["EXPTIME"] * np.asarray(master_dark_hdu.data, dtype=np.float64)
        fully_subtracted_science = bias_subtracted_science - adjusted_master_dark

        # Load the master flat and divide by it
        master_flat = np.asarray(fits.open(f"{image_folder}/FLAT/master_flat-{filter_name}.fits")[0].data, dtype=np.float64)
        master_flat[master_flat <= 0] = np.nan
        divided_science = fully_subtracted_science / master_flat

        # Rotate and translate the science image
        if "Rotate 180" in imshifts[f"{file_prefix}{number}-{filter_name}.fits"][6]:
            shifted_science = imshift(
                divided_science,
                int(imshifts[f"{file_prefix}{number}-{filter_name}.fits"][2]),
                int(imshifts[f"{file_prefix}{number}-{filter_name}.fits"][1]),
                True
            )
        else:
            shifted_science = imshift(
                divided_science,
                int(imshifts[f"{file_prefix}{number}-{filter_name}.fits"][2]),
                int(imshifts[f"{file_prefix}{number}-{filter_name}.fits"][1])
            )

        # Add the calibrated and shifted science image to the list
        sciences.append(shifted_science)
        total_exptime += exptime
        print(f"Calibrated and shifted image {number}")

    """
    plot_adu_distribution(
        sciences,
        bins=2000,
        use_log=True,
        clip_percentile=99.999,
        title=f"{filter_name} ADU Distribution — Night {image_folder}"
    )
    plt.savefig(f"{image_folder}/LIGHT/{filter_name}_ADU_Distribution.png")
    """

    # Stack the science images together
    master_science = np.nansum(sciences, axis=0)

    # Save the master FITS science image
    hdu = fits.PrimaryHDU(master_science)
    hdu.header["EXPTIME"] = total_exptime
    hdu.writeto(f"{image_folder}/LIGHT/master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")

calibrate_science_images("20250908_07in_NGC6946", 10, "g'")
calibrate_science_images("20250928_07in_NGC6946", 10, "g'","NGC6946_")
calibrate_science_images("20251015_07in_NGC6946", 23, "g'", "LIGHT_NGC6946_")

calibrate_science_images("20250928_07in_NGC6946", 10, "ha","NGC6946_")
calibrate_science_images("20251003_07in_NGC6946", 15, "ha", "LIGHT_NGC6946_")
calibrate_science_images("20251009_07in_NGC6946", 18, "ha", "LIGHT_NGC6946_")
calibrate_science_images("20251015_07in_NGC6946", 23, "ha", "LIGHT_NGC6946_")

#%% Merging the science images from each observing session

def final_shift(image_folders, filter_name):
    # Load imshifts.txt, the file containing image shifting information
    imshifts = autostrip(
        { row[0]: row[1:] for row in np.loadtxt("imshifts.txt", delimiter = ",", skiprows=1, dtype=str) }
    )

    # Load and shift the science images
    sciences = []
    total_exptime = 0

    for image_folder in image_folders:
        # Try loading the image
        try:
            science_hdu = fits.open(f"{image_folder}/LIGHT/master_science-{filter_name}.fits")[0]
            exptime = science_hdu.header["EXPTIME"]
        except FileNotFoundError:
            continue

        # Translate the image
        translated_science = imshift(
            np.asarray(science_hdu.data, dtype=np.float64),
            int(imshifts[f"{image_folder}-{filter_name}.fits"][2]),
            int(imshifts[f"{image_folder}-{filter_name}.fits"][1])
        )

        # Rotate the image
        rotated_science = rotate_about_point(
            translated_science,
            float(imshifts[f"{image_folder}-{filter_name}.fits"][3]),
            (
                int(imshifts[f"{image_folder}-{filter_name}.fits"][5]) - int(imshifts[f"{image_folder}-{filter_name}.fits"][2]),
                int(imshifts[f"{image_folder}-{filter_name}.fits"][4]) - int(imshifts[f"{image_folder}-{filter_name}.fits"][1])
            )
        )

        # Add the shifted science image to the list
        sciences.append(rotated_science)
        total_exptime += exptime

    # Stack the science images together
    master_science = np.nansum(sciences, axis=0)

    # Save the master FITS science image
    science_hdu = fits.PrimaryHDU(master_science)
    science_hdu.header["EXPTIME"] = total_exptime
    science_hdu.writeto("master_science-{filter_name}.fits", overwrite=True)
    print("Saved combined and calibrated image")

final_shift(
    [
        "20250908_07in_NGC6946",
        "20250928_07in_NGC6946",
        "20251015_07in_NGC6946"
    ], "g'"
)
final_shift(
    [
        "20250928_07in_NGC6946",
        "20251003_07in_NGC6946",
        "20251009_07in_NGC6946",
        "20251015_07in_NGC6946"
    ], "ha"
)

#%% Determining the sky annulus outer radius bound

def manually_draw_ellipse(image, col0, row0, rad1_0, rad2_0, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap='plasma')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ax.set_title("Adjust Ellipse for Initial Guess")

    # Draw the ellipse
    p = np.linspace(0, 2*np.pi, 400)
    xc = col0 + rad1_0*np.cos(p)
    yc = row0 + rad2_0*np.sin(p)
    ax.plot(xc, yc, 'w', lw=2)

    plt.show()

    print("After closing the window, use these initial values:")
    print(f"col = {col0}")
    print(f"row = {row0}")
    print(f"rad1 = {rad1_0:.1f}")
    print(f"rad2 = {rad2_0:.1f}")

# Load the master science images
master_science_g = np.asarray(fits.open("master_science-g'.fits")[0].data, dtype=np.float64)
master_science_ha = np.asarray(fits.open("master_science-ha.fits")[0].data, dtype=np.float64)

# Manually test various aperture sizes
manually_draw_ellipse(
    master_science_g,
    col0=3320,
    row0=2190,
    rad1_0=1050,
    rad2_0=1050,
    vmin=200000,
    vmax=204000
)

manually_draw_ellipse(
    master_science_ha,
    col0=3320,
    row0=2190,
    rad1_0=1050,
    rad2_0=1050,
    vmin=1600,
    vmax=1700
)

# 1050 is determined to be the edge of the background, so 1040 will be used as the sky annulus's outer radius
r_max = 1040
center = (2190, 3320)

#%% Determining the radius and performing photometry

def photometry(master_science, num_tests, r_min):
    test_apertures = np.linspace(r_max - num_tests, r_max + num_tests, 1 + 2*num_tests)
    noises = []
    fluxes = []
    snrs = []

    # Loop through each test aperture
    for test_aperture in test_apertures:
        # Using EGAIN, found from header of uncalibrated science images
        flux, noise = aperE(master_science, center[1], center[0], test_aperture, test_aperture, r_min, r_min, r_max, r_max, 1/0.242862924933434)
        noises.append(noise)
        fluxes.append(flux)
        snrs.append(flux/noise) # SNR is flux divided by noise

    # Get the change in SNR and flux
    d_snrs = []
    d_fluxes = []

    for i in range(len(snrs) - 1):
        d_snr = snrs[i + 1] - snrs[i]
        d_snrs.append(d_snr)

    for i in range(len(fluxes) - 1):
        d_flux = fluxes[i + 1] - fluxes[i]
        d_fluxes.append(d_flux)

    test_apertures = np.asarray(test_apertures, dtype=float)
    snrs = np.asarray(snrs, dtype=float)

    # Mask out invalid entries
    valid = np.isfinite(snrs)

    # Mask within requested aperture window
    in_window = (test_apertures >= r_min) & (test_apertures <= r_max)
    mask = valid & in_window
    if not mask.any():
        return None, None, None

    # Get the maximum SNR location
    max_index_local = np.nanargmax(snrs[mask])

    # Get the "global" index at that location
    global_indices = np.nonzero(mask)[0]
    max_index_global = int(global_indices[max_index_local])

    # Print out the photometry results
    print(f"Aperture = {float(test_apertures[max_index_global])} px")
    print(f"Flux = {float(fluxes[max_index_global])}")
    print(f"SNR = {float(snrs[max_index_global])}")

    return max_index_global, float(test_apertures[max_index_global]), float(snrs[max_index_global])

# 1010 pixels is used as the minimum radius
index_g, radius_g, snr_g = photometry(master_science_g, 300, 1010)
index_ha, radius_ha, snr_ha = photometry(master_science_ha, 300, 1010)
