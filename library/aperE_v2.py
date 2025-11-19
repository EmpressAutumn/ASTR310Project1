#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



def aperE(im,
    col,row,
    rad1,rad2,
    ir1,ir2,
    or1,or2,
    Kccd,saturation=np.inf,
    plot=True,save_name="aperE.png"):
    
    """
    Perform elliptical aperture photometry on a 2D image.
    
    The function measures the total flux of an object within an elliptical
    aperture and subtracts the background estimated from an elliptical annulus.
    Optionally, it can display and save a plot showing the apertures.
    
    Parameters
    ----------
    im : ndarray
        2D image array in ADU.
    col, row : float, float
        Center coordinates (x, y) of the target object in pixels.
    rad1, rad2 : float, float
        Semi-major and semi-minor axes of the target aperture.
    ir1, ir2 : float, float
        Inner semi-major and semi-minor axes of the sky annulus.
    or1, or2 : float, float
        Outer semi-major and semi-minor axes of the sky annulus.
    Kccd : float
        CCD gain in ADU per electron.
    saturation : float, optional
        Saturation level of the CCD. Defaults to infinity.
    plot : bool, optional
        Whether to generate a plot of the aperture and sky annulus. Defaults to True.
    save_name : str or bool, optional
        Filename for saving the plot. Set to False or 0 to disable saving. Defaults to saving plot as "aperE.png".
    
    Returns
    -------
    flx : float
        Total flux of the object in electrons.
    total_err : float
        Uncertainty in the measured flux.
    
    Notes
    -----
    Before using `aperE`, rotate the image so the major axis of the object
    is parallel or perpendicular to the x or y axis.
    
    Original MATLAB code by Professor Alberto Bolatto, edited by Alyssa Pagan, and
    translated to Python by ChongChong He, further edited by Orion Guiffreda. 
    Revised by Yugadeep Kanaparthy to improve readability and error propagation.
    """
    
    # Aperture Photometry
    a, b = im.shape

    xx, yy = np.meshgrid(range(b), range(a))

    ixsrc = ((xx - col) / rad1) ** 2 + ((yy - row) / rad2) ** 2 <= 1  # returns a boolean array same size as the image where True is part of the target aperture

    ixsky = np.logical_and(
        (((xx - col) / or1) ** 2) + (((yy - row) / or2) ** 2) <= 1,
        (((xx - col) / ir1) ** 2) + (((yy - row) / ir2) ** 2) >= 1,
    )  # returns a boolean array same size as the image where True is part of the sky annulus and False is not.

    src_pixels = im[ixsrc]  # returns a 1D array of pixel values in ADUs of the target aperture pixels
    num_src = len(src_pixels)
    src_err = np.sqrt(
        src_pixels / Kccd
    )  # Poisson read noise. dividing by the Kccd converts ADU to electrons.

    sky_pixels = im[ixsky]  # returns a 1D array of pixel values of the sky annulus
    num_sky = len(sky_pixels)
    sky = np.median(sky_pixels)  # Median value of the sky annulus is taken as 'the' sky brightness. To be subtracted from source pixels.
    sky_err = np.sqrt(
        sky_pixels * num_src / num_sky / Kccd
    )  # Sky error normalized and scaled by the number of sky pixels in the target aperture, and then converted to electrons

    net_pixels = (src_pixels - sky)  # each src pixel is corrected by subtracting the median sky signal

    flx = np.sum(net_pixels) / Kccd  # final flux value of the source in electrons
    total_err = (
        np.sqrt(np.sum(src_err**2) + np.sum(sky_err**2)) / Kccd
    )  # final error through error propagation on the flx term.

    # Checking for saturated pixels
    issat = 0
    if np.max(src_pixels) > saturation:  # checks if any src pixel is saturated
        issat = 1

    fw = np.copy(or1)
    ix = np.where(
        np.logical_and(
            np.logical_and(
                np.logical_and(xx >= col - 2 * fw, xx <= col + 2 * fw),
                yy >= row - 2 * fw,
            ),
            yy <= row + 2 * fw,
        )
    )

    aa = np.sum(np.logical_and(xx[0, :] >= col - 2 * fw, xx[0, :] <= col + 2 * fw))
    bb = np.sum(np.logical_and(yy[:, 0] >= row - 2 * fw, yy[:, 0] <= row + 2 * fw))
    px = np.reshape(xx[ix], (bb, aa))
    py = np.reshape(yy[ix], (bb, aa))
    pz = np.reshape(im[ix], (bb, aa))

    # Plotting and saving
    if plot:
        plt.figure()
        plt.imshow(
            pz, extent=[px[0, 0], px[0, -1], py[0, 0], py[-1, 0]], origin="lower"
        ) # added the origin='lower' argument so that the y-axis is origin is at the bottom left, as is expected.
        plt.tight_layout()
        p = np.arange(360) * np.pi / 180
        xc = np.cos(p)
        yc = np.sin(p)
        plt.plot(col + rad1 * xc, row + rad2 * yc, "w")
        plt.plot(col + ir1 * xc, row + ir2 * yc, "r")
        plt.plot(col + or1 * xc, row + or2 * yc, "r")

        if issat:
            plt.text(
                col,
                row,
                "CHECK SATURATION",
                ha="center",
                color="w",
                va="top",
                fontweight="bold",
            )
            print("At the peak this source has {:0.0f} counts.".format(max(src_pixels)))
            print("Judging by the number of counts, if this is a single exposure the")
            print("source is likely to be saturated. If this is the coadding of many")
            print(
                "short exposures, check in one of them to see if this message appears."
            )
            print("If it does, you need to flag the source as bad in this output file.")
            plt.tight_layout()

        if save_name:
            plt.savefig(save_name)
            print(f"Plot saved as: {save_name}")

    return flx, total_err



