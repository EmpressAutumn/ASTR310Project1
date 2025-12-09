import numpy as np

def imshift(im,nr,nc, rotate=False):
    """Shifts an image by nr rows and nc columns
    (which can be either positive or negative)"""
    # Image rotation ability by Autumn
    if rotate:
        out = np.zeros(im.shape)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                out[i, im.shape[1] - 1 - j] = im[im.shape[0] - 1 - i, j]
        im = out

    a,b=im.shape
    imr=np.zeros(im.shape)
    ir1 = max(0, -nr)
    ir2 = min(a, a-nr)
    it1 = max(0, -nc)
    it2 = min(b, b-nc)
    r1=max(0,nr)
    r2=min(a,nr+a)
    c1=max(0,nc)
    c2=min(b,nc+b)
    imr[r1:r2, c1:c2] = im[ir1:ir2, it1:it2]

    return imr