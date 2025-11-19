import numpy as np

def imshift(im,nr,nc):
    """Shifts an image by nr rows and nc columns
    (which can be either positive or negative)"""

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
