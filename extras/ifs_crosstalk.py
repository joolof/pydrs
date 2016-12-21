import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve as conv
# --------------------------------------------------------------
def sph_ifs_crosstalk(img, remove_large_scale = True):
    """
    Adapted from A. Vigan's IDL scripts
    """
    dimimg = 2048
    sepmax = 20 # dimension of the matrix

    dimmat = sepmax * 2 + 1
    matsub = np.zeros(shape=(dimmat, dimmat))
    pc = sepmax
    bfac = 0.727986/1.8
    # --------------------------------------------------------------
    # Define a matrix to be used around each pixel
    # --------------------------------------------------------------
    for k in range(dimmat):
        for j in range(dimmat):
            if ((np.abs(pc - k) > 1.) | (np.abs(pc - j) > 1.)):
                rdist = np.sqrt((1. * pc - 1.* k)**2. + (1. * pc - 1.* j)**2.)
                matsub[j,k] = 1. / (1. + (rdist/bfac)**3.)
    # --------------------------------------------------------------
    # Convolution and subtraction
    # --------------------------------------------------------------
    kernel = conv(img, matsub, mode='same')
    imgsub = img - kernel
    # --------------------------------------------------------------
    # Remove the large scale structure
    # --------------------------------------------------------------
    if remove_large_scale:
        imgfin = imgsub.copy()
        # --------------------------------------------------------------
        # Step 2: calculation of the cross-talk on large scale
        # on sub-images of 64x64 pixels
        # --------------------------------------------------------------
        img = imgfin.copy()
        dimsub = 64
        dimimgct = dimimg / dimsub

        valinix = np.zeros(dimimgct * dimimgct, dtype = int)
        valfinx = np.zeros(dimimgct * dimimgct, dtype = int)
        valiniy = np.zeros(dimimgct * dimimgct, dtype = int)
        valfiny = np.zeros(dimimgct * dimimgct, dtype = int)
        konta = 0
     
        # Define the positions of the subimages
        for j in range(dimimgct):
            for k in range(dimimgct):
                valinix[konta] = k * dimsub
                valfinx[konta] = valinix[konta] + dimsub - 1
                valiniy[konta] = j * dimsub
                valfiny[konta] = valiniy[konta] + dimsub - 1
                konta = konta+1

        mdnimg = np.median(img)
        for k in range(dimimgct):
            for j in range(dimimgct):
                if np.abs(img[k,j]) > 30000.:
                    img[k,j] = mdnimg


        # For each subimage it creates an histogram and defines
        # the value of the maximum of the pixel counts distribution
        stephist = 10.e0
        imgct = np.zeros(shape=(dimimgct,dimimgct))
        for k in range(dimimgct * dimimgct):
            imgsub  = img[valinix[k]:valfinx[k],valiniy[k]:valfiny[k]]
            ncount, bin_edges = np.histogram(imgsub, bins = np.int((np.max(imgsub) - np.min(imgsub))/stephist))
            vcount = bin_edges[:-1]
            vcount += stephist/2.e0

            rs = np.where(ncount == np.max(ncount))[0][0]
            valct = vcount[rs]
            cy = np.int(np.float(k)/np.float(dimimgct))
            cx = k - cy * dimimgct
            imgct[cx,cy] = valct
        # --------------------------------------------------------------
        # Step 3: Subtraction of the large scale cross talk
        # --------------------------------------------------------------
        img = imgfin.copy()
        imgsub = np.zeros(shape=(dimimg,dimimg))
        for k in range(dimimgct * dimimgct):
            cy = np.int(np.float(k)/np.float(dimimgct))
            cx = k - cy * dimimgct
            imgsub[valinix[k]:valfinx[k],valiniy[k]:valfiny[k]] = img[valinix[k]:valfinx[k],valiniy[k]:valfiny[k]] - imgct[cx,cy]
        # imgsub = np.float(imgsub)
  
    return imgsub




