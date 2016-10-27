# pydrs

This is a simple attempt to make a data reduction pipeline for VLT/SPHERE observations. At the moment, it is only functional for IRDIS observations. 

The ESO esorex software must be installed separately, along with the SPHERE recipes. At the moment, I've used the 0.15.0 version of the recipes (important intermediate products are not saved in version 0.18.0).

The pipeline is imported as:
> from pydrs import DRS

And is called as:
> data_red = DRS('name of the star', 'path to data')

The pipeline will prompt questions along the way, to confirm the use of the fits files it identified. For each step (e.g., dark, flat, centering), there are ways to define "blacklist" for the pipeline not to consider some files. 


MANDATORY INPUTS:
    + name of the star: a string that should match the name in the header of the science frames
    + path to data: a string pointing to the directory where the fits files are stored. The fits.Z files should be unzipped (no ".fits.Z"). Fits files for several stars or runs can be in the same directory, the pipeline will try to identify the proper calibration files as a function of the date of the observations. I advise not to run the script where the fits files are stored (the pipeline cleans some intermediate products and may move your fits files).

OPTIONAL INPUTS:
    + During the initialization of the DRS class, it will try to create three directories (by default "sof", "cosmetics", and "science"). These will be created in the directory "./" where the script is run from. You can change the names of these directory with the following keywords:
        - dir_cosmetics: a string for the name of the directory that will contain the dark, flat, starcenter products
        - dir_science: a string for the name of the direcotry that will contain the reduced science frames
        - dir_sof: a string for the name of the directory that will contain the files for the esorex recipes
    + summary: a boolean to plot a quick summary of the observations (to be improved)
    + width: a float for the width, in pixels, of a 2D gaussian to convolve the science frames with [ONLY for DPI observations]
    + kernel_width: an integer for median filtering of all columns of the science frames, to remove "hot" pixels that were not flagged in the bad pixel map [ONLY for DPI observations]
    + corono: a boolean to ignore the centering frames (will not apply the star_centering in the cosmetics for the science frames). Default is True (i.e., a coronagraph was used).

DEPENDENCIES:
    + VIP: if VIP is not installed, the pipeline will not try to improve the determination of the centering from the waffle pattern
    + astropy
    + photutils
    + matplotlib
    + scipy
    + numpy
    

