import sys
import clip_sigma
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion
# --------------------------------------------------------------        
# To print some error messages and exit smoothly
# --------------------------------------------------------------        
def error_msg(message):
    print message
    sys.exit()
# --------------------------------------------------------------        
# Check the header of a fits file for the proper keyword
# --------------------------------------------------------------        
def check_lamp(hdr):
    """
    Check the header for proper keywords to identify which lamp was used
    """
    idlamp = ''
    if 'ESO INS2 LAMP1 ST' in hdr.keys():
        idlamp = '_l1'
    if 'ESO INS2 LAMP2 ST' in hdr.keys():
        idlamp = '_l2'
    if 'ESO INS2 LAMP3 ST' in hdr.keys():
        idlamp = '_l3'
    if 'ESO INS2 LAMP4 ST' in hdr.keys():
        idlamp = '_l4'
    if 'ESO INS2 LAMP5 ST' in hdr.keys():
        idlamp = '_l5'
    return idlamp
# --------------------------------------------------------------        
# Custom routine, adapted from an IDL script written by A. Vigan
# --------------------------------------------------------------        
def sph_ifs_detector_flat_manual(sof_file, ffname, bpname):
    """
    From Arthur Vigan's IDL script. All credits go to him.
    Paper to be cited:
        Vigan et al., 2015, MNRAS, 454, 129

    Furthermore, the following papers should also be cited:
        IFS general descripton: Claudi et al., 2008, SPIE, 7014
        Performance: Mesa et al., 2015, A&A, 576, 121
    """
    # --------------------------------------------------------------        
    # Read the SOF file (it sould definitely exists, I just created it)
    # --------------------------------------------------------------        
    f = open(sof_file, 'r')
    lines = f.readlines()
    f.close()
    nl = len(lines)
    fits_name = []
    fits_type = []
    for i in range(nl):
        fits_name.append(lines[i].split()[0])
        fits_type.append(lines[i].split()[1])
    fits_name = np.array(fits_name)
    fits_type = np.array(fits_type)
    # --------------------------------------------------------------        
    sel_ff = np.where(fits_type == 'IFS_DETECTOR_FLAT_FIELD_RAW')[0]
    sel_bb = np.where(fits_type == 'IFS_STATIC_BADPIXELMAP')[0]
    nff = len(sel_ff)
    nbb = len(sel_bb)
    ff_file = fits_name[sel_ff]
    if nbb != 0:
        bpm_file = fits_name[sel_bb]
        hdu = fits.open(bpm_file[0])
        bpm = hdu[0].data
        for i in range(nbb):
            hdu = fits.open(bpm_file[i])
            tmp = hdu[0].data
            hdu.close()
            bpm = bpm|tmp
        # bpm = np.byte(np.abs(1-bpm))
    else:
        print 'You should provide at least one bad pixel map.'
        sys.exit
    # --------------------------------------------------------------        
    # Read the lamp files
    # --------------------------------------------------------------        
    hdu = fits.open(ff_file[0])
    hdr0 = hdu[0].header
    ff0 = hdu[0].data
    hdu.close()
    hdu = fits.open(ff_file[1])
    hdr1 = hdu[0].header
    ff1 = hdu[0].data
    hdu.close()

    ilamp0 = check_lamp(hdr0)
    ilamp1 = check_lamp(hdr1)
    if ilamp0 != ilamp1:
        print 'The two flat were not taken with the same lamp.'
        sys.exit
    # --------------------------------------------------------------        
    # Create the master flat
    # --------------------------------------------------------------        
    dit0 = hdr0['ESO DET SEQ1 DIT']
    dit1 = hdr1['ESO DET SEQ1 DIT']
    # Order by increasing dit
    if (dit1 < dit0)    :
        tmp = dit1
        dit1 = dit0
        dit0 = tmp

        tmp = ff1
        ff1 = ff0
        ff0 = tmp
    # Average along the third dimension, if there is any
    if len(np.shape(ff0)) == 3:
        ff0 = np.median(ff0, axis = 0)
    if len(np.shape(ff1)) == 3:
        ff1 = np.median(ff1, axis = 0)

    flat_sub = ff1 - ff0
    fake_flat = np.ones(shape=(2048, 2048))
    fake_dark = np.zeros(shape=(2048, 2048))
    # --------------------------------------------------------------        
    # First pass
    # --------------------------------------------------------------        
    print '\tMasking and interpolating the flat: first pass ...'
    nflat = clip_sigma.sigma_filter(flat_sub, bpm, neighbor_box = 6, min_neighbors = 5)
    print '\tSigma clipping: first pass ...'
    nflat = clip_sigma.median_clip(nflat, 5., num_neighbor = 5)
    print '\tSigma clipping: second pass ...'
    nflat = clip_sigma.median_clip(nflat, 3., num_neighbor = 5)
    nflat = nflat / np.median(nflat)
    # --------------------------------------------------------------        
    # Second pass
    # --------------------------------------------------------------        
    new_bpm = np.zeros(shape=(2048, 2048))
    fbpm = ((nflat < 0.9) | (nflat > 1.1))
    new_bpm[fbpm] = 1.
    print '\tMasking and interpolating the flat: second pass ...'
    nflat = clip_sigma.sigma_filter(nflat, new_bpm, neighbor_box = 6, min_neighbors = 5)
    print '\tSigma clipping: third pass ...'
    nflat = clip_sigma.median_clip(nflat, 5., num_neighbor = 5)
    # --------------------------------------------------------------        
    # Final flat
    # --------------------------------------------------------------        
    final_flat = nflat / np.median(nflat)
    final_bpm = np.zeros(shape=(2048, 2048))
    fbpm = ((final_flat < 0.95) | (final_flat > 1.1))
    final_bpm[fbpm] = 1.

    Xin, Yin = np.mgrid[0:10, 0:10] - 4.5
    dist = np.sqrt(Xin**2. + Yin**2.)
    kern = np.zeros(shape=(10, 10), dtype=int)
    kern[(dist <= 5.)] = 1
    mask = binary_dilation(binary_dilation(binary_erosion(final_bpm, structure = kern), structure = kern), structure = kern)
    final_flat[mask] = 1.

    fits.writeto(ffname, final_flat, clobber = True, output_verify = "ignore", header = hdr0)
    fits.writeto(bpname, final_bpm, clobber = True, output_verify = "ignore", header = hdr0)


# --------------------------------------------------------------        
# Custom routine, adapted from an IDL script written by A. Vigan
# --------------------------------------------------------------        
def sph_ifs_preprocess(sof_file, coll = False, bkgsub = False, bpcor = False, xtalk = False, colltyp = 'MEAN', collval = 0.5, colltol = 0.05):
    """
    From Arthur Vigan's IDL script. All credits go to him.
    Paper to be cited:
        Vigan et al., 2015, MNRAS, 454, 129

    Furthermore, the following papers should also be cited:
        IFS general descripton: Claudi et al., 2008, SPIE, 7014
        Performance: Mesa et al., 2015, A&A, 576, 121
    """
    # --------------------------------------------------------------        
    # Check the inputs
    # --------------------------------------------------------------        
    if type(sof_file) is not str: error_msg('The sof_file should be a string.') 
    if type(colltyp) is not str: error_msg('colltyp should be a string.') 
    if type(collval) is not float: error_msg('collval should be a float.') 
    if type(colltol) is not float: error_msg('colltol should be a float.') 
    if type(coll) is not bool: error_msg('colltol should be a boolean.') 
    if type(bkgsub) is not bool: error_msg('bkgsub should be a boolean.') 
    if type(bpcor) is not bool: error_msg('bpcor should be a boolean.') 
    if type(xtalk) is not bool: error_msg('xtalk should be a boolean.') 
    if collval <=0: error_msg('collval should be >0')
    if colltol <=0: error_msg('colltol should be >0')
    # --------------------------------------------------------------        
    # Read the SOF file (it sould definitely exists, I just created it)
    # --------------------------------------------------------------        
    f = open(sof_file, 'r')
    lines = f.readlines()
    f.close()
    nl = len(lines)
    fits_name = []
    fits_type = []
    for i in range(nl):
        fits_name.append(lines[i].split()[0])
        fits_type.append(lines[i].split()[1])
    fits_name = np.array(fits_name)
    fits_type = np.array(fits_type)
    # --------------------------------------------------------------        
    sel_ff = np.where(fits_type == 'IFS_RAW')[0]
    sel_dd = np.where(fits_type == 'IFS_MASTER_DARK')[0]
    sel_bb = np.where(fits_type == 'IFS_STATIC_BADPIXELMAP')[0]
    nff = len(sel_ff)
    ndd = len(sel_dd)
    nbb = len(sel_bb)
    # --------------------------------------------------------------        
    # Check the files provided
    # --------------------------------------------------------------        
    # Science data
    if nff == 0:
        error_msg('No science frames provided.')
    raw_file = fits_name[sel_ff]
    # Background files
    if ndd == 1:
        hdu = fits.open(fits_name[sel_dd[0]])
        bkg = hdu[0].data
        hdu.close()
    else:
        if bkgsub:
            print 'No background (or more than one) file were provided.'
            print 'Skipping background subtraction'
            bkgsub = False
    # Bad pixels maps
    if nbb != 0:
        bpm_file = fits_name[sel_bb]
        hdu = fits.open(bpm_file[0])
        bpm = hdu[0].data
        for i in range(nbb):
            hdu = fits.open(bpm_file[i])
            tmp = hdu[0].data
            hdu.close()
            bpm = bpm|tmp
    else:
        if bpcor:
            print 'No bad pixel map was provided.'
            print 'Skipping BP correction'
            bpcor = False
    # --------------------------------------------------------------        
    # Make a dictionary that will be useful somehow
    # --------------------------------------------------------------        
    nframes = 0
    for i in range(nff):
        hdr = fits.getheader(raw_file[i])
        if 'NAXIS3' in hdr.keys():
            ndit = hdr['NAXIS3']
        else:
            ndit = 1
        nframes += 1
    frames = {}
    frames['file'] = []
    frames['dit'] = np.zeros(nframes)
    frames[''] = 
    frames[''] = 
    frames[''] = 
    frames[''] = 
    frames[''] = 
    frames[''] = 


# d['filename'] = []