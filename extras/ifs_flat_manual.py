import numpy as np
import ifs_misc as misc
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_erosion
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
    fits_name, fits_type = misc.read_sof(sof_file)
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
    else:
        misc.error_msg('You should provide at least one bad pixel map.')
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
        misc.error_msg('The two flat were not taken with the same lamp.')
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
    nflat = misc.sigma_filter(flat_sub, bpm, neighbor_box = 6, min_neighbors = 5)
    print '\tSigma clipping: first pass ...'
    nflat = misc.median_clip(nflat, 5., num_neighbor = 5)
    print '\tSigma clipping: second pass ...'
    nflat = misc.median_clip(nflat, 3., num_neighbor = 5)
    nflat = nflat / np.median(nflat)
    # --------------------------------------------------------------        
    # Second pass
    # --------------------------------------------------------------        
    new_bpm = np.zeros(shape=(2048, 2048))
    fbpm = ((nflat < 0.9) | (nflat > 1.1))
    new_bpm[fbpm] = 1.
    print '\tMasking and interpolating the flat: second pass ...'
    nflat = misc.sigma_filter(nflat, new_bpm, neighbor_box = 6, min_neighbors = 5)
    print '\tSigma clipping: third pass ...'
    nflat = misc.median_clip(nflat, 5., num_neighbor = 5)
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
