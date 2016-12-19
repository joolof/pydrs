import os.path
import numpy as np
import ifs_misc as misc
import ifs_crosstalk
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import astropy.coordinates as coord
# --------------------------------------------------------------        
# Get relevant info from the header
# --------------------------------------------------------------
def header_info(hdr):
    """
    Get the relevant info from the header
    """
    # Get the longitude for the parallactic angle
    geolon = hdr['ESO TEL GEOLON']
    geolat = hdr['ESO TEL GEOLAT']
    # Get the DIT and NDIT
    dit  = hdr['ESO DET SEQ1 DIT']
    if 'NAXIS3' in hdr.keys():
        ndit = hdr['NAXIS3']
    else:
        ndit = 1
    # Get the RA and DEC during the observations
    ra_drot  = hdr['ESO INS4 DROT1 RA']
    dec_drot = hdr['ESO INS4 DROT1 DEC']

    ra0 = np.int(ra_drot/10000.)
    ra1 = np.int((ra_drot - ra0 * 10000.) / 100)
    ra2 = ra_drot - ra0 * 10000. - ra1 * 100.
    ra = coord.Angle(str(ra0) + 'h' + str(ra1) + 'm' + str(ra2)+'s').value

    dec0 = np.int(dec_drot / 10000.)
    dec1 = np.int((dec_drot - dec0 * 10000.) / 100)
    dec2 = dec_drot - dec0 * 10000. - dec1 * 100.
    dec = coord.Angle(str(dec0) + 'd' + str(np.abs(dec1)) + 'm' + str(np.abs(dec2))+'s').value

    tobs = Time([hdr['DATE-OBS'], hdr['ESO DET FRAM UTC']])
    jul_in = tobs.jd[0]
    jul_out = tobs.jd[1]
    delta = (jul_out - jul_in) / (ndit * 1.)

    pa_beg = np.zeros(ndit)
    pa_mid = np.zeros(ndit)
    pa_end = np.zeros(ndit)

    for i in range(ndit):
        # Time for each DIT
        time_beg = jul_in + delta * i
        time_mid = jul_in + delta * i + (dit / 24.e0 / 3600.e0 / 2.e0)
        time_end = jul_in + delta * i + (dit / 24.e0 / 3600.e0)

        lst_b = Time(time_beg, format='jd', location = (geolon, geolat)).sidereal_time('mean').value
        lst_m = Time(time_mid, format='jd', location = (geolon, geolat)).sidereal_time('mean').value
        lst_e = Time(time_end, format='jd', location = (geolon, geolat)).sidereal_time('mean').value

        pa_beg[i] = parangle(lst_b - ra, dec, geolat)
        pa_mid[i] = parangle(lst_m - ra, dec, geolat)
        pa_end[i] = parangle(lst_e - ra, dec, geolat)

    return ndit, pa_beg, pa_mid, pa_end

# --------------------------------------------------------------        
# Method to compute the parralactic angle
# --------------------------------------------------------------        
def parangle(ha, dec, latitude):
    """
    HA is in hours decimal, dec in degrees and latitude in degrees
    """
    r2d = 180.e0 / np.pi
    d2r = np.pi/180.e0
    had = ha * 15.e0
    para = -r2d * np.arctan2(-np.sin(d2r*had), np.cos(d2r*dec)*np.tan(d2r*latitude)-np.sin(d2r*dec)*np.cos(d2r*had))
    return para

# --------------------------------------------------------------        
# Custom routine, adapted from an IDL script written by A. Vigan
# --------------------------------------------------------------        
def sph_ifs_preprocess(sof_file, folder, coll = False, bkgsub = True, bpcor = True, xtalk = False, colltyp = 'mean', collval = 0.5, colltol = 0.05):
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
    if (colltyp != 'mean'):
        error_msg('colltyp should be \"mean\". The other methods have to be implemented')
    # if ((colltyp != 'mean') & (colltyp != 'angle') & (colltyp != 'coadd')):
    #     error_msg('colltyp should be either \"mean\", \"angle\", or \"coadd".')
    # --------------------------------------------------------------        
    # Read the SOF file (it sould definitely exists, I just created it)
    # --------------------------------------------------------------        
    fits_name, fits_type = misc.read_sof(sof_file)
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
    # Will have a look at it later to see how useful that is
    # --------------------------------------------------------------        
    # nframes = 0
    # for i in range(nff):
    #     hdr = fits.getheader(raw_file[i])
    #     if 'NAXIS3' in hdr.keys():
    #         ndit = hdr['NAXIS3']
    #     else:
    #         ndit = 1
    #     nframes += 1
    # frames = {}
    # frames['file'] = []
    # frames['dit'] = np.zeros(nframes)
    # --------------------------------------------------------------        
    # Doing the pre-processing, here we go
    # --------------------------------------------------------------        
    print '\tPre-processing ...'
    for j in range(nff):
        print '\t\tFile ' + str(j+1) + ' of ' + str(nff) + ' ...'
        # --------------------------------------------------------------        
        # Read the data
        # --------------------------------------------------------------        
        hdu = fits.open(raw_file[j])
        hdr = hdu[0].header
        img = hdu[0].data
        hdu.close()
        # --------------------------------------------------------------        
        # Get relevant info from the header
        # --------------------------------------------------------------
        ndit, pa_beg, pa_mid, pa_end = header_info(hdr)
        # --------------------------------------------------------------        
        # Apply the background subtraction
        # --------------------------------------------------------------        
        if bkgsub:
            print '\t\tApplying the background correction ...'
            if ndit > 1:
                for i in range(ndit):
                    frame = img[i,].copy()
                    frame -= bkg
                    img[i,] = frame
            else:
                img -= bkg
        # --------------------------------------------------------------        
        # Collapse or bin the data
        # --------------------------------------------------------------        
        if coll:
            final_pa_start = 0.
            final_pa_end = 0.
            # --------------------------------------------------------------
            # For the MEAN
            # --------------------------------------------------------------
            if colltyp == 'mean':
                print '\t\tCollapse algorithm: mean'
                if ndit > 1:
                    img = np.mean(img, axis = 0)
                    idx = ndit - 1
                    ndit = 1
                else:
                    idx = 0
                # Probably need to update the parallactic angle ... will see later on
                final_pa_start = pa_beg[0]
                final_pa_end = pa_end[idx]
                # if final_pa_start/final_pa_end < 0.:
                #     if final_pa_start > final_pa_end:
                #         final_pa_end += 360.
                #     else:
                #         final_pa_start += 360.
                print '\t\tPA start: ' + format(final_pa_start, '0.2f')
                print '\t\tPA end: ' + format(final_pa_end, '0.2f')
            # --------------------------------------------------------------
            # For the ANGLE
            # --------------------------------------------------------------
            elif colltyp == 'angle':
                print '\t\tCollapse algorithm: angle (with collval = ' + format(collval, '0.2f') + ' deg).'
                imin = 0
                imax = 0
                nbin = 0
                val_min = [np.nan]
                idx_min = [-1]
                idx_max = [-1]
                # for d in range(ndit):
                #     if d >= imin:
                #         delta = pa_beg[imin] - pa_end[imin:]
                #         minang = np.min([np.abs(np.abs(delta) - collval), imax])
                #         if minang <= colltol:
                #             val_min = np.concatenate(val_min, [minang])
                #             idx_min = np.concatenate(idx_min, [imin])
                #             idx_max = np.concatenate(idx_max, [imin + imax])
                #             imin = imin + imax + 1
                #             nbin += 1
            else:
                error_msg('Not other options for colltyp')
            # --------------------------------------------------------------
            # Bad pixel correction
            # --------------------------------------------------------------
            if bpcor:
                print '\t\tDoing the bad pixel map correction ...'
                if ndit > 1:
                    for i in range(ndit):
                        frame = img[i,].copy()
                        print '\tMasking and interpolating the flat: first pass ...'
                        frame = clip_sigma.sigma_filter(frame, bpm, neighbor_box = 6, min_neighbors = 5)
                        print '\tSigma clipping: first pass ...'
                        frame = clip_sigma.median_clip(frame, 5., num_neighbor = 5)
                        print '\tSigma clipping: second pass ...'
                        frame = clip_sigma.median_clip(frame, 3., num_neighbor = 5)
                        img[i,] = frame
                else:
                    img = clip_sigma.sigma_filter(img, bpm, neighbor_box = 6, min_neighbors = 5)
                    print '\tSigma clipping: first pass ...'
                    img = clip_sigma.median_clip(img, 5., num_neighbor = 5)
                    print '\tSigma clipping: second pass ...'
                    img = clip_sigma.median_clip(img, 3., num_neighbor = 5)
            # --------------------------------------------------------------
            # Now the freaking cross talk ...
            # --------------------------------------------------------------
            if xtalk:
                print '\t\tDoing the cross talk correction ...'
                if ndit > 1:
                    for i in range(ndit):
                        frame = img[i,].copy()
                        frame = ifs_crosstalk.sph_ifs_crosstalk(frame)
                        img[i,] = frame
                else:
                    img = ifs_crosstalk.sph_ifs_crosstalk(img)
            # --------------------------------------------------------------
            # Save the fits file
            # --------------------------------------------------------------
            hdr['HIERARCH ESO TEL PARANG START'] = final_pa_start
            hdr['HIERARCH ESO TEL PARANG END'] = final_pa_end
            suffix = '_preproc'
            fname = os.path.basename(raw_file[j])
            fname.replace('.fits', '')
            fname = folder + '/' + fname + suffix + '.fits'
            fits.writeto(fname, img, clobber = True, output_verify = "ignore", header = hdr)
