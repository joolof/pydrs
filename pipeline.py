import sys
import glob
import scipy
import os.path
import datetime
import subprocess
import numpy as np
import scipy.ndimage
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, daofind
from scipy.signal import fftconvolve, medfilt
try:
    import vip
    is_vip = True
except ImportError:
    is_vip = False
# -----------------------------------------------------------------------------
class DRS(object):
    """
    Class for the SPHERE data reduction. The module should be imported as follows:

    > from pydrs import DRS

    and can be called as

    > data_red = DRS('Starname', 'path to data')

    Everything should be automatic from there and the pipeline will ask
    you some questions along the way.

    MANDATORY INPUTS:
        + name of the star: should match the name in the header of the science frames
        + directory where the fits files are stored (no ".ftis.Z")

    OPTIONAL INPUTS:
        + dir_cosmetics: a string for the name of the directory that will contain the 
                         dark, flat, starcenter products
        + dir_science: a string for the name of the direcotry that will contain the
                       reduced science frames
        + dir_sof: a string for the name of the directory that will contain the files
                   for the esorex recipes
        + summary: a boolean to plot a quick (to be improved) summary of the observations
        + width: a float for the width, in pixels, of a 2D gaussian to convolve the 
                 science frames [ONLY for DPI observations]
        + kernel_width: an integer for median filtering of all columns of the science 
                        frames, to remove "hot" pixels that were not flagged in the bad
                        pixel map [ONLY for DPI observations]
        + corono: a boolean to ignore the centering frames (will not apply the
                  star_centering in the cosmetics for the science frames). Default
                  is True (i.e., a coronagraph was used).


    """
    def __init__(self, starname, path_to_fits, dir_cosmetics = 'cosmetics', dir_science = 'science', dir_sof = 'sof', summary = False, width = 0., kernel_width = 9, corono = True):
        # --------------------------------------------------------------        
        # Check the inputs
        # --------------------------------------------------------------        
        assert type(starname) is str, 'The name of the star should be a string'
        assert type(summary) is bool, 'The variable \'summary\' should be True or False'
        assert type(corono) is bool, 'The variable \'corono\' should be True or False'
        assert type(dir_cosmetics) is str, 'The name of the cosmetics directory should be a string'
        assert type(dir_science) is str, 'The name of the science directory should be a string'
        assert type(dir_sof) is str, 'The name of the sof directory should be a string'
        assert type(path_to_fits) is str, 'The name of the directory containing the fits file should be a string'
        assert type(width) is float, 'The value of width should be a float or an integer'
        assert type(kernel_width) is int, 'The width of the kernel for the readout smoothing should be an integer'
        if not os.path.isdir(path_to_fits):
            self._error_msg('The directory containing the fits files cannot be found.')
        # --------------------------------------------------------------        
        # Define some variables
        # --------------------------------------------------------------        
        self._km = kernel_width
        self._nx = 1024
        self._starname = starname
        self._width = width
        self._path_to_fits = path_to_fits
        self._csv_red = False
        self._is_dark = False
        self._is_flat = False
        self._is_center = False
        self._is_flux = False
        self._is_science = False
        self._corono = corono
        # --------------------------------------------------------------
        # Check for existing directories, if not there try to create them
        # --------------------------------------------------------------
        if not os.path.isdir(dir_cosmetics):
            try:
                os.mkdir(dir_cosmetics)
            except:
                self._error_msg("Cannot create the directory \"" + dir_cosmetics +"\"")
        if not os.path.isdir(dir_science):
            try:
                os.mkdir(dir_science)
            except:
                self._error_msg("Cannot create the directory \"" + dir_science +"\"")
        if not os.path.isdir(dir_sof):
            try:
                os.mkdir(dir_sof)
            except:
                self._error_msg("Cannot create the directory \"" + dir_sof +"\"")
        self._dir_cosm = dir_cosmetics
        self._dir_sci = dir_science
        self._dir_sof = dir_sof
        # --------------------------------------------------------------        
        # Check if the star has observations in the directory
        # --------------------------------------------------------------        
        self._check_star()
        if not self._csv_red:
            self._error_msg('Could not read the csv file properly.')
        # --------------------------------------------------------------
        # Get the proper mode, MJD and DIT for the science observations
        # --------------------------------------------------------------
        self.obs_mode = self._get_mode()
        self.obs_mjd = self._get_mjd()
        self.obs_dit = self._get_dit()
        self.obs_filter = self._get_filter()
        # --------------------------------------------------------------
        # Print a brief summary of what we have.
        # --------------------------------------------------------------
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & 
            (self._mjd == self.obs_mjd) & (self._dit == self.obs_dit) & (self._filter == self.obs_filter) & 
            ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
        nframes = len(sel)
        txt = '\n' + '-' * 80 + '\n' + self._starname + ' - ' + self.obs_mode + '\nMJD ' + str(self.obs_mjd) + ': ' + str(nframes) + ' frames\n'
        txt += 'DIT: ' + str(self.obs_dit) + ' sec\nFilter: ' + self.obs_filter
        print txt
        print '-'*80
        # --------------------------------------------------------------
        # Plot a summary of the center and science observations
        # --------------------------------------------------------------
        if summary:
            self._plot_summary()
        # --------------------------------------------------------------
        # Check which recipes have been run already
        # --------------------------------------------------------------
        self._check_prod()
        # --------------------------------------------------------------
        # Run the recipes that needs to be run
        # --------------------------------------------------------------
        if not self._is_dark: self._master_dark()
        if not self._is_flat: self._flat()
        if self._corono:
            if not self._is_center: self._center()
        if not self._is_flux: self._flux()
        if not self._is_science: self._science()
        # --------------------------------------------------------------
        # Merge the frames if necessary
        # --------------------------------------------------------------
        if ((self.obs_mode == 'IMAGE,DUAL') | (self.obs_mode == 'IMAGE,CLASSICAL')):
            self._merge()
        if self.obs_mode == 'POLARIMETRY':
            self._reduce_dpi()

    # --------------------------------------------------------------
    # Method to integrate something
    # --------------------------------------------------------------
    def _integrate(self,x,fx):
        return np.sum( (x[:-1] - x[1:]) * (fx[:-1] + fx[1:]) * 0.5)

    # --------------------------------------------------------------
    # Method for the master dark
    # --------------------------------------------------------------
    def _master_dark(self):
        """
        Method to create the master dark frame
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        # fits_dark = self._find_fits('SKY')
        fits_dark = self._find_fits('DARK')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/master_dark.sof', 'w')
        for i in range(len(fits_dark)):
            f.write(self._path_to_fits + '/' + fits_dark[i] + '\tIRD_DARK_RAW\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ird_master_dark', 
                    # '--ird.master_dark.clean_mean.reject_low=0',
                    # '--ird.master_dark.clean_mean.reject_high=0',
                    # '--ird.master_dark.sigma_clip=1.0',
                    '--ird.master_dark.outfilename=' + self._dir_cosm + '/master_dark.fits', 
                    '--ird.master_dark.save_addprod=TRUE',
                    '--ird.master_dark.badpixfilename=' + self._dir_cosm + '/static_badpixels.fits',
                    self._dir_sof + '/master_dark.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/master_dark.fits'):
                self._error_msg('It seems a master_dark.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------        
    # Method for the flat
    # --------------------------------------------------------------        
    def _flat(self):
        """
        Method to produce the flat
        """
        # --------------------------------------------------------------        
        # First, identify the proper files
        # --------------------------------------------------------------        
        fits_flat = self._find_fits('FLAT')
        # fits_bck = self._find_fits('BACKGROUND')
        # --------------------------------------------------------------        
        # Write the SOF file
        # --------------------------------------------------------------        
        f = open(self._dir_sof + '/flat.sof', 'w')
        for i in range(len(fits_flat)):
            f.write(self._path_to_fits + '/' + fits_flat[i] + '\tIRD_FLAT_FIELD_RAW\n')
        # f.write(self._dir_cosm + '/master_dark.fits\tIRD_MASTER_DARK\n')
        f.write(self._dir_cosm + '/static_badpixels.fits\tIRD_STATIC_BADPIXELMAP\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ird_instrument_flat',
                    # '--ird.instrument_flat.threshold=0.9',
                    '--ird.instrument_flat.save_addprod=TRUE',
                    '--ird.instrument_flat.badpixfilename=' + self._dir_cosm + '/instr_flat_badpixels.fits',
                    '--ird.instrument_flat.outfilename=' + self._dir_cosm + '/irdis_flat.fits',
                    self._dir_sof + '/flat.sof']
            doflat = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/irdis_flat.fits'):
                self._error_msg('It seems a irdis_flat.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------
    # Method to read the filter curves
    # --------------------------------------------------------------
    def _read_filters(self, nd):
        """
        Method to read the filters files and such.
        If there are no ND filter (nd == 'OPEN'), there is no scaling other than the 
        possible DIT differences.
        """
        f_left, f_right = 1.0, 1.0
        # --------------------------------------------------------------
        # Check if the proper files are here
        # --------------------------------------------------------------
        dir_fil = os.path.dirname(os.path.abspath(__file__)) + '/filters'
        if not os.path.isfile(dir_fil + '/SPHERE_IRDIS_' + self.obs_filter + '.dat'):
            self._error_msg('The file SPHERE_IRDIS_' + self.obs_filter + '.dat seems to be missing in ' + dir_fil)
        if not os.path.isfile(dir_fil + '/SPHERE_CPI_ND.dat'):
            self._error_msg('The file SPHERE_CPI_ND.dat for ND filters seems to be missing in ' + dir_fil)
        # --------------------------------------------------------------
        # Read the ND curve
        # --------------------------------------------------------------
        data = ascii.read(dir_fil + '/SPHERE_CPI_ND.dat', data_start = 1)
        nd_wave = data['wavelength']
        nd_tr_tmp = data[nd]
        # --------------------------------------------------------------
        # Read the filter curves
        # --------------------------------------------------------------
        data = ascii.read(dir_fil + '/SPHERE_IRDIS_' + self.obs_filter + '.dat', data_start = 1)
        fl_wave = data['wavelength']
        if 'DB' in self.obs_filter: # then we are in dual mode
            fl_left = data['left']
            fl_right = data['right']
        elif 'BB' in self.obs_filter:
            fl_left = data['broad']
            fl_right = data['broad']
        else:
            self._error_msg('Are these dual band or broad band data ? Could not read the filter file')
        # --------------------------------------------------------------
        # Interpolate the ND curve over the wavelength of the filter curves
        # --------------------------------------------------------------
        nd_tr = np.interp(fl_wave, nd_wave, nd_tr_tmp)
        f_left = self._integrate(fl_wave, fl_left) / self._integrate(fl_wave, fl_left * nd_tr)
        f_right = self._integrate(fl_wave, fl_right) / self._integrate(fl_wave, fl_right * nd_tr)
        return f_left, f_right


    # --------------------------------------------------------------
    # Method for the flux calibration
    # --------------------------------------------------------------
    def _flux(self):
        """
        Method for the flux calibration. The goal is to extract the stellar psf
        and correct for the DIT and Neutral Density filters.
        """
        if self.obs_mode == 'POLARIMETRY':
            print 'Skipping flux calibration in DPI mode'
            print '-'*80
        else:
            # --------------------------------------------------------------
            # First, identify the proper files
            # --------------------------------------------------------------
            fits_flux = self._find_fits('FLUX', verbose = True)
            fits_bkg = self._find_fits('BACKGROUND', verbose = False)
            # --------------------------------------------------------------
            # Get the average of the flux measurements
            # --------------------------------------------------------------
            for i in range(len(fits_flux)):
                hdu = fits.open(self._path_to_fits + '/' + fits_flux[i])
                idfit = np.where(self._fitsname == fits_flux[i])[0][0]
                if i == 0:
                    f_dit = self._dit[idfit]
                    f_filt = self._filter[idfit]
                    f_nd = self._nd[idfit]
                    flux = hdu[0].data[0,]
                else:
                    flux += hdu[0].data[0,]
                    if self._dit[idfit] != f_dit: self._error_msg('Different DIT in the flux measurements')
                    if self._filter[idfit] != f_filt: self._error_msg('Different filters in the flux measurements')
                    if self._nd[idfit] != f_nd: self._error_msg('Different ND filters in the flux measurements')
                hdu.close()
            flux = flux / np.float(len(fits_flux))
            txt = 'Found ' + str(len(fits_flux)) + ' FLUX (' + f_nd + ')'
            # --------------------------------------------------------------
            # Get the average of the background measurements
            # --------------------------------------------------------------
            fits_pp_bkg = [] # list of proper fits file with the correct DIT compared to the FLUX
            for i in range(len(fits_bkg)):
                idfit = np.where(self._fitsname == fits_bkg[i])[0][0]
                if ((self._dit[idfit] == f_dit) & (self._filter[idfit] == f_filt)):
                    fits_pp_bkg.append(fits_bkg[i])
            # --------------------------------------------------------------
            # Check if there are any BACKGROUND measurements
            # --------------------------------------------------------------
            if len(fits_pp_bkg) == 0:
                self._error_msg('Could not find any BACKGROUND measurements matching the FLUX observations')
            # --------------------------------------------------------------
            # Get the average
            # --------------------------------------------------------------
            nb = 0
            for i in range(len(fits_pp_bkg)):
                hdu = fits.open(self._path_to_fits + '/' + fits_pp_bkg[i])
                if hdu[0].header['HIERARCH ESO INS4 FILT2 NAME'] == 'OPEN':
                    if nb == 0:
                        bkg = hdu[0].data[0,]
                        nb += 1
                    else:
                        bkg += hdu[0].data[0,]
                hdu.close()
            if nb == 0: 
                self._error_msg('Could not find BACKGROUND files among the following:')
                fits_bkg = self._find_fits('BACKGROUND', verbose = True)
            bkg = bkg / np.float(nb)
            txt += ' and ' + str(nb) + ' BACKGROUND measurements with DITs of ' + str(f_dit) + '.\n' + '-'*80
            print txt
            # --------------------------------------------------------------
            # Do the thing for the ND filter
            # --------------------------------------------------------------
            fl, fr = self._read_filters(f_nd)
            # --------------------------------------------------------------
            # Read the flat and divide the whole thing by it
            # --------------------------------------------------------------
            hdu = fits.open(self._dir_cosm + '/irdis_flat.fits')
            flat = hdu[0].data
            hdu.close()
            flux = (flux - bkg)/flat
            # --------------------------------------------------------------
            # Account for the DIT difference
            # --------------------------------------------------------------
            flux = flux / f_dit * self.obs_dit
            # --------------------------------------------------------------
            # Find the point sources
            # --------------------------------------------------------------
            thrs = 5.
            sources = daofind(flux, fwhm = 4.0, threshold = thrs * np.std(flux))
            if len(sources) < 2:
                self._error_msg('Found less than 2 point sources ... that is a problem. Maybe change the threshold in the DRS.py file')
            txt = 'Found ' + str(len(sources)) + ' point sources above ' + str(thrs) + ' sigma. The brightest two are:\n'
            bright = np.argsort(sources['peak'])[-2:]
            for i in range(len(bright)):
                txt += 'x: ' + format(sources['xcentroid'][bright[i]], '0.1f') 
                txt += '\ty: ' + format(sources['ycentroid'][bright[i]], '0.1f')
                txt += '\tpeak: ' + format(sources['peak'][bright[i]], '0.2f')
                if i != len(bright)-1:
                    txt += '\n'
            print txt
            print '-'*80
            # --------------------------------------------------------------
            x0, x1 = np.int(np.round(sources['xcentroid'][bright[0]])), np.int(np.round(sources['xcentroid'][bright[1]]))
            y0, y1 = np.int(np.round(sources['ycentroid'][bright[0]])), np.int(np.round(sources['ycentroid'][bright[1]]))
            b0, b1 = sources['peak'][bright[0]], sources['peak'][bright[1]]
            # --------------------------------------------------------------
            # Crop the psf
            # --------------------------------------------------------------
            psf0 = flux[y0 - 20:y0 + 20, x0 - 20:x0 + 20]
            psf1 = flux[y1 - 20:y1 + 20, x1 - 20:x1 + 20]
            if x0 < x1:
                b0 *= fl
                b1 *= fr
                fitsname0 = self._dir_cosm + '/psf_left.fits'
                fitsname1 = self._dir_cosm + '/psf_right.fits'
            else:
                b0 *= fr
                b1 *= fl
                fitsname0 = self._dir_cosm + '/psf_right.fits'
                fitsname1 = self._dir_cosm + '/psf_left.fits'
            # --------------------------------------------------------------
            # Save the first one
            # --------------------------------------------------------------
            hdu = fits.PrimaryHDU(data = psf0)
            hdu.header.append(('Peak', b0, 'Peak of the psf'),end=True)
            hdu.writeto(fitsname0, clobber = True)
            # --------------------------------------------------------------
            # Save the second one
            # --------------------------------------------------------------
            hdu = fits.PrimaryHDU(data = psf1)
            hdu.header.append(('Peak', b1, 'Peak of the psf'),end=True)
            hdu.writeto(fitsname1, clobber = True)

    # --------------------------------------------------------------        
    # Method to get the observations date
    # --------------------------------------------------------------        
    def _get_obs_time(self, fitsname):
        """
        Will return a datetime.datetime
        """
        hdu = fits.open(self._path_to_fits + '/' + fitsname)
        date_hdr = hdu[0].header['DATE-OBS']
        hdu.close()
        date_str = date_hdr.split('T')[0].split('-')
        time_str = date_hdr.split('T')[1].split(':')
        dateobs = datetime.datetime(np.int(date_str[0]), np.int(date_str[1]), np.int(date_str[2]), np.int(time_str[0]), np.int(time_str[1]), np.int(time_str[2][:2]))
        return dateobs

    # --------------------------------------------------------------        
    # Method for the star center
    # --------------------------------------------------------------        
    def _center(self):
        """
        Method to produce the star center
        """
        # --------------------------------------------------------------        
        # First, identify the proper files
        # --------------------------------------------------------------        
        fits_center = self._find_fits('CENTER')
        # --------------------------------------------------------------        
        # Select the proper center
        # --------------------------------------------------------------        
        id_waffle = 0
        if len(fits_center) > 1:
            print 'There are two waffle files:'
            for i in range(len(fits_center)):
                print '[' + str(i) + '] ' + fits_center[i]
            select = raw_input('Which one do you want to use: ')
            id_waffle = np.int(select) 
            if id_waffle > len(fits_center)-1:
                raise ValueError('The ID number is too high')
        # --------------------------------------------------------------        
        # Write the SOF file
        # --------------------------------------------------------------        
        f = open(self._dir_sof + '/star_center.sof', 'w')
        f.write(self._path_to_fits + '/' + fits_center[id_waffle] + '\tIRD_STAR_CENTER_WAFFLE_RAW\n')
        f.write(self._dir_cosm + '/static_badpixels.fits\tIRD_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/master_dark.fits\tIRD_MASTER_DARK\n')
        f.write(self._dir_cosm + '/irdis_flat.fits\tIRD_FLAT_FIELD\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ird_star_center', 
                    '--ird.star_center.outfilename=' + self._dir_cosm + '/starcenter.fits',
                    self._dir_sof + '/star_center.sof']
            star_c = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            # --------------------------------------------------------------        
            # Run esorex science_dbi to clean up the center frame
            # --------------------------------------------------------------        
            f = open(self._dir_sof + '/center_clean.sof', 'w')
            f.write(self._path_to_fits + '/' + fits_center[id_waffle] + '\tIRD_SCIENCE_DBI_RAW\n')
            f.write(self._dir_cosm + '/master_dark.fits\tIRD_MASTER_DARK\n')
            f.write(self._dir_cosm + '/static_badpixels.fits\tIRD_STATIC_BADPIXELMAP\n')
            f.write(self._dir_cosm + '/irdis_flat.fits\tIRD_FLAT_FIELD\n')
            f.write(self._dir_cosm + '/starcenter.fits\tIRD_STAR_CENTER\n')
            f.close()
            args = ['esorex', 'sph_ird_science_dbi',
                        self._dir_sof + '/center_clean.sof']
            sci_dbi = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            # --------------------------------------------------------------        
            # Clean up the fits files
            # --------------------------------------------------------------        
            # Rename left file
            mv_left = glob.glob('SPHER*_left.fits')
            args = ['mv', mv_left[0], self._dir_cosm + '/center_left.fits']
            mvfits = subprocess.Popen(args).wait()
            # Rename right file
            mv_right = glob.glob('SPHER*_right.fits')
            args = ['mv', mv_right[0], self._dir_cosm + '/center_right.fits']
            mvfits = subprocess.Popen(args).wait()
            # Changing for save the *.txt of the center images ---> My attempt
            mv_txt = glob.glob('SPHER*fctable*txt')
            args = ['mv', mv_txt[0], self._dir_cosm + '/cor.txt']
            mvtxt = subprocess.Popen(args).wait()
            # Remove other files
            rmfiles = glob.glob('SPHER*.*')
            for i in range(len(rmfiles)):
                args = ['rm', rmfiles[i]]
                rmtxt = subprocess.Popen(args).wait()
            args = ['rm', 'science_dbi.fits']
            rmtxt = subprocess.Popen(args).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/starcenter.fits'):
                self._error_msg('It seems a starcenter.fits file was not produced. Check the log above')
        else:
            sys.exit()
        # --------------------------------------------------------------
        # If VIP could be imported then update the centering
        # --------------------------------------------------------------
        if is_vip:
            self._update_center()


    # --------------------------------------------------------------
    # Method to update the center
    # --------------------------------------------------------------
    def _update_center(self):
        """
        Method to call VIP and update the centering
        """
        # --------------------------------------------------------------
        # Read the fits file used for the centering
        # --------------------------------------------------------------
        hdu = fits.open(self._dir_cosm + '/center_left.fits')
        if len(np.shape(hdu[0].data)) == 2:
            waffle_left = hdu[0].data
        elif len(np.shape(hdu[0].data)) == 3:
            waffle_left = np.mean(hdu[0].data, axis = 0)
        else:
            self._error_msg('Weird shape for the center_left fits file')
        hdu.close()

        hdu = fits.open(self._dir_cosm + '/center_right.fits')
        if len(np.shape(hdu[0].data)) == 2:
            waffle_right = hdu[0].data
        elif len(np.shape(hdu[0].data)) == 3:
            waffle_right = np.mean(hdu[0].data, axis = 0)
        else:
            self._error_msg('Weird shape for the center_right fits file')
        hdu.close()
        # --------------------------------------------------------------
        # Read the original center
        # --------------------------------------------------------------
        hdu = fits.open(self._dir_cosm + '/starcenter.fits')
        off_left = self._vip_centering(waffle_left)
        off_right = self._vip_centering(waffle_right)
        print '-' * 80
        # --------------------------------------------------------------
        # Update the starcenter.fits file
        # --------------------------------------------------------------
        hdu[1].data['CENTRE_LEFT_X'] += off_left[0]
        hdu[1].data['CENTRE_LEFT_Y'] += off_left[1]
        hdu[1].data['CENTRE_RIGHT_X'] += off_right[0]
        hdu[1].data['CENTRE_RIGHT_Y'] += off_right[1]
        hdu.writeto(self._dir_cosm + '/starcenter.fits', clobber = True)
        hdu.close()

    # --------------------------------------------------------------
    # VIP routine for the updated centering.
    # --------------------------------------------------------------
    def _vip_centering(self, array):
        """
        Method that uses the VIP centering stuff
        """
        txt_x = 'X: 512 -> '
        txt_y = 'Y: 512 -> '
        cenxy = [512, 512]
        offset = np.zeros(2)
        # --------------------------------------------------------------
        # Do the first search over a large grid
        # --------------------------------------------------------------
        dist = 15.
        fr_broad = vip.calib.frame_crop(array, 151, cenxy = cenxy, verbose = False)
        rough = vip.calib.frame_center_radon(fr_broad, cropsize = 141, wavelet=False, mask_center=None, 
            hsize = dist, step=1., nproc=2, verbose = False, plot = False)
        if ((np.abs(rough[0]) >= dist) | (np.abs(rough[1]) >= dist)):
            self._error_msg('Increase the box size for the first centering.')
        cenxy[0] -= np.int(rough[1])
        cenxy[1] -= np.int(rough[0])
        offset[0] -= np.int(rough[1])
        offset[1] -= np.int(rough[0])
        # --------------------------------------------------------------
        # Do the second search over a smaller grid.
        # --------------------------------------------------------------
        dist = 2.
        fr_broad = vip.calib.frame_crop(array, 151, cenxy = cenxy, verbose = False)
        rough = vip.calib.frame_center_radon(fr_broad, cropsize = 141, wavelet=False, mask_center=None, 
            hsize = dist, step=.1, nproc=2, verbose = False, plot = False)
        if ((np.abs(rough[0]) >= dist) | (np.abs(rough[1]) >= dist)):
            self._error_msg('Increase the box size for the second centering.')
        offset[0] -= rough[1]
        offset[1] -= rough[0]
        print 'Offset: dx = ' + str(offset[0]) + '\t\t dy = ' + str(offset[1])
        return offset

    # --------------------------------------------------------------
    # Method for the science
    # --------------------------------------------------------------        
    def _science(self):
        """
        Method to process the science frames
        """
        # --------------------------------------------------------------        
        # First, identify the proper files
        # --------------------------------------------------------------        
        fits_science = self._find_fits('SCIENCE')
        # --------------------------------------------------------------        
        # Write the SOF file
        # --------------------------------------------------------------        
        f = open(self._dir_sof + '/science.sof', 'w')
        for i in range(len(fits_science)):
            if self.obs_mode == 'POLARIMETRY':
                f.write(self._path_to_fits + '/' + fits_science[i] + '\tIRD_SCIENCE_IMAGING_RAW\n')
            else:
                f.write(self._path_to_fits + '/' + fits_science[i] + '\tIRD_SCIENCE_DBI_RAW\n')
            # f.write(self._path_to_fits + '/' + fits_science[i] + '\tIRD_SCIENCE_DBI_RAW\n')
        f.write(self._dir_cosm + '/master_dark.fits\tIRD_MASTER_DARK\n')
        # f.write(self._dir_cosm + '/instr_flat_badpixels.fits\tIRD_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/static_badpixels.fits\tIRD_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/irdis_flat.fits\tIRD_FLAT_FIELD\n')
        if self._corono:
            f.write(self._dir_cosm + '/starcenter.fits\tIRD_STAR_CENTER\n')
        f.close()
        # --------------------------------------------------------------        
        # Either run the DBI esorex method or the polarimetric one
        # --------------------------------------------------------------        
        if self.obs_mode == 'POLARIMETRY':
            self._esorex_dpi()
        else:
            self._esorex_dbi()
        # self._esorex_dbi()

    # --------------------------------------------------------------
    # Method to plot a summary
    # --------------------------------------------------------------
    def _plot_summary(self):
        """
        Method to plot a summary of the center and science frames
        """
        # --------------------------------------------------------------
        # Get the science fits files
        # --------------------------------------------------------------
        fits_science = self._find_fits('SCIENCE', blacklist = False, verbose = False)
        date_science = [] 
        for i in range(len(fits_science)):
            date_science.append(self._get_obs_time(fits_science[i]))
        date_science = np.array(date_science)
        # --------------------------------------------------------------
        # Get the center fits files
        # --------------------------------------------------------------
        fits_center = self._find_fits('CENTER', blacklist = False, verbose = False)
        date_center = []
        for i in range(len(fits_center)):
            date_center.append(self._get_obs_time(fits_center[i]))
        date_center = np.array(date_center)

        plt.figure()
        plt.plot(date_science, np.zeros(len(fits_science)), 'wo')
        plt.plot(date_center, np.ones(len(fits_center)), 'ro')
        for i in range(len(fits_center)):
            plt.plot([date_center[i], date_center[i]], [0.,1],lw=1,color='r')
        plt.ylim(-0.1, 1.1)

        plt.show()

    # --------------------------------------------------------------
    # Method to get the STOKES keywords from the fits files. Since 
    # this may be done several times, it should limit the reading time.
    # --------------------------------------------------------------
    def _get_stokes(self):
        """
        Read all the fits files in science.sof and extract the stokes keyword
        """
        list_files = glob.glob(self._dir_sci + '/SPHER*_total.fits')
        bl = self._read_blacklist('DPI_blacklist.txt')
        # --------------------------------------------------------------
        # Read the sof file and check the ordering of the fits files.
        # --------------------------------------------------------------
        fitsname, stokes, date = [], [], []
        for i in range(len(list_files)):
            if list_files[i] not in bl:
                fitsname.append(list_files[i])
                hdu = fits.open(list_files[i])
                stokes.append(hdu[0].header['HIERARCH ESO OCS DPI H2RT STOKES'])
                date.append(hdu[0].header['DATE-OBS'])
                hdu.close()
        return fitsname, stokes, date

    # --------------------------------------------------------------
    # Method to clean for the readout noise before averaging the frames.
    # --------------------------------------------------------------
    def _clean_frame(self, frame):
        """
        Will subtract the average of a few pixels at the top and bottom of each column and
        will run a median filter for each column. This is done for all DITs in a given frame 
        and will return the median
        """
        # --------------------------------------------------------------
        # For the selection of the pixels on which the mean readout will be estimated
        # --------------------------------------------------------------
        dx, bx = 100, 30
        xr = np.arange(self._nx)
        sel_readout = ((xr > bx ) & (xr < bx + dx) | (xr < self._nx - bx - 1) & (xr > self._nx - bx - 1 - dx))
        ndit = np.shape(frame)[0]
        cleaned = np.zeros(shape=(ndit, self._nx, self._nx))
        # --------------------------------------------------------------
        # 
        # --------------------------------------------------------------
        for i in range(ndit):
            med_col = np.median(frame[i,sel_readout,:],axis=0)
            for j in range(self._nx):
                cleaned[i,:,j] = frame[i,:,j] - med_col[j]
                if self._km > 0:
                    cleaned[i,:,j] = medfilt(cleaned[i,:,j], self._km)
        return np.median(cleaned, axis = 0)

    # --------------------------------------------------------------
    # Method for the DPI data reduction
    # --------------------------------------------------------------
    def _reduce_dpi(self):
        """
        Method to reduce the DPI data, based on Tomas Stolker's script
        """
        # --------------------------------------------------------------
        # Some parameters for the optimization
        # --------------------------------------------------------------
        # For the instrumental polarization
        instPolMin = 5
        instPolMax =  30
        instPolStep = 0.5
        # For the HWP plate
        thetaMin = -10.0
        thetaMax = 10.0
        thetaStep = 0.01
        # For the plot
        vmin = -0.5
        vmax = 5.5
        # --------------------------------------------------------------
        # Get the Stokes parameters
        # --------------------------------------------------------------
        fitsname, stokes, date = self._get_stokes()
        # --------------------------------------------------------------
        # Ask for confirmation
        # --------------------------------------------------------------
        print 'Will use the following frames:'
        for i in range(len(fitsname)):
            print fitsname[i] + '\t' + stokes[i] + '\t' + date[i].split('T')[1]
        txt = '\n(create a \'DPI_blacklist\' if you want to exclude some frames.)'
        print txt
        print '-'*80
        runit = raw_input('\nProceed [Y]/n: ')
        if runit == 'n':
            sys.exit()
        # --------------------------------------------------------------
        # Get a gaussian kernel if I want to smooth things up
        # --------------------------------------------------------------
        if self._width > 0.:
            Xin, Yin = np.mgrid[0:self._nx, 0:self._nx]
            kernel = self._gaussian((self._nx)/2, (self._nx)/2, self._width , self._width)(Xin, Yin)
        # --------------------------------------------------------------
        # Here we go !
        # --------------------------------------------------------------
        print 'Processing all the science frames ...'
        Qp = np.zeros(shape=(2, self._nx, self._nx))
        Qm = np.zeros(shape=(2, self._nx, self._nx))
        Up = np.zeros(shape=(2, self._nx, self._nx))
        Um = np.zeros(shape=(2, self._nx, self._nx))
        dQp, dQm, dUp, dUm = 0., 0., 0., 0.
        for i in range(len(fitsname)):
            fitsfile = fitsname[i].replace('_total.fits','')
            # --------------------------------------------------------------
            # Read the left frame
            # --------------------------------------------------------------
            hdu = fits.open(fitsfile+'_left.fits')
            stokes = hdu[0].header['HIERARCH ESO OCS DPI H2RT STOKES']
            ordinary = self._clean_frame(hdu[0].data) #np.mean(hdu[0].data, axis=0)
            hdu.close()
            # --------------------------------------------------------------
            # Read the right frame
            # --------------------------------------------------------------
            hdu = fits.open(fitsfile+'_right.fits')
            stokes = hdu[0].header['HIERARCH ESO OCS DPI H2RT STOKES']
            extraord = self._clean_frame(hdu[0].data) #np.mean(hdu[0].data, axis=0)
            hdu.close()
            # --------------------------------------------------------------
            # Remove some NaN
            # --------------------------------------------------------------
            ordinary = np.nan_to_num(ordinary)
            extraord = np.nan_to_num(extraord)
            # --------------------------------------------------------------
            # Fill in the array and update the dit
            # --------------------------------------------------------------
            if stokes == "Qplus":
                Qp[0,] += ordinary
                Qp[1,] += extraord
                dQp += 1.
            elif stokes == "Qminus":
                Qm[0,] += ordinary
                Qm[1,] += extraord
                dQm += 1.
            elif stokes == "Uplus":
                Up[0,] += ordinary
                Up[1,] += extraord
                dUp += 1.
            elif stokes == "Uminus":
                Um[0,] += ordinary
                Um[1,] += extraord
                dUm += 1.
        # --------------------------------------------------------------
        # Normalize the array.
        # --------------------------------------------------------------
        Qp = Qp/dQp
        Qm = Qm/dQm
        Up = Up/dUp
        Um = Um/dUm
        print 'Number of frames used: ' + format(dQp, '0.0f') + ' | ' + format(dQm, '0.0f') + ' | ' + format(dUp, '0.0f') + ' | ' + format(dUm, '0.0f')
        print '-'*80
        # --------------------------------------------------------------
        # Double difference, Canovas et al. (2011)
        # --------------------------------------------------------------
        Q_mean = ( (Qp[0,] - Qp[1,]) - (Qm[0,] - Qm[1,]) ) / 2.
        U_mean = ( (Up[0,] - Up[1,]) - (Um[0,] - Um[1,]) ) / 2.
        Iq_mean = (Qp[0,] + Qp[1,] + Qm[0,] + Qm[1,]) / 2.
        Iu_mean = (Up[0,] + Up[1,] + Um[0,] + Um[1,]) / 2.
        # --------------------------------------------------------------
        # Adapted from Tomas Stolker's script
        # --------------------------------------------------------------
        x = np.linspace(-self._nx/2+0.5, self._nx/2-0.5, self._nx)
        y = np.linspace(-self._nx/2+0.5, self._nx/2-0.5, self._nx)
        xv, yv = np.meshgrid(x,y)
        rr = np.sqrt(xv *xv + yv * yv)
        tt = np.arctan2(yv, xv)
        # --------------------------------------------------------------
        # Instrumental polarization correction
        # --------------------------------------------------------------
        print 'Correction for the instrumental polarization ...'
        rIn = np.arange(instPolMin,instPolMax,instPolStep) # [pixels]
        rOut = np.arange(instPolMin,instPolMax,instPolStep) # [pixels]
        UphiInst = np.zeros(shape=(len(rIn),len(rOut)))
        UphiMin, r1Min, r2Min = 1e100, 0., 0.
        # Loop over a wide range of annulii
        for i in range(len(rIn)):
            for j in range(len(rOut)):
                r1 = rIn[i]
                r2 = rOut[j]
                if r2 > r1:
                    annulus = CircularAnnulus((self._nx/2., self._nx/2.), r1, r2)
                    factor_q = self._annulus_phot(Q_mean, annulus) / self._annulus_phot(Iq_mean, annulus)
                    factor_u = self._annulus_phot(U_mean, annulus) / self._annulus_phot(Iu_mean, annulus)
                    Q_mean_inst = Q_mean - factor_q * Iq_mean
                    U_mean_inst = U_mean - factor_u * Iu_mean

                    Uphi_corrected = Q_mean_inst * np.sin(2.*(tt)) - U_mean_inst * np.cos(2.*(tt))
                    UphiSum = np.nansum(np.abs(Uphi_corrected))
                    UphiInst[i,j] = UphiSum
                    if UphiSum < UphiMin:
                        r1Min = r1
                        r2Min = r2
                        UphiMin = UphiSum
        annulus = CircularAnnulus((self._nx/2., self._nx/2.), r1Min, r2Min)
        factor_q = self._annulus_phot(Q_mean, annulus) / self._annulus_phot(Iq_mean, annulus)
        factor_u = self._annulus_phot(U_mean, annulus) / self._annulus_phot(Iu_mean, annulus)
        Q_mean -= factor_q * Iq_mean
        U_mean -= factor_u * Iu_mean
        print 'Annulus rIn-rOut [pixels] = '+str(r1Min)+'-'+str(r2Min)
        print '-'*80
        # --------------------------------------------------------------
        # Uphi minimization for the HWP offset
        # --------------------------------------------------------------
        print 'Correction for the HWP offset ...'
        offset = np.arange(thetaMin,thetaMax,thetaStep)
        UphiMin = 1e100
        UphiThetaMin = 0.
        for i in range(len(offset)):
            theta = offset[i] * np.pi/180.
            Uphi_corrected = Q_mean * np.sin(2.*(tt + theta)) - U_mean * np.cos(2.*(tt + theta))
            UphiSum = np.nansum(np.abs(Uphi_corrected))
            if UphiSum < UphiMin:
                UphiThetaMin = theta
                UphiMin = UphiSum
        print 'Theta offset [deg] = ' + str(UphiThetaMin * 180. / np.pi)
        print '-'*80
        # --------------------------------------------------------------
        # Qphi and Uphi calculations
        # --------------------------------------------------------------
        self.Qphi =  Q_mean * np.cos(2.*(tt + UphiThetaMin)) + U_mean * np.sin(2.*(tt + UphiThetaMin))
        self.Uphi = -Q_mean * np.sin(2.*(tt + UphiThetaMin)) + U_mean * np.cos(2.*(tt + UphiThetaMin))
        # --------------------------------------------------------------
        # Convolve with a kernal if needed
        # --------------------------------------------------------------
        if self._width > 0.:
            self.Qphi = fftconvolve(self.Qphi, kernel, mode='same')
            self.Uphi = fftconvolve(self.Uphi, kernel, mode='same')
            Q_mean = fftconvolve(Q_mean, kernel, mode='same')
            U_mean = fftconvolve(U_mean, kernel, mode='same')
        # --------------------------------------------------------------
        # Plot all the things !!!
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(12,12))
        plt.subplots_adjust(hspace=0.1,wspace=0.1,top=0.95,left=0.1,right=0.95)
        ax1 = plt.subplot2grid((2,2), (0, 0))
        cb = ax1.imshow(self.Qphi, origin = 'lower', cmap = 'afmhot', vmin = vmin, vmax = vmax)
        ax1.set_ylabel('$\delta$ [$\,^{\prime\prime}$]')
        ax1.set_title('$Q_\phi$')

        ax2 = plt.subplot2grid((2,2), (0, 1))
        cb = ax2.imshow(self.Uphi, origin = 'lower', cmap = 'afmhot', vmin = vmin, vmax = vmax)
        ax2.set_title('$U_\phi$')

        ax3 = plt.subplot2grid((2,2), (1, 0))
        cb = ax3.imshow(Q_mean, origin = 'lower', cmap = 'afmhot', vmin = vmin, vmax = vmax)
        ax3.set_xlabel('$\\alpha$ [$\,^{\prime\prime}$]')
        ax3.set_ylabel('$\delta$ [$\,^{\prime\prime}$]')
        ax3.set_title('$Q$')

        ax4 = plt.subplot2grid((2,2), (1, 1))
        cb = ax4.imshow(U_mean, origin = 'lower', cmap = 'afmhot', vmin = vmin, vmax = vmax)
        ax4.set_xlabel('$\\alpha$ [$\,^{\prime\prime}$]')
        ax4.set_title('$U$')
        plt.show()
        # --------------------------------------------------------------
        # Save fits files
        # --------------------------------------------------------------
        print 'Writing fits files \'Qphi.fits\' \'Uphi.fits\' ...'
        fits.writeto('Qphi.fits', self.Qphi, clobber = True)
        fits.writeto('Uphi.fits', self.Uphi, clobber = True)
        print '-'*80

    # --------------------------------------------------------------        
    # Method for the photometry in an annulus
    # --------------------------------------------------------------        
    def _annulus_phot(self, data, ring):
        """
        Simple method to derive the photometry in a given annulus
        """
        phot_table = aperture_photometry(data, ring, method='exact')
        return phot_table['aperture_sum']


    # --------------------------------------------------------------        
    # Method for the DPI data reduction
    # --------------------------------------------------------------        
    def _esorex_dpi(self):
        """
        Method to run the esorex recipe on the DPI observations
        (using the science_imaging recipe.)
        """
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ird_science_imaging', 
                        self._dir_sof + '/science.sof']
            sci_dbi = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            # --------------------------------------------------------------        
            # Clean up the fits files
            # --------------------------------------------------------------        
            mvfiles = glob.glob('SPHER*.fits')
            for i in range(len(mvfiles)):
                args = ['mv', mvfiles[i], self._dir_sci + '/']
                mvfits = subprocess.Popen(args).wait()
            mvfiles = glob.glob('SPHER*.txt')
            for i in range(len(mvfiles)):
                args = ['rm', mvfiles[i]]
                rmtxt = subprocess.Popen(args).wait()
            print '-'*80
        else:
            sys.exit()
    # --------------------------------------------------------------        
    # Method for the DBI data reduction
    # --------------------------------------------------------------        
    def _esorex_dbi(self):
        """
        Method to run the esorex recipe on the DBI observations
        """
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ird_science_dbi', 
                        '--ird.science_dbi.use_adi=FALSE', 
                        '--ird.science_dbi.use_sdi=FALSE',
                        self._dir_sof + '/science.sof']
            # sci_dbi = subprocess.Popen(args).wait()
            sci_dbi = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            # --------------------------------------------------------------        
            # Clean up the fits files
            # --------------------------------------------------------------        
            mvfiles = glob.glob('SPHER*.fits')
            for i in range(len(mvfiles)):
                args = ['mv', mvfiles[i], self._dir_sci + '/']
                mvfits = subprocess.Popen(args).wait()
            mvfiles = glob.glob('SPHER*.txt')
            for i in range(len(mvfiles)):
                args = ['rm', mvfiles[i]]
                rmtxt = subprocess.Popen(args).wait()
            print '-'*80
        else:
            sys.exit()


    # --------------------------------------------------------------
    # Method to check for blacklisted fits files
    # --------------------------------------------------------------
    def _read_blacklist(self, filename):
        """
        Method to check if a file exists and read its contents
        """
        if type(filename) is not str: self._error_msg('The filename should be a string')
        bl_list = []
        if os.path.isfile(filename):
            f = open(filename, 'r')
            lines = f.readlines()
            f.close()
            nl = len(lines)
            for i in range(nl):
                bl_list.append(lines[i].split()[0])
            return bl_list
        else:
            return bl_list


    # --------------------------------------------------------------
    # Method to check if there are the necessary producs
    # --------------------------------------------------------------
    def _check_prod(self):
        """
        Check if the different recipes were run beforehand
        """
        if os.path.isfile(self._dir_cosm + '/master_dark.fits'):
            print 'Found a DARK in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_dark = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/irdis_flat.fits'):
            print 'Found a FLAT in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/starcenter.fits'):
            print 'Found a STAR CENTER in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_center = True
            print '-'*80
        if (os.path.isfile(self._dir_cosm + '/psf_left.fits') & os.path.isfile(self._dir_cosm + '/psf_right.fits')):
            print 'Found a STELLAR PSF in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flux = True
            print '-'*80
        if ((self.obs_mode == 'IMAGE,DUAL') | (self.obs_mode == 'IMAGE,CLASSICAL')):
            if os.path.isfile('science_dbi.fits'):
                print 'Found a science frame in this directory. Erase it if you want to recalculate it.'
                self._is_science = True
                print '-'*80
        elif self.obs_mode == 'POLARIMETRY':
            if os.path.isfile('science_imaging.fits'):
                print 'Found a science frame in \'' + self._dir_sci + '/\'. Erase it if you want to recalculate it.'
                self._is_science = True
                print '-'*80
            if os.path.isfile('science_dbi.fits'):
                print 'Found a science frame in this directory. Erase it if you want to recalculate it.'
                self._is_science = True
                print '-'*80

    # --------------------------------------------------------------
    # Method to make a summary of what can be done.
    # --------------------------------------------------------------
    def _find_fits(self, action, blacklist = True, verbose = True):
        """
        Method to identify what can be done (dark, flat, distortion, star center)
        """
        # --------------------------------------------------------------
        # For the DARK measurements
        # --------------------------------------------------------------
        if action == 'DARK':
            # catg, arm, tp, dit, bl_txt = 'CALIB', 'IRDIS', 'DARK,BACKGROUND', True, 'dark_blacklist'
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IRDIS', 'DARK', True, 'dark_blacklist'
        # --------------------------------------------------------------
        # For the SKY measurements
        # --------------------------------------------------------------
        elif action == 'SKY':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IRDIS', 'SKY', True, 'sky_blacklist'
        # --------------------------------------------------------------
        # For the BACKGROUND measurements. With restrictions on the DIT for the flux calibration, but with restrictions for the flat-field
        # --------------------------------------------------------------
        elif action == 'BACKGROUND':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IRDIS', 'DARK,BACKGROUND', False, 'bkg_blacklist'
        # --------------------------------------------------------------
        # For the FLAT measurements
        # --------------------------------------------------------------
        elif action == 'FLAT':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IRDIS', 'FLAT,LAMP', False, 'flat_blacklist'
        # --------------------------------------------------------------
        # For the CENTER measurements
        # --------------------------------------------------------------
        elif action == 'CENTER':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IRDIS', 'OBJECT,CENTER', True, 'center_blacklist'
        # --------------------------------------------------------------
        # For the FLUX measurements
        # --------------------------------------------------------------
        elif action == 'FLUX':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IRDIS', 'OBJECT,FLUX', False, 'flux_blacklist'
        # --------------------------------------------------------------
        # For the SCIENCE measurements
        # --------------------------------------------------------------
        elif action == 'SCIENCE':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IRDIS', 'OBJECT', True, 'science_blacklist'
        # --------------------------------------------------------------
        # Make the selection. For the FLAT and DISTORTION, the DIT is irrelevant. For the DARK, the filter seems to be irrelevant also.
        # --------------------------------------------------------------
        if action == 'DARK':
            sel = np.where((self._catg == catg) & (self._arm == arm) & (self._type == tp) & (self._dit == self.obs_dit))[0]
        else:
            if dit:
                sel = np.where((self._catg == catg) & (self._filter == self.obs_filter) & (self._arm == arm) & (self._type == tp) & (self._dit == self.obs_dit))[0]
            else:
                sel = np.where((self._catg == catg) & (self._filter == self.obs_filter) & (self._arm == arm) & (self._type == tp))[0]
        # --------------------------------------------------------------
        # Check if there are any measurements available and be verbose about it
        # --------------------------------------------------------------
        if len(sel) == 0:
            sel = np.where((self._catg == catg) & (self._arm == arm) & (self._type == tp))[0]
            txt = ''
            for i in range(len(sel)):
                txt += '\n' + self._fitsname[sel[i]] + '\t' + str(self._catg[sel[i]]) + '\t' + str(self._type[sel[i]]) + '\t' + str(self._tech[sel[i]]) + '\t' + str(self._mjd[sel[i]]) + '\t' + str(self._dit[sel[i]]) + '\t' + str(self._filter[sel[i]])
            self._error_msg('Could not find proper ' + str(action) + ' measurements:'+txt)
        # --------------------------------------------------------------
        # Find the frames that were taken closest to the science observations
        # --------------------------------------------------------------
        mjd_diff = np.abs(self._mjd[sel] - self.obs_mjd)
        sel = sel[np.where(mjd_diff == np.min(mjd_diff))[0]]
        # --------------------------------------------------------------
        # Check against any blacklsit.
        # --------------------------------------------------------------
        if blacklist:
            bl = self._read_blacklist(bl_txt + '.txt')
            if len(bl) >0:
                print 'The following frames for the ' + action +' have been blacklisted:\n'
                for i in range(len(sel)):
                    if self._fitsname[sel[i]] in bl:
                        print self._fitsname[sel[i]] + '\t' + str(self._dit[sel[i]]) + '\t' + str(np.abs(self._mjd[sel[i]] - self.obs_mjd)) + '\t' + str(self._tech[sel[i]])
                print ' '
        else:
            bl = []
        # --------------------------------------------------------------
        # Select the proper files that will be used.
        # --------------------------------------------------------------
        list_selected = []
        txt = 'Will use the following frames for the ' + action +':\n'
        for i in range(len(sel)):
            if self._fitsname[sel[i]] not in bl:
                txt += self._fitsname[sel[i]] + '\t' + str(self._dit[sel[i]]) + '\t' + str(np.abs(self._mjd[sel[i]] - self.obs_mjd)) + '\t' + str(self._filter[sel[i]]) + '\t' + str(self._tech[sel[i]]) + '\n'
                # txt += self._fitsname[sel[i]] + '\t' + str(self._dit[sel[i]]) + '\t' + str(np.abs(self._mjd[sel[i]] - self.obs_mjd)) + '\t' + str(self._tech[sel[i]]) + '\t' + str(self._filter[sel[i]]) + '\n'
                list_selected.append(self._fitsname[sel[i]])
        txt += '\n(create a \'' + bl_txt + '.txt\' if you want to exclude some frames.)'
        if verbose:
            print txt
            print '-'*80
        return list_selected

    # --------------------------------------------------------------        
    # Method to get the mode of the observations. If there are several, will ask you to pick the one you want
    # --------------------------------------------------------------
    def _get_mode(self):
        """
        Method to get the observing mode you are interested in. Either DUAL, CLASSICAL, or POLARIMETRY.
        """
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT'))[0]
        # --------------------------------------------------------------
        # Check which modes are available
        # --------------------------------------------------------------
        dual = False
        dpi = False
        classical = False
        mode_obs = np.unique(self._tech[sel])
        nb_mode = 0
        id_mode = 0
        final_mode = None
        if (('IMAGE,DUAL' in mode_obs) | ('IMAGE,DUAL,CORONOGRAPHY' in mode_obs)):
            nb_mode += 1
            dual = True
            id_mode += 1
        if (('POLARIMETRY' in mode_obs) | ('POLARIMETRY,CORONOGRAPHY' in mode_obs)):
            nb_mode += 1
            dpi = True
            id_mode += 2
        if (('IMAGE,CLASSICAL' in mode_obs) | ('IMAGE,CLASSICAL,CORONOGRAPHY' in mode_obs)):
            nb_mode += 1
            classical = True
            id_mode += 4
        # --------------------------------------------------------------
        # Print some info and pick the mode you need
        # --------------------------------------------------------------
        if id_mode == 1:
            final_mode = 'IMAGE,DUAL'
        elif id_mode == 2:
            final_mode = 'POLARIMETRY'
        elif id_mode == 4:
            final_mode = 'IMAGE,CLASSICAL'
        elif id_mode == 3:
            print '-'*80
            print 'Found dual-band and polarimetric observations. '
            select = raw_input('Which one do you want to use?\n[0] DUAL BAND\n[1] POLATIMETRIC')
            idm = np.int(select)
            if ((idm != 0 ) and (idm != 1)):
                self._error_msg('The answer should be 0 or 1.')
            if idm == 0:
                final_mode = 'IMAGE,DUAL'
            if idm == 1:
                final_mode = 'POLARIMETRY'
        elif id_mode == 5:
            print '-'*80
            print 'Found classical and dual-band observations. '
            select = raw_input('Which one do you want to use?\n[0] CLASSICAL\n[1] DUAL BAND')
            idm = np.int(select)
            if ((idm != 0 ) and (idm != 1)):
                self._error_msg('The answer should be 0 or 1.')
            if idm == 0:
                final_mode = 'IMAGE,CLASSICAL'
            if idm == 1:
                final_mode = 'IMAGE,DUAL'
        elif id_mode == 6:
            print '-'*80
            print 'Found classical and polarimetric observations. '
            select = raw_input('Which one do you want to use?\n[0] CLASSICAL\n[1] POLATIMETRIC')
            idm = np.int(select)
            if ((idm != 0 ) and (idm != 1)):
                self._error_msg('The answer should be 0 or 1.')
            if idm == 0:
                final_mode = 'IMAGE,CLASSICAL'
            if idm == 1:
                final_mode = 'POLARIMETRY'
        elif id_mode == 6:
            print '-'*80
            print 'Found dual-band, classical, and polarimetric observations. '
            select = raw_input('Which one do you want to use?\n[0] DUAL BAND \n[1] CLASSICAL\n[2] POLATIMETRIC')
            idm = np.int(select)
            if ((idm != 0 ) and (idm != 1) and (idm != 2)):
                self._error_msg('The answer should be 0, 1, or 2.')
            if idm == 0:
                final_mode = 'IMAGE,DUAL'
            if idm == 1:
                final_mode = 'IMAGE,CLASSICAL'
            if idm == 2:
                final_mode = 'POLARIMETRY'
        # print '-'*80
        return final_mode
    # --------------------------------------------------------------        
    # Method to get the MJD of the observations. If there are several, will ask you to pick the one you want
    # --------------------------------------------------------------
    def _get_mjd(self):
        """
        Method to get the MJD of the observations. If multiple DITs are available, will ask for the one you want.
        """
        final_mjd = None
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
        mjd_obs = np.unique(self._mjd[sel]) # the MJD when there were observations
        # --------------------------------------------------------------
        # First case, there were observations only on the same MJD
        # --------------------------------------------------------------
        if len(mjd_obs) == 1:
            # print 'Found ' + str(len(sel)) + ' observations on MJD ' + str(mjd_obs[0]) + ' ['+self.obs_mode+']'
            final_mjd = mjd_obs[0]
        else:
            txt = 'Which MJD do you want to choose:\n'
            for i in range(len(mjd_obs)):
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._mjd == mjd_obs[i]) & (self._arm == 'IRDIS') & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
                txt += '[' + str(i) + '] MJD ' + str(mjd_obs[i]) + ' (' + str(len(sel)) + ' frames)' + '\n'

            select = raw_input(txt)
            idm = np.int(select)
            if ((idm < 0 ) or (idm > len(mjd_obs))):
                self._error_msg('Select the proper number (' + str(idm) + ')')
            else:
                final_mjd = mjd_obs[idm]
        # print '-'*80
        return final_mjd
    # --------------------------------------------------------------        
    # Method to get the DIT of the observations. If there are several, will ask you to pick the one you want
    # --------------------------------------------------------------
    def _get_dit(self):
        """
        Method to get the MJD of the observations. If multiple DITs are available, will ask for the one you want.
        """
        final_dit = None
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & (self._mjd == self.obs_mjd) & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
        dit_obs = np.unique(self._dit[sel]) # the DIT of observations
        # --------------------------------------------------------------
        # First case, there were observations only on the same MJD
        # --------------------------------------------------------------
        if len(dit_obs) == 1:
            # print 'Found ' + str(len(sel)) + ' observations with a DIT of ' + str(dit_obs[0]) + ' s'
            final_dit = dit_obs[0]
        else:
            txt = 'Which one do you want to choose:\n'
            for i in range(len(dit_obs)):
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & (self._dit == dit_obs[i]) & (self._mjd == self.obs_mjd) & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
                txt += '[' + str(i) + '] DIT = ' + str(dit_obs[i]) + ' s (' + str(len(sel)) +  ' frames)\n'

            select = raw_input(txt)
            idm = np.int(select)
            if ((idm < 0 ) or (idm > len(dit_obs))):
                self._error_msg('Select the proper number (' + str(idm) + ')')
            else:
                final_dit = dit_obs[idm]
        # print '-'*80
        return final_dit
    # --------------------------------------------------------------        
    # Method to get the observing filter
    # --------------------------------------------------------------
    def _get_filter(self):
        """
        Method to get the filter of the observations
        """
        final_filter = ''
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & (self._mjd == self.obs_mjd) & (self._dit == self.obs_dit) & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
        filter_obs = np.unique(self._filter[sel]) # the filter of the observations
        # --------------------------------------------------------------
        # First case, there were observations in the same filter
        # --------------------------------------------------------------
        if len(filter_obs) == 1:
            # print 'Found ' + str(len(sel)) + ' observations with a DIT of ' + str(filter_obs[0]) + ' s'
            final_filter = filter_obs[0]
        else:
            txt = 'Which one do you want to choose:\n'
            for i in range(len(filter_obs)):
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IRDIS') & (self._dit == self.obs_dit) & (self._filter == filter_obs[i]) & (self._mjd == self.obs_mjd) & ((self._tech == self.obs_mode) | (self._tech == self.obs_mode + ',CORONOGRAPHY')))[0]
                txt += '[' + str(i) + '] filter = ' + str(filter_obs[i]) + ' (' + str(len(sel)) +  ' frames)\n'

            select = raw_input(txt)
            idm = np.int(select)
            if ((idm < 0 ) or (idm > len(filter_obs))):
                self._error_msg('Select the proper number (' + str(idm) + ')')
            else:
                final_filter = filter_obs[idm]
        # print '-'*80
        return final_filter

    # --------------------------------------------------------------        
    # Method to shift individual frame
    # --------------------------------------------------------------        
    def _shift_array(self, frame):
        """
        Method to find the bright spot and shift it to the center
        """
        thrs = 5.
        sources = daofind(frame, fwhm = 4.0, threshold = thrs * np.std(frame))
        bright = np.argsort(sources['peak'])[-1:]
        new_frame = scipy.ndimage.interpolation.shift(frame, [512-sources['ycentroid'][bright[0]],512-sources['xcentroid'][bright[0]]])
        return new_frame        

    # --------------------------------------------------------------        
    # Method to do "manual" recentering
    # --------------------------------------------------------------        
    def _recenter_dao(self, cube):
        """
        Use daofind to locate the source and recenter it (only use for non coronagraphic observations)
        """
        if len(np.shape(cube)) == 3:
            nf = np.shape(cube)[0]
            for i in range(nf):
                if i == 0:
                    frame = self._shift_array(cube[i,])
                else:
                    frame += self._shift_array(cube[i,])
            frame = frame / np.float(nf)
        else:
            frame = self._shift_array(cube)
        return frame



    # --------------------------------------------------------------        
    # Method to read the science fits files
    # --------------------------------------------------------------        
    def _read_fits(self, filename, peak_left, peak_right):
        filename = filename.replace('_total.fits', '')
        # --------------------------------------------------------------        
        # Read the right image
        # --------------------------------------------------------------        
        hdu = fits.open(filename + '_right.fits')
        p_r = 0.5 * (hdu[0].header['HIERARCH ESO TEL PARANG END'] + hdu[0].header['HIERARCH ESO TEL PARANG START'])
        if self._corono:
            data_right = np.mean(hdu[0].data, axis = 0) / peak_right
        else:
            data_right = self._recenter_dao(hdu[0].data)/ peak_right
        hdu.close()
        # --------------------------------------------------------------        
        # Read the left image
        # --------------------------------------------------------------        
        hdu = fits.open(filename + '_left.fits')
        p_l = 0.5 * (hdu[0].header['HIERARCH ESO TEL PARANG END'] + hdu[0].header['HIERARCH ESO TEL PARANG START'])
        if self._corono:
            data_left = np.mean(hdu[0].data, axis = 0) / peak_left
        else:
            data_left = self._recenter_dao(hdu[0].data)/ peak_left
        hdu.close()
        # --------------------------------------------------------------        
        # The total image is the mean of left and right
        # --------------------------------------------------------------        
        data_total = (data_right + data_left)/2.
        # --------------------------------------------------------------        
        # Check if there is any difference between the sizes of the left and right images
        # --------------------------------------------------------------        
        if ((np.shape(data_right)[1] != np.shape(data_left)[1]) & (np.shape(data_right)[0] != np.shape(data_left)[0])):
            print 'Different sizes between the left and right sides'
            return 0,0
        # --------------------------------------------------------------        
        # Check if the parallactic angles are the same or not
        # --------------------------------------------------------------        
        if p_r != p_l:
            print 'problem with the angle'
            return 0,0
        return data_right, data_left, data_total, p_r

    # --------------------------------------------------------------        
    # Method to merge the fits files
    # --------------------------------------------------------------        
    def _merge(self):
        """
        Method to merge fitsfile
        """
        peak_left, peak_right = 1., 1.
        if ((os.path.isfile(self._dir_cosm + '/psf_left.fits')) & (os.path.isfile(self._dir_cosm + '/psf_right.fits'))):
            print 'Found psf, will normalize the individual frames.'
            hdu = fits.open(self._dir_cosm + '/psf_left.fits')
            peak_left = hdu[0].header['Peak']
            hdu.close()
            hdu = fits.open(self._dir_cosm + '/psf_right.fits')
            peak_right = hdu[0].header['Peak']
            hdu.close()

        if not os.path.isfile(self._starname + '_total.fits'):
            print 'Merging all the frames together.'
            print '-'*80
            list_fits = glob.glob(self._dir_sci + '/*_total.fits')
            nl = len(list_fits)
            newx = 1024
            cube_right = np.zeros(shape=(nl,newx,newx))
            cube_left = np.zeros(shape=(nl,newx,newx))
            cube_total = np.zeros(shape=(nl,newx,newx))
            para = np.zeros(nl)
            for i in range(nl):
                cube_right[i,], cube_left[i,], cube_total[i,], para[i] = self._read_fits(list_fits[i], peak_left, peak_right)
            # --------------------------------------------------------------        
            # Save the fits 
            # --------------------------------------------------------------        
            fits.writeto(self._starname + '_right.fits', cube_right, clobber=True)
            fits.writeto(self._starname + '_left.fits', cube_left, clobber=True)
            fits.writeto(self._starname + '_total.fits', cube_total, clobber=True)
            fits.writeto('para.fits', para, clobber=True)

    # --------------------------------------------------------------        
    # Check if the star has some observations in the directory containing the fits files
    # --------------------------------------------------------------        
    def _check_star(self):
        """
        Method to check is the star has some observations
        """
        # --------------------------------------------------------------
        # First check if the directory is there
        # --------------------------------------------------------------        
        if not os.path.isdir(self._path_to_fits):
            self._error_msg('The directory does not seem to exists.')
        # --------------------------------------------------------------        
        # First check if there is a csv file listing all the fits files.
        # If not, create it
        # --------------------------------------------------------------        
        if not os.path.isfile(self._path_to_fits + '/DRS_list.csv'):
            self._list_fits()
        # --------------------------------------------------------------        
        # Read the list of files
        # --------------------------------------------------------------        
        data = ascii.read(self._path_to_fits + '/DRS_list.csv', delimiter = ';', data_start = 1)
        self._fitsname = data['Name']
        self._obj = data['OBJECT']
        self._arm = data['SEQ.ARM']
        self._catg = data['DPR.CATG']
        self._type = data['DPR.TYPE']
        self._tech = data['DPR.TECH']
        self._dit = data['SEQ1.DIT']
        self._filter = data['FILTER']
        self._nd = data['ND']
        mjd = np.array(data['MJD.OBS'])
        mjd -= 0.5
        self._mjd = mjd.astype(int)
        self._csv_red = True
        # --------------------------------------------------------------        
        # Check if the star has some observations (at least one fits files with the name of the star.)
        # --------------------------------------------------------------        
        sel_star = np.where(self._obj == self._starname)[0]
        if len(sel_star) == 0:
            msg = 'Could no find SCIENCE observations for ' + self._starname + ' in this directory. \nThe following stars have SCIENCE frames:\n'            
            sel_sci = np.where(self._catg == 'SCIENCE')[0]
            uni_stars = np.unique(self._obj[sel_sci])
            for i in range(len(uni_stars)):
                msg += '- ' + uni_stars[i] + '\n'
            self._error_msg(msg)

    def _error_msg(self, message):
        """
        A method to print some error message (a string) and stop the program.
        """
        print message
        print '-'*80
        print 'Stopping.'
        print '-'*80
        sys.exit()
    # --------------------------------------------------------------        
    # Extract relevant informations from the header of the fits files
    # --------------------------------------------------------------        
    def _list_fits(self):
        """
        Method to list the fits file and save some parameters in the same directory
        """
        list_files = glob.glob(self._path_to_fits + '/*.fits')
        nf = len(list_files)
        f = open(self._path_to_fits + '/DRS_list.csv', 'w')
        f.write('Name;OBJECT;SEQ.ARM;DPR.CATG;DPR.TYPE;DPR.TECH;SEQ1.DIT;MJD.OBS;FILTER;ND\n')
        for i in range(nf):
            hdu = fits.open(list_files[i])
            header = hdu[0].header
            hdu.close()
            # if 'NAXIS1' in header.keys():
            try:
                if 'ESO SEQ ARM' in header.keys():
                    txt = list_files[i].replace(self._path_to_fits, '') + ';' + header['OBJECT'] + ';' + header['ESO SEQ ARM'] + ';' + header['ESO DPR CATG'] + ';'
                    txt += header['ESO DPR TYPE'] + ';' + header['ESO DPR TECH'] + ';' 
                    txt += str(header['ESO DET SEQ1 DIT']) + ';' + str(header['MJD-OBS']) + ';' + header['HIERARCH ESO INS COMB IFLT'] + ';' + header['HIERARCH ESO INS4 FILT2 NAME'] + '\n'
                    f.write(txt)
            except:
                print list_files[i]
        f.close()
        # print redok, notred
    # --------------------------------------------------------------        
    # 2D gaussian for the convolution
    # --------------------------------------------------------------        
    def _gaussian(self, center_x, center_y, width_x, width_y):
        """
        Returns a gaussian function with the given parameters. That should be normalized
        """
        width_x = float(width_x)
        width_y = float(width_y)
        amp = 1. / (2.e0 * np.pi * width_x * width_y)
        return lambda x,y: amp*np.exp(-(((center_x-x)/width_x)**2. + ((center_y-y)/width_y)**2.)/2.)



