import sys
import glob
import scipy
import os.path
import datetime
import subprocess
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from .extras import ifs_flat_manual, ifs_preprocess, ifs_misc
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, daofind
from scipy.signal import fftconvolve, medfilt
try:
    import vip
    is_vip = True
except ImportError:
    is_vip = False
# -----------------------------------------------------------------------------
class IFS(object):
    """
    Class for the SPHERE/IFS data reduction. The module should be imported as follows:

    > from pydrs import IFS

    and can be called as

    > data_red = IFS('Starname', 'path to data')

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


    THINGS TO BE DONE:
        + check the SKY measurments
        + add the pre-processing of the SCIENCE data
        + add the pre-processing of the wave calibration files.
        + check for observations with YJH

    """
    def __init__(self, starname, path_to_fits, dir_pre_proc = 'preprocess', dir_cosmetics = 'cosmetics', 
                 dir_science = 'science', dir_ready = 'sci_ready', dir_sof = 'sof', summary = False):
        # --------------------------------------------------------------        
        # Check the inputs
        # --------------------------------------------------------------        
        assert type(starname) is str, 'The name of the star should be a string'
        assert type(summary) is bool, 'The variable \'summary\' should be True or False'
        assert type(dir_cosmetics) is str, 'The name of the cosmetics directory should be a string'
        assert type(dir_ready) is str, 'The name of the \"science ready\" directory should be a string'
        assert type(dir_science) is str, 'The name of the science directory should be a string'
        assert type(dir_pre_proc) is str, 'The name of the preprocess directory should be a string'
        assert type(dir_sof) is str, 'The name of the sof directory should be a string'
        assert type(path_to_fits) is str, 'The name of the directory containing the fits file should be a string'
        if not os.path.isdir(path_to_fits):
            self._error_msg('The directory containing the fits files cannot be found.')
        # --------------------------------------------------------------        
        # Define some variables
        # --------------------------------------------------------------        
        self._nx = 1024
        self._starname = starname
        self._path_to_fits = path_to_fits
        self._csv_red = False
        self._is_dark_sky = False
        self._is_dark_psf = False
        self._is_dark_cen = False
        self._is_dark_corono = False
        self._is_flat_white = False
        self._is_flat_1020 = False
        self._is_flat_1230 = False
        self._is_flat_1300 = False
        self._is_flat_1550 = False
        self._is_wave_cal = False
        self._is_specpos = False
        self._is_ifuflat = False
        self._is_science = False
        self._is_wave_preproc = False
        self._is_clean = False

        self._is_dark = False
        self._is_flat = False
        self._is_center = False
        self._is_flux = False
        # --------------------------------------------------------------
        # Check for existing directories, if not there try to create them
        # --------------------------------------------------------------
        if not os.path.isdir(dir_ready):
            try:
                os.mkdir(dir_ready)
            except:
                self._error_msg("Cannot create the directory \"" + dir_ready +"\"")
        if not os.path.isdir(dir_pre_proc):
            try:
                os.mkdir(dir_pre_proc)
            except:
                self._error_msg("Cannot create the directory \"" + dir_pre_proc +"\"")
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
        self._dir_pre_proc = dir_pre_proc
        self._dir_cosm = dir_cosmetics
        self._dir_sci = dir_science
        self._dir_ready = dir_ready
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
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & 
            (self._mjd == self.obs_mjd) & (self._dit == self.obs_dit) & (self._filter == self.obs_filter) & 
            (self._tech == self.obs_mode))[0]
        nframes = len(sel)
        txt = '\n' + '-' * 80 + '\n' + self._starname + ' - ' + self.obs_mode + '\nMJD ' + str(self.obs_mjd) + ': ' + str(nframes) + ' frames\n'
        txt += 'DIT: ' + str(self.obs_dit) + ' sec\nFilter: ' + self.obs_filter
        print txt
        print '-'*80
        # --------------------------------------------------------------
        # Plot a summary of the center and science observations
        # --------------------------------------------------------------
        # if summary:
        #     self._plot_summary()
        # --------------------------------------------------------------
        # Check which recipes have been run already
        # --------------------------------------------------------------
        self._check_prod()
        # --------------------------------------------------------------
        # Run the cascade
        # --------------------------------------------------------------
        if not self._is_dark_sky: self._dark_sky()
        if not self._is_dark_psf: self._dark_psf()
        if not self._is_dark_cen: self._dark_center()
        if not self._is_dark_corono: self._dark_corono()
        if not self._is_flat_white: self._flat('white')
        if not self._is_flat_1020: self._flat('1020')
        if not self._is_flat_1230: self._flat('1230')
        if not self._is_flat_1300: self._flat('1300')
        if self.obs_filter != 'OBS_YJ':
            print ' '
            print ' '
            print 'To be checked more thorougly'
            print ' '
            print ' '
            if not self._is_flat_1550: self._flat('1550')
        if not self._is_wave_cal: self._specpos()
        if not self._is_specpos: self._wave_cal()
        if not self._is_ifuflat: self._ifu_flat()
        if not self._is_wave_preproc: self._wave_collapse()
        if not self._is_science: self._sci()
        if not os.path.isfile(self._dir_sci + '/center_pos.csv'): self._get_center()
        self._center_science()
        # if not self._is_clean: self._clean_sci()


    # --------------------------------------------------------------
    # Method to integrate something
    # --------------------------------------------------------------
    def _integrate(self,x,fx):
        return np.sum( (x[:-1] - x[1:]) * (fx[:-1] + fx[1:]) * 0.5)

    # --------------------------------------------------------------
    # Method for the dark sky
    # --------------------------------------------------------------
    def _dark_sky(self):
        """
        Method to create the sky dark frame
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_dark = self._find_fits('DARK', force_dit = 1.650726)
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/dark_sky.sof', 'w')
        for i in range(len(fits_dark)):
            f.write(self._path_to_fits + '/' + fits_dark[i] + '\tIFS_DARK_RAW\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ifs_master_dark',
                    '--ifs.master_dark.coll_alg=2',
                    '--ifs.master_dark.sigma_clip=5.0',
                    '--ifs.master_dark.smoothing=5',
                    '--ifs.master_dark.min_acceptable=0.0',
                    '--ifs.master_dark.max_acceptable=2000.0',
                    '--ifs.master_dark.outfilename=' + self._dir_cosm + '/dark_cal.fits', 
                    '--ifs.master_dark.badpixfilename=' + self._dir_cosm + '/dark_bpm_cal.fits',
                    self._dir_sof + '/dark_sky.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/dark_cal.fits'):
                self._error_msg('It seems a dark_cal.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------
    # Method for the master dark
    # --------------------------------------------------------------
    def _dark_psf(self):
        """
        Method to create the master dark frame
        """
        # --------------------------------------------------------------
        # First, identify the DIT of the FLUX frames
        # --------------------------------------------------------------
        fits_flux = self._find_fits('FLUX', verbose = False)
        for i in range(len(fits_flux)):
            idfit = np.where(self._fitsname == fits_flux[i])[0][0]
            if i == 0:
                f_dit = self._dit[idfit]
            else:
                if self._dit[idfit] != f_dit: self._error_msg('Different DIT in the flux measurements')
        # --------------------------------------------------------------
        # Find the DARK measurements with the proper DIT f_dit
        # --------------------------------------------------------------
        fits_dark_tmp = self._find_fits('DARK', verbose = True, force_dit = f_dit)
        fits_dark = [] # list of proper fits file with the correct DIT compared to the FLUX
        for i in range(len(fits_dark_tmp)):
            idfit = np.where(self._fitsname == fits_dark_tmp[i])[0][0]
            if (self._dit[idfit] == f_dit):
                fits_dark.append(fits_dark_tmp[i])
        if len(fits_dark) == 0:
            self._error_msg('Could not find a DARK measurment with a DIT of:' + format(f_dit, '0.1f'))
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/master_dark.sof', 'w')
        for i in range(len(fits_dark)):
            f.write(self._path_to_fits + '/' + fits_dark[i] + '\tIFS_DARK_RAW\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ifs_master_dark', 
                    '--ifs.master_dark.coll_alg=2',
                    '--ifs.master_dark.sigma_clip=3.0',
                    '--ifs.master_dark.smoothing=5',
                    '--ifs.master_dark.min_acceptable=0.0',
                    '--ifs.master_dark.max_acceptable=2000.0',
                    '--ifs.master_dark.outfilename=' + self._dir_cosm + '/dark_psf.fits',
                    '--ifs.master_dark.badpixfilename=' + self._dir_cosm + '/dark_bpm_psf.fits',
                    self._dir_sof + '/master_dark.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/dark_psf.fits'):
                self._error_msg('It seems a dark_psf.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------
    # Method for the dark star center
    # --------------------------------------------------------------
    def _dark_center(self):
        """
        Method to create the sky dark frame
        """
        # --------------------------------------------------------------
        # First, identify the DIT of the CENTER frames
        # --------------------------------------------------------------
        fits_center = self._find_fits('CENTER', verbose = False)
        for i in range(len(fits_center)):
            idfit = np.where(self._fitsname == fits_center[i])[0][0]
            if i == 0:
                c_dit = self._dit[idfit]
            else:
                if self._dit[idfit] != c_dit: self._error_msg('Different DIT in the CENTER frames')
        # --------------------------------------------------------------
        # Find the DARK measurements with the proper DIT f_dit
        # --------------------------------------------------------------
        fits_dark_tmp = self._find_fits('DARK', verbose = True, force_dit = c_dit)
        fits_dark = [] # list of proper fits file with the correct DIT compared to the CENTER
        for i in range(len(fits_dark_tmp)):
            idfit = np.where(self._fitsname == fits_dark_tmp[i])[0][0]
            if (self._dit[idfit] == c_dit):
                fits_dark.append(fits_dark_tmp[i])
        if len(fits_dark) == 0:
            self._error_msg('Could not find a DARK measurment with a DIT of:' + format(c_dit, '0.1f'))
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/dark_center.sof', 'w')
        for i in range(len(fits_dark)):
            f.write(self._path_to_fits + '/' + fits_dark[i] + '\tIFS_DARK_RAW\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ifs_master_dark',
                    '--ifs.master_dark.coll_alg=2',
                    '--ifs.master_dark.sigma_clip=3.0',
                    '--ifs.master_dark.smoothing=5',
                    '--ifs.master_dark.min_acceptable=0.0',
                    '--ifs.master_dark.max_acceptable=2000.0',
                    '--ifs.master_dark.outfilename=' + self._dir_cosm + '/dark_cen.fits',
                    '--ifs.master_dark.badpixfilename=' + self._dir_cosm + '/dark_bpm_cen.fits',
                    self._dir_sof + '/dark_center.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/dark_cen.fits'):
                self._error_msg('It seems a dark_cen.fits file was not produced. Check the log above')
        else:
            sys.exit()


    # --------------------------------------------------------------
    # Method for the dark star center
    # --------------------------------------------------------------
    def _dark_corono(self):
        """
        Method to create the corongraphic dark frame
        """
        # --------------------------------------------------------------
        # First, identify the DIT of the CENTER frames
        # --------------------------------------------------------------
        fits_sci = self._find_fits('SCIENCE', verbose = False)
        for i in range(len(fits_sci)):
            idfit = np.where(self._fitsname == fits_sci[i])[0][0]
            if i == 0:
                s_dit = self._dit[idfit]
            else:
                if self._dit[idfit] != s_dit: self._error_msg('Different DIT in the SCIENCE frames')
        # --------------------------------------------------------------
        # Find the DARK measurements with the proper DIT f_dit
        # --------------------------------------------------------------
        fits_dark_tmp = self._find_fits('DARK', verbose = True, force_dit = s_dit)
        fits_dark = [] # list of proper fits file with the correct DIT compared to the CENTER
        for i in range(len(fits_dark_tmp)):
            idfit = np.where(self._fitsname == fits_dark_tmp[i])[0][0]
            if (self._dit[idfit] == s_dit):
                fits_dark.append(fits_dark_tmp[i])
        if len(fits_dark) == 0:
            self._error_msg('Could not find a DARK measurment with a DIT of:' + format(s_dit, '0.1f'))
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/dark_corono.sof', 'w')
        for i in range(len(fits_dark)):
            f.write(self._path_to_fits + '/' + fits_dark[i] + '\tIFS_DARK_RAW\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ifs_master_dark',
                    '--ifs.master_dark.coll_alg=2',
                    '--ifs.master_dark.sigma_clip=3.0',
                    '--ifs.master_dark.smoothing=5',
                    '--ifs.master_dark.min_acceptable=0.0',
                    '--ifs.master_dark.max_acceptable=2000.0',
                    '--ifs.master_dark.outfilename=' + self._dir_cosm + '/dark_cor.fits',
                    '--ifs.master_dark.badpixfilename=' + self._dir_cosm + '/dark_bpm_cor.fits',
                    self._dir_sof + '/dark_corono.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/dark_cor.fits'):
                self._error_msg('It seems a dark_cor.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------        
    # Check the header of a fits file for the proper keyword
    # --------------------------------------------------------------        
    def _check_lamp(self, fitsfile):
        """
        Check the header for proper keywords to identify which lamp was used
        """
        hdu = fits.open(self._path_to_fits + '/' + fitsfile)
        hdr = hdu[0].header
        hdu.close()
        idlamp = None
        if 'ESO INS2 LAMP1 ST' in hdr.keys():
            idlamp = 'l1'
        if 'ESO INS2 LAMP2 ST' in hdr.keys():
            idlamp = 'l2'
        if 'ESO INS2 LAMP3 ST' in hdr.keys():
            idlamp = 'l3'
        if 'ESO INS2 LAMP4 ST' in hdr.keys():
            idlamp = 'l4'
        if 'ESO INS2 LAMP5 ST' in hdr.keys():
            idlamp = 'l5'
        if idlamp is None:
            self._error_msg('Could not dientify the lamp used for the flat.')
        return idlamp

    # --------------------------------------------------------------        
    # Method for the flat
    # --------------------------------------------------------------        
    def _flat(self, flatname):
        """
        Method to produce the white flat
        """
        print ' '
        print 'This should be updated to the custom IDL routines.'
        print ' '
        if type(flatname) is not str: 
            self._error_msg('The \"flatname\" should be a string.')
        if flatname == 'white':
            flat_filter, suffix, wave = 'CAL_BB_2', 'flat_white', -1.0
        elif flatname == '1020':
            flat_filter, suffix, wave = 'CAL_NB1_1', 'flat_1020', 1.020
        elif flatname == '1230':
            flat_filter, suffix, wave = 'CAL_NB2_1', 'flat_1230', 1.230
        elif flatname == '1300':
            flat_filter, suffix, wave = 'CAL_NB3_1', 'flat_1300', 1.300
        elif flatname == '1550':
            flat_filter, suffix, wave = 'CAL_NB4_1', 'flat_1550', 1.550
        else:
            self._error_msg('Weird flatname ...')
        # --------------------------------------------------------------        
        # First, identify the proper files
        # --------------------------------------------------------------        
        fits_flat = self._find_fits('FLAT', flat_filter = flat_filter)
        if len(fits_flat) != 2:
            self._error_msg('There should be two files for the white flat. Problem ...')
        else:
            suffix_lamp = self._check_lamp(fits_flat[0])
            if suffix_lamp != self._check_lamp(fits_flat[1]):
                self._error_msg('The flats were taken with different lamps.')
        # --------------------------------------------------------------        
        # Write the SOF file
        # --------------------------------------------------------------        
        sof_file = self._dir_sof + '/' + suffix  +'.sof'
        f = open(sof_file, 'w')
        for i in range(len(fits_flat)):
            f.write(self._path_to_fits + '/' + fits_flat[i] + '\tIFS_DETECTOR_FLAT_FIELD_RAW\n')
        f.write(self._dir_cosm + '/dark_bpm_cal.fits   IFS_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/dark_bpm_cen.fits   IFS_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/dark_bpm_cor.fits   IFS_STATIC_BADPIXELMAP\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the manual script
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            ffname = self._dir_cosm + '/master_detector_' + suffix + '.fits'
            bpname = self._dir_cosm + '/dff_badpixelname_' + suffix + '.fits'
            ifs_flat_manual.sph_ifs_detector_flat_manual(sof_file, ffname, bpname)
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/master_detector_' + suffix + '.fits'):
                self._error_msg('It seems a master_detector_' + suffix + '.fits file was not produced. Check the log above')
        else:
            sys.exit()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        # runit = raw_input('\nProceed [Y]/n: ')
        # if runit != 'n':
        #     args = ['esorex', 'sph_ifs_master_detector_flat',
        #             '--ifs.master_detector_flat.save_addprod=TRUE',
        #             '--ifs.master_detector_flat.outfilename=' + self._dir_cosm + '/master_detector_' + suffix + '_drh.fits',
        #             '--ifs.master_detector_flat.lss_outfilename=' + self._dir_cosm + '/large_scale_' + suffix + '_drh.fits',
        #             '--ifs.master_detector_flat.preamp_outfilename=' + self._dir_cosm +   '/preamp_' + suffix + '_drh.fits',
        #             '--ifs.master_detector_flat.badpixfilename=' + self._dir_cosm + '/dff_badpixelname_' + suffix + '_drh.fits',
        #             '--ifs.master_detector_flat.lambda=' + str(wave),
        #             '--ifs.master_detector_flat.smoothing_length=10.0',
        #             '--ifs.master_detector_flat.smoothing_method=1',
        #             self._dir_sof + '/' + suffix + '.sof']
        #     doflat = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
        #     print '-'*80
        #     # --------------------------------------------------------------
        #     # Rename the fits files
        #     # --------------------------------------------------------------
        #     flat_list = glob.glob(self._dir_cosm + '/*' + suffix_lamp + '.fits')
        #     for i in range(len(flat_list)):
        #         tmp_name = flat_list[i]
        #         args = ['mv', tmp_name, tmp_name.replace('_drh_'+suffix_lamp, '')]
        #         mvfits = subprocess.Popen(args).wait()
        #     # --------------------------------------------------------------
        #     # Check if it actually produced something
        #     # --------------------------------------------------------------
        #     if not os.path.isfile(self._dir_cosm + '/master_detector_' + suffix + '.fits'):
        #         self._error_msg('It seems a master_detector_' + suffix + '_drh.fits file was not produced. Check the log above')
        # else:
        #     sys.exit()

    # --------------------------------------------------------------
    # Method for the wavelength calibration
    # --------------------------------------------------------------
    def _specpos(self):
        """
        Method for the wavelength calibration
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_spec = self._find_fits('SPECPOS')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/specpos.sof', 'w')
        for i in range(len(fits_spec)):
            f.write(self._path_to_fits + '/' + fits_spec[i] + '\tIFS_SPECPOS_RAW\n')
        f.write(self._dir_cosm + '/dark_cal.fits   IFS_MASTER_DARK\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            if self.obs_filter == 'OBS_YJ':
                Hmode = 'FALSE'
            else:
                Hmode = 'TRUE'
                print ' '
                print ' '
                print 'Check the mode !'
                print ' '
                print ' '
                # Hmode should be TRUE only for OBS_YJH, but I need to be sure about the syntax of the mode
                sys.exit()
            args = ['esorex', 'sph_ifs_spectra_positions',
                     '--ifs.spectra_positions.outfilename=' + self._dir_cosm + '/spectra_positions.fits',
                     '--ifs.spectra_positions.hmode=' + Hmode,
                    self._dir_sof + '/specpos.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/spectra_positions.fits'):
                self._error_msg('It seems a spectra_positions.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------
    # Method for the wavelength calibration
    # --------------------------------------------------------------
    def _wave_cal(self):
        """
        Method for the wavelength calibration
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_wave = self._find_fits('WAVECAL')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/wave_cal.sof', 'w')
        for i in range(len(fits_wave)):
            f.write(self._path_to_fits + '/' + fits_wave[i] + '\tIFS_WAVECALIB_RAW\n')
        f.write(self._dir_cosm + '/spectra_positions.fits   IFS_SPECPOS\n')
        f.write(self._dir_cosm + '/dark_cal.fits   IFS_MASTER_DARK\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            if self.obs_filter == 'OBS_YJ':
                args = ['esorex', 'sph_ifs_wave_calib',
                        '--ifs.wave_calib.number_lines=3',
                        '--ifs.wave_calib.outfilename=' + self._dir_cosm + '/wave_calib.fits',
                        '--ifs.wave_calib.wavelength_line1=0.9877',
                        '--ifs.wave_calib.wavelength_line2=1.1237',
                        '--ifs.wave_calib.wavelength_line3=1.3094',
                        self._dir_sof + '/wave_cal.sof']
            else:
                args = ['esorex', 'sph_ifs_wave_calib',
                        '--ifs.wave_calib.number_lines=4',
                        '--ifs.wave_calib.outfilename=' + self._dir_cosm + '/wave_calib.fits',
                        '--ifs.wave_calib.wavelength_line1=0.9877',
                        '--ifs.wave_calib.wavelength_line2=1.1237',
                        '--ifs.wave_calib.wavelength_line3=1.3094',
                        '--ifs.wave_calib.wavelength_line4=1.5451',
                        self._dir_sof + '/wave_cal.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/wave_calib.fits'):
                self._error_msg('It seems a wave_calib.fits file was not produced. Check the log above')
        else:
            sys.exit()


    # --------------------------------------------------------------
    # Method for the IFU flat
    # --------------------------------------------------------------
    def _ifu_flat(self):
        """
        Method for the wavelength calibration
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_ifu = self._find_fits('IFUFLAT')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/ifu_flat.sof', 'w')
        for i in range(len(fits_ifu)):
            f.write(self._path_to_fits + '/' + fits_ifu[i] + '\t IFS_FLAT_FIELD_RAW\n')
        f.write(self._dir_cosm + '/wave_calib.fits                     IFS_WAVECALIB\n')
        f.write(self._dir_cosm + '/master_detector_flat_1020.fits      IFS_MASTER_DFF_LONG1\n')
        f.write(self._dir_cosm + '/master_detector_flat_1230.fits      IFS_MASTER_DFF_LONG2\n')
        f.write(self._dir_cosm + '/master_detector_flat_1300.fits      IFS_MASTER_DFF_LONG3\n')
        if self.obs_filter != 'OBS_YJ':
            # Int he original script, it was:
            # if [ ${MODE} = 'YJH' ]; then
            # fi
            print ' '
            print 'Check the observing mode in ifu_flat !!!'
            print ' '
            sys.exit()
            f.write(self._dir_cosm + '/master_detector_flat_1550.fits      IFS_MASTER_DFF_LONG4\n')
        f.write(self._dir_cosm + '/master_detector_flat_white.fits     IFS_MASTER_DFF_LONGBB\n')
        f.write(self._dir_cosm + '/master_detector_flat_white.fits     IFS_MASTER_DFF_SHORT\n')
        f.write(self._dir_cosm + '/dark_cal.fits                       IFS_MASTER_DARK\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            args = ['esorex', 'sph_ifs_instrument_flat',
                    '--ifs.instrument_flat.ifu_filename=' + self._dir_cosm + '/ifu_flat.fits',
                    '--ifs.instrument_flat.nofit=TRUE',
                    self._dir_sof + '/ifu_flat.sof']
            master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
            print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            if not os.path.isfile(self._dir_cosm + '/ifu_flat.fits'):
                self._error_msg('It seems a ifu_flat.fits file was not produced. Check the log above')
        else:
            sys.exit()

    # --------------------------------------------------------------
    # Method for the SCI and CENTER files
    # --------------------------------------------------------------
    def _sci(self):
        """
        Method for the PSF science files
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_wave = glob.glob(self._dir_pre_proc + '/SPHER*wave*.fits')
        print "Will use the following frames for the WAVECAL:"
        for i in range(len(fits_wave)):
            print fits_wave[i]
        fits_psf = self._find_fits('FLUX')
        fits_center = self._find_fits('CENTER')
        fits_sci = self._find_fits('SCIENCE')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            for i in range(len(fits_wave)):
                self._run_ifs_science(fits_wave[i], self._dir_sof + '/wave_cal.sof', 'wave')
            for i in range(len(fits_psf)):
                self._run_ifs_science(fits_psf[i], self._dir_sof + '/psf_sci.sof', 'flux')
            for i in range(len(fits_center)):
                self._run_ifs_science(fits_center[i], self._dir_sof + '/science_frames.sof', 'center')
            for i in range(len(fits_sci)):
                self._run_ifs_science(fits_sci[i], self._dir_sof + '/science_frames.sof', 'sci')
        else:
            sys.exit()


    def _run_ifs_science(self, filename, sof_file, suffix):
        """
        Method to run the sph_ifs_science_dr esorex recipe
        """
        f = open(sof_file, 'w')
        if self._dir_pre_proc in filename:
            f.write(filename + '\tIFS_SCIENCE_DR_RAW\n')
        else:
            f.write(self._path_to_fits + '/' + filename + '\tIFS_SCIENCE_DR_RAW\n')

        f.write(self._dir_cosm + '/master_detector_flat_1020.fits      IFS_MASTER_DFF_LONG1\n')
        f.write(self._dir_cosm + '/master_detector_flat_1230.fits      IFS_MASTER_DFF_LONG2\n')
        f.write(self._dir_cosm + '/master_detector_flat_1300.fits      IFS_MASTER_DFF_LONG3\n')
        if self.obs_filter != 'OBS_YJ':
            # Int he original script, it was:
            # if [ ${MODE} = 'YJH' ]; then
            # fi
            print ' '
            print 'Check the observing mode in ifu_flat !!!'
            print ' '
            sys.exit()
            f.write(self._dir_cosm + '/master_detector_flat_1550.fits      IFS_MASTER_DFF_LONG4\n')
        f.write(self._dir_cosm + '/master_detector_flat_white.fits     IFS_MASTER_DFF_LONGBB\n')
        f.write(self._dir_cosm + '/master_detector_flat_white.fits     IFS_MASTER_DFF_SHORT\n')
        if suffix =='flux':
            f.write(self._dir_cosm + '/dark_psf.fits                       IFS_MASTER_DARK\n')
            f.write(self._dir_cosm + '/dark_bpm_psf.fits                       IFS_STATIC_BADPIXELMAP\n')
        elif suffix == 'center':
            f.write(self._dir_cosm + '/dark_cen.fits                       IFS_MASTER_DARK\n')
            f.write(self._dir_cosm + '/dark_bpm_cen.fits                       IFS_STATIC_BADPIXELMAP\n')
        elif suffix == 'sci':
            f.write(self._dir_cosm + '/dark_cor.fits                       IFS_MASTER_DARK\n')
            f.write(self._dir_cosm + '/dark_bpm_cor.fits                       IFS_STATIC_BADPIXELMAP\n')
        f.write(self._dir_cosm + '/wave_calib.fits                     IFS_WAVECALIB\n')
        f.write(self._dir_cosm + '/ifu_flat.fits                     IFS_IFU_FLAT_FIELD\n')
        f.close()
        # --------------------------------------------------------------        
        # Run the esorex pipeline
        # --------------------------------------------------------------        
        args = ['esorex', 'sph_ifs_science_dr',
                '--ifs.science_dr.use_adi=0',
                '--ifs.science_dr.spec_deconv=FALSE',
                sof_file]
        master = subprocess.Popen(args, stdout = open(os.devnull, 'w')).wait()
        # Rename and move other files
        mvfiles = glob.glob('SPHER*.*')
        for i in range(len(mvfiles)):
            if self._dir_pre_proc in filename:
                tmp_name = mvfiles[i]
            else:
                tmp_name = mvfiles[i].replace('_', '_' + suffix + '_')
            args = ['mv', mvfiles[i], self._dir_sci + '/' + tmp_name]
            mvf = subprocess.Popen(args).wait()
        args = ['rm', 'ifs_science_dr.fits']
        rmcompiled = subprocess.Popen(args).wait()
        print '-'*80
            # --------------------------------------------------------------
            # Check if it actually produced something
            # --------------------------------------------------------------
            # if not os.path.isfile(self._dir_cosm + '/ifu_flat.fits'):
            #     self._error_msg('It seems a ifu_flat.fits file was not produced. Check the log above')

    # --------------------------------------------------------------
    # Wavelength calibration collapse
    # --------------------------------------------------------------
    def _wave_collapse(self):
        """
        Use the pre-process routine to collapse the cube 
        """
        # --------------------------------------------------------------
        # First, identify the proper files
        # --------------------------------------------------------------
        fits_wave = self._find_fits('WAVECAL')
        # --------------------------------------------------------------
        # Write the SOF file
        # --------------------------------------------------------------
        f = open(self._dir_sof + '/preproc.sof', 'w')
        for i in range(len(fits_wave)):
            f.write(self._path_to_fits + '/' + fits_wave[i] + '\tIFS_RAW\n')
        f.write(self._dir_cosm + '/dark_cal.fits\tIFS_MASTER_DARK\n')
        f.write(self._dir_cosm + '/dark_bpm_psf.fits\tIFS_STATIC_BADPIXELMAP\n')
        f.close()

        ifs_preprocess.sph_ifs_preprocess(self._dir_sof + '/preproc.sof', self._dir_pre_proc, coll = True, bkgsub = False, bpcor = True, xtalk = True, colltyp = 'mean', update_pa = False, catg = 'wave')

    # --------------------------------------------------------------
    # Method to update the center
    # --------------------------------------------------------------
    def _get_center(self):
        """
        Method to find the center position from the waffle pattern.
        """
        # --------------------------------------------------------------
        # Find the fits files for the center
        # --------------------------------------------------------------
        fits_center = glob.glob(self._dir_sci + '/SPHER*center*.fits')
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
        # Get the positions
        # --------------------------------------------------------------
        print '\tDetermining the positions of the satellite spots ...'
        centers = self._center_pos(fits_center[id_waffle])
        # --------------------------------------------------------------
        # Write all that in a file
        # --------------------------------------------------------------
        f = open(self._dir_sci + '/center_pos.csv', 'w')
        f.write('#cx,cy\n')
        for i in range(len(centers)):
            f.write(str(centers[i,0]) + ',' + str(centers[i,1]) + '\n')
        f.close()



    # --------------------------------------------------------------
    # Method to find the intersection of the 4 spots
    # --------------------------------------------------------------
    def _line(self, p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def _intersection(self, x0, y0, x1, y1, x2, y2, x3, y3):
        L1 = self._line([x0, y0], [x2, y2])
        L2 = self._line([x1, y1], [x3, y3])
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False
    # --------------------------------------------------------------
    # Method to find the satellite spot
    # --------------------------------------------------------------
    def _get_spot(self, cx_int, cy_int, frame, ext = 15., thrs = 3., fwhm = 4.0):
        sub = frame[cy_int - ext:cy_int + ext, cx_int - ext:cx_int + ext]
        sources = daofind(sub, fwhm = fwhm, threshold = thrs * np.std(sub))
        bright = np.argsort(sources['peak'])[-2:]
        if len(sources) != 1:
            print 'found ' + str(len(sources)) + ' sources ...'
        if len(sources) == 0.:
            plt.figure()
            plt.imshow(sub, origin = 'lower')
            plt.show()
        cx = (sources['xcentroid'][bright[0]]) + cx_int - ext
        cy = (sources['ycentroid'][bright[0]]) + cy_int - ext
        return cx, cy
    # --------------------------------------------------------------
    # Method to find the four satellite spots
    # --------------------------------------------------------------
    def _center_pos(self, fitsname):
        """
        Method to identify the positions of the spots
        """
        hdu = fits.open(fitsname)
        img = hdu[0].data
        hdr = hdu[0].header
        hdu.close()

        taille = np.shape(img)[1]
        nlambda = np.shape(img)[0]
        pixel   = 7.3  # pixel size [mas]

        wave_min = hdr['HIERARCH ESO DRS IFS MIN LAMBDA']
        wave_max = hdr['HIERARCH ESO DRS IFS MAX LAMBDA']
        wave_step = (wave_max - wave_min) / (nlambda-1)
        wavelength = wave_min + wave_step * np.arange(nlambda * 1.)   
        waffle_orientation = hdr['HIERARCH ESO OCS WAFFLE ORIENT']

        lsurD  = wavelength * 1.e-6/8.e0 * 180/ np.pi * 3600. *1000. / pixel

        centers  = np.zeros(shape=(nlambda,2))
        # spot_centers = np.zeros(shape=(nlambda,4,2))
        # distance = np.zeros(shape=(nlambda,6))

        for i in range(nlambda):
            frame = img[i,].copy()

            if (waffle_orientation == '+'):
                offset = np.pi / 4.e0
            else:
                offset = 0.e0
            orient = 57. * np.pi / 180.0 + offset
            freq   = 10. * np.sqrt(2.) * 0.97
            ext    = 10.

            if (i == 0):
                # initial guess at the frame center
                #  ==> can be modified manually if necessary
                cx_int = np.int(taille / 2.)
                cy_int = np.int(taille / 2.)
            else:
                # subsequent guess is center of previous channel
                cx_int = np.int(centers[i-1,0])
                cy_int = np.int(centers[i-1,1])

            # Spot 0
            cx0_int = np.int(cx_int + freq * lsurD[i] * np.cos(orient))
            cy0_int = np.int(cy_int + freq * lsurD[i] * np.sin(orient))
            cx0, cy0 = self._get_spot(cx0_int, cy0_int, frame)#, fwhm = 4.0)

            # Spot 1
            cx1_int = np.int(cx_int + freq * lsurD[i] * np.cos(orient + np.pi/2))
            cy1_int = np.int(cy_int + freq * lsurD[i] * np.sin(orient + np.pi/2))
            cx1, cy1 = self._get_spot(cx1_int, cy1_int, frame)#, fwhm = lsurD[i])

            
            # Spot 2
            cx2_int = np.int(cx_int + freq * lsurD[i] * np.cos(orient + np.pi))
            cy2_int = np.int(cy_int + freq * lsurD[i] * np.sin(orient + np.pi))
            cx2, cy2 = self._get_spot(cx2_int, cy2_int, frame)#, fwhm = lsurD[i])

            # Spot 3
            cx3_int = np.int(cx_int + freq * lsurD[i] * np.cos(orient + 3. * np.pi/2))
            cy3_int = np.int(cy_int + freq * lsurD[i] * np.sin(orient + 3. * np.pi/2))
            cx3, cy3 = self._get_spot(cx3_int, cy3_int, frame)#, fwhm = lsurD[i])

            # spot_centers[i,0,:] = cx0, cy0
            # spot_centers[i,1,:] = cx1, cy1
            # spot_centers[i,2,:] = cx2, cy2
            # spot_centers[i,3,:] = cx3, cy3
            
            cen = self._intersection(cx0, cy0, cx1, cy1, cx2, cy2, cx3, cy3)
            if cen:
                centers[i,] = cen[0], cen[1]
            else:
                print 'Could not find the intersection'

        return centers



    # --------------------------------------------------------------
    # Method to recenter the science frames
    # --------------------------------------------------------------
    def _center_science(self):
        """
        Method to find the center position from the waffle pattern.
        """
        # --------------------------------------------------------------
        # Read the center positions
        # --------------------------------------------------------------
        data = ascii.read(self._dir_sci + '/center_pos.csv')
        cx = data['cx']
        cy = data['cy']
        nl = len(cx) # This should be the number of wavelength
        nx = 291 # this will be checked on later
        # --------------------------------------------------------------
        # Find the fits files for the center
        # --------------------------------------------------------------
        print '\tApplying the center shift to the science data ...'
        fits_sci = glob.glob(self._dir_sci + '/SPHER*sci*.fits')
        for j in range(nl):
            print '\t\tWavelength ' + str(j+1) + ' out of ' + str(nl) + ' ...'
            cube = np.zeros(shape=(len(fits_sci), nx, nx))
            para = np.zeros(len(fits_sci))
            for i in range(len(fits_sci)):
                hdu = fits.open(fits_sci[i])
                img =hdu[0].data
                hdr = hdu[0].header
                hdu.close()
                if np.shape(img)[0] != nl:
                    print 'nl:', nl
                    print 'img', np.shape(img)[0]
                    self._error_msg('dimension problem when merging')
                if ((np.shape(img)[1] != nx) | (np.shape(img)[2] != nx)):
                    self._error_msg('dimension problem when merging')
                # --------------------------------------------------------------
                # Find the fits files for the center
                # --------------------------------------------------------------
                ndit, pa_beg, pa_mid, pa_end = ifs_preprocess.header_info(hdr)
                id_dit = np.int(fits_sci[i].split('_')[-1].replace('.fits',''))
                para[i] = pa_mid[id_dit]
                # print para[i]

                frame = img[j,].copy()
                shiftx = (np.shape(img)[1]-1)/2 - cx[j]
                shifty = (np.shape(img)[1]-1)/2 - cy[j]
                frame = scipy.ndimage.interpolation.shift(frame, [shifty,shiftx])
                cube[i,] = frame
            # --------------------------------------------------------------
            # Save the fits file
            # --------------------------------------------------------------
            fname = self._dir_ready + '/' + self._starname + '_' + format(j, '03d') + '.fits'
            fits.writeto(fname, cube, clobber = True)
            fits.writeto(self._dir_ready + '/para.fits', para, clobber=True)




    # --------------------------------------------------------------
    # VIP routine for the updated centering.
    # --------------------------------------------------------------
    def _vip_centering(self, array):
        """
        Method that uses the VIP centering stuff
        """
        txt_x = 'X: 145 -> '
        txt_y = 'Y: 145 -> '
        cenxy = [145, 145]
        offset = np.zeros(2)
        # --------------------------------------------------------------
        # Do the first search over a large grid
        # --------------------------------------------------------------
        dist = 20.
        # fr_broad = vip.calib.frame_crop(array, 141, cenxy = cenxy, verbose = False)
        rough = vip.calib.frame_center_radon(array, cropsize = 141, wavelet=False, mask_center=None, 
            hsize = dist, step=1., nproc=2, verbose = True, plot = True)
        if ((np.abs(rough[0]) >= dist) | (np.abs(rough[1]) >= dist)):
            self._error_msg('Increase the box size for the first centering.')
        cenxy[0] -= np.int(rough[1])
        cenxy[1] -= np.int(rough[0])
        offset[0] -= np.int(rough[1])
        offset[1] -= np.int(rough[0])
        # --------------------------------------------------------------
        # Do the second search over a smaller grid.
        # --------------------------------------------------------------
        # dist = 2.
        # fr_broad = vip.calib.frame_crop(array, 141, cenxy = cenxy, verbose = False)
        # rough = vip.calib.frame_center_radon(fr_broad, cropsize = 131, wavelet=False, mask_center=None, 
        #     hsize = dist, step=.1, nproc=2, verbose = True, plot = True)
        # if ((np.abs(rough[0]) >= dist) | (np.abs(rough[1]) >= dist)):
        #     self._error_msg('Increase the box size for the second centering.')
        # offset[0] -= rough[1]
        # offset[1] -= rough[0]
        print 'Offset: dx = ' + str(offset[0]) + '\t\t dy = ' + str(offset[1])
        return offset


    # --------------------------------------------------------------
    # Clean the science and center frames with sigma clipping
    # --------------------------------------------------------------
    def _clean_sci(self):
        """
        Use sigma clipping and masking to clean the SCI and CENTER frames
        """
        fits_center = glob.glob(self._dir_sci + '/SPHER*center*.fits')
        fits_sci = glob.glob(self._dir_sci + '/SPHER*sci*.fits')
        print "Will clean the following frames:"
        for i in range(len(fits_center)):
            print fits_center[i] + '\tCENTER'
        for i in range(len(fits_sci)):
            print fits_sci[i] + '\tSCIENCE'
        # --------------------------------------------------------------
        # Do the cleaning !
        # --------------------------------------------------------------
        runit = raw_input('\nProceed [Y]/n: ')
        if runit != 'n':
            self._clean_cube(fits_center[0])
            # for i in range(len(fits_center)):
            #     print '\tCleaning CENTER '  +str(i+1) + ' of ' + str(len(fits_center)) + ' ...'
            #     self._clean_cube(fits_center[i])
            # for i in range(len(fits_sci)):
            #     print '\tCleaning SCIENCE '  +str(i+1) + ' of ' + str(len(fits_sci)) + ' ...'
            #     self._clean_cube(fits_sci[i])
        else:
             sys.exit()   


    def _clean_cube(self, fitsname):
        """
        Read a single fits file and clean it
        """
        hdu = fits.open(fitsname)
        hdr = hdu[0].header
        img = hdu[0].data
        hdu.close()
        # --------------------------------------------------------------
        # Get the NDIT information
        # --------------------------------------------------------------
        frame = img[0,].copy()
        img[0,] = self._clip_and_mask(frame)
        # if len(np.shape(img)) == 3:
        #     for i in range(np.shape(img)[0]):
        #         frame = img[i,].copy()
        #         img[i,] = self._clip_and_mask(frame)
        # else:
        #     frame = img.copy()
        #     img = self._clip_and_mask(frame)
        fname = fitsname.replace('.fits', '_clean.fits')
        fits.writeto(fname, img, clobber = True, output_verify = "ignore", header = hdr)


    def _clip_and_mask(self, frame):
        """
        Method to do sigma clipping, create a bad pixel map from the output and 
        re-evaluate the data for those pixels
        """
        # print '\t\t\tSigma clipping: first pass ...'
        frame_out = ifs_misc.median_clip(frame, 3.5, num_neighbor = 7)
        sel = (frame == 0.)
        frame_out[sel] = 0.
        sel = (frame_out != frame)
        new_bpm = np.zeros(shape=(np.shape(frame)[0],np.shape(frame)[1]))
        new_bpm[sel] = 1.
        # print '\t\t\tMasking and interpolating the frame'
        frame_out = ifs_misc.sigma_filter(frame, new_bpm, neighbor_box = 6, min_neighbors = 5)
        return frame_out


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
        if os.path.isfile(self._dir_cosm + '/dark_cal.fits'):
            print 'Found a \"dark sky\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_dark_sky = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/dark_psf.fits'):
            print 'Found a \"dark psf\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_dark_psf = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/dark_cen.fits'):
            print 'Found a \"dark center\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_dark_cen = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/dark_cor.fits'):
            print 'Found a \"dark corono\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_dark_corono = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/master_detector_flat_white.fits'):
            print 'Found a \"flat white\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat_white = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/master_detector_flat_1020.fits'):
            print 'Found a \"flat 1020\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat_1020 = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/master_detector_flat_1230.fits'):
            print 'Found a \"flat 1230\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat_1230 = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/master_detector_flat_1300.fits'):
            print 'Found a \"flat 1300\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat_1300 = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/master_detector_flat_1550.fits'):
            print 'Found a \"flat 1550\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_flat_1550 = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/wave_calib.fits'):
            print 'Found a \"wave cal\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_wave_cal = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/spectra_positions.fits'):
            print 'Found a \"spectra positions\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_specpos = True
            print '-'*80
        if os.path.isfile(self._dir_cosm + '/ifu_flat.fits'):
            print 'Found a \"IFU flat\" in \'' + self._dir_cosm + '/\'. Erase it if you want to recalculate it.'
            self._is_ifuflat = True
            print '-'*80

        list_wave = glob.glob(self._dir_pre_proc + '/SPHER*wave*')
        if (len(list_wave) != 0):
            print 'Found a pre-processed WAVE frame in \"' + self._dir_pre_proc + '\". Erase if you want to recalculate them.'
            self._is_wave_preproc = True
            print '-'*80

        list_wave = glob.glob(self._dir_sci + '/SPHER*wave*')
        list_flux = glob.glob(self._dir_sci + '/SPHER*flux*')
        list_center = glob.glob(self._dir_sci + '/SPHER*center*')
        list_sci = glob.glob(self._dir_sci + '/SPHER*sci*')
        if ((len(list_wave) != 0) & (len(list_flux) != 0) & (len(list_center) != 0) & (len(list_sci) != 0)):
            print 'Found WAVE, FLUX, CENTER, and SCIENCE frames in \"' + self._dir_sci + '\". Erase if you want to recalculate them.'
            self._is_science = True
            print '-'*80

        list_clean = glob.glob(self._dir_pre_proc + '/SPHER*clean.fits')
        if (len(list_clean) != 0):
            print 'Found \"cleaned\" fits files in \"' + self._dir_sci + '\". Erase if you want to recalculate them.'
            self._is_clean = True
            print '-'*80


    # --------------------------------------------------------------
    # Method to make a summary of what can be done.
    # --------------------------------------------------------------
    def _find_fits(self, action, blacklist = True, verbose = True, force_dit = None, flat_filter = None):
        """
        Method to identify what can be done (dark, flat, distortion, star center)
        """
        # --------------------------------------------------------------
        # For the DARK measurements
        # --------------------------------------------------------------
        if action == 'DARK':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'DARK', False, 'dark_blacklist'
            if force_dit is None:
                self._error_msg('You should provide a value for force_dit for the dark reduction')
        # --------------------------------------------------------------
        # For the FLAT measurements
        # --------------------------------------------------------------
        elif action == 'FLAT':
            if flat_filter is None:
                self._error_msg('You should provide a value for flat_filter for the flat reduction')
            else:
                if type(flat_filter) is not str:
                    self._error_msg('The flat_filter should be a string (find_fits method).')
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'FLAT,LAMP', False, 'flat_blacklist'
            if self.obs_filter == 'OBS_YJ':
                filt = flat_filter + '_YJ'
            else:
                self._error_msg('Update the find_fits method for that observing mode.')
        # --------------------------------------------------------------
        # For the SPECPOS measurements
        # --------------------------------------------------------------
        elif action == 'SPECPOS':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'SPECPOS,LAMP', False, 'specpos_blacklist'
        # --------------------------------------------------------------
        # For the WAVECAL measurements
        # --------------------------------------------------------------
        elif action == 'WAVECAL':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'WAVE,LAMP', False, 'wavecal_blacklist'
        # --------------------------------------------------------------
        # For the IFUFLAT measurements
        # --------------------------------------------------------------
        elif action == 'IFUFLAT':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'FLAT,LAMP', False, 'ifu_flat_blacklist'
        # --------------------------------------------------------------
        # For the SKY measurements
        # --------------------------------------------------------------
        elif action == 'SKY':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IFS', 'SKY', True, 'sky_blacklist'
        # --------------------------------------------------------------
        # For the BACKGROUND measurements. With restrictions on the DIT for the flux calibration, but with restrictions for the flat-field
        # --------------------------------------------------------------
        elif action == 'BACKGROUND':
            catg, arm, tp, dit, bl_txt = 'CALIB', 'IFS', 'DARK,BACKGROUND', False, 'bkg_blacklist'
        # --------------------------------------------------------------
        # For the CENTER measurements
        # --------------------------------------------------------------
        elif action == 'CENTER':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IFS', 'OBJECT,CENTER', True, 'center_blacklist'
        # --------------------------------------------------------------
        # For the FLUX measurements
        # --------------------------------------------------------------
        elif action == 'FLUX':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IFS', 'OBJECT,FLUX', False, 'flux_blacklist'
        # --------------------------------------------------------------
        # For the SCIENCE measurements
        # --------------------------------------------------------------
        elif action == 'SCIENCE':
            catg, arm, tp, dit, bl_txt = 'SCIENCE', 'IFS', 'OBJECT', True, 'science_blacklist'
        # --------------------------------------------------------------
        # Make the selection. For the FLAT and DISTORTION, the DIT is irrelevant. For the DARK, the filter seems to be irrelevant also.
        # --------------------------------------------------------------
        if (action == 'DARK'):
            sel = np.where((self._catg == catg) & (self._arm == arm) & (self._type == tp) & (self._dit == force_dit))[0]
        elif (action == 'FLAT'):
                sel = np.where((self._catg == catg) & (self._filter == filt) & (self._arm == arm) & (self._type == tp))[0]
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
        # else:
        #     txt = 'Found: \n'
        #     for i in range(len(sel)):
        #         txt += '\n' + self._fitsname[sel[i]] + '\t' + str(self._catg[sel[i]]) + '\t' + str(self._type[sel[i]]) + '\t' + str(self._tech[sel[i]]) + '\t' + str(self._mjd[sel[i]]) + '\t' + str(self._dit[sel[i]]) + '\t' + str(self._filter[sel[i]])
        #     print txt
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
        mode_obs = np.unique(self._tech[sel])
        final_mode = None
        if 'IFU' in mode_obs:
            final_mode = 'IFU'
        else:
            self._error_msg('No IFU data in the available raw data.')
        return final_mode
    # --------------------------------------------------------------        
    # Method to get the MJD of the observations. If there are several, will ask you to pick the one you want
    # --------------------------------------------------------------
    def _get_mjd(self):
        """
        Method to get the MJD of the observations. If multiple DITs are available, will ask for the one you want.
        """
        final_mjd = None
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & (self._tech == self.obs_mode))[0]
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
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._mjd == mjd_obs[i]) & (self._arm == 'IFS') & (self._tech == self.obs_mode))[0]
                txt += '[' + str(i) + '] MJD ' + str(mjd_obs[i]) + ' (' + str(len(sel)) + ' frames)' + '\n'

            select = raw_input(txt)
            idm = np.int(select)
            if ((idm < 0 ) or (idm > len(mjd_obs))):
                self._error_msg('Select the proper number (' + str(idm) + ')')
            else:
                final_mjd = mjd_obs[idm]
        return final_mjd
    # --------------------------------------------------------------        
    # Method to get the DIT of the observations. If there are several, will ask you to pick the one you want
    # --------------------------------------------------------------
    def _get_dit(self):
        """
        Method to get the MJD of the observations. If multiple DITs are available, will ask for the one you want.
        """
        final_dit = None
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & (self._mjd == self.obs_mjd) & (self._tech == self.obs_mode))[0]
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
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & (self._dit == dit_obs[i]) & (self._mjd == self.obs_mjd) & (self._tech == self.obs_mode))[0]
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
        sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & (self._mjd == self.obs_mjd) & (self._dit == self.obs_dit) & (self._tech == self.obs_mode))[0]
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
                sel = np.where((self._obj == self._starname) & (self._type == 'OBJECT') & (self._arm == 'IFS') & (self._dit == self.obs_dit) & (self._filter == filter_obs[i]) & self._mjd == self.obs_mjd) & ((self._tech == self.obs_mode))[0]
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
        data_right = np.mean(hdu[0].data, axis = 0) / peak_right
        hdu.close()
        # --------------------------------------------------------------        
        # Read the left image
        # --------------------------------------------------------------        
        hdu = fits.open(filename + '_left.fits')
        p_l = 0.5 * (hdu[0].header['HIERARCH ESO TEL PARANG END'] + hdu[0].header['HIERARCH ESO TEL PARANG START'])
        data_left = np.mean(hdu[0].data, axis = 0) / peak_left
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
        if not os.path.isfile(self._path_to_fits + '/IFS_list.csv'):
            self._list_fits()
        # --------------------------------------------------------------        
        # Read the list of files
        # --------------------------------------------------------------        
        data = ascii.read(self._path_to_fits + '/IFS_list.csv', delimiter = ';', data_start = 1)
        self._fitsname = data['Name']
        self._obj = data['OBJECT']
        self._arm = data['SEQ.ARM']
        self._catg = data['DPR.CATG']
        self._type = data['DPR.TYPE']
        self._tech = data['DPR.TECH']
        self._dit = data['SEQ1.DIT']
        self._filter = data['INS2.COMB.IFS']
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
        f = open(self._path_to_fits + '/IFS_list.csv', 'w')
        f.write('Name;OBJECT;SEQ.ARM;DPR.CATG;DPR.TYPE;DPR.TECH;SEQ1.DIT;INS2.COMB.IFS;MJD.OBS\n')
        for i in range(nf):
            hdu = fits.open(list_files[i])
            header = hdu[0].header
            hdu.close()
            # if 'NAXIS1' in header.keys():
            try:
                if 'ESO SEQ ARM' in header.keys():
                    txt = list_files[i].replace(self._path_to_fits, '') + ';' + header['OBJECT'] + ';' 
                    txt += header['ESO SEQ ARM'] + ';' + header['ESO DPR CATG'] + ';'
                    txt += header['ESO DPR TYPE'] + ';' + header['ESO DPR TECH'] + ';' 
                    txt += str(header['ESO DET SEQ1 DIT']) + ';' + header['ESO INS2 COMB IFS'] + ';'
                    txt += str(header['MJD-OBS']) + '\n'
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



