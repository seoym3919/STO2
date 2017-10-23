from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
import numpy as np
import scipy as sp
import glob
import math
from astropy import constants as const
from grid_otf import grid_otf
import sys
from progressbar import ProgressBar

import math
import numpy
import sys
from scipy import interpolate
import scipy
import time
import os,errno

def silentremove(filename):
# remove files without raising error
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred



def call_fits(dir):
	file = glob.glob(dir+'*.fits')
	nfile = len(file)
	hdu_tot = []
	
	print('Reading calibrated spectra', flush = True)
	bar0 = ProgressBar()
	for i in bar0(range(0,nfile)):
		hdu = fits.open(file[i])
		hdu_tot.append(hdu)

	return hdu_tot

def make_line_array(dir):
	file = glob.glob(dir+'*.fits')
	nfile = len(file)
	hdu0 = fits.open(file[0])
	nchan = hdu0[1].header['MAXIS1']
	data_hdu = (hdu0[1].data).copy()
	restfreq = (data_hdu.field('RESTFREQ')).copy()
	nline = len(restfreq)

	arr_line = np.zeros(nfile*nline*nchan).reshape(nline,nfile,nchan)
	xpos = np.zeros(nfile*nline).reshape(nline,nfile)
	ypos = np.zeros(nfile*nline).reshape(nline,nfile)

	print('Creating spectral line array', flush = True)

	bar = ProgressBar()
	for i0 in bar(range(0,nfile)):
		hdu_OTF = fits.open(file[i0])
		data_hdu = hdu_OTF[1].data
		int_OTF = data_hdu.field('DATA')
		for j in range(nline):
			arr_line[j,i0,:] = int_OTF[j,:]
		
		pos0 = SkyCoord((np.float(hdu_OTF[1].header['UDP_RA'])+np.float(data_hdu.field('CDELT2')[0]))*u.deg, (np.float(hdu_OTF[1].header['UDP_DEC'])+np.float(data_hdu.field('CDELT3')[0]))*u.deg, frame='icrs')
		pos1 = SkyCoord((np.float(hdu_OTF[1].header['UDP_RA'])+np.float(data_hdu.field('CDELT2')[1]))*u.deg, (np.float(hdu_OTF[1].header['UDP_DEC'])+np.float(data_hdu.field('CDELT3')[1]))*u.deg, frame='icrs')
		pos2 = SkyCoord( np.float(hdu_OTF[1].header['UDP_RA'])*u.deg, np.float(hdu_OTF[1].header['UDP_DEC'])*u.deg, frame='icrs')
		
		pos = [pos0, pos1, pos2]
		for j in range(nline):
			pos_out = pos[j].galactic
			xpos[j,i0] = pos_out.l.degree
			ypos[j,i0] = pos_out.b.degree
		
		del data_hdu
		del int_OTF
		hdu_OTF.close()
		
	return arr_line, xpos, ypos, nchan

	
	
def get_restfreq(hdu):
	data_hdu = hdu[1].data
	restfreq = data_hdu.field('RESTFREQ')
	return restfreq

def get_vel_freq(hdu):
	header = hdu[1].header
	data_hdu = hdu[1].data
	n_pixl = data_hdu.field('MAXIS1')[0]
	restfreq = data_hdu.field('RESTFREQ')
	n_line = len(restfreq)
	vv = np.zeros(n_pixl*n_line).reshape(n_line, n_pixl)
	freq = np.zeros(n_pixl*n_line).reshape(n_line, n_pixl)
	for j0 in range(0,n_line):
		vv[j0,:] = (np.float(hdu[1].header['CRVAL1']) + (1 + np.arange(n_pixl) - data_hdu.field('CRPIX1')[j0]) * data_hdu.field('CDELT1')[j0]) 
		vv[j0,:] = vv[j0,:]*1.e2
		freq[j0,:] = restfreq[j0]* (1.- vv[j0,:]/const.c.cgs.value)
	return vv, freq
		
def make_header(xref, yref, xsize, ysize, pix_scale, xref_pix, yref_pix, coordType, radesys, equinox, frest, faxis, beam_fwhm, veldef, specsys, proj="SFL"):

    hdr = fits.Header()

    # BASIC stuff, the WCS code needs this
    hdr['SIMPLE'] = True
    hdr['NAXIS'] = 4
    hdr['NAXIS1'] = xsize
    hdr['NAXIS2'] = ysize
    hdr['NAXIS3'] = len(faxis)
    hdr['NAXIS4'] = 1

    ctypeDashes = '----'

    xctype = coordType[0] + ctypeDashes[len(coordType[0]):]
    yctype = coordType[1] + ctypeDashes[len(coordType[1]):]

    # MAKE THE POSITION AXES
    hdr['CTYPE1'] = xctype + '-' + proj
    hdr['CRVAL1'] = xref
    hdr['CRPIX1'] = xref_pix
    hdr['CDELT1'] = -1.0*pix_scale

    hdr['CTYPE2'] = yctype + '-' + proj
    hdr['CRVAL2'] = yref
    hdr['CRPIX2'] = yref_pix
    hdr['CDELT2'] = pix_scale

    # MAKE THE VELOCITY AXIS (ALONG THE THIRD DIMENSION)
    # the frame is now indicated via SPECSYS.  Check on any other
    # needed WCS keywords for use here.
    hdr['CTYPE3'] = 'FREQ'
    hdr['CUNIT3'] = 'Hz'
    hdr['CRVAL3'] = faxis[0]
    hdr['CRPIX3'] = 1.0
    hdr['CDELT3'] = faxis[1]-faxis[0]

    # STOKES axis - always I
    hdr['CTYPE4'] = 'STOKES'
    hdr['CRVAL4'] = 1.0
    hdr['CRPIX4'] = 1.0
    hdr['CDELT4'] = 1.0

    hdr['SPECSYS'] = specsys

    # AIPS velocity type
    hdr['VELREF'] = 0
    if specsys == "LSRK":
        hdr['VELREF'] = 1
    elif specsys == "HELIOCEN":
        hdr['VELREF'] = 2
    elif specsys == "TOPOCENT":
        hdr['VELREF'] = 3
    # no others are defined in the original AIPS memo, should search for updates
    # for now, leave everything else at 0
    if veldef == "RADI":
        # radio definition adds 256
        hdr['VELREF'] = hdr['VELREF'] + 256
    # AIPS memo doesn't say what to do for relativistic velocity definition

    # Set the ALT* axis keywords if possible
    if hdr['CDELT3'] != 0. and frest > 0.:
        # zero velocity
        hdr['ALTRVAL'] = 0.0
        # is at channel here the frequency axis equals the rest frequency
        hdr['ALTRPIX'] = hdr['CRPIX3'] + (frest - hdr['CRVAL3'])/hdr['CDELT3']
        
    hdr['RESTFRQ'] = frest

    # ADD THE RADESYS and EQUINOX when appropriate
    if radesys is not None and len(radesys) > 0:
        hdr['RADESYS'] = radesys
    if equinox is not None and equinox > 0.0:
        hdr['EQUINOX'] = equinox

    return hdr
	
def create_wcsheader(header,xpos,ypos,restfreq,freq,line,pix_scale,beam_fwhm):
	# image size
	xRange = np.max(xpos)-np.min(xpos)
	yRange = np.max(ypos)-np.min(ypos)
	xsize = int(math.ceil(xRange*1.1/pix_scale))+20
	ysize = int(math.ceil(yRange*1.1/pix_scale))+20
	# set image center
	refXsky = np.min(xpos)+0.5*xRange
	refYsky = np.min(ypos)+0.5*yRange
	refXpix = math.ceil(xsize*0.5)
	refYpix = math.ceil(ysize*0.5)
	# set coordinates and projection
	xcoord = header['CTYPE2']
	xcoord, proj_temp = xcoord.split('--')
	ycoord = header['CTYPE3']
	ycoord, proj_temp = ycoord.split('--')
	specSysDict = {'OBS':'TOPOCENT','GEO':'GEOCENTR','BAR':'BARYCENT','HEL':'HELIOCEN','GAL':'GALACTOC','LSD':'LSRD','LSR':'LSRK','LGR':'LOCALGRP','COB':'CMBDIPOL'}
	coordType = [xcoord,ycoord]
	radesys = ''
	equinox = 0.
	veldef = 'RADI'
	specsys = specSysDict[header['VELFRAME']]
	# create header for the spectral cube
	hdr = make_header(refXsky, refYsky, xsize, ysize, pix_scale, refXpix, refYpix, coordType, radesys, equinox, restfreq, freq, beam_fwhm, veldef, specsys)
	# create wcs object from STO2 header (non-trivial header)
	w = wcs.WCS(hdr,relax=True)
	return hdr, w, xsize, ysize

#
# begine of main program
#	

dir = 'C:/Users/seo39/Drive/OBS/STO2/'
dir_data0 = dir+'BASE/NII_1/CAL_DATA/' 
dir_data1 = dir+'BASE/NII_2/CAL_DATA/' 
dir_data2 = dir+'BASE/CII/CAL_DATA/' 
dir_write = dir+'gridder/'

# read all calibrated fits data
arr_line0, xpos0, ypos0, nchan0 = make_line_array(dir_data0)
arr_line1, xpos1, ypos1, nchan1 = make_line_array(dir_data1)
#
# header sample
file = glob.glob(dir_data0+'*.fits')
nfile= len(file)
hdu0 = fits.open(file[0])
header = hdu0[1].header
#
## get rest frequency from header of file
restfreq = get_restfreq(hdu0)
#
## velocity
vv, freq = get_vel_freq(hdu0)
#
# dish size of STO2 in cm
dish_diam = 80.
# wavelength of lines in cm
wavelength = const.c.cgs.value/restfreq
#
# beam size in array 
beam_fwhm = 1.2 * wavelength/dish_diam * np.rad2deg(1.)

# create spectra array to put in regridder
#arr_line0, xpos0, ypos0, nchan0 = make_line_array(hdu)
#
# pixel size
pixPerBeam = 6.
#
# mask nan channels and arrange variables for header and re-gridding
line = 0
mask_nan = np.isfinite(arr_line0[line,nfile-10,:])
vv_in = vv[line,mask_nan]
nchan_in = len(vv_in)
# 
arr_line_in = np.vstack([arr_line0[0,:,:],arr_line1[1,:,:]])
arr_line_in = arr_line_in[:,mask_nan]
restfreq_in = restfreq[line]
freq_in = freq[line,mask_nan]
xpos_in = np.append(xpos0[0,:],xpos1[1,:])
ypos_in = np.append(ypos0[0,:],ypos1[1,:])
beam_fwhm_in = beam_fwhm[line]
pix_scale = np.int(3600.0*beam_fwhm[line]/pixPerBeam)/3600.0
#
# create header, wcs, and image size from header and given parameters
hdr, wcsObj, xsize, ysize = create_wcsheader(header,xpos_in,ypos_in,restfreq_in,freq_in,line,pix_scale,beam_fwhm_in)
#
# create spectral map 
cube, weight, beam_size = grid_otf(arr_line_in, xpos_in, ypos_in, wcsObj, nchan_in, xsize, ysize, pix_scale, beam_fwhm_in)
#
#
hdr['CTYPE3'] = 'VELO-LSR'
hdr['CUNIT3'] = 'km/s'
hdr['CRVAL3'] = vv_in[0]/1.e5
hdr['CRPIX3'] = 1.0
hdr['CDELT3'] = (vv_in[1]-vv_in[0])/1.e5
#
silentremove(dir_write+'cube_NII.fits')
hdu_cube_out = fits.PrimaryHDU(cube, header = hdr)
hdu_cube_out.writeto(dir_write+'cube_NII.fits')
#
#
#
#