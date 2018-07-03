import sys
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import Imputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import FastICA
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy as sp
from scipy.linalg import solve
from scipy import signal
import scipy.optimize as sop
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks_cwt
import glob
import os, errno
import shutil
import time
import matplotlib.pyplot as plt
import pywt

def silentremove(filename):
# remove files without raising error
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def wT_lowfreq(arr):
# Find lowfrequency structure in spectra. Wipe out any structure less than 8 channel frequency.
# Do not use this if channel is not evenly spaced. 
	coeff = pywt.wavedec(arr,'db5','zpd',5)
	coeff[1][:] = 0.
	coeff[2][:] = 0.
	coeff[3][:] = 0.
	coeff[4][:] = 0.
	coeff[5][:] = 0.
	arr_wT = pywt.waverec(coeff,'db5','zpd')
	return arr_wT

def fixdata_raw(arr, badpix, buffer=50, limit = 1.,deg = 3, verbose = False):
#	This function fixes bad channels in data with the polynomial fitting
#	input:
#		arr : the data arry
#	 	buffer : number of channel for buffer windows
#		limit : sigma level in finding bad channels in the sigma-clipping 
#	output:
#		arr : array with bad channels fixed
	size = arr.shape
	mask_window = np.zeros(size[0],dtype = bool)
	mask_window[np.min(badpix)-buffer:np.max(badpix)+buffer] = True
	mask_bad = np.zeros(size[0],dtype = bool)
	mask_bad[badpix] = True 
	mask_buffer = mask_window & np.invert(mask_bad)
	std_buffer = np.std(arr[mask_buffer])
	mean_buffer = np.mean(arr[mask_buffer])
	#sigma clip
	badbad = (arr[mask_bad] < mean_buffer-limit*std_buffer) | (arr[mask_bad] > mean_buffer+limit*std_buffer)
	#update
	mask_bad[badpix] = badbad[:]
	mask_good = np.invert(mask_bad) & mask_window
	# set x-axis values
	x = np.arange(len(arr))
	x_in = x[mask_good]
	y_in = arr[mask_good]
	p0 = np.ones(deg+1)*0.1
	err_poly = lambda p0, x_in, y_in: y_in - free_poly(p0, x_in)
	p1, cov_x, infodict, mesg, ier = sop.leastsq(err_poly, p0[:], args=(x_in, y_in),full_output=1)
	y_polyfit = free_poly(p1,x[mask_bad])
	arr[mask_bad] = y_polyfit[:]
	if verbose:
		print(np.arange(size[0])[mask_bad],flush = True)
	return arr			

def free_poly(p, arr):
	y = np.zeros(len(arr))
	for i0 in range(0,len(arr)):
		y[i0] = (np.array(p[:])*arr[i0]**np.arange(len(p))).sum() 
	return y
	
dir = '/mnt/c/Users/seo39/Drive/OBS/STO2/'
dir_base = dir + 'Gum31/'
dir_data_in = dir_base+'TIME_DATA/'
dir_data_out = dir_base+'TIME_DATA/'

nchan =1024

hdu_time_hot = fits.open(dir_data_in+'time_HOT.fits')
time_hot = hdu_time_hot[0].data
hdu_hot_t = fits.open(dir_data_in+'HOT_t.fits')
hot_t = hdu_hot_t[0].data
size_hot_t = hot_t.shape

hdu_time_sky = fits.open(dir_data_in+'time_SKY.fits')
time_sky = hdu_time_sky[0].data
hdu_sky_t = fits.open(dir_data_in+'SKY_t.fits')
sky_t = hdu_sky_t[0].data
size_sky_t = sky_t.shape

hdu_time_hot_otf = fits.open(dir_data_in+'time_OTF_HOT.fits')
time_hot_otf = hdu_time_hot_otf[0].data
hdu_hot_otf_t = fits.open(dir_data_in+'HOT_OTF_t.fits')
hot_otf_t = hdu_hot_otf_t[0].data
size_hot_otf_t = hot_otf_t.shape

time_all = np.sort(np.concatenate([time_hot,time_sky,time_hot_otf]))
size_tot = len(time_all)
start_time = np.min(time_all)

# fix data first
for k in range(0,size_hot_t[0]):
	hot_t[k,:] = fixdata_raw(hot_t[k,:],badpix)
	hot_t[k,badpix] = wT_lowfreq(hot_t[k,:])[badpix]
hot_t[:,1023] = hot_t[:,1022]
for k in range(0,size_sky_t[0]):
	sky_t[k,:] = fixdata_raw(sky_t[k,:],badpix)
	sky_t[k,badpix] = wT_lowfreq(sky_t[k,:])[badpix]
sky_t[:,1023] = sky_t[:,1022]
for k in range(0,size_hot_otf_t[0]):
	hot_otf_t[k,:] = fixdata_raw(hot_otf_t[k,:],badpix)
	hot_otf_t[k,badpix] = wT_lowfreq(hot_otf_t[k,:])[badpix]
hot_otf_t[:,1023] = hot_otf_t[:,1022]

valid1 = np.zeros(size_hot_t[0],dtype=bool)
for k in range(0,size_hot_t[0]):
	valid1[k] = np.min(hot_t[k,:]) > 0.

valid2 = np.zeros(size_sky_t[0],dtype=bool)
for k in range(0,size_sky_t[0]):
	valid2[k] = np.min(sky_t[k,:]) > 0.

valid3 = np.zeros(size_hot_otf_t[0],dtype=bool)
for k in range(0,size_hot_otf_t[0]):
	valid3[k] = np.min(hot_otf_t[k,:]) > 0.

mask_in1 = (time_all >= np.min(time_hot[valid1])) & (time_all <= np.max(time_hot[valid1])) 
mask_out1 = np.invert(mask_in1)
mask_in2 = (time_all >= np.min(time_sky[valid2])) & (time_all <= np.max(time_sky[valid2])) 
mask_out2 = np.invert(mask_in2)
mask_in3 = (time_all >= np.min(time_hot_otf[valid3])) & (time_all <= np.max(time_hot_otf[valid3])) 
mask_out3 = np.invert(mask_in3)
	
	
hot_interp = np.zeros([size_tot,nchan])
sky_interp = np.zeros([size_tot,nchan])
hot_otf_interp = np.zeros([size_tot,nchan])
for i0 in range(0,nchan):
	#print(i0)
	func0 = sp.interpolate.interp1d((time_hot[valid1]-start_time)/1.e3,hot_t[valid1,i0], kind = 'linear')
	func1 = sp.interpolate.interp1d((time_hot[valid1]-start_time)/1.e3,hot_t[valid1,i0], kind = 'nearest',fill_value ='extrapolate')
	hot_interp[mask_in1,i0] = func0((time_all[mask_in1]-start_time)/1.e3)
	hot_interp[mask_out1,i0] = func1((time_all[mask_out1]-start_time)/1.e3)
	func0 = sp.interpolate.interp1d((time_sky[valid2]-start_time)/1.e3,sky_t[valid2,i0], kind = 'linear')
	func1 = sp.interpolate.interp1d((time_sky[valid2]-start_time)/1.e3,sky_t[valid2,i0], kind = 'nearest',fill_value ='extrapolate')
	sky_interp[mask_in2,i0] = func0((time_all[mask_in2]-start_time)/1.e3)
	sky_interp[mask_out2,i0] = func1((time_all[mask_out2]-start_time)/1.e3)
	func0 = sp.interpolate.interp1d((time_hot_otf[valid3]-start_time)/1.e3,hot_otf_t[valid3,i0], kind = 'linear')
	func1 = sp.interpolate.interp1d((time_hot_otf[valid3]-start_time)/1.e3,hot_otf_t[valid3,i0], kind = 'nearest',fill_value ='extrapolate')
	hot_otf_interp[mask_in3,i0] = func0((time_all[mask_in3]-start_time)/1.e3)
	hot_otf_interp[mask_out3,i0] = func1((time_all[mask_out3]-start_time)/1.e3)

silentremove(dir_data_out+'time_tot.fits')	
hdu = fits.PrimaryHDU(time_all)
hdu.writeto(dir_data_out+'time_tot.fits')
silentremove(dir_data_out+'hot_interp.fits')	
hdu = fits.PrimaryHDU(hot_interp)
hdu.writeto(dir_data_out+'hot_interp.fits')
	
silentremove(dir_data_out+'sky_interp.fits')	
hdu = fits.PrimaryHDU(sky_interp)
hdu.writeto(dir_data_out+'sky_interp.fits')

silentremove(dir_data_out+'hot_otf_interp.fits')	
hdu = fits.PrimaryHDU(hot_otf_interp)
hdu.writeto(dir_data_out+'hot_otf_interp.fits')


y_interp = hot_interp[:,:]/sky_interp[:,:]


silentremove(dir_data_out+'y_interp.fits')	
hdu = fits.PrimaryHDU(y_interp)
hdu.writeto(dir_data_out+'y_interp.fits')

#mask_in_hothot = (time_all >= np.min(time_hot[valid1])) & (time_all <= np.max(time_hot[valid1])) 
#mask_out_hothot = np.invert(mask_in_hothot)
#
#hot_vs_hot = np.zeros([len(time_all),nchan])
#for i0 in range(0,nchan):
#	func0 = sp.interpolate.interp1d((time_hot[valid1]-start_time)/1.e3,hot_t[valid1,i0], kind = 'linear')
#	func1 = sp.interpolate.interp1d((time_hot[valid1]-start_time)/1.e3,hot_t[valid1,i0], kind = 'nearest',fill_value ='extrapolate')
#	hot_vs_hot[mask_in_hothot,i0] = func0((time_all - start_time)/1.e3)/hot_otf_temp[mask_in_hothot,i0]
#	hot_vs_hot[mask_out_hothot,i0] = func1((time_all - start_time)/1.e3)/hot_otf_temp[mask_out_hothot,i0]
#
#	
#
#
#silentremove(dir_data_out+'hot_vs_hot.fits')	
#hdu = fits.PrimaryHDU(hot_vs_hot)
#hdu.writeto(dir_data_out+'hot_vs_hot.fits')
