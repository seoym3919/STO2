import numpy as np
import scipy as sp
import scipy.optimize as sop
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from scipy.linalg import solveh_banded
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import pywt
import peakutils
import glob
import os, errno
import shutil
import time
import matplotlib.pyplot as plt

def silentremove(filename):
# remove files without raising error
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

			
def runaverage(arr,bin):
# run over average data
#	input:
#		arr : 1D data array
#		bin : # of channels to bin
#	output:
#		averaged : averaged data array 
	old = arr
	dbin=np.int(bin/2)
	averaged = np.zeros(len(arr))
	for i0 in range (dbin,len(arr)-dbin):
		averaged[i0] = sum(old[i0-dbin+1:i0+dbin+1]) / np.float(bin)
	return averaged

def smoothing(x,window_len=11,window='hanning'):
#	smooth the data using a window with requested size.
#	This method is based on the convolution of a scaled window with the signal.
#	The signal is prepared by introducing reflected copies of the signal 
#	(with the window size) in both ends so that transient parts are minimized
#	in the begining and end part of the output signal.
#	input:
#		x: the input signal 
#		window_len: the dimension of the smoothing window; should be an odd integer
#		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#			flat window will produce a moving average smoothing.
#	output:
#		the smoothed signal
#	example:
#	t=linspace(-2,2,0.1)
#	x=sin(t)+randn(len(t))*0.1
#	y=smooth(x)
#	see also: 
#	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#	scipy.signal.lfilter
#	TODO: the window parameter could be the window itself if an array instead of a string
#	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#	
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[np.int((window_len-1)/2):np.int(len(x)+(window_len-1)/2)]



def read_scan(file):
# read in all spectra within am OTF scan
	nfile = len(file)
	hdu_tot = []
	for i in range(0,nfile):
		hdu = fits.open(file[i])
		hdu_tot.append(hdu)
	return hdu_tot

		
def find_nearest(array,value):
#	find the nearest value within an array to the given value
#	input:
#		array: float array
#		value: given value
#	output:
#		array[idx]: the nearest value to the given value
    idx = (np.abs(array-value)).argmin()
    return array[idx]


#=============================================================================
#=============================================================================
# Main program
#=============================================================================
#=============================================================================

dir = '/mnt/c/Users/seo39/Drive/OBS/STO2/'
dir_base = dir+'Gum31/'
dir_data_in = dir_base+'TIME_DATA/'
dir_scans_in = dir+'DATA/LEVEL0.6/etaCar5_4591-4733/'
dir_write = dir_base+'LINES/'
dir_wdata = dir_base+'CAL_DATA/'

shutil.rmtree(dir_wdata, ignore_errors=True)
os.mkdir(dir_wdata)

# read the time of observations
hdu_time_y = fits.open(dir_data_in+'time_tot.fits')
time_all = hdu_time_y[0].data

# read fitted y-factor
hdu_y = fits.open(dir_data_in+'y_interp.fits')
y_factor_all = hdu_y[0].data

# read fitted HOT spectra within OTF scans
hdu_OTF_HOT = fits.open(dir_data_in+'hot_otf_interp.fits')
OTF_HOT_all = hdu_OTF_HOT[0].data

Tsky =45.
# Choose which line to process. 2 is CII, 0 & 1 are NII
lin = 2
nchan =1024

start_time = np.min(time_all)

# reading data folder and make array containing OTF scans and REf scans
folder_name = os.listdir(dir_scans_in)
scans = np.sort(np.array(folder_name,dtype=int))
OTF_mask = np.zeros(len(scans),dtype = bool)
REF_mask = np.zeros(len(scans),dtype = bool)

count = 0
for i0 in scans:
	dir_data = dir_scans_in+'{0:05d}'.format(i0)+'/'
	# find HOT0 and REF scans
	file_REH = glob.glob(dir_data+'HOT*')
	file_REC = glob.glob(dir_data+'REF*')
	file_OTF = glob.glob(dir_data+'OTF*')
	if len(file_OTF) > 1:
		OTF_mask[count] = True
	if len(file_REC) > 0:
		REF_mask[count] = True
	count = count + 1
	
OTF_scans = scans[OTF_mask]
REF_scans = scans[REF_mask]

# bad channels
badpix = [480,481,482,483,484,485,486,487,488,489,490]

#================================================================
# Start TA conversion
for i0 in OTF_scans:
#for i0 in range(4591,4592):
	print(i0, flush = True)
	dir_data = dir_scans_in+'{0:05d}'.format(i0)+'/'
	file_OTF = glob.glob(dir_data+'OTF*')
	file_OTFHOT = glob.glob(dir_data+'HOT*')
	ndump_OTF = len(file_OTF)

	shutil.rmtree(dir_wdata+'{0:05d}'.format(i0), ignore_errors=True)
	os.mkdir(dir_wdata+'{0:05d}'.format(i0))
	
	if ndump_OTF > 0:
	
		# read all OTF dumps within a scan
		hdu_tot = read_scan(file_OTF)
	
		# setup variable to process spectra
		TA_original = np.zeros([ndump_OTF,nchan])
		TA_binned = np.zeros([ndump_OTF,nchan])
		time_in_scan = np.zeros(ndump_OTF)#		
	
		for i2 in range(0,ndump_OTF):
			# get data
			hdu_OTF = hdu_tot[i2]
			data_OTF = hdu_OTF[1].data
			int_time_OTF = np.float(hdu_OTF[1].header['OBSTIME'])
			obs_time_OTF = np.float(hdu_OTF[1].header['UNIXTIME'])
			int_OTF = (data_OTF.field('DATA')).copy()/int_time_OTF
			time_in_scan[i2] = obs_time_OTF

			# Get velocity axis
			n_pixl = data_OTF.field('MAXIS1')[lin]
			vv = (np.float(hdu_OTF[1].header['CRVAL1']) + (1 + np.arange(n_pixl) - data_OTF.field('CRPIX1')[lin]) * data_OTF.field('CDELT1')[lin]) 
			vv = vv/1.e3
			
			# Get spatial position of CII in RADec(pos2) and Galactic coord.(pos3)
			pos2 = SkyCoord( np.float(hdu_OTF[1].header['UDP_RA'])*u.deg, np.float(hdu_OTF[1].header['UDP_DEC'])*u.deg, frame='icrs')
			pos3 = pos2.galactic
			
			# get ambient temperature from the first HOT scan within OTF column
			if len(file_OTFHOT) > 0:
				hdu_OTFHOT = fits.open(file_OTFHOT[0])
				Thot=np.float(hdu_OTFHOT[1].header['CONELOAD'])
			else:
				dir_REF = dir_scans_in+'{0:05d}'.format(find_nearest(REF_scans,i0))+'/'
				file_REH = glob.glob(dir_REF+'HOT*')
				hdu_REH = fits.open(file_REH[0])
				Thot=np.float(hdu_REH[1].header['CONELOAD'])
			
			# construct reference spectra for the observation time
			# y_fit: y-factor at the obs. time
			# y_OTF: y-factor at the obs. time. HOT_OTF(t)/ON(t) We will use this without emission to correct y-fit 
			# REF_fit: reference scan at the obs. time
			y_fit = np.zeros(1024)
			y_OTF = np.zeros(1024)
			REF_fit = np.zeros(1024)
			
			# interpolate HOT scan and y factor
			for j0 in range(0,1024):
				interp_func1 = sp.interpolate.interp1d((time_all[:]-start_time)/1.e3, y_factor_all[:,j0], fill_value = 'extrapolate')
				y_fit[j0] = interp_func1((obs_time_OTF-start_time)/1.e3)
				interp_func2 = sp.interpolate.interp1d((time_all[:]-start_time)/1.e3, OTF_HOT_all[:,j0], fill_value = 'extrapolate')
				REF_fit[j0] = interp_func2((obs_time_OTF-start_time)/1.e3)/y_fit[j0]
	
			y_fit = runaverage(y_fit,2)
			
			# Tsys
			Tsys = (Thot - y_fit * Tsky) / (y_fit - 1.)
	
			# Calculate TA
			TA = (int_OTF[lin,:]-REF_fit[:]) / REF_fit[:]* Tsys[:]
					
			# offset correction. Offset is adjusted to be nearly 0
			nan_mask = np.isfinite(TA) == True
			if (i2 == 0):
				TA0 = TA
				TA0_median = np.median(TA0[nan_mask])
			residue_median = np.median(TA[nan_mask]-TA0[nan_mask])
			TA = TA - residue_median - TA0_median
			
			# 2 channel roll average
			TA = runaverage(TA,2)
	
			# Double Side Band correction
			TA = TA*2.
	
			TA_temp = TA[90:900]
			nan_mask = (np.isfinite(TA_temp) == True)

			# outputs TA converted spectra without baseline correction in the fits format 
			data_temp = hdu_OTF[1].data
			data_temp.field('DATA')[2,:] = TA[:]
			hdu_OTF[1].header['TUNIT24'] = 'K'
			hdu_OTF.writeto(dir_wdata+'{0:05d}'.format(i0)+'/'+'cal_0th_'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i2)+'.fits')

			# plot spectra
			plt.title('Scan'+'{0:05d}'.format(i0)+'--'+'{0:03f}'.format((obs_time_OTF-start_time)/1.e3)+r'[10$^3$ seconds]')
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.axis([-120,60,-50,50])
			plt.plot(vv, TA, drawstyle='steps',alpha=0.5)
	
		plt.text(-80.,45.,'{:.5f}'.format(pos3.l.degree)+', '+'{:.5f}'.format(pos3.b.degree))
		plt.savefig(dir_write+'all'+'{0:05d}'.format(i0)+'.png',dpi=500)
		plt.close("all")
	
		
#			
#		
