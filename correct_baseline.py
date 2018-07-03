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

	
	
def fixdata(arr, win_min, win_max, clip = 100., limit = 5.,iter = 2):
#	This function fixes bad channels in data with the polynomial fitting
#	input:
#		arr : the data arry
#	 	win_min, win_max : a range of channels containing bad channels
#		limit : sigma level in finding bad channels in the sigma-clipping 
#	output:
#		arr : array with bad channels fixed

	arr_cut0 = arr[win_min:win_max]
	badpix = (arr_cut0 >= clip) | (arr_cut0 <= -clip)
	goodpix = (arr_cut0 < clip) & (arr_cut0 > -clip)

	# set x-axis values
	vv = np.arange(len(arr))
	vv_cut = vv[win_min:win_max]

	sig = np.std(arr_cut0[goodpix])
	med = np.median(arr_cut0[goodpix])
	
	badpix = (arr_cut0 >= med+limit*sig) | (arr_cut0 <= med-limit*sig)
	goodpix = (arr_cut0 < med+limit*sig) & (arr_cut0 > med-limit*sig)
	
	if (goodpix.any() == False):
		return arr
	
	p0 = np.polyfit(vv_cut[goodpix],arr_cut0[goodpix],2)
	arr_cut_fitted = p0[2] + p0[1]*vv_cut + p0[0]*vv_cut**2
	
	count = 0
	while (len(badpix) > 0) & (count <= iter):
		arr_cut0[badpix] = arr_cut_fitted[badpix]
		arr[win_min:win_max] =arr_cut0
		arr_ave = runaverage(arr,24)
		residue = arr[win_min:win_max]-arr_ave[win_min:win_max]
		sig = np.std(residue)
		med = np.median(residue)
		badpix = (residue >= med+limit*sig) | (residue <= med-limit*sig)
		p0 = np.polyfit(vv_cut,arr_cut0,2)
		arr_cut_fitted = p0[2] + p0[1]*vv_cut + p0[0]*vv_cut**2
		count = count +1
		
	if (len(badpix) > 0):
		arr_cut0[badpix] = arr_cut_fitted[badpix]
		arr[win_min:win_max] = arr_cut0
	return arr	
		
	
	
def replacedata(arr, badpix):
# badpix is array containing bad channels
	badpix_min = np.min(badpix)
	badpix_max = np.max(badpix)
	dpix = len(badpix)

	arr_ave =runaverage(arr,4)
	xx = np.arange(len(arr))
	
	xx_cut = np.append(xx[badpix_min-5*dpix:badpix_min],xx[badpix_max:badpix_max+5*dpix])
	yy_cut = np.append(arr_ave[badpix_min-5*dpix:badpix_min],arr_ave[badpix_max:badpix_max+5*dpix])

	pars = np.polyfit(xx_cut,yy_cut,2)
	arr[badpix] = pars[2] + pars[1]*xx[badpix] + pars[0]*xx[badpix]**2
	return arr
	
def baseline(arr):
# find a constant, lowest flat baseline 
#	input:
#		arr : data array
#	output :
#		arr_ave: baseline
#		arr_sig: standard deviation from the baseline

	arr_ave = np.mean(arr)
	arr_sig = np.std(arr)
	arr = arr[(arr >= arr_ave - 2*arr_sig)]
	N_LAST = 0
	while (len(arr) != N_LAST):
		N_LAST = len(arr)
		arr_ave = np.mean(arr)
		if (len(arr) > 1):
			arr_sig = np.std(arr) 
		else:
			arr_sig = 0.
		arr = arr[( arr <= arr_ave + 2*arr_sig )]
	return arr_ave, arr_sig

	
def find_good_spec(arr, win_min, win_max, first_limit = 100., chisq_limit = 16.):
# Sorting out bad spectra by iterative process. This firstly estimates baseline structures(fringes) and find bad spectra deviating considerably from the baseline 
#	input:
#		arr : all spectra within a scan, array shape should be 2D with a shape of (# of spectra, # of channels)
#		win_min : low index of windows to check bad spectra
#		win_max : high index of windows to check bad spectra
#		first_limit : limit for clipping. unit is in Kelvin. If emission/absorption is expected larger than 100K/-100K, please use this argument
#		chisq_limit : limit for sigma-clipping in finding good spectra, Default = 16 which is 4-sigma
#	output:
#		good_spec: boolean array telling which is a good spectra. True is good one and False is bad one. Array size is # of spectra

	size = arr.shape
	base_mean = np.zeros(size[1]) 
	base_stdev = np.zeros(size[1])
	# find baseline and standard deviation from the baseline with only good spectra within the first explicit limit in Kelvin 
	for i3 in range(0,size[1]):
		goodpix = (arr[:,i3] > -first_limit) & (arr[:,i3] < first_limit)
		if (goodpix.any() == True):
			base_mean[i3] = np.median(arr[goodpix,i3])
			base_stdev[i3] = np.std(arr[goodpix,i3]) 

	base_mean = runaverage(base_mean,24)
	
	# the fisrt iteration of finding good spectra by the sigma-clipping
	# estimated chi-sqaure value of the whole spectrum compared to the baseline
	chisq = np.zeros(size[0])
	for i3 in range(0,size[0]):
		chisq[i3] = ((arr[i3,win_min:win_max]-base_mean[win_min:win_max])**2/base_stdev[win_min:win_max]).sum()/(win_max-win_min)
	# find good spectra
	good_spec = (chisq <= chisq_limit)

	# the second iteration of finding good spectra by the sigma-clipping
	# re-estimate baseline and standard deviation using good spectra found with the sigma-clipping
	if (good_spec.any() != False):
		for i3 in range(0,size[1]):
			base_mean[i3] = np.median(arr[good_spec,i3])
			base_stdev[i3] = np.std(arr[good_spec,i3]) 
		# find good spectra
		for i3 in range(0,size[0]):
			chisq[i3] = ((arr[i3,win_min:win_max]-base_mean[win_min:win_max])**2/base_stdev[win_min:win_max]).sum()/(win_max-win_min)
		good_spec = (chisq <= chisq_limit)
	return good_spec



def read_scan(file):
# read in all spectra within am OTF scan
	nfile = len(file)
	hdu_tot = []
	for i in range(0,nfile):
		hdu = fits.open(file[i])
		hdu_tot.append(hdu)
	return hdu_tot
	
def free_poly(p, arr):
	y = np.zeros(len(arr))
	for i0 in range(0,len(arr)):
		y[i0] = (np.array(p[:])*arr[i0]**np.arange(len(p))).sum() 
	return y

def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,max_iters=50, conv_thresh=1e-5, verbose=False):
	'''Computes the asymmetric least squares baseline.
	* http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
	smoothness_param: Relative importance of smoothness of the predicted response.
	asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
	Setting p=1 is effectively a hinge loss.
	'''
	smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
	# Rename p for concision.
	p = asymmetry_param
	# Initialize weights.
	w = np.ones(intensities.shape[0])
	for i in range(max_iters):
		z = smoother.smooth(w)
		mask = intensities > z
		new_w = p*mask + (1-p)*(~mask)
		conv = np.linalg.norm(new_w - w)
		if verbose:
			print (i+1, conv)
		if conv < conv_thresh:
			break
		w = new_w
	else:
		print('ALS did not converge in %d iterations' % max_iters)
	return z


class WhittakerSmoother(object):
	def __init__(self, signal, smoothness_param, deriv_order=1):
		self.y = signal
		assert deriv_order > 0, 'deriv_order must be an int > 0'
		# Compute the fixed derivative of identity (D).
		d = np.zeros(deriv_order*2 + 1, dtype=int)
		d[deriv_order] = 1
		d = np.diff(d, n=deriv_order)
		n = self.y.shape[0]
		k = len(d)
		s = float(smoothness_param)
		
		# Here be dragons: essentially we're faking a big banded matrix D,
		# doing s * D.T.dot(D) with it, then taking the upper triangular bands.
		diag_sums = np.vstack([np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant') for i in range(1, k+1)])
		upper_bands = np.tile(diag_sums[:,-1:], n)
		upper_bands[:,:k] = diag_sums
		for i,ds in enumerate(diag_sums):
			upper_bands[i,-i-1:] = ds[::-1][:i+1]
		self.upper_bands = upper_bands
		
	def smooth(self, w):
		foo = self.upper_bands.copy()
		foo[-1] += w  # last row is the diagonal
		return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

		
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

	
def wT_denoise(arr):
# Denoise using the wavelet transform. Reduce noise less than 2 channels 
# Do not use this if channel is not evenly spaced. 
	coeff = pywt.wavedec(arr,'db5','zpd',5)
	coeff_all = np.concatenate([coeff[0],coeff[1],coeff[2],coeff[3],coeff[4],coeff[5]])
	coeff[5][:] = 0.
	arr_wT = pywt.waverec(coeff,'db5','zpd')
	return arr_wT

	
	
def iterative_polynimial_baseline(x,y,win_min,win_max,deg_min=1,deg_max=10,chisq_limit=2.):
	chisq = 50.
	deg = deg_min
	# try high degree polynomial if baseline fitting is not good enough
	while ((chisq > chisq_limit) & (deg < deg_max)):
		frac1 = vv <= win_min
		frac2 = vv >= win_max
		y_runave = runaverage(y, 12)
		
		x_in = np.transpose(np.append(x[frac2], x[frac1]))
		y_in = np.transpose(np.append(y_runave[frac2], y_runave[frac1]))
		p0 = np.ones(deg+1)*0.1
		err_poly = lambda p0, x_in, y_in: y_in - free_poly(p0, x_in)
		p1, cov_x, infodict, mesg, ier = sop.leastsq(err_poly, p0[:], args=(x_in, y_in),full_output=1)
		y_polyfit = free_poly(p1,x_in)
	
		y_partial = np.append(y_runave[frac2], y_runave[frac1])
		y_base = np.append(y_polyfit[frac2], y_polyfit[frac1])
		stdev = []
		for j1 in range(0,len(y_partial),10):
				stdev.append(np.std(y_partial[j1:j1+10]))
		stdev_min = np.min(stdev)
		chisq = ((y_partial - y_base)**2).sum()/stdev_min**2/(len(y_partial)-4.-1.)
		deg =deg + 1
	return y_polyfit
	
	
def gauss_func(x,amp,center,sigma):
	y = amp*np.exp(-0.5*(x-center)**2/sigma**2)
	return y


def multi_gauss_func(x,par,ngauss):
# returns multiple Gaussian curves, requires gauss_func routine
	if ngauss == 0:
		print('Number of Gaussian curves is 0',flush=True)
	if len(par) < 3 * ngauss:
		print('Number of elements are not sufficient for  '+'{0:02d}'.format(ngauss)+' Gaussian curves.',flush=True)
		print('Number of parameter is '+'{0:02d}'.format(len(par)),flush=True)

	xsize = len(x)
	gauss_array = np.zeros([ngauss,xsize])
	for i0 in range(0,ngauss):
		gauss_array[i0,:] = gauss_func(x, par[i0*3],par[i0*3+1],par[i0*3+2])

	multi_gauss = gauss_array.sum(axis =0) 
	return multi_gauss

	
def multi_gauss_func_fixed_center(x,par,center,ngauss):
# returns multiple gaussian curves when center positions of Gaussian curves are fixed
	if ngauss == 0:
		print('Number of Gaussian curves is 0',flush=True)

	if (len(par) < 2 * ngauss):
		print('Number of elements are not sufficient for  '+'{0:02d}'.format(ngauss)+' Gaussian curves.',flush=True)
		print('Number of parameter is '+'{0:02d}'.format(len(par)),flush=True)

	xsize = len(x)
	gauss_array = np.zeros([ngauss,xsize])
	for i0 in range(0,ngauss):
		gauss_array[i0,:] = gauss_func(x, par[i0*2],center[i0],par[i0*2+1])

	multi_gauss = gauss_array.sum(axis =0) 
	return multi_gauss

	
def peak_fit(x,y,peak_pos,npeaks, init_amp = 0.1, init_width = 1.):
# fits gaussian to peaks found by peakutils. Peak positions should be inputed and are fixed in the fitting.
# input:
#	x: 1D x axis array (velocity or frequency array for spectra)
#	y: 1D y axis array (intensity array for spectra)
#	peak_pos: position of peaks in velocity. Should be obtained from peak-detecting algorithm. Peak position will be fixed for the fitting. 
#	npeaks: number of peaks 
# output:
#	gauss_peaks: summation of fitted gaussian curves
#	p1: paramters for Gaussian curves. Only has amplitudes and widths of the curves.   
	xsize = len(x)
	ysize = len(x)
	peak_pos_size = len(peak_pos)
	if xsize != ysize:
		print('x and y arryas should have same dimensiona and size')

	if peak_pos_size == 0:
		print('peak position is not provided. Fitting cannot continue')


		
	# create input parameter array for fitting
	par = np.zeros(npeaks*2)
	for i0 in range(0,npeaks):
		par[i0*2] = init_amp
		par[i0*2+1] = init_width
	
	# make error function and fit using scipy least square fitting	
	errfunc = lambda par, x, y: y - multi_gauss_func_fixed_center(x, par[0:npeaks*2], peak_pos, npeaks)
	p1, cov_x, infodict, mesg, ier = sop.leastsq(errfunc, par, args=(x, y),full_output=1,ftol=1.5e-10,xtol=1.5e-10)
	gauss_peaks = multi_gauss_func_fixed_center(x, p1, peak_pos, npeaks)
	return gauss_peaks, p1


def masking_peaks(x, center, sigma, factor = 3., buffer = 0.):
# Create mask for the Gaussian peaks the and baseline
# input:
#	x: input x-axis array
#	center: mu values of the Gaussian curves
#	sigma: sigma values of the Gaussian curves
#	factor: controls mask width,  factor * sigma
#	buffer: additional width to peak mask
# output
#	mask_peaks: boolean array for Gaussian peaks
#	mask_base:boolean array for baseline
#
	if len(center) != len(sigma):
		print('Error: the dimension of center and sigma is not same',flush=True)
	if len(center) == 0:
		print('Error: empty data',flush=True)
	mask_temp = np.zeros([len(center),len(x)],dtype = bool)
	mask_peaks = np.zeros(len(x),dtype = bool)
	for i0 in range(0,len(center)):
		mask_temp[i0,:] = (x <= center[i0] + factor * sigma[i0] + buffer) & (x >= center[i0] - factor * sigma[i0] - buffer)
	
	for j0 in range(0,len(x)):
		mask_peaks[j0] = mask_temp[:,j0].any()
	mask_base = np.invert(mask_peaks)
	return mask_peaks, mask_base


def find_peaks(y,npeaks,steps = 0.01):
# find the strongest peaks upto npeaks 
# input:
#	y: input array to find peaks
#	npeaks: number of peaks
#	steps: interval of limit to find peaks in normalized spectra (highest peak value is 1)
# output:
#	peak_indexes: array containing peak index 
	limit = 1.
	peak_indexes = []
	while (len(peak_indexes) < npeaks) & (limit > 0.):		
		limit = limit - steps
		peak_indexes = peakutils.peak.indexes(y,limit,20)
	# limit number of peaks to npeaks if it happen to find more than npeaks.  
	if len(peak_indexes) >= npeaks+1:
		temp = y[peak_indexes]
		temp_sort_index = np.argsort(temp)
		peak_indexes = peak_indexes[temp_sort_index[::-1][0:npeaks]]	
	return peak_indexes


def find_window(x, arr, npeaks, win_peak, win_limit, factor, buffer, plot = False, scan= 0, dir = None):
# Finds windows using spectra in array
# input:
#	x: velocity array in 1D
#	arr: spectra intensity array in 2D. Dimension is [# of spectra, # of channels]
#	npeaks: number of peaks to be found
#	win_peak: window to find peaks, 1D array with 2 elements [min, max]
#	win_limit: force additional limit to window. Sets region of interest. 1D array with 2 elements [min,max].
#	factor: controls width of peaks; half width of window is factor * sigma
#	buffer: additional width to the windows. width of window + 2* buffer
#	plot: if true, it will generate plot for standard deviation and Gaussian fitting
#	scan: scan number to be written in file name and title of plot
#	dir: output directory. If it is none, it will save plot in the local folder
# output:
#	win_gauss_fit:
#	win_baseline_fit:
#	npeaks_roi:
#
	if dir == None:
		dir = './'

	size = arr.shape
	nspec = size[0]
	nchan = size[1]
	# estimate standard deviation for each channel
	stdev = np.zeros(nchan)
	for j0 in range (0,nchan):
		stdev[j0] = np.std(arr[:,j0])
	# get lowfrequency structure using the wavelet transform
	stdev_wT = wT_lowfreq(stdev)
	# estimate baseline using the asymmetric least square smoothing. This will eliminate broad features due to large fringes
	als_stdev = als_baseline(stdev_wT,asymmetry_param=0.03)
	# subtract baseline
	stdev_smooth = stdev_wT - als_stdev

	# setting  region for peak-finding
	cut_peakfind = (x > win_peak[0]) & (x < win_peak[1])
	x_peakfind = x[cut_peakfind]
	stdev_smooth_peakfind = stdev_smooth[cut_peakfind]

	# find peaks
	peak_indexes = find_peaks(stdev_smooth_peakfind, npeaks, steps = 0.01)
	
	# number of peaks within region of interest
	peaks_in_vel = x_peakfind[peak_indexes]
	npeaks_roi = len(peaks_in_vel[(peaks_in_vel >= win_limit[0]) & (peaks_in_vel <= win_limit[1])])
	
	gauss_fit, p1 = peak_fit(x,stdev_smooth, peaks_in_vel, npeaks, init_amp = 0.1, init_width = 1.)

	sigma = np.zeros(npeaks)
	for j0 in range(0,npeaks):
		sigma[j0] = np.abs(p1[j0*2+1])
		if sigma[j0] > np.abs(p1[j0*2])* 10.:
			sigma[j0] = 0.
	
	mask_peaks, maks_base = masking_peaks(x, peaks_in_vel, sigma, factor = factor, buffer = buffer)
	
	win_limit = (x <= win_limit_max) & (x >= win_limit_min) 
	
	win_gauss_fit = mask_peaks & win_limit
	# Get channels for baseline fitting by inverting windows for the Gaussian fit
	win_baseline_fit = np.invert(win_gauss_fit)
	
	if plot:
		plt.axis([np.min(x),np.max(x),-5, 2.*np.max(stdev)])
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Standard Deviation [K]')
		plt.title('Scan = '+'{0:05d}'.format(scan))
		plt.plot(x,stdev, label = 'Standard Deviation')
		plt.plot(x,gauss_fit, label = 'Gaussian')
		plt.plot(x,als_stdev, label = 'ALS')
		plt.plot(x_peakfind,stdev_smooth_peakfind, label = 'wT')
		plt.plot(x_peakfind[peak_indexes], stdev_smooth_peakfind[peak_indexes], 'ro',label = 'Peaks')
		plt.legend(frameon=False)
		plt.savefig(dir+'stdev_'+'{0:05d}'.format(scan)+'.png')
		plt.close("all")	
	
	return win_gauss_fit, win_baseline_fit, npeaks_roi
	
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
dir_data_in = dir_base+'CAL_DATA/'
dir_scans_in = dir+'DATA/LEVEL0.6/etaCar5_4591-4733/'
dir_write = dir_base+'LINES/'
dir_wdata = dir_base+'BASE_CORRECTED/'

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


shutil.rmtree(dir_wdata, ignore_errors=True)
os.mkdir(dir_wdata)

# Choose which line to process. 2 is CII, 0 & 1 are NII
lin = 2
nchan =1024

plot_1stcal = True
plot_2ndcal = True

# bad channels
badpix = [479,480,481,482,483,484,485,486,487,488,489,490,491,492]

#================================================================
# Start baseline correction
#for i0 in OTF_scans:
for i0 in range(4591,4592):
	file_OTF = glob.glob(dir_data_in+'{0:05d}'.format(i0)+'/'+'*.fits')
	hdu_tot = read_scan(file_OTF)
		
	TA_original = np.zeros([len(file_OTF), nchan])
	
	shutil.rmtree(dir_wdata+'{0:05d}'.format(i0), ignore_errors=True)
	os.mkdir(dir_wdata+'{0:05d}'.format(i0))
	
	for i2 in range(0,len(file_OTF)):
		print(file_OTF[i2], flush = True)
		hdu_OTF = hdu_tot[i2]
		data_OTF = hdu_OTF[1].data
		TA_0th = (data_OTF['DATA'])[lin].copy()
	
		# Get velocity axis
		n_pixl = data_OTF.field('MAXIS1')[lin]
		vv = (np.float(hdu_OTF[1].header['CRVAL1']) + (1 + np.arange(n_pixl) - data_OTF.field('CRPIX1')[lin]) * data_OTF.field('CDELT1')[lin]) 
		vv = vv/1.e3
		
		# Get spatial position of CII in RADec(pos2) and Galactic coord.(pos3)
		pos2 = SkyCoord( np.float(hdu_OTF[1].header['UDP_RA'])*u.deg, np.float(hdu_OTF[1].header['UDP_DEC'])*u.deg, frame='icrs')
		pos3 = pos2.galactic	
		
		# replace bad pixels
		TA_badpix_removed = replacedata(TA_0th, badpix)
		TA_badpix_removed[1023] = TA_badpix_removed[1020]
		TA_badpix_removed[1022] = TA_badpix_removed[1021]
		
		TA_original[i2,:] = TA_badpix_removed[:]
		
		if plot_1stcal:
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.axis([-120,60,-50,50])
			plt.plot(vv, TA_badpix_removed, drawstyle='steps',lw=0.5)
			plt.text(-80.,45.,'{:.5f}'.format(pos3.l.degree)+', '+'{:.5f}'.format(pos3.b.degree))
			plt.savefig(dir_write+'cal_1st'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i2)+'.png',dpi=500)
			plt.close("all")
			
		# outputs TA converted spectra without baseline correction in the fits format 
		data_temp = hdu_OTF[1].data
		data_temp.field('DATA')[2,:] = TA_badpix_removed[:]
		hdu_OTF[1].header['TUNIT24'] = 'K'
		hdu_OTF.writeto(dir_wdata+'{0:05d}'.format(i0)+'/'+'cal_1st_'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i2)+'.fits')
		
	# Sorting out bad spectra based on the median spectra and standar deviation
	good_spec = find_good_spec(TA_original, 100, 400, 100., 25.)
	good_spec_index = np.arange(len(file_OTF),dtype = int)[good_spec]
	
	for i3 in good_spec_index:
		TA = TA_original[i3,:].copy()
		als_guess = als_baseline(wT_lowfreq(TA),asymmetry_param=0.15)
		TA_base_corrected = TA - als_guess
		
		if plot_2ndcal:
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.axis([-120,60,-50,50])
			plt.plot(vv, wT_lowfreq(TA), drawstyle='steps',lw=0.3)
			plt.plot(vv, als_guess, drawstyle='steps',lw=0.3)
			plt.plot(vv, TA, drawstyle='steps',lw=0.3)
			plt.plot(vv, TA_base_corrected, drawstyle='steps',lw=0.7, color='r')
			plt.text(-80.,45.,'{:.5f}'.format(pos3.l.degree)+', '+'{:.5f}'.format(pos3.b.degree))
			plt.savefig(dir_write+'cal_2st'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i3)+'.png',dpi=500)
			plt.close("all")
		
		# outputs TA converted spectra without baseline correction in the fits format 
		hdu_OTF = hdu_tot[i3]
		data_temp = hdu_OTF[1].data
		data_temp.field('DATA')[2,:] = TA_base_corrected[:]
		hdu_OTF[1].header['TUNIT24'] = 'K'
		hdu_OTF.writeto(dir_wdata+'{0:05d}'.format(i0)+'/'+'cal_2nd_'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i3)+'.fits')
		
		
		
		

