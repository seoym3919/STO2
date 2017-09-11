#
#	Collection of filters and mathmatical functions for STO2 pipeline
#	written by Youngmin Seo  
#	September-07-2017
#
# math & science util
import numpy as np
import pywt

#====================================================================================================================================================
# Filters and Functions
#====================================================================================================================================================
def wT_lowfreq(arr):
	#	1D noise reduction algorithm using wavelet
	#	Find lowfrequency structure in spectra. Wipe out any structure less than 8 channel frequency.
	#	Do not use this if channel is not evenly spaced. 
	#	input:
	#		arr: float 1D array, number of array should be even number
	# 	output:
	#		arr_wT: noise-reduced
	if (len(arr) % 2 == 1):
		raise Exception('Number of array should be even.')
	#
	coeff = pywt.wavedec(arr,'db5','zpd',5)
	coeff[1][:] = 0.
	coeff[2][:] = 0.
	coeff[3][:] = 0.
	coeff[4][:] = 0.
	coeff[5][:] = 0.
	arr_wT = pywt.waverec(coeff,'db5','zpd')
	return arr_wT

def wT_denoise(arr):
	#	Denoise using the wavelet transform. Reduce noise less than 2 channels 
	#	Do not use this if channel is not evenly spaced.
	#	input:
	#		arr: float, input array
	#	output:
	#		arr_wT: float, denoised array
	coeff = pywt.wavedec(arr,'db5','zpd',5)
	coeff_all = np.concatenate([coeff[0],coeff[1],coeff[2],coeff[3],coeff[4],coeff[5]])
	coeff[5][:] = 0.
	arr_wT = pywt.waverec(coeff,'db5','zpd')
	return arr_wT

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

def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,max_iters=50, conv_thresh=1e-5, verbose=False):
	#	Computes the asymmetric least squares baseline.
	#	* http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
	#	smoothness_param: Relative importance of smoothness of the predicted response.
	#	asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
	#	Setting p=1 is effectively a hinge loss.
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

def gauss_func(x,amp,center,sigma):
	#	Returns Gaussian functions
	#	input:
	#		x: float, array containing abscissa values
	#		amp: float, value of Gaussian profile amplitude
	#		center: float, value of Gaussian profile center
	#		sigma: float, value of Gaussian profile sigma
	#	output:
	#		y: float, array of Gaussian profile
	y = amp*np.exp(-0.5*(x-center)**2/sigma**2)
	return y

def multi_gauss_func(x,par,ngauss):
	# Returns multiple Gaussian curves, requires gauss_func routine
	#	input:
	#		x: float, array containing abscissa values
	#		par: float, parameter array of multiple Gaussian profiles
	#		ngauss: integer, number of Gaussian 
	#	output:
	#		multi_gauss: float, array containing multiple Gaussian profiles
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
	#	input:
	#		x: float, array containing abscissa values
	#		par: float, parameter array of multiple Gaussian profiles other than profile centers
	#		ngauss: integer, number of Gaussian 
	#	output:
	#		y: float, array containing multiple Gaussian profiles with given centers
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

def FiniteDiff(x,y):
	# Evaluate centered finite difference
	#	input:
	#		x: float array, array containing abscissa
	#		y: float array, array containing data value
	#	output:
	#		dydx: float array, differentiated values
	if (len(x) != len(y)):
		raise Exception('FiniteDiff: lengths of x and y do not match.')
	delx= x[2]-x[1]
	size = len(y)
	y_rshift = np.zeros(size)
	y_lshift = np.zeros(size)
	y_lshift[0:size-1] = y[1:size]
	y_lshift[size-1] = y[size-1]
	y_rshift[1:size] = y[0:size-1]
	y_rshift[0] = y[0]
	dydx = 0.5*(y_lshift-y_rshift)/delx
	return dydx		

def ExpandMask(input, iters):
	#	Expands the True area in an array 'input'.
	#	Expansion occurs in the horizontal and vertical directions by one
	#	cell, and is repeated 'iters' times.
	#	input:
	#		input: boolean array
	#		iters: number of channels to expand mask
	#	output:
	#		output: boolean array with expanded mask
	Len = len(input)
	output = input.copy()
	for iter in range(0,iters):
		for y in range(1,Len-1):
			if (input[y]): 
				output[y-1] = True
				output[y]   = True
				output[y+1] = True
		input = output.copy()
	return output
	
def free_poly(p, arr):
	#	Returns 1D polynomial array
	#	input:
	#		p: float array, parameters for polynomial. p[0] is the zeroth order
	#		arr: float array, array of x-axis
	#	output:
	#		y: float array, polynomial array
	y = np.zeros(len(arr))
	for i0 in range(0,len(arr)):
		y[i0] = (np.array(p[:])*arr[i0]**np.arange(len(p))).sum() 
	return y
	
def find_nearest(array,value):
	#	find the nearest array value to a given value 
	#	input:
	#		array: float array, array for searching the nearest value
	#		value: float, input value
	#	output:
	#		array[idx]: float, the array value nearest to the input value 
    idx = (np.abs(array-value)).argmin()
    return array[idx]
	

def find_nearest_index(array,value):
	#	find the nearest array index to a given value 
	#	input:
	#		array: float array, array for searching the nearest value
	#		value: float, input value
	#	output:
	#		idx: integer, the array index nearest to the input value 
    idx = (np.abs(array-value)).argmin()
    return idx

def masking_peaks(x, center, sigma, factor = 3., buffer = 0.):
	#	Create mask for the Gaussian peaks the and baseline
	#	input:
	#		x: float, x-axis array
	#		center: float, mu values of the Gaussian curves
	#		sigma: float, sigma values of the Gaussian curves
	#		factor: float, controls mask width,  factor * sigma
	#		buffer: float, additional width to peak mask
	#	output
	#		mask_peaks: boolean, array for Gaussian peaks
	#		mask_base:boolean, array for baseline
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
	#	 find the strongest peaks upto npeaks 
	#	 input:
	#		y: float, input array to find peaks
	#		npeaks: integer, number of peaks
	#		steps: float, interval of limit to find peaks in normalized spectra (highest peak value is 1)
	#	 output:
	#		peak_indexes: float, array containing peak index 
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

def peak_fit(x,y,peak_pos,npeaks, init_amp = 0.1, init_width = 1.):
	#	fits gaussian to peaks found by peakutils. Peak positions should be inputed and are fixed in the fitting.
	#	input:
	#		x: float, 1D x axis array (velocity or frequency array for spectra)
	#		y: float, 1D y axis array (intensity array for spectra)
	#		peak_pos: float, position of peaks in velocity. Should be obtained from peak-detecting algorithm. Peak position will be fixed for the fitting. 
	#		npeaks: integer, number of peaks 
	#	output:
	#		gauss_peaks: float, summation of fitted gaussian curves
	#		p1: float, array containing paramters for Gaussian curves. Only has amplitudes and widths of the curves.   
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
	
class WhittakerSmoother(object):
	# Class cotaining functions required for Asymmetric Least Square baseline correction
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

