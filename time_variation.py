### Script for reading Single-Dish Observation Data from STO2

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.optimize as sop
import pywt
import os


lin = 2


dir='/mnt/c/Users/seo39/Drive/OBS/STO2/Gum31/'
dir_write=dir+'TIME_DATA/'
dir_scans = '/mnt/c/Users/seo39/Drive/OBS/STO2/DATA/LEVEL0.6/etaCar5_4591-4733/'

#
# reading data folder and make array containing OTF scans and REf scans
folder_name = os.listdir(dir_scans)
scans = np.sort(np.array(folder_name,dtype=int))
OTF_mask = np.zeros(len(scans),dtype = bool)
REF_mask = np.zeros(len(scans),dtype = bool)
#
count = 0
for i0 in scans:
	dir_data = dir_scans+'{0:05d}'.format(i0)+'/'
	# find HOT0 and REF scans
	file_REH = glob.glob(dir_data+'HOT*')
	file_REC = glob.glob(dir_data+'REF*')
	file_OTF = glob.glob(dir_data+'OTF*')
	if len(file_OTF) > 1:
		OTF_mask[count] = True
	if (len(file_REC) > 0) & (len(file_REH) > 0):
		REF_mask[count] = True
	count = count + 1
OTF_scans = scans[OTF_mask]
REF_scans = scans[REF_mask]

nREF= len(REF_scans)

hot_t = np.zeros(nREF*1024).reshape(nREF,1024)
sky_t = np.zeros(nREF*1024).reshape(nREF,1024)
time = np.zeros(nREF)


#
# Collecting HOT and SKY observations in REF scans
#
count = 0
for i0 in REF_scans:
	dir_REF1 = dir_scans+'{0:05d}'.format(i0)+'/'
	print(dir_REF1)
	
	# find HOT0 and REF scans
	file_REH = glob.glob(dir_REF1+'HOT*')
	file_REC = glob.glob(dir_REF1+'REF*')
	
	hdu_REH = fits.open(file_REH[0])
	data_REH = hdu_REH[1].data
	int_time_REH = np.float(hdu_REH[1].header['OBSTIME'])
	obs_time_REH = np.float(hdu_REH[1].header['UNIXTIME'])
	int_REH = data_REH.field('DATA')/int_time_REH

	hot_t[count,:] = int_REH[2,:]
	time[count] = obs_time_REH 
	
	nchan = hdu_REH[1].header['MAXIS1']
	
	hdu_REC = fits.open(file_REC[0])
	data_REC = hdu_REC[1].data
	int_time_REC = np.float(hdu_REC[1].header['OBSTIME'])
	obs_time_REC = np.float(hdu_REC[1].header['UNIXTIME'])
	int_REC = data_REC.field('DATA')/int_time_REC

	sky_t[count,:] = int_REC[2,:]
	
	count =count + 1

#
# Collecting HOT observations in OTF scans
#	
nOTF= len(OTF_scans)

hot_t_OTF = np.zeros(nOTF*5*1024).reshape(nOTF*5,1024)
time_OTF = np.zeros(nOTF*5)
count = 0
for i0 in OTF_scans:
	dir_data = dir_scans+'{0:05d}'.format(i0)+'/'
	file_HOT = glob.glob(dir_data+'HOT*')

	hdu_HOT1 = fits.open(file_HOT[0])
	data_HOT1 = hdu_HOT1[1].data
	int_time_HOT1 = np.float(hdu_HOT1[1].header['OBSTIME'])
	obs_time_HOT1 = np.float(hdu_HOT1[1].header['UNIXTIME'])
	int_HOT1 = data_HOT1.field('DATA')/int_time_HOT1

	hot_t_OTF[count,:] = int_HOT1[2,:]
	time_OTF[count] = obs_time_HOT1
	count = count + 1
	
	if (len(file_HOT) >= 2.):
		hdu_HOT2 = fits.open(file_HOT[1])
		data_HOT2 = hdu_HOT2[1].data
		int_time_HOT2 = np.float(hdu_HOT2[1].header['OBSTIME'])
		obs_time_HOT2 = np.float(hdu_HOT2[1].header['UNIXTIME'])
		int_HOT2 = data_HOT2.field('DATA')/int_time_HOT2
		hot_t_OTF[count,:] = int_HOT2[2,:]
		time_OTF[count] = obs_time_HOT2
		count = count + 1

		
#		
#	Collecting OTF observations	
#
ON_t = np.zeros(nOTF*150*1024).reshape(nOTF*150,1024)
time_ON = np.zeros(nOTF*150)
count = 0
for i0 in OTF_scans:
	dir_data = dir_scans+'{0:05d}'.format(i0)+'/'
	file_OTF = glob.glob(dir_data+'OTF*')
	print(dir_data)
	
	for i1 in range(0,len(file_OTF)):
		hdu_OTF = fits.open(file_OTF[i1])
		data_OTF = hdu_OTF[1].data
		int_time_OTF = np.float(hdu_OTF[1].header['OBSTIME'])
		obs_time_OTF = np.float(hdu_OTF[1].header['UNIXTIME'])
		int_OTF = data_OTF.field('DATA')/int_time_OTF

		ON_t[count,:] = int_OTF[2,:]
		time_ON[count] = obs_time_OTF
		count = count + 1

# Get only valid data
		
channel =100
		
valid = (time_OTF > 0.)

hot_t_OTF = hot_t_OTF[valid,:]
time_OTF = time_OTF[valid]

valid = (time_ON > 0.)
time_ON = time_ON[valid]
ON_t = ON_t[valid,:]


y_max=np.max(hot_t_OTF[:,channel]/1.e6)
plt.axis([0,17,y_max-3,y_max+0.1])
plt.plot((time-time[0])/1.e3,hot_t[:,channel]/1.e6,'r.',markersize=5)
plt.plot((time_OTF-time[0])/1.e3,hot_t_OTF[:,channel]/1.e6,'b.',markersize=5)
plt.xlabel('Time [s]')
plt.ylabel('Count/s')
plt.savefig(dir+'hot_time_variation.png')
plt.close("all")
#
hdu = fits.PrimaryHDU(time)
hdu.writeto(dir_write+'time_HOT.fits')
hdu = fits.PrimaryHDU(hot_t)
hdu.writeto(dir_write+'HOT_t.fits')

hdu = fits.PrimaryHDU(time_OTF)
hdu.writeto(dir_write+'time_OTF_HOT.fits')
hdu = fits.PrimaryHDU(hot_t_OTF)
hdu.writeto(dir_write+'HOT_OTF_t.fits')

hdu = fits.PrimaryHDU(time)
hdu.writeto(dir_write+'time_sky.fits')
hdu = fits.PrimaryHDU(sky_t)
hdu.writeto(dir_write+'SKY_t.fits')

hdu = fits.PrimaryHDU(time_ON)
hdu.writeto(dir_write+'time_ON.fits')
hdu = fits.PrimaryHDU(ON_t)
hdu.writeto(dir_write+'ON_t.fits')


for channel in range(0,1024):
	y_max=np.max(hot_t_OTF[:,channel]/1.e6)
	plt.axis([0,17,0.,y_max*1.1])
	plt.plot((time-time[0])/1.e3,hot_t[:,channel]/1.e6,'r.',markersize=5)
	plt.plot((time_OTF-time[0])/1.e3,hot_t_OTF[:,channel]/1.e6,'r.',markersize=5)
	plt.plot((time_ON-time[0])/1.e3,ON_t[:,channel]/1.e6,'g.',markersize=2)
	plt.plot((time-time[0])/1.e3,sky_t[:,channel]/1.e6,'b.',markersize=5)
	plt.xlabel(r'Time [$10^3$ seconds]')
	plt.ylabel(r'$10^6$ Count/s')
	plt.savefig(dir_write+'sky_time_variation_'+'{0:05d}'.format(channel)+'.png')
	plt.close("all")
	
