from __future__ import division
import numpy as np

def delay_to_dm(delay, error):
    c = 299792458 #m/s
    r_e = 2.81794033e-15 #m
    bandwidth = 16e6 #Hz
    f = np.array([311.25e6, 327.25e6, 343.25e6]) #Hz
    f_central = f + bandwidth/2
    k_DM = c * r_e / (2*np.pi)

    print 'Max delay within lowest band is '+format(2*bandwidth/f_central[0]*np.amax(delay),'.2f')+' us.'

    dm = f_central**2/k_DM * delay * 1e-6 / 3.086e22 #pc/cm^3
    ddm = f_central**2/k_DM * error *1e-6 / 3.086e22 #pc/cm^3

    dm = np.average(dm, weights = 1/ddm**2, axis=-1)
    ddm = np.sqrt(np.sum(ddm**2, axis=-1) / 3)

    return dm, ddm

f_open = open('filelist.txt','r')
filelist = f_open.read()
f_open.close()
folder = 'dm_2014-06-15_v2/2s/'
bin_factor = 1244

for i in range(14):
    timestamp = filelist.splitlines()[3*i+0][-28:]
    print 'Processing '+timestamp

    delay = np.load(folder+'delay_'+format(bin_factor,'04')+'_'+timestamp)
    error = np.load(folder+'error_'+format(bin_factor,'04')+'_'+timestamp)

    dm, ddm = delay_to_dm(delay,error)
    np.save(folder+'DM_'+format(bin_factor,'04')+'_'+timestamp, dm)
    np.save(folder+'dDM_'+format(bin_factor,'04')+'_'+timestamp, ddm)
