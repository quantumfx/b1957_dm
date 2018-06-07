from __future__ import division
from scipy.optimize import minimize, fmin
import numpy as np
import sys

# Nikhil's code, for the most part
t = np.linspace(0,1,512,endpoint=False)#[:-1]
t += (t[1] - t[0])/2
ppdt = t[1] - t[0]

# sum over polarizations
#meanpp = np.load('timing/pp512.npy').mean(-1)
highpp = np.load('timing/pphi512.npy').mean(-1)
lowpp = np.load('timing/pplo512.npy').mean(-1)

def pp_shift(dt, pp):
    fpp = np.fft.rfft(pp,axis=0)
    fshift = np.exp(-2*np.pi*1j*dt*np.fft.rfftfreq(512, ppdt))
#     return np.fft.irfft(fpp*fshift[:, np.newaxis],axis=0) # summed over polarizations
    return np.fft.irfft(fpp*fshift,axis=0)

def check_mode(p0, pp):
    a, b, c= p0
    cpp = a*lowpp + b*highpp + c
    return ((pp - cpp)**2).sum()

def chi_check_high(X, z, varpp, N):
    dt, a, b = X
    model = a*highpp + b
    model_shift = pp_shift(dt, model)
    return (((z - model_shift)**2)/(varpp/N)).sum()

def chi_check_low(X, z, varpp, N):
    dt, a, b = X
    model = a*lowpp + b
    model_shift = pp_shift(dt, model)
    return (((z - model_shift)**2)/(varpp/N)).sum()

def chi_check_mode(X, z, varpp, N):
    dt, a, b, c = X
    model = a*lowpp + b*highpp + c
    model_shift = pp_shift(dt, model)
    return (((z - model_shift)**2)/(varpp/N)).sum()

def find_chi_error_high(dt, p0, target, z, varpp, N):
    chisq = chi_check_high(list(dt) + list(p0), z, varpp, N)
    return (chisq - target)**2

def find_chi_error_low(dt, p0, target, z, varpp, N):
    chisq = chi_check_low(list(dt) + list(p0), z, varpp, N)
    return (chisq - target)**2

def find_chi_error_mode(dt, p0, target, z, varpp, N):
    chisq = chi_check_mode(list(dt) + list(p0), z, varpp, N)
    return (chisq - target)**2

def which_mode(pnum, modes):
    if modes[pnum//583,0] > modes[pnum//583,1]:
        return 'low'
    elif modes[pnum//583,0] < modes[pnum//583,1]:
        return 'high'
    else:
        print('Fitting both modes... Check what\'s wrong')
        return 'mode'

def delay_to_dm(delay, error):
    c = 299792458 #m/s
    r_e = 2.81794033e-15 #m
    bandwidth = 16e6 #Hz
    f = np.array([311.25e6, 327.25e6, 343.25e6]) #Hz
    f_central = f + bandwidth/2
    k_DM = c * r_e / (2*np.pi)

    print('Max delay within lowest band is '+format(2*bandwidth/f_central[0]*np.amax(delay),'.2f')+' us.')

    dm = f_central**2/k_DM * delay * 1e-6 / 3.086e22 #pc/cm^3
    ddm = f_central**2/k_DM * error *1e-6 / 3.086e22 #pc/cm^3

    dm = np.average(dm, weights = 1/ddm**2, axis=-1)
    #ddm = np.sqrt(np.sum(ddm**2, axis=-1) / 3) #unweighted
    ddm = 1/np.sqrt(np.sum(1/ddm**2,axis=-1))

    return dm, ddm

def find_mode(data, bin_factor_modes = 583):
    # bin the data in 1s, to find the modes
    print('Folding '+format(bin_factor_modes,'03')+' pulses to find mode')
    data_binned = data[: np.shape(data)[0]//bin_factor_modes * bin_factor_modes].reshape(-1,bin_factor_modes, 512, 3).mean(-1)
    data_binned_var = np.var(data_binned, axis=1)
    N = bin_factor_modes
    data_binned = data_binned.sum(1)

    mode_scale = np.zeros((np.shape(data_binned)[0],2))

    for j in range(np.shape(data_binned)[0]):
        X_mode = fmin(chi_check_mode, x0=[0., 0.5, 0.5, 0.], args=(data_binned[j,:], data_binned_var[j,:], N), disp=0)
        mode_scale[j,0] = X_mode[1]
        mode_scale[j,1] = X_mode[2]
    return mode_scale

def fit_delay(bin_factor, data):
    # print("Using Nikhil's code to fit DMs")
    print('Folding '+format(bin_factor,'04')+' pulses to find delays')
    print('Discarding', np.shape(data)[0]-np.shape(data)[0]//bin_factor * bin_factor, 'from', np.shape(data)[0], 'data points to fold into', np.shape(data)[0]//bin_factor, 'points.')
    data_binned = data[: np.shape(data)[0]//bin_factor * bin_factor].reshape(-1,bin_factor, 512, 3)
    data_binned_var = np.var(data_binned, axis=1)
    N = bin_factor
    data_binned = data_binned.sum(1)

    delay = np.zeros((np.shape(data_binned)[0],3))
    error = np.zeros(np.shape(delay))

    err_check = 0.00005

    for j in range(np.shape(data_binned)[0]):
        pnum = j*bin_factor
        print(j, 'out of', np.shape(data_binned)[0], 'bin_factor', bin_factor, 'pnum', pnum)
        fitmode = which_mode(pnum, modes)
        if fitmode == 'low' and bin_factor < 583:
            fchi = chi_check_low
            fchierr = find_chi_error_low
            dof = 512 - 2
            xguess = [0., 1., 0.]
        elif fitmode == 'high' and bin_factor < 583:
            fchi = chi_check_high
            fchierr = find_chi_error_high
            dof = 512 - 2
            xguess = [0., 1., 0.]
        else:
            fchi = chi_check_mode
            fchierr = find_chi_error_mode
            dof = 512 - 3
            xguess = [0., 0.5, 0.5, 0.]
        print(fchi, dof)
        for band in range(3):
            X_mode, F_mode = fmin(fchi, x0=xguess, args=(data_binned[j,:,band], data_binned_var[j,:,band], N), full_output=True, disp=0,maxiter=1500,maxfun=3000)[:2]
            dtmode = X_mode[0]
            p0mode = X_mode[1:]

            dterr0 = minimize(fchierr, x0=dtmode - err_check, args=(p0mode, F_mode * (1 + 1/dof), data_binned[j,:,band], data_binned_var[j,:,band], N), bounds=[(None,dtmode)], method='L-BFGS-B').x[0]
            dterr1 = minimize(fchierr, x0=dtmode + err_check, args=(p0mode, F_mode * (1 + 1/dof), data_binned[j,:,band], data_binned_var[j,:,band], N), bounds=[(dtmode,None)], method='L-BFGS-B').x[0]
            avg_err_mode = (dterr1 - dterr0)/2

            delay[j,band] = dtmode*1607
            error[j,band] = np.abs(avg_err_mode)*1607
        print(j, pnum, fitmode, delay[j], error[j])
    return delay, error

if len(sys.argv) != 4:
    print('Wrong args: fold_dm.py bin_factor fname outfolder')
    exit()

bin_factor = int(sys.argv[1])
fname = sys.argv[2]
outfolder = sys.argv[3]

f_open = open(fname,'r')
filelist = f_open.read().splitlines()
f_open.close()

nikdir = '/mnt/raid-cita/mahajan/Pulsars/Ar_B1957_IndividualPulses/'

#bin_factor = 62
#outfolder = 'dm_2014-06-13/'

# print(filelist.splitlines()[0])

if len(filelist)%3 != 0:
    print('Files not divisible by 3, check filelist')
    exit()

#egr1 is 6
for i in range(len(filelist)//3):

    #print(filelist.splitlines()[3*timestamp+0], filelist.splitlines()[3*timestamp+1], filelist.splitlines()[3*timestamp+2],)
    timestamp = filelist[3*i+0][-28:]
    print('Processing '+timestamp)

    pulsefile = np.dstack( (np.load(nikdir+filelist[3*i+0]).mean(-1),
                            np.load(nikdir+filelist[3*i+1]).mean(-1),
                            np.load(nikdir+filelist[3*i+2]).mean(-1)))[1:-1]
    #print(np.shape(pulsefile))

    pulsefile -= np.median(pulsefile, axis=1, keepdims=True)

    modes = find_mode(pulsefile)

    delay, error = fit_delay(bin_factor, pulsefile)
    np.save(outfolder+'delay_'+format(bin_factor,'04')+'_'+timestamp, delay)
    np.save(outfolder+'error_'+format(bin_factor,'04')+'_'+timestamp, error)

    dm, ddm = delay_to_dm(delay,error)
    np.save(outfolder+'DM_'+format(bin_factor,'04')+'_'+timestamp, dm)
    np.save(outfolder+'dDM_'+format(bin_factor,'04')+'_'+timestamp, ddm)
