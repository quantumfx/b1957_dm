#from __future__ import division
from scipy.optimize import minimize, fmin
import numpy as np
import sys

t = np.linspace(0,1,512,endpoint=False)#[:-1]
t += (t[1] - t[0])/2
ppdt = t[1] - t[0]

# sum over polarizations
#meanpp = np.load('timing/pp512.npy').mean(-1)
highpp  = np.load('timing/pphi512.npy').mean(-1)
lowpp   = np.load('timing/pplo512.npy').mean(-1)

highpp_f = np.fft.rfft(highpp)
lowpp_f  = np.fft.rfft(lowpp)
freqs    = np.fft.rfftfreq(512, ppdt)

#import GP list
egr1gp = np.load('GPlist/egr1GPlist.npy')
egr2gp = np.load('GPlist/egr2GPlist.npy')


def off_gates(max_gate):
    #compute relative off gates
    #max_gate0 = highpp.argmax()
    relative_offgates = np.concatenate( (np.arange(0,100), np.arange(290,400) ) ) + max_gate
    #print(max_gate)
    return relative_offgates%512

def chi_check_high_f(X, z_f, z_var):
    dt, a = X
    num   = (z_f - a * highpp_f * np.exp(-1j*2*np.pi*freqs*dt))[1:] # only consider k>0
    chisq = np.sum( np.abs(num)**2 ) / (z_var * 256)
    return chisq

def chi_check_high_jac(X, z_f, z_var):
    dt, a = X
    num = (z_f * highpp_f.conj() * np.exp(1j*2*np.pi*freqs*dt)) # only consider k != 0
    dchisqdt = -a*np.sum( (1j*2*np.pi*freqs*(num-num.conj())) ).real/(z_var*256)
    dchisqda = np.sum( (2*a*highpp_f*highpp_f.conj() - num - num.conj())[1:] ).real / (z_var*256)
    return np.array([dchisqdt, dchisqda])

def chi_check_low_f(X, z_f, z_var):
    dt, a = X
    num   = (z_f - a * lowpp_f * np.exp(-1j*2*np.pi*freqs*dt))[1:] # only consider k>0
    chisq = np.sum( np.abs(num)**2 ) / (z_var * 256)
    return chisq

def chi_check_low_jac(X, z_f, z_var):
    dt, a = X
    num = (z_f * lowpp_f.conj() * np.exp(1j*2*np.pi*freqs*dt)) # only consider k != 0
    dchisqdt = -a*np.sum( (1j*2*np.pi*freqs*(num-num.conj())) ).real/(z_var*256)
    dchisqda = np.sum( (2*a*lowpp_f*lowpp_f.conj() - num - num.conj())[1:] ).real / (z_var*256)
    return np.array([dchisqdt, dchisqda])

def chi_check_mode_f(X, z_f, z_var):
    dt, a, b = X
    num   = (z_f - (a * lowpp_f + b * highpp_f) * np.exp(-1j*2*np.pi*freqs*dt))[1:] # only consider k>0
    chisq = np.sum( np.abs(num)**2 ) / (z_var * 256)
    return chisq

def multi_min(fun, x0, args, methods, jac, bounds):
    fval = np.zeros(len(methods))
    xmin = np.zeros((len(methods),len(x0)))
    for i in range(len(methods)):
        if jac:
            out_full = minimize(fun = fun, x0 = x0, args = args, method = methods[i], jac = jac, bounds = bounds)
        else:
            out_full = minimize(fun = fun, x0 = x0, args = args, method = methods[i], bounds = bounds)
        fval[i]   = out_full.fun
        xmin[i,:] = out_full.x
    #print('method:',methods[np.argmin(fval)])
    return xmin[np.argmin(fval),:]

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

    dm  = f_central**2/k_DM * delay * 1e-6 / 3.086e22 #pc/cm^3
    ddm = f_central**2/k_DM * error * 1e-6 / 3.086e22 #pc/cm^3

    dm  = np.average(dm, weights = 1/ddm**2, axis=-1)
    #ddm = np.sqrt(np.sum(ddm**2, axis=-1) / 3) #unweighted
    ddm = 1/np.sqrt(np.sum(1/ddm**2,axis=-1))

    return dm, ddm

def find_mode(data, bin_factor_modes = 583):
    # bin the data in ~1s, to find the modes
    print('Folding '+format(bin_factor_modes,'03')+' pulses to find mode and max gates.')
    data_binned = data[: np.shape(data)[0]//bin_factor_modes * bin_factor_modes].reshape(-1,bin_factor_modes, 512, 3)
    #guessed offgates, taking into account delays
    offgates = np.concatenate( (np.arange(50,100), np.arange(360,410)) )

    # warning caused by variance of offgates in GPs, all is okay
    #data_binned_var = data_binned[:,:,offgates,:].var(axis=2).sum(1).sum(-1)/3
    data_binned_var = np.nansum(np.nansum(np.nanvar(data_binned[:,:,offgates,:], 2), 1), -1)/3

    #data_binned = data_binned.sum(1).mean(-1)
    data_binned = np.nanmean(np.nansum(data_binned, 1), -1)

    mode_scale = np.zeros((np.shape(data_binned)[0],2))
    max_gates  = np.zeros(np.shape(data_binned)[0], dtype=int)
    for j in range(np.shape(data_binned)[0]):
        z_f   = np.fft.rfft(data_binned[j,:])
        z_var = data_binned_var[j]
        dtmode, *p0mode = fmin(chi_check_mode_f, x0=[0., 0.5, 0.5], args=(z_f, z_var), disp = 0, maxiter = 1500, maxfun = 3000)
        #X_mode = fmin(chi_check_mode_f, x0=[0., 0.5, 0.5, 0.], args=(data_binned[j,:], data_binned_var[j,:], N), disp=0)
        mode_scale[j,0] = p0mode[0]
        mode_scale[j,1] = p0mode[1]
        max_gates[j] = int(dtmode*512)

    return mode_scale, max_gates

def fit_delay(bin_factor, data):
    print('Folding '+format(bin_factor,'04')+' pulses to find delays')
    print('Discarding', np.shape(data)[0]-np.shape(data)[0]//bin_factor * bin_factor, 'from', np.shape(data)[0], 'data points to fold into', np.shape(data)[0]//bin_factor, 'points.')
    data_binned_temp = data[: np.shape(data)[0]//bin_factor * bin_factor].reshape(-1,bin_factor, 512, 3)
    #offgates = np.concatenate( np.arange(10,100), np.arange(310,410) )

    #data_binned = data_binned_temp.sum(1)
    data_binned = np.nansum(data_binned_temp, 1)

    delay = np.zeros((np.shape(data_binned)[0],3))
    error = np.zeros(np.shape(delay))

    #minimization methods
    methods = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP']
    bounds  = ((None,None),(0,None))

    for j in range(np.shape(data_binned)[0]):
        pnum = j*bin_factor
        #print(j, 'out of', np.shape(data_binned)[0], 'bin_factor', bin_factor, 'pnum', pnum)
        fitmode = which_mode(pnum, modes)
        offgates = off_gates(max_gates[pnum//583])
        #print(offgates)
        #data_binned_var = data_binned_temp[j,:,offgates,:].var(axis=2).sum(1)

        if fitmode == 'low' and bin_factor < 583:
            fchi = chi_check_low_f
            fchi_jac = chi_check_low_jac
            pp_f = lowpp_f
            dof = 512 - 2
            xguess = [0., 1.]
        elif fitmode == 'high' and bin_factor < 583:
            fchi = chi_check_high_f
            fchi_jac = chi_check_high_jac
            pp_f = highpp_f
            dof = 512 - 2
            xguess = [0., 1.]
        else:
            fchi = chi_check_mode_f
            fchi_jac = False
            dof = 512 - 3
            xguess = [0., 0.5, 0.5]
        print('Using', fchi, 'with', dof, 'degrees of freedom.')
        for band in range(3):
            z_f   = np.fft.rfft(data_binned[j,:,band])
            #numpy funky dimension switching??? data_binned_temp[0,:,offgates,0] has dimension (offgates.size,data_binned_temp.shape[1])...
            #z_var = data_binned_temp[j,:,offgates,band].var(0).sum(0)
            z_var = np.nansum(np.nanvar(data_binned_temp[j,:,offgates,band], 0), 0)
            #dtmode, p0mode = fmin(fchi, x0=xguess, args=(z_f, z_var), disp = 0, maxiter = 1500, maxfun = 3000)
            dtmode, p0mode = multi_min(fun = fchi, x0 = xguess, jac = fchi_jac, args = (z_f, z_var), methods = methods, bounds = bounds)
            # add offset
            p0mode = np.append(p0mode, (z_f[0] - p0mode * pp_f[0]).real/512)
            dterr = np.sqrt( (z_var*256)/p0mode[0] / np.sum( ((2*np.pi*freqs)**2*(z_f*pp_f.conj()*np.exp(1j*2*np.pi*freqs*dtmode)+z_f.conj()*pp_f*np.exp(-1j*2*np.pi*freqs*dtmode)))[1:] ).real )

            #print(fchi((dtmode,p0mode[0]),np.fft.rfft(data_binned[j,:,band]+p0mode[1]),z_var))
            #print(fchi((dtmode+dterr,p0mode[0]),np.fft.rfft(data_binned[j,:,band]+p0mode[1]),z_var))

            delay[j,band] = dtmode*1607
            error[j,band] = np.abs(dterr)*1607
        #print('IF                 :', band)
        print('Index              :', j)
        print('Start pulse number :', pnum)
        print('Bin factor         :', bin_factor)
        print('Mode               :', fitmode)
        print('Delay IF0          :', delay[j,0], '+-', error[j,0])
        print('Delay IF1          :', delay[j,1], '+-', error[j,1])
        print('Delay IF2          :', delay[j,2], '+-', error[j,2])
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
    timestamp = filelist[3*i+0][-28:-9]
    print('Processing '+timestamp)

    pulsefile = np.dstack( (np.load(nikdir+filelist[3*i+0]).mean(-1),
                            np.load(nikdir+filelist[3*i+1]).mean(-1),
                            np.load(nikdir+filelist[3*i+2]).mean(-1)))[1:-1]
    #print(np.shape(pulsefile))

    pulsefile -= np.median(pulsefile, axis=1, keepdims=True)

    #ignore GPs, -1 to account for ignoring the first pulse:
    if timestamp == '2014-06-15T06:36:50':
        print('We\'re in egr1, ignoring GPs...')
        pulsefile[egr1gp-1, ...] = np.nan
    if timestamp == '2014-06-15T06:46:02':
        print('We\'re in egr2, ignoring GPs...')
        pulsefile[egr2gp-1, ...] = np.nan

    modes, max_gates = find_mode(pulsefile)

    delay, error = fit_delay(bin_factor, pulsefile)
    np.save(outfolder+'delay_'+format(bin_factor,'04')+'_'+timestamp, delay)
    np.save(outfolder+'error_'+format(bin_factor,'04')+'_'+timestamp, error)

    dm, ddm = delay_to_dm(delay,error)
    np.save(outfolder+'DM_'+format(bin_factor,'04')+'_'+timestamp, dm)
    np.save(outfolder+'dDM_'+format(bin_factor,'04')+'_'+timestamp, ddm)
