from __future__ import print_function; from __future__ import division; import time;
import matplotlib; matplotlib.use('Agg'); from pylab import *; import sys; import os;
import astropy.units as u; from baseband import mark4; from pulsar.predictor import Polyco;
from pyfftw.interfaces.numpy_fft import rfft, irfft;
_fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 32)), 'planner_effort': 'FFTW_ESTIMATE'}
#_fftargs = {}

# Parameters
DM = 29.11680 * 1.00007 * u.pc / u.cm**3; D = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc;
SR = 32. * u.MHz; dt = (1./SR).to(u.s);
fedge = np.array([311.25, 327.25, 343.25, 359.25])*u.MHz; fref = 359.540 * u.MHz;
epoch = sys.argv[1]; file_num = int(sys.argv[2]);
assert epoch in ('a','b','c','d');
ar_data = 'mountpoint/Ar/gp052%c_ar_no00%02d'%(epoch, file_num);
#ar_data = 'Files/gp052{0}_ar_no00{1:02d}'.format(epoch, file_num);
#polyco = Polyco('packages/scintellometry/scintellometry/trials/evn_14jun/data/polycob1957+20_ao.dat');
polyco = Polyco('polycob1957+20_corrTasc.dat');
N = 2**28; DN = 759*(2**14); ngate = 512; T = 67; #T = 67;

block_length = int((N - DN)/SR.decompose().value);

with mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=32*u.MHz) as fh:
    fh.seek(0); t0 = fh.tell(unit='time');
    offset = 320*(int(1E5*ceil(t0.unix)) - int(1E5*t0.unix));
    fh.seek(offset)
    t00 = fh.tell(unit='time')

f = fedge[0] + rfftfreq(N, dt); dang = D * DM * u.cycle * f * (1./fref - 1./f)**2;
with u.set_enabled_equivalencies(u.dimensionless_angles()): dd_coh1 = np.exp(dang * 1j).conj().astype(np.complex64);
f = fedge[1] + rfftfreq(N, dt); dang = D * DM * u.cycle * f * (1./fref - 1./f)**2;
with u.set_enabled_equivalencies(u.dimensionless_angles()): dd_coh2 = np.exp(dang * 1j).conj().astype(np.complex64);
f = fedge[2] + rfftfreq(N, dt); dang = D * DM * u.cycle * f * (1./fref - 1./f)**2;
with u.set_enabled_equivalencies(u.dimensionless_angles()): dd_coh3 = np.exp(dang * 1j).conj().astype(np.complex64);
del f; del dang;

print('Done Chores\n--- --- ---\nUsing File {0}\n--- --- ---'.format(ar_data));

# Insert dynamic spectrum you wish to weigh by
# Must be perfectly aligned in time and frequency, 
# should probably make general using astropy time and units
dynspec = np.load('b1957_dynspec.py')

first = True

for i in range(T):
    with mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=32*u.MHz, thread_ids=[0, 1, 2, 3, 4, 5]) as fh:
        fh.seek(offset + i*(N - DN));
        t0 = fh.tell(unit='time'); phasepol = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True);
        print('Went to time {0} (i = {1})'.format(t0.isot, i))
        z = fh.read(N)
        print('... Read {0} samples, ended at raw offset {1}.'.format(N, fh.fh_raw.tell()))
        z = rfft(z, axis=0, **_fftargs);
        z[..., (0,1)] *= dd_coh1[:, np.newaxis]; # Dedisperse Band 1!
        z[..., (2,3)] *= dd_coh2[:, np.newaxis]; # Dedisperse Band 2!
        z[..., (4,5)] *= dd_coh3[:, np.newaxis]; # Dedisperse Band 3!
        z = irfft(z, axis=0, **_fftargs)[:-DN]; # This chops off the wraparound
        print('... Dedispersed!');

    # Reshape, FFT, weigh by scintillation pattern, and IFFT
    z = z.reshape(-1, 4000, z.shape[-1])
    z = rfft(z, axis=1, **_fftargs)
    # need to change this line, since dynamic spectrum can go negative
    # sqrt since z is the electric field, dynspec is I = E^2
    z *= np.sqrt(dynspec[i])
    z = irfft(z, axis=1, **_fftargs)
    # Changes over, now have weighted timestream

    z = z*z; #z = z[...,(0,2,4)] + z[...,(1,3,5)];
    npol = z.shape[-1];

    phasepol = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True);
    phase = phasepol(np.arange(z.shape[0]) * dt.to(u.s).value);
    phase -= np.floor(phase[0]); ncycle = np.floor(phase[-1]) + 1;
    iphase = np.remainder(phase*ngate, ngate*ncycle).astype(np.int);

    profile = np.zeros((ncycle, ngate, npol), dtype=np.float32);

    for j in range(npol):
        profile[..., j] = np.bincount(iphase, z[..., j], minlength=ngate*ncycle).reshape(ncycle, ngate);
    icount = np.bincount(iphase, minlength=ngate*ncycle).reshape(ncycle, ngate)

    if first:
        profiles = profile; icounts = icount; first = not first;
    else:
        if icounts[-1,-1] > 0 and icount[0,0] > 0:
            profiles = np.append(profiles, profile, axis=0);
            icounts = np.append(icounts, icount, axis=0);
        else:
            profiles[-1] += profile[0]; icounts[-1] += icount[0];
            profiles = np.append(profiles, profile[1:], axis=0);
            icounts = np.append(icounts, icount[1:], axis=0);

icounts[icounts == 0] = 1
zz = profiles/icounts[...,np.newaxis];
np.save('B1957pol3_{0}g_{1}+{2}s.npy'.format(ngate, t00.isot, T*block_length), zz);
