
from wave import Wave_write
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
# from scipy.io.wavfile import write as wavWrite
import scipy.io.wavfile as wavefile

fs = 48000
def pseudorandom_from_string(s):
    rng = default_rng(list(s.encode('utf-8')))
    return rng.normal(scale=np.sqrt(1/20), size = int(fs / 10) )

call = 'WA5FRF'
maidenhead = 'EL09nn'
call_static = pseudorandom_from_string(call)
grid_static = pseudorandom_from_string(maidenhead)
static = np.concatenate((call_static, grid_static))

samp_static = np.zeros( (96000), dtype=np.float32)

# Downsample the static to 2400 Hz (the original was ~ 48 kHz) NOTE: Kristina's code downsampled to 4800 Hz
for i in range(0, len(static), 1):
    samp_static[i * 20 : (i * 20) + 20] = static[i] # 20 gives ~ 2500 BW PN

samp_static = 1.5 * samp_static
static_FFT = np.fft.fft(samp_static)
freq = np.fft.fftfreq(len(static_FFT), d=1/fs)
static_corr = np.correlate(samp_static, samp_static, mode='full')

fig, axs = plt.subplots(1)
fig.set_figheight(8)
fig.set_figwidth(15)
fig.suptitle("PN filtered is in Orange ")
axs.set_xlabel("Seconds")
axs.set_ylabel("Normalized Amplitude")
axs.plot(np.linspace(0,1,2 * int(fs / 10) ), static)
axs.plot(np.linspace(0,1,2 * int(fs / 10) ), samp_static[0 : 9600])
plt.show()

np.save('samp_static', samp_static)

# zero pad the burst for Steve's initial tess
do_zero_pad = False
if do_zero_pad:
    burst_padded = np.zeros((len(samp_static) + 48000), dtype=np.float32)
    for i in range(0, len(samp_static), 1):
       burst_padded[i+24000] = samp_static[i]
    samp_static = burst_padded

wavefile.write("BW wide_2 PN.wav", fs, samp_static)

# TEST CODE -- read in filtered burst
fs, filtered_static = wavefile.read('BW_2500_PN_filtered500.wav')



filtered_static_FFT = np.fft.fft(filtered_static)
freq = np.fft.fftfreq(len(filtered_static_FFT), d=1/fs)
# plt.plot(freq, filtered_static_FFT)
# plt.title("FFT samp static__FILTERED")
# plt.show ()

filtered_static_corr = np.correlate(filtered_static, filtered_static, mode='full')
# plt.plot(filtered_static_corr)
# plt.title("Correlation of samp static__FILTERED")
# plt.show()


fig, axs = plt.subplots(2)
# set figure size
fig.set_figheight(10)
fig.set_figwidth(15)
axs[0].set_title('fft before filter')
axs[0].plot(freq,static_FFT)
axs[0].set_xlim(0, 5000)
# axs[0].set_ylim(0, 2000)
axs[0].grid(True)
# axs[0].set_xticks(np.arange(0, 32, 1))
axs[1].plot(freq,filtered_static_FFT)
axs[1].set_title('fft after filter')
axs[1].set_xlim(0, 5000)
axs[1].grid(True)
plt.show()

fig, axs = plt.subplots(2)
# set figure size
fig.set_figheight(10)
fig.set_figwidth(15)
axs[0].set_title('correlation before filter')
axs[0].plot(static_corr)
axs[0].set_xlim(95000, 97000)
# axs[0].set_ylim(0, 2000)
axs[0].grid(True)
# axs[0].set_xticks(np.arange(0, 32, 1))
axs[1].plot(filtered_static_corr)
axs[1].set_title('correlation after filter')
axs[1].set_xlim(95000, 97000)
axs[1].grid(True)
plt.show()




print("END")
