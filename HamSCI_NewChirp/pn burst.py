
from wave import Wave_write
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.io.wavfile import write as wavWrite

fs = 48000
def pseudorandom_from_string(s):
    rng = default_rng(list(s.encode('utf-8')))
    return rng.normal(scale=np.sqrt(1/20), size = int(fs / 10) )

# %% [markdown]
# The results of this function should be repeatable no matter how many times you run this function, or restart the notebook kernel.

# %%
# pseudorandom_from_string(call)

# # %%
# pseudorandom_from_string(call)

# %%

call = 'WA5FRF'
maidenhead = 'EL09nn'
call_static = pseudorandom_from_string(call)
grid_static = pseudorandom_from_string(maidenhead)
static = np.concatenate((call_static, grid_static))

# %%
plt.plot(np.linspace(0,1,2 * int(fs / 10) ), static)
plt.show()

samp_static = np.zeros( (96000), dtype=np.float32)
for i in range(0, len(static), 1):
    samp_static[i * 10 : (i * 10) + 10] = static[i]

plt.plot(np.linspace(0,1,2 * fs ), samp_static)
plt.title("sample static")
plt.show()

static_FFT = np.fft.fft(samp_static)
freq = np.fft.fftfreq(len(static_FFT), d=1/fs)
plt.plot(freq, static_FFT)
plt.title("FFT samp static")
plt.show ()

# for i in range(0, 100, 1):
#     # print("static = ", static[i], "    samp_static = " , samp_static[i : i + 10])
#     print("samp_static = ", samp_static[i])

fig, axs = plt.subplots(1)
fig.set_figheight(10)
fig.set_figwidth(15)
fig.suptitle("Received Signal After DCO and Normalization  -  ")
axs.set_xlabel("Seconds")
axs.set_ylabel("Normalized Amplitude")
axs.plot(np.linspace(0,1,2 * int(fs / 10) ), static)
axs.plot(np.linspace(0,1,2 * int(fs / 10) ), samp_static[0 : 9600])
plt.show()

np.save('samp_static', samp_static)
wavWrite("BW wide_2 PN.wav", fs, samp_static)


dummy = 0