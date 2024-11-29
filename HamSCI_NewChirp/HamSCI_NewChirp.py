from operator import truediv
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavWrite
from scipy import signal


# Define some global input values
low_freq = 1000  # in hz
high_freq = 2000
BW = high_freq - low_freq
fs = 48000  # same rate
sweep_rate = 10  # hz/ms
chirp_length = (BW / sweep_rate) / 1000.  # in seconds

# DEBUG functions run if True
DEBUG = True


def remove_DC_normlz(sig):
    sig_RemDC = (sig - np.average(sig))  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< works better with AVERAGE vs MEAN
    sig_RemDC = sig_RemDC / np.max(np.abs(sig_RemDC))
    sig = sig_RemDC
    return (sig)


class chirp_element:
    def __init__(self, low_freq, high_freq, fs, sweep_rate, name):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.sweep_rate = sweep_rate
        self.BW = high_freq - low_freq
        self.length = (BW / sweep_rate) / 1000.  # in seconds
        self.t_chirp = np.linspace(0, chirp_length, int(chirp_length * fs))
        self.chirp_array = chirp(self.t_chirp, f0=low_freq, f1=high_freq, t1=chirp_length, method='linear')
        self.name = name

    def plot_it(self):
        plt.plot(self.t_chirp, self.chirp_array)
        plt.title(self.name)
        plt.show()
        return ("didit")

    def save_wave(self):
        wavWrite(self.name + ".wav", fs, self.chirp_array)


class up_down_chirp_class:
    def __init__(self, up_chirp, down_chirp, name):
        self.up = up_chirp
        self.down = down_chirp
        self.chirp_array = 0.707 * np.concatenate((self.up.chirp_array, self.down.chirp_array))
        self.chirp_length = len(self.chirp_array) / fs
        self.t_chirp = np.linspace(0, self.chirp_length, int(self.chirp_length * fs))
        self.name = name

    def plot_it(self):
        plt.plot(self.t_chirp, self.chirp_array)
        plt.title(self.name)
        plt.show()
        return ("didit")

    def save_wave(self):
        wavWrite(self.name + ".wav", fs, self.chirp_array)


if __name__ == "__main__":

    # create up and down chirp elements
    up_chirp = chirp_element(low_freq, high_freq, 48000, 10, "Up Chirp")
    down_chirp = chirp_element(high_freq, low_freq, 48000, 10, "Down Chirp")
    up_down_chirp = up_down_chirp_class(up_chirp, down_chirp, "Up Down Chirp")
    up_down_chirp.save_wave()

    wavWrite("up chirp.wav", fs, up_chirp.chirp_array)

    if DEBUG:
        up_down_chirp.plot_it()

    # create a chain of up down chirps for a total 8 second signal
    chirp_chain = up_down_chirp.chirp_array
    for i in range(0, 40, 1):
        chirp_chain = np.concatenate((chirp_chain, up_down_chirp.chirp_array))

    fft_chirp_chain = np.fft.fft(chirp_chain)
    freq = np.fft.fftfreq(len(fft_chirp_chain), d=1 / fs)
    plt.plot(freq, np.abs(fft_chirp_chain))
    plt.title("FFT chirp chain")
    plt.show()

    # zero pad the chirp chain for Steve's initial tess
    do_zero_pad = True
    if do_zero_pad:
        chirp_chain_padded = np.zeros(len(chirp_chain) + 48000)
        for i in range(0, len(chirp_chain), 1):
            chirp_chain_padded[i+24000] = chirp_chain[i]

    wavWrite("chirpChain_10hz_8sec_padded.wav", fs, chirp_chain_padded)

    # create a 2 ms delayed version of the chirp chain
    chirp_chain_delayed = np.zeros(len(chirp_chain) + 96)
    for i in range(0, len(chirp_chain), 1):
        chirp_chain_delayed[i + 96] = chirp_chain[i]

    tdoa = np.zeros(len(chirp_chain_delayed))
    for j in range(0, len(chirp_chain), 1):
        tdoa[j] = 0.5 * chirp_chain_delayed[j] + 0.5 * chirp_chain[j]

    wavWrite("tdoa_2ms.wav", fs, tdoa)

    fft_tdoa = np.fft.fft(tdoa)
    freq = np.fft.fftfreq(len(fft_tdoa), d=1 / fs)
    plt.plot(freq, np.abs(fft_tdoa))
    plt.title("FFT tdoa")
    plt.show()

    # look at the envelope of the tdoa signal
    # chirp_end = chirp_start + 8.0  # chirp_length
    # chirp = sig[int(chirp_start * fs_s): int(chirp_end * fs_s)]

    chirp = tdoa

    plt.plot(chirp)
    plt.title("chirp")
    plt.show()


    chirp_analytic = signal.hilbert(chirp)
    chirp_real = np.real(chirp_analytic)
    chirp_imag = np.imag(chirp_analytic)
    chirp_env = np.abs(chirp_analytic)

    chirp_env = chirp_env * chirp_env


    # Try some filtering LP
    wp = 250.
    ws = 1.1 * wp
    gpass = 3.
    gstop = 40.
    N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=fs)
    sos = signal.butter(N, Wn, 'low', fs=fs, output='sos')
    envFiltered = signal.sosfiltfilt(sos, chirp_env)

    # Apply Hanning window
    hanning_window = np.hanning(len(envFiltered))
    windowed_signal = envFiltered  # * hanning_window
    envFiltered = windowed_signal



    # remove the DC offset and normalize the envelope
    chirp_env_DC_removed = remove_DC_normlz(envFiltered)

    plt.plot(chirp_env_DC_removed)
    plt.title("env_dc removed")
    plt.show()



    window = np.hanning(len(envFiltered))
    envFilteredWindowed = window * envFiltered

    # envUNfilteredWindowed = chirp_env * window
    # chirp_fft_UNfiltered = np.fft.fft(envUNfilteredWindowed)
    # f_un = np.fft.fftfreq(envUNfilteredWindowed.size, (1 / fs_s))

    chirp_fft_filtered = (np.fft.fft(envFilteredWindowed  , n=2**18))

    f = np.fft.fftfreq(chirp_fft_filtered.size, (1 / fs))
    fft_time = f / 10000.0
    plt.plot(f, np.abs(chirp_fft_filtered))
    plt.xlim(0,40)
    plt.xlabel("Beat Freq (hz)")
    plt.title("long chirp fft")
    plt.show()