
from tkinter import END
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavWrite

# Define some global input values
low_freq = 500  # in hz
high_freq = 2500
BW = high_freq - low_freq
fs = 48000 # same rate
sweep_rate = 10 # hz/ms
chirp_length = (BW /  sweep_rate) / 1000. # in seconds


def plot_it(chirp, name, time):
    chirp_size = len(chirp)
    plt.plot(time, chirp)
    plt.title(name)
    plt.show()
    return

def save_wave(name, chirp):
    wavWrite(name + ".wav", fs, chirp)


# Calculate up and down chirp
t_chirp = np.linspace(0, chirp_length, int(chirp_length * fs) )
up_chirp = chirp(t_chirp, f0=low_freq, f1=high_freq, t1=chirp_length, method='linear')
plot_it(up_chirp, "calc chirp", t_chirp)
save_wave("calc chirp", up_chirp)
down_chirp = chirp(t_chirp, f0=high_freq, f1=low_freq, t1=chirp_length, method='linear')
plot_it(down_chirp, "calc chirp", t_chirp)
save_wave("calc chirp", down_chirp)

up_down_chirp = 0.707 * np.concatenate( (up_chirp, down_chirp) )        # WEIRD, but concat seems to want the two arrays as a TUPLE
up_down_chirp_length = 2.0 * chirp_length
t_up_down_chirp = np.linspace(0, up_down_chirp_length, int(up_down_chirp_length * fs) )
plot_it(up_down_chirp, "up down", t_up_down_chirp)
save_wave("upDown", up_down_chirp)


# double the up down
double_chirp = np.concatenate( (up_down_chirp, up_down_chirp) )
double_chirp_length = 2.0 * up_down_chirp_length
t_d = np.linspace(0, double_chirp_length, int(double_chirp_length * fs) )
plot_it(double_chirp, "double", t_d)
save_wave("double", double_chirp)
