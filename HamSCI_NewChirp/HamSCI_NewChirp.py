import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavWrite

# Define some global input values
low_freq = 1000  # in hz
high_freq = 2000
BW = high_freq - low_freq
fs = 48000 # same rate
sweep_rate = 10 # hz/ms
chirp_length = (BW /  sweep_rate) / 1000. # in seconds

# DEBUG functions run if True
DEBUG = True


class chirp_element:
    def __init__(self, low_freq, high_freq, fs, sweep_rate, name):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.sweep_rate = sweep_rate
        self.BW = high_freq - low_freq
        self.length = (BW / sweep_rate) / 1000.  # in seconds
        self.t_chirp = np.linspace(0, chirp_length, int(chirp_length * fs) )
        self.chirp_array = chirp(self.t_chirp, f0=low_freq, f1=high_freq, t1=chirp_length, method='linear')
        self.name = name
        
    def plot_it(self):
        plt.plot(self.t_chirp, self.chirp_array)
        plt.title(self.name)
        plt.show()
        return("didit")
    
    def save_wave(self):
        wavWrite(self.name + ".wav", fs, self.chirp_array)
        
        
class up_down_chirp_class:
    def __init__(self, up_chirp, down_chirp, name):
        self.up = up_chirp
        self.down = down_chirp
        self.chirp_array = 0.707 * np.concatenate((self.up.chirp_array, self.down.chirp_array)) 
        self.chirp_length = len(self.chirp_array) / fs
        self.t_chirp = np.linspace( 0, self.chirp_length, int( self.chirp_length * fs) )
        self.name = name
        
    def plot_it(self):
        plt.plot(self.t_chirp, self.chirp_array)
        plt.title(self.name)
        plt.show()
        return("didit")
    
    def save_wave(self):
        wavWrite(self.name + ".wav", fs, self.chirp_array)



if __name__ == "__main__":

    # create up and down chirp elements
    up_chirp = chirp_element(low_freq, high_freq, 48000, 10, "Up Chirp")   
    down_chirp = chirp_element(high_freq, low_freq, 48000, 10, "Down Chirp")
    up_down_chirp = up_down_chirp_class(up_chirp, down_chirp, "Up Down Chirp")
    up_down_chirp.save_wave() 
    
    if DEBUG:
        up_down_chirp.plot_it()
        
          
    # create a chain of up down chirps for a total 8 second signal
    chirp_chain = up_down_chirp.chirp_array
    for i in range(0, 100, 1):
        chirp_chain = np.concatenate( (chirp_chain, up_down_chirp.chirp_array) )

    fft_chirp_chain = np.fft.fft(chirp_chain)
    freq = np.fft.fftfreq(len(fft_chirp_chain), d=1/fs)
    plt.plot(freq, np.abs(fft_chirp_chain))
    plt.title("FFT chirp chain")
    plt.show()
    
    # wavWrite("chirpChain.wav", fs, chirp_chain)


