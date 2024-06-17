
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavWrite

# # # Generate a linear chirp from 6 Hz to 1 Hz over 10 seconds
# t = np.linspace(0, 10, 1500)
# w = chirp(t, f0=6, f1=1, t1=10, method='linear')

# # Plot the waveform
# plt.plot(t, w)
# plt.title("Linear Chirp, f(0)=6, f(10)=1")
# plt.xlabel('t (sec)')
# plt.show()

t = np.linspace(0, .5, 24000)
up = chirp(t, f0=1000, f1=2000, t1=0.5, method='linear')

# Plot the waveform
plt.plot(t, up)
plt.title("up")
plt.xlabel('t (sec)')
plt.show()

down = chirp(t, f0=2000, f1=1000, t1=0.5, method='linear')
plt.plot(t, down)
plt.title("down")
plt.xlabel('t (sec)')
plt.show()

# test = np.concatenate((up,down))  #######################################################################################################
# #######################################################################################################################################
# wavWrite("test.wav", 48000, test)

upDown = np.zeros((48000), dtype=np.float64)
# for i in range(0,24000,1):
#     upDown[i] = up[i]
    
# for j in range(0,  24000, 1):
#     upDown[j +24000 - 1] = down[j]

upDown = np.concatenate((up, down))
    
t_comp = np.linspace(0, 1.0, 48000)

plt.plot(t_comp, upDown)
plt.title("UpDown")
plt.xlabel('t (sec)')
plt.show()

wavWrite("upDownChirp.wav", 48000, upDown)
# wavWrite("upDown_24k.wav", 24000, upDown)

twoSecChirp = np.concatenate((upDown, upDown)) * .25
wavWrite("twoSecChirp.wav", 48000, twoSecChirp)