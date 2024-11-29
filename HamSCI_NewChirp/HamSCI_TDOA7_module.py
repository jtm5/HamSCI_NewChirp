import array
from fileinput import filename
from multiprocessing import dummy
import random
import hashlib
from sqlite3 import Row
from sys import path_hooks, path_importer_cache
from tabnanny import check
from tkinter import FIRST
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy import signal
import scipy
from scipy.io import wavfile
import os
import csv
from GLOBALS_config import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import pandas as pd


def get_params(sig_raw, IQ, fs_sig_raw):
    global default_fs
    fs_s = default_fs
    data_type = sig_raw.dtype
    if (data_type != "float32"):
        print(
            "DATA TYPE ERROR IN SIGNAL FILE.......CORRECTING TO FLOAT32...................")
        sig_raw = np.float32(sig_raw)

    number_channels = len(sig_raw.shape)
    if ((number_channels == 2) and ((IQ) == False)):
        sig = sig_raw[:, 0]
        number_channels = 1

    elif ((number_channels == 2) and (IQ == True)):
        sig = sig_raw
        # Add code to demod the I/Q data
        print("add demod code")
        exit()
    else:
        sig = sig_raw

    # If the signal sample rate is different from template, resample it
    if (fs_sig_raw != default_fs):
        print("Sampling rate error............resampling..............")
        t_sig = np.arange(len(sig)) * (1.0 / fs_sig_raw)
        num_resampled_samples = int(len(sig) / fs_sig_raw * default_fs)
        fs_upsampled_sig = default_fs
        upsampled_sig, t_upsampled_sig = scipy.signal.resample(
            sig, num_resampled_samples, t=t_sig)

        # Ok, it's resampled,
        sig = upsampled_sig
        fs_s = fs_upsampled_sig

    toop = (sig, data_type, IQ, number_channels, fs_s)
    return (toop)


def remove_DC_normlz(sig):
    sig_RemDC = (sig - np.average(sig))  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< works better with AVERAGE vs MEAN
    sig_RemDC = sig_RemDC / np.max(np.abs(sig_RemDC))
    sig = sig_RemDC
    return (sig)


def find_chirps_start(sig, fs_s):

    path_data = "D:\\Data\\Ham Radio\\HamSCI_TDOA_2024\Raw Data Recordings\\Eclipse Day Recordings\\Test Signal For N6RFM\\"

    fs_test_sig, test_sig = wavfile.read(
        path_data + "SEQP Test Signal_v2_WA5FRF EL09nn.wav")

    # clip the first chirp out of test_sig , starts at 19.370 secs, .5 sec long

    start = int(19.370 * fs_test_sig)
    end = int(start + 0.5 * fs_test_sig)

    test_chirp = test_sig[start:end]

    cross_corr = signal.correlate(sig, test_chirp)
    # t_cross_corr=np.arange(len(cross_corr)) * 1.0 / fs_s
    # plt.plot(t_cross_corr, cross_corr)
    # plt.title("cross corr find chirps")
    # plt.show()

    peaks = signal.find_peaks(cross_corr, height=700, distance=300)[0]

    chirp_times = (peaks / 48000.) - .5
    # print(chirp_times)

    return (chirp_times)


def hard_limit_sig(sig, limit):
    clipped_sig = np.zeros(len(sig), dtype=np.float32)
    for n in range(1, len(sig), 1):
        if (sig[n] > limit):
            clipped_sig[n] = limit
        elif (sig[n] < -limit):
            clipped_sig[n] = -limit
        else:
            clipped_sig[n] = sig[n]
    return clipped_sig


def find_burst_start(sig, fs_s):

    # EXPERIMENAL CODE TO FIND THE START OF THE PN BURST
    # We know this was unrealiable with anything but perfect signals
    # and the best TX/RX wide filters
    # Will try to use the chirp start to find the PN burst start  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Also will try hard limiting to see if that helps

    # RESULTS:
    # Hard limiting the signal and the test signal burst did not help
    # The PN burst start was not found by the cross correlation method with anyh real reliability
    # The PN burst is not usable for TDOA calculations except with the very best signals - EVEN WITH
    #       having found the exact PN burst start time in reference to the chirp start time

    path_data = "D:\\Data\\Ham Radio\\HamSCI_TDOA_2024\Raw Data Recordings\\Eclipse Day Recordings\\Test Signal For N6RFM\\"

    fs_test_sig, test_sig = wavfile.read(path_data + "SEQP Test Signal_v2_WA5FRF EL09nn.wav")
    # plt.plot(test_sig)
    # plt.show()

    # clip the burst out of test_sig , starts at  14.459 secs, 2.0 sec long
    start = int(14.459 * fs_test_sig)
    end = int(start + 2.0 * fs_test_sig)  # EXPERIMENTAL - try using only the first .621 of burst -
    # proportional to shorting of chirps by NW limits
    # <<<<<<<<<<<<< NO GOOD >>>>>>>>>>>>>>>>>>>>>>>>
    test_burst = test_sig[start:end]
    # plt.plot(test_burst)
    # plt.show()

    start_of_burst_in_sig = find_chirps_start(sig, fs_s)[0] - 4.934  # 4.934 is the time from chirp start to burst start in the test signal
    sig_burst_start = int(start_of_burst_in_sig * fs_s)
    sig_burst = sig[sig_burst_start:sig_burst_start + len(test_burst)]

    # Hard limit the test burst
    clipped_test_burst = hard_limit_sig(test_burst, 0.01)
    # t_clipped_test_burst=np.arange(len(clipped_test_burst)) * 1.0 / fs_s
    # plt.plot(t_clipped_test_burst, clipped_test_burst)
    # plt.title("clipped test burst")
    # plt.show()

    # Hard limit the signal
    clipped_sig_burst = hard_limit_sig(sig_burst, 0.01)
    # t_clipped_sig=np.arange(len(clipped_sig_burst)) * 1.0 / fs_s
    # plt.plot(t_clipped_sig, clipped_sig_burst)
    # plt.title("clipped sig")
    # plt.show()

    cross_corr = signal.correlate(clipped_sig_burst, clipped_test_burst)
    # t_corr = np.arange(len(cross_corr)) * 1.0 / fs_s
    # plt.plot(t_corr, cross_corr)
    # plt.title("cross corr")
    # plt.show()

    return (start_of_burst_in_sig)


def auto_correlate_burst(sig, declare_burst_start, burst_length, fs_s, fn_signal):

    # Create burst copy for auto correlation
    burst_start = int(declare_burst_start * fs_s)  # 26.082  # declare_sig_start #+ 14.459 #14.459 for wa5frf
    # burst_start = int(burst_start * fs_s)
    burst_end = burst_start + int(burst_length * fs_s)
    burst = sig[burst_start: burst_end]

    # Do the correlation
    # [burst_start : burst_end]
    corr_burst = signal.correlate(sig[burst_start: burst_end], burst)
    t_burst = np.arange(len(corr_burst)) * 1.0 / fs_s
    t_burst_max = np.where(np.abs(corr_burst) ==
                           np.max(np.abs(corr_burst)))[0][0]
    string_t_burst_max = "%.5f" % (t_burst_max / fs_s)
    print(f"Time of maximum corr_burst = { (t_burst_max / fs_s):.4f} seconds")

    # Plot result
    t_sig = np.arange(len(sig)) * 1.0 / fs_s
    burst_corr_array = np.zeros(len(t_burst), dtype=np.float32)
    string_corr_max = "%.4f" % t_burst_max  # str(declare_sig_start)
    burst_corr_array[int(t_burst_max)] = 1500.0
    plt.plot(t_burst, corr_burst)
    plt.plot(t_burst, burst_corr_array)
    plt.annotate('Max Correlation = ' + string_t_burst_max, xy=(t_burst_max * fs_s, 1000.0),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.show()
    plt.close()

    # Find the PN burst correlation peaks
    aop_list = []
    aop_list.append(fn_signal)

    copy_of_corr = corr_burst

    array_of_peak_bins, peaks = signal.find_peaks(corr_burst, height=250)
    array_of_peak_times = array_of_peak_bins / fs_s
    print("array of peaks = ", array_of_peak_bins)
    print("time of peaks = ", array_of_peak_times)
    print("corr(arrayof)", corr_burst[array_of_peak_bins])

    ################################################################################################################################
    # The following if is needed to assure that cases with no valid delays are not processed
    ###################################################################################################################################

    if (len(array_of_peak_bins) > 1):

        # the 240.0 is a swag at max delay expected
        if ((len(array_of_peak_bins) > 2) & (array_of_peak_bins[2] < burst_end + 240.0)):
            print("second peak is at = ", array_of_peak_bins[1])
            if (array_of_peak_bins[2] < burst_end + 240.0):
                tdoa = array_of_peak_times[2] - array_of_peak_times[1]
                print("TDOA (msec) = ", tdoa)

                aop_list.append(tdoa)
                for item in array_of_peak_bins:
                    aop_list.append(item)
                # new_row = [filename, array_of_peaks]

                with open('test.csv', 'a', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(aop_list)

            else:
                with open('test.csv', 'a', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(aop_list)

        else:
            aop_list.append(
                ">>>>>>>>>>>>>>>>>>>>>> NO VALID DELAYED PN PEAKS FOUND <<<<<<<<<<<<<<<<<<")
            with open('test.csv', 'a', newline='') as f:
                write = csv.writer(f)
                write.writerow(aop_list)

    else:
        aop_list.append(
            ">>>>>>>>>>>>>>>>>>>>>> NO VALID DELAYED PN PEAKS FOUND <<<<<<<<<<<<<<<<<<")
        with open('test.csv', 'a', newline='') as f:
            write = csv.writer(f)
            write.writerow(aop_list)

    #################################################################################################################################
    # The following if is needed to assure that cases with no valid delays are not processed
    ###################################################################################################################################

    if (len(array_of_peak_bins) > 1):

        # Zoom in to display delay peaks
        start_t = t_burst_max - int(.005 * fs_s)
        end_t = t_burst_max + int(.005 * fs_s)
        fig, axs = plt.subplots(1)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        fig.suptitle("Peak Correlation and Delay (if found)  -  ")
        axs.set_xlabel("Seconds After Signal Start")
        axs.set_ylabel("Correlation Value")
        axs.plot(t_burst[start_t: end_t], corr_burst[start_t: end_t])
        axs.plot(t_burst[start_t: end_t], burst_corr_array[start_t: end_t])
        # plt.annotate('Start of Signal = ' +string_start,xy=(declare_sig_start, 1.0),
        #              arrowprops=dict(facecolor='red', shrink=0.05))

        plt.annotate('Max Correlation = ' + string_t_burst_max, xy=(t_burst_max / fs_s, peaks["peak_heights"][1]),
                     arrowprops=dict(facecolor='red', shrink=0.05))
        plt.annotate('Delay peak = ' + str("%.5f" % (array_of_peak_times[2])), xy=(array_of_peak_times[2], peaks["peak_heights"][2]),
                     arrowprops=dict(facecolor='red', shrink=0.05))

        plt.annotate('TDOA = ' + str("%.5f" % (array_of_peak_times[2] - array_of_peak_times[1])), xy=(array_of_peak_times[1] + .001, 1500.0),
                     arrowprops=dict(facecolor='red', shrink=0.05))

        plt.savefig("test")
        plt.show()


def auto_correlate_chirp(sig, chirp_start, fs_s, fn_signal):   # This code is OBE - see auto_correlate_all_chirps() for current version

    chirp_end = chirp_start + 8.0  # chirp_length
    chirp = sig[int(chirp_start * fs_s): int(chirp_end * fs_s)]

    plt.plot(chirp)
    plt.title("chirp")
    plt.show()


    chirp_analytic = signal.hilbert(chirp)
    chirp_real = np.real(chirp_analytic)
    chirp_imag = np.imag(chirp_analytic)
    chirp_env = np.abs(chirp_analytic)

    # # # plt.plot(chirp_env)
    # # # plt.title("chirp env")
    # # # plt.show()

    chirp_env = chirp_env * chirp_env

    # # # plt.plot(chirp_env)
    # # # plt.title("chirp env squared")
    # # # plt.show()

    # # # chirp_env = remove_DC_normlz(chirp_env)
    # # # plt.plot(chirp_env)
    # # # plt.title("chirp env DC removed")
    # # # plt.show()

    # Try some filtering LP
    wp = 250.
    ws = 1.1 * wp
    gpass = 3.
    gstop = 40.
    N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=fs_s)
    sos = signal.butter(N, Wn, 'low', fs=fs_s, output='sos')
    envFiltered = signal.sosfiltfilt(sos, chirp_env)

    # Apply Hanning window
    hanning_window = np.hanning(len(envFiltered))
    windowed_signal = envFiltered  # * hanning_window
    envFiltered = windowed_signal

    # plt.plot(envFiltered)
    # plt.title("env filtered windowed")
    # plt.show()

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

    chirp_fft_filtered = (np.fft.fft(envFilteredWindowed)) #   , n=2**24))

    f = np.fft.fftfreq(chirp_fft_filtered.size, (1 / fs_s))
    fft_time = f / 10000.0
    plt.plot(f, np.abs(chirp_fft_filtered))
    plt.xlim(0,40)
    plt.xlabel("Beat Freq (hz)")
    plt.title("long chirp fft")
    plt.show()

    return (1)


def auto_correlate_all_chirps(sig, start_of_chirps, fs_s, fn_signal, path_results, waterfall_fn, filename):
    print("start of chirps = ", start_of_chirps)

    #################################################################################################################################
    # EXPERIMENTAL CODE TO PLAY WITH OPTIONS TO IMPROVE RESULTS WITH CROSS CORR OF CHIRPS
    #       CONCLUSION: So far, nothing seems to help enough with cross correlation
    #               The BW issues of ham rigs just mangle the chirps too much
    #################################################################################################################################
    # Get a chirp from the test signal for use as a template
    # path_data = "D:\\Data\\Ham Radio\\HamSCI_TDOA_2024\Raw Data Recordings\\Eclipse Day Recordings\\Test Signal For N6RFM\\"
    # path_data = "D:\\Data\Ham Radio\\HamSCI_TDOA_2024\\Raw Data Recordings\\Eclipse Day Recordings\\N5DUP\\"  # 40M_TX-WA5FRF_RX-N5DUP_318km\\"

    # fs_test_sig, test_sig = wavfile.read(path_data + "SEQP Test Signal_v2_WA5FRF EL09nn.wav")
    # # plt.plot(test_sig)
    # # plt.show()

    # # We know that the first chirp starts at 19.370 secs, .5 sec long, but we only want same len as the global chirp_length
    # chirp_template = test_sig[int(19.370 * fs_test_sig): int((19.370 + chirp_length) * fs_test_sig)]
    # # plt.plot(chirp_template)
    # # plt.title("chirp template")
    # # plt.show()

    rec_time = filename[11:15]  # [18:22]for N5DUP    # filename[11:15] for N6RFM data
    list_of_TDOAs = []  # list to hold the TDOAs for each chirp in a recording

    # NOTE:  below calcs chirps from knowledge of template test signal
    chirp_spacing = 0.75  # time between start of a chirp and start of next chirp
    # time between start of last chirp of first group of 5 and start of next group of chirps
    gap_spacing = 1.11
    array_of_chirps = np.zeros((10), dtype=np.float32)
    array_of_chirps[0] = start_of_chirps
    for i in range(1, 10, 1):
        if (i == 5):
            array_of_chirps[i] = array_of_chirps[i - 1] + \
                gap_spacing  # + .09 * i
        else:
            array_of_chirps[i] = array_of_chirps[i - 1] + \
                chirp_spacing  # + .009 * i

    # print("array of chirps calculated= ", array_of_chirps)

    # now, do auto corr for each of the chirps and add to array
    # first, find out how long the corr is

    test_chirp_start = array_of_chirps[0]
    if TEST_LONG_CHIRP:
        test_chirp_start = 34.95

    # TODO: set up an input parameter for this
    test_chirp_end = test_chirp_start + chirp_length
    chirp = sig[int(test_chirp_start * fs_s): int(test_chirp_end * fs_s)]

    if TEST_LONG_CHIRP:
        plt.plot(chirp)
        plt.title("chirp")
        plt.show()

    # test_corr = signal.correlate(chirp, chirp_template) <<<<----------------------------------------------------------ONLY FOR CROSS CORR!!!!!!!!!!!!!!
    test_corr = signal.correlate(chirp, chirp, mode='same')  # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>############
    corr_len = len(test_corr)
    if TEST_LONG_CHIRP:
        tPlot = np.arange(corr_len) * 1.0 / fs_s
        plt.plot(tPlot - chirp_length / 2.0, test_corr)  # [11000:12500]
        plt.title("test corr")
        plt.show()

    array_of_chirp_corr = np.zeros((10, corr_len), dtype=np.float32)
    for k in range(0, 10, 1):
        chirp_start = array_of_chirps[k]
        chirp_end = chirp_start + chirp_length
        chirp = sig[int(chirp_start * fs_s): int(chirp_end * fs_s)]

        # fig, axs = plt.subplots(2)
        # Plot data on each subplot
        # axs[0].plot(chirp)
        # axs[0].set_title('chirp')
        # axs[1].plot(chirp)
        # axs[1].set_title('chirp template')
        # plt.show()

        corr = signal.correlate(chirp, chirp, mode='same')
        # np.save("test template corr", corr)   #one time only to save a template for chirp auto corr to plot later
        array_of_chirp_corr[k, :] = corr
        t_corr = np.arange(len(corr)) * 1.0 / fs_s
        # plt.plot(t_corr[14962:15100], corr[14962:15100])  # [24000:24200] is for .5 sec chirp,  [14962:15100] is for .312 sec chirp
        # plt.plot(corr, 'orange')
        # plt.title("chirp corr")
        # plt.show()

        # Do an FFT for each chirp
        # pad the chirp - only for the fft - keep normal for ac

        padded_chirp = np.zeros((65536), dtype=np.float32)

        for n in range(0, len(chirp), 1):
            padded_chirp[n + 8200] = chirp[n]  # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????

        # plt.plot(chirp)
        # plt.title("chirp")
        # # plt.show()
        # plt.savefig(path_results + "Chirp" + str(k) + ".png")
        # plt.close()

        # plt.plot(padded_chirp)
        # plt.title("padded chirp")
        # plt.show()

        # ******************************************************************************************************************
        #   THIS NEEDS CAREFUL REVIEWING:
        #       Look at using and not using the padding of the chirp in computing the analytic signal
        #         There seems to be artifacts introdued at FFT ~2.1 hz when padding is used
        #            However!!!  this happened but now is not, I have lost config control of this section of code
        #               Look at the interaction of padding, windowing and DC component removal
        ###################################################################################################################
        ###################################################################################################################

        chirp_analytic = signal.hilbert(chirp)
        chirp_real = np.real(chirp_analytic)
        chirp_imag = np.imag(chirp_analytic)
        chirp_env = np.abs(chirp_analytic)

        # # # plt.plot(chirp_env)
        # # # plt.title("chirp env")
        # # # plt.show()

        chirp_env = chirp_env * chirp_env

        # # # plt.plot(chirp_env)
        # # # plt.title("chirp env squared")
        # # # plt.show()

        # # # chirp_env = remove_DC_normlz(chirp_env)
        # # # plt.plot(chirp_env)
        # # # plt.title("chirp env DC removed")
        # # # plt.show()

        # Try some filtering LP
        wp = 250.
        ws = 1.1 * wp
        gpass = 3.
        gstop = 40.
        N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=fs_s)
        sos = signal.butter(N, Wn, 'low', fs=fs_s, output='sos')
        envFiltered = signal.sosfiltfilt(sos, chirp_env)

        # Apply Hanning window
        hanning_window = np.hanning(len(envFiltered))
        windowed_signal = envFiltered  # * hanning_window
        envFiltered = windowed_signal

        # plt.plot(envFiltered)
        # plt.title("env filtered windowed")
        # plt.show()

        # remove the DC offset and normalize the envelope
        chirp_env_DC_removed = remove_DC_normlz(envFiltered)

        # # # plt.plot(chirp_real)
        # # # plt.title("chirp real")
        # # # plt.show()

        # # # plt.plot(chirp_env)
        # # # plt.title("chirp env")
        # # # plt.show()

        # # # plt.plot(chirp)
        # # # plt.title(filename + "  chirp" + str(k + 1))
        # # # plt.show()
        # # # # plt.savefig(path_results + filename + "Chirp " + str(k+1) + ".jpg")
        # # # plt.close()

        # plt.plot(chirp_env_DC_removed)
        # plt.title(filename + "  chirp" + str(k + 1) + " Env DC removed")
        # plt.show()
        # # plt.savefig(path_results + filename + "Chirp " + str(k+1) + "Env.jpg")
        # plt.close()

        # The below det_filtered was from David Kazdan. Works, OK, but not as well as original
        det_filtered = np.empty(len(envFiltered), dtype=np.float32)
        det_filtered[0] = 0.5
        for i in range(1, len(envFiltered)):
            det_filtered[i] = max(det_filtered[i - 1] * 0.893736, envFiltered[i])

        # det_filtered = remove_DC_normlz(det_filtered)
        # plt.plot(det_filtered)
        # plt.title("det filtered")
        # plt.show()

        # envFiltered = det_filtered

        window = np.hanning(len(envFiltered))
        envFilteredWindowed = window * envFiltered

        # envUNfilteredWindowed = chirp_env * window
        # chirp_fft_UNfiltered = np.fft.fft(envUNfilteredWindowed)
        # f_un = np.fft.fftfreq(envUNfilteredWindowed.size, (1 / fs_s))

        chirp_fft_filtered = (np.fft.fft(envFilteredWindowed, n=2**18))

        # np.save('Test Template FFT',chirp_fft_filtered)  #one time only to save a template for chirp FTT to plot later
        # f = np.fft.fftfreq(chirp_fft_filtered.size, 1 / fs_s)
        # plt.figure(figsize=(12, 8))
        # plt.plot(f, np.abs(chirp_fft_filtered))
        # plt.title('FFT filtered')
        # plt.xlim(0, 32)
        # # plt.ylim(0, 1200)
        # plt.show()

        if (do_FFT_corr_each_chirp):
            f = np.fft.fftfreq(chirp_fft_filtered.size, (1 / fs_s))
            # plt.figure(figsize=(12, 8))
            # plt.plot(f/10000, np.abs(chirp_fft_filtered))
            # plt.title("chirp fft" + str(k))
            # plt.xlim(0, .0032)
            # # plt.ylim(0, 1200)
            # plt.grid(True)
            # plt.xticks(np.arange(0, .0032, .001))
            # plt.show()

            # plt.plot(f, np.abs(chirp_fft_filtered))
            # plt.title("chirp fft" + str(k))
            # plt.show()

            # fig, axs = plt.subplots(2)
            # # set figure size
            # fig.set_figheight(10)
            # fig.set_figwidth(15)
            # axs[0].set_title('Chirp Env FFT')
            # axs[0].plot(f, np.abs(chirp_fft_filtered), 'red', label='Chirp FFT')
            # template_fft = np.load("test template fft.npy")
            # axs[0].fill_between(f, np.abs(template_fft), color='lightgray', label='Zero Delay Template')
            # axs[0].legend()
            # axs[0].set_xlim(0, 32)
            # axs[0].set_ylim(0, 1500)
            # axs[0].grid(True)
            # axs[0].set_xticks(np.arange(0, 32, 1))
            # axs[1].set_title('Chirp Correlation')
            # # Next line is deprecated - was when NOT used corr mode='same
            # # axs[1].plot(t_corr[14962:15100]-.312, np.abs(corr[14962:15100]))  # [24000:24200] is for .5 sec chirp,  [14962:15100] is for .312 sec chirp
            # axs[1].plot(t_corr[11999:12200] - chirp_length / 2.0, (corr[11999:12200]), 'red', label='Chirp Corr')  # [11999:12200] for .5 sec chirp, [7488:7636] for .312 sec chirp
            # # plot the template corr to judge the chirp corr
            # template_corr = np.load("test template corr.npy")
            # axs[1].fill_between(t_corr[11999:12200] - chirp_length / 2.0, (template_corr[11999:12200]), color='lightgray', label='Zero Delay Template')
            # axs[1].legend()
            # axs[1].set_xlim(0.0, .0032)
            # axs[1].set_ylim(0, 1000)
            # axs[1].grid(True)
            # plt.suptitle('Chirp FFT and Correlation\n' + filename + '  Chirp: ' + str(k + 1))
            # plt.show()
            # # plt.savefig(filename + str(k + 1) + '.jpg')
            # plt.close()

            # now, plot the fft and time domain on the same chart
            fft_time = f / 10000.0
            plt.figure(figsize=(18, 10))
            plt.title('Chirp FFT and Correlation\n' + filename + '  Chirp: ' + str(k + 1))

            plt.plot(fft_time, np.abs(chirp_fft_filtered), 'red', label='Chirp FFT')
            plt.plot(t_corr[11999:12200] - chirp_length / 2.0, (corr[11999:12200]), 'green', label='Chirp Corr')  # [11999:12200] for .5 sec chirp, [7488:7636] for .312 sec chirp
            plt.legend()
            plt.xlim(0, .0032)
            plt.ylim(0, 2500)
            plt, plt.xlabel('Time (s) For Correlation or Frequency / 10000 (hz) For FFT')
            plt.ylabel('Magnitude')
            plt.xticks(np.arange(0, .0042, .00025))
            plt.grid(which='both')
            # plt.show()
            plt.savefig(path_results + filename + str(k + 1) + '.jpg')
            # plt.pause(.1)
            plt.close()

        # fig, axs = plt.subplots(2)  # Two vertically stacked subplots
        # axs[0].plot(f_un, np.abs(chirp_fft_UNfiltered))
        # axs[1].plot(f, np.abs(chirp_fft_filtered))
        # plt.suptitle('Unfiltered and Filtered Chirp FFTs')
        # plt.show()

    # this is only for making the sidelobe cancellation template from the test signal:
        # 23jul24 -- the template is for WA5FRF
        # sidelobe_cancel_template = array_of_chirp_corr[0, :]
        # np.save("sidelobe_cancel_template-500ms", sidelobe_cancel_template)

    # use saved cancellation template to zero out sidelobes
    # <<<<<<<<<<<<<<<<<<  BE SURE TO USE THE CORRECT CHIRP LENGTH TEMPLATE  >>>
    if (do_sidelobe_cancellation == True):
        template = np.load('sidelobe_cancel_template-500ms.npy')

    # scale the template to each chirp's max corr value and subtract from that chirp_corr array

        template_max = np.max(template)
        for n in range(0, 10, 1):
            chirp_max = np.max(array_of_chirp_corr[n, :])
            scale_factor = chirp_max / template_max

            print("scale factor = ", scale_factor)

            before = array_of_chirp_corr[n, :]  # 23999:24150])
            # plt.plot(before)
            # plt.title("chirp corr before")
            # plt.show()
            scaled_template = (template[:] * scale_factor)
            # array_of_chirp_corr[n, :] = array_of_chirp_corr[n, :] - scaled_template[:]
            after = array_of_chirp_corr[n, :] - scaled_template[:]  # 23999:24150])

            tslc = np.arange(0, 151 * 1.0 / fs_s, 1.0 / fs_s)
            fig, axs = plt.subplots(3)
            # set figure size
            fig.set_figheight(10)
            fig.set_figwidth(15)
            axs[0].set_title('before and after slc')
            axs[0].plot(tslc, before[23999:24150])
            # axs[0].set_xlim(0, 32)
            # axs[0].set_ylim(0, 2000)
            axs[0].grid(True)
            # axs[0].set_xticks(np.arange(0, 32, 1))
            axs[1].plot(tslc, scaled_template[23999:24150])
            axs[1].set_title('scaled template')
            axs[1].grid(True)
            # axs[0].set_title('Before')
            axs[2].plot(tslc, after[23999:24150])
            axs[2].set_title('After')
            # axs[1].set_xlim(0.5, 0.5031)
            # axs[1].set_ylim(0, 1000)
            axs[2].grid(True)
            # plt.suptitle('Chirp FFT and Correlation\n' + filename + '  Chirp: ' + str(k + 1))
            plt.show()

            array_of_chirp_corr[n, :] = np.copy(after)

    # if (first_recording == True):
    out_header = [filename, 'Auto Correlation Magnitude vs Time']
    with open(path_results + waterfall_fn, 'a') as f:
        write = csv.writer(f)
        write.writerow(' ')
        write.writerow(out_header)

    # we only need one half of the corr results - they are mirror image about t=0
    corr_center = int((corr_len + 1) / 2)
    pruned_corr_array = array_of_chirp_corr[:, corr_center: corr_center + 150]

    # CLIP VALUES BELOW ZERO AND ABOVE 1000 FOR A BETTER PLOT

    for x in range(0, 10, 1):
        for y in range(0, 150, 1):
            if (pruned_corr_array[x, y] < 0.0):
                pruned_corr_array[x, y] = 0.0
            if (pruned_corr_array[x, y] > 1000.0):
                pruned_corr_array[x, y] = 1000.0

    first_corr_array = pruned_corr_array[1, :]

    # plt.plot(first_corr_array)
    # plt.show()

    # Look for TDOA peaks in all 10 chirps
    # Create lists to hold the TDOA values for all chirps of this record
    TDOA_times_list = [filename]
    TDOA_mags_list = [filename]
    for chirp_number in range(0, 10, 1):
        TDOAs, _ = signal.find_peaks(pruned_corr_array[chirp_number, :], height=20)  # 20
        # plt.plot(pruned_corr_array[num, :])
        # plt.plot(TDOAs, pruned_corr_array[num, TDOAs], "x")
        # plt.title(filename)
        # plt.show()

        TDOA_mags = pruned_corr_array[chirp_number, TDOAs]
        TDOA_times = TDOAs * (1 / fs_s)

        tdoa_pairs = np.zeros((len(TDOA_mags), 2), dtype=np.float32)
        for m in range(0, len(TDOA_mags), 1):
            tdoa_pairs[m, 0] = TDOA_times[m]
            tdoa_pairs[m, 1] = TDOA_mags[m]
        # print("tdoa pairs: ", tdoa_pairs)

        # ok, for this chirp, pick the highest mag peak in range of .5 - 1.2 ms
        e_holds = []
        e_layer_tdoas = np.zeros((len(TDOA_mags), 2), dtype=np.float32)

        # The global PEAK_SEARCH_LIMITS sets the limits for automatically finding TDOA peaks
        # 'all' is for from 0.5 ms to 2.8 ms
        # 'low' is for from 0.5 ms to 1.1 ms
        # 'medium' is for from 1.1 ms to 2.2 ms
        # 'high' is for from 2.2 ms to 3.0 ms

        if (PEAK_SEARCH_LIMITS == 'all'):
            low_limit = .0005
            high_limit = .0028

        if (PEAK_SEARCH_LIMITS == 'low'):
            low_limit = .0005
            high_limit = .0011

        if (PEAK_SEARCH_LIMITS == 'medium'):
            low_limit = .0011
            high_limit = .0022

        if (PEAK_SEARCH_LIMITS == 'high'):
            low_limit = .0022
            high_limit = .0030

##########################################################################################################################
##########################################################################################################################
#                            THIS NEEDS A <<VERY>> CAREFUL LOOK
#                            Seeing instances where the max TDOA is not the best when compared
#                                to what I see in the waterfall plots
##########################################################################################################################
##########################################################################################################################

        for i in range(0, len(TDOA_mags), 1):
            if (low_limit < tdoa_pairs[i, 0] <= high_limit):
                if (DEBUG):
                    print("in range")
                e_layer_tdoas[i, :] = tdoa_pairs[i, :]
                e_holds.append(tdoa_pairs[i, :].tolist())
            else:
                if (DEBUG):
                    print("out of range")
        # <<<<<<< WORKING - this holds the elayer stuff for each chirp
        if (DEBUG):
            print("eholds: ", e_holds)

        # find best of the eholds IN THIS CHIRP
        temp_ehold = []
        for k in range(0, len(e_holds), 1):
            max_value = max(max(row) for row in e_holds)
            if (e_holds[k][1] == max_value):
                best_ehold = e_holds[k]
                # print("chirp number = ", chirp_number +
                # 1, "best low TDOA = ", best_ehold)
                # temp = rec_time + ", " + str(chirp_number + 1) + ", " + str(best_ehold[0]) + ", " + str(best_ehold[1])
                # temp_ehold.append((chirp_number, best_ehold[0], best_ehold[1]))

                list_of_TDOAs.append((int(rec_time), chirp_number + 1, best_ehold[0], best_ehold[1]))
                #################################################################################
                # list_of_TDOAs.append((1200, chirp_number + 1, best_ehold[0], best_ehold[1]))  ##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #######################################################################################
                if (DEBUG):
                    print("list of modes : ", list_of_TDOAs)

    # write list of TDOAs to a text file
    # with open(path_results + 'TDOA_list.txt', 'a') as f:
    #     for item in list_of_low_TDOAs:
    #         f.write("%s\n" % item)

    # Now find the CHIRP with the BEST TDOA in the low range for this recording
    test = 0
    best_tdoa = []
    for n in range(0, len(list_of_TDOAs), 1):
        if (list_of_TDOAs[n][3] > test):
            test = list_of_TDOAs[n][3]
            best_tdoa = list_of_TDOAs[n]

    # below list is a global dfined in GLOBALS
    list_of_best_TDOAs.append(list(best_tdoa))

    chirp_times = TDOA_times.tolist()
    TDOA_times_list.append(chirp_times)
    chirp_mags = TDOA_mags.tolist()
    TDOA_mags_list.append(chirp_mags)

    if (do_chirp_scatter_plots):
        for n in range(0, len(TDOA_mags_list[1]), 1):
            plt.scatter(TDOA_mags_list[1][n], TDOA_times_list[1][n])
            plt.title(filename)
            plt.xlim(0.0, 500.)
            plt.ylim(0.0, .0035)
            plt.grid()
            plt.show()

        # Get ready for 3D plots of all chirps for this recording
    len_each_corr_array = len(first_corr_array)
    time_bins = np.zeros((len_each_corr_array), dtype=np.float32)
    for inc in range(0, len(time_bins), 1):
        time_bins[inc] = inc * .020833

    chirp_bins = np.arange(0, 10, 1)

    X, Y = np.meshgrid(time_bins, chirp_bins)
    Z = pruned_corr_array

    # Plot the surface
    # fig = plt.figure()

    if (do_3D_plots):
        fig = plt.figure(figsize=(15, 15))  # Set the figure size
        ax = fig.add_subplot(111, projection='3d')
        # You can choose a different colormap if you like
        ax.plot_surface(X, Y, Z, cmap=cm.viridis)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Chirps')
        ax.set_zlabel('Correlation Mag')
        # ax.set_zlim3d(0, 600.)
        # ax.set_zlim(bottom=0.0)
        ax.set_xticks(np.arange(0, 3.0, .25))
        ax.view_init(30, -72, 0)
        fig.savefig(path_results + filename + '_TEST.jpg')
        plt.show()

    for m in range(0, 1, 1):
        with open(path_results + waterfall_fn, 'a') as f:
            np.savetxt(f, pruned_corr_array, delimiter=",", newline='\n')

    # plt.plot(pruned_corr_array[3])
    # plt.show()

    return (len(array_of_chirps))
