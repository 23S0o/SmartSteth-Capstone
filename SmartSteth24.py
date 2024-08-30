# directory: Pi/project
#  with folders: Heart Sound, Lung Sound
#  with scripts: SmartSteth22.py, HS_class.py, LS_class.py
# Heart Sound: btbwar, tmp, tflite, labels.txt
# Lung Sound: btbwar, tmp, tflite, labels.txt

import os
import wave
import time
import threading
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.font as tkFont
import subprocess
import io
from scipy import signal
import pandas as pd
import cv2
import pyaudio
import matplotlib.image as mpimg
import scipy.fftpack
from scipy.signal import chirp 
import math
from IPython.display import Audio
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
import glob
from scipy.io import wavfile
from PIL import Image
import subprocess

root = tk.Tk()
myFont = tkFont.Font(family="Helvetica", size=15, weight="bold")
root.title("SmartSteth")
#root.configure(bg="#db8e9b")
root.configure(bg="#efbfcf")
root.geometry("1000x300")

ax = plt.subplot(111)  # Create axes object outside the functions


def H_record():
    global scaled_amplitudes, raw_audio
    duration_seconds = 5
    ax.clear()  # Clear previous plot
    command = ["arecord", "-D", "hw:2,0", "-d", str(duration_seconds), "-f", "S16_LE", "-r", "44100", "-t", "wav", "-"]
    
    try:
        # Run the arecord command and capture its output
        output = subprocess.check_output(command)
        print("Recording finished.")
        # return output
 
        raw_audio = np.frombuffer(output, dtype=np.int16)
        
        fs = 44100
        lowcut_BPF = 150  # Lower cutoff frequency (in Hz)
        highcut_BPF = 200  # Upper cutoff frequency (in Hz)
        
        # Filter order
        order_BPF = 2
        
        # Generate the filter coefficients
        b, a = signal.butter(order_BPF, [lowcut_BPF, highcut_BPF], btype='band', fs=fs)
        
        # Apply the filter to the test signal
        filtered_signal_BPF = signal.lfilter(b, a, raw_audio)

        # Band Reject Filter (BRF):
        lowcut_BRF = 50  # Lower cutoff frequency (in Hz)
        highcut_BRF = 1000  # Upper cutoff frequency (in Hz)
        order_BRF = 2
        
        c, d = signal.butter(order_BRF, [lowcut_BRF, highcut_BRF], btype='bandstop', fs=fs)
        
        filtered_signal_BRF = signal.lfilter(c, d, filtered_signal_BPF)
        
        scaled_amplitudes = np.int16(filtered_signal_BRF * 10) #32767
     
        time_scaled = np.linspace(0, len(scaled_amplitudes) / 44100, num=len(scaled_amplitudes))
        ax.plot(time_scaled, scaled_amplitudes , color='#ad305d')
        ax.set_xlabel("Time (seconds)", fontweight='bold')
        ax.set_ylabel("Amplitude", fontweight='bold')
        canvas.draw()
        
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        
def L_record():
    global scaled_amplitudes
    duration_seconds = 5
    ax.clear()  # Clear previous plot
    command = ["arecord", "-D", "hw:2,0", "-d", str(duration_seconds), "-f", "S16_LE", "-r", "44100", "-t", "wav", "-"]
    
    try:
        # Run the arecord command and capture its output
        output = subprocess.check_output(command)
        print("Recording finished.")
        # return output
 
        raw_audio = np.frombuffer(output, dtype=np.int16)
        
        fs = 44100
        lowcut_BPF = 50  # Lower cutoff frequency (in Hz)
        highcut_BPF = 1000  # Upper cutoff frequency (in Hz)
        
        # Filter order
        order_BPF = 2
        
        # Generate the filter coefficients
        b, a = signal.butter(order_BPF, [lowcut_BPF, highcut_BPF], btype='band', fs=fs)
        
        # Apply the filter to the test signal
        filtered_signal_BPF = signal.lfilter(b, a, raw_audio)

        # Band Reject Filter (BRF):
        lowcut_BRF = 20  # Lower cutoff frequency (in Hz)
        highcut_BRF = 200  # Upper cutoff frequency (in Hz)
        order_BRF = 2
        
        c, d = signal.butter(order_BRF, [lowcut_BRF, highcut_BRF], btype='bandstop', fs=fs)
        
        filtered_signal_BRF = signal.lfilter(c, d, filtered_signal_BPF)
        
        scaled_amplitudes = np.int16(filtered_signal_BRF * 10) #32767
     
        time_scaled = np.linspace(0, len(scaled_amplitudes) / 44100, num=len(scaled_amplitudes))
        ax.plot(time_scaled, scaled_amplitudes , color='#ad305d')
        ax.set_xlabel("Time (seconds)", fontweight='bold')
        ax.set_ylabel("Amplitude", fontweight='bold')
        canvas.draw()
        
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        
def HS_save_audio():  # consider it a new button
    global current_recording_number, i, scaled_amplitudes
    exists = True
    i = current_recording_number
    while exists:
        if os.path.exists(f"./Heart Sound/btbwar/Heart Sound {i}.wav"):
            i += 1
        else:
            exists = False

    sound_file = wave.open(f"./Heart Sound/btbwar/Heart Sound {i}.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(2)  # 16-bit audio
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(scaled_amplitudes ))
    sound_file.close()

    current_recording_number = i
    print("HS saved completed")


    def scalogram():
        global i, word

        print("scalogram!")

        def cwt2(data, nv=10, sr=1., low_freq=0.):
            global word
            try:
                data -= np.mean(data)
                n_orig = data.size
                ds = 1 / nv
                dt = 1 / sr
    
            # Pad data symmetrically
                padvalue = n_orig // 2
                x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
                n = x.size
    
            # Define scales
                _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)
                num_scales = wavscales.size
    
            # Frequency vector sampling the Fourier transform of the wavelet
                omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
                omega *= (2 * np.pi) / n
                omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))
    
            # Compute FFT of the (padded) time series
                f = np.fft.fft(x)
    
            # Loop through all the scales and compute wavelet Fourier transform
                psift, freq = waveft(omega, wavscales)
    
            # Inverse transform to obtain the wavelet coefficients.
                cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
                cfs = cwtcfs[:, padvalue:padvalue + n_orig]
                freq = freq * sr
    
                return cfs, freq
            except np.core._exceptions._ArrayMemoryError as e:
                word = 'Please try again'
                print(word)
                update_label()
                return None, None

        def getDefaultScales(n, ds, sr, low_freq):
            nv = 1 / ds
        # Smallest useful scale (default 2 for Morlet)
            s0 = 2

        # Determine longest useful scale for wavelet
            max_scale = n // (np.sqrt(2) * s0)
            if max_scale <= 1:
                max_scale = n // 2
            max_scale = np.floor(nv * np.log2(max_scale))
            a0 = 2 ** ds
            scales = s0 * a0 ** np.arange(0, max_scale + 1)

        # filter out scales below low_freq
            fourier_factor = 6 / (2 * np.pi)
            frequencies = sr * fourier_factor / scales
            frequencies = frequencies[frequencies >= low_freq]
            scales = scales[0:len(frequencies)]

            return s0, ds, scales

        print("get defult!")

        def waveft(omega, scales):
            num_freq = omega.size
            num_scales = scales.size
            wft = np.zeros([num_scales, num_freq])

            gC = 6
            mul = 2
            for jj, scale in enumerate(scales):
                expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)
                wft[jj,] = mul * np.exp(expnt) * (omega > 0)

            fourier_factor = gC / (2 * np.pi)
            frequencies = fourier_factor / scales

            return wft, frequencies

        DB_RANGE = 100  # dynamic range to show in dB
        SR = 22050
        CMAP = 'magma'

        def show_sigx3(d):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # FFT
            N, T = SR, 1. / SR
            x = np.linspace(0.0, int(N * T), N)
            yf = scipy.fftpack.fft(d * np.hamming(len(d)))
            xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
            axes[0].plot(xf, 20 * np.log10(2.0 / N * np.abs(yf[:N // 2])))
            axes[0].set_ylim(-80, 0)
            axes[0].set_title('FFT')
        # Spectrogram
            f, t, Sxx = signal.spectrogram(d, SR)
            axes[1].pcolormesh(t, f, 20 * np.log10(Sxx), shading='auto', cmap=CMAP, vmax=-60, vmin=-60 - DB_RANGE)
            axes[1].set_title('Spectrogram')

            cs, f = cwt2(d, nv=12, sr=SR, low_freq=40)
            axes[2].imshow(20 * np.log10(np.abs(cs)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-DB_RANGE)
            axes[2].set_title('Scaleogram')
            plt.show()

        SR = 22050  # sample rate in Hz

        def rd_file(fname, offset=0, duration=60):
            #data, _ = librosa.load(fname, sr=SR, mono=True, offset=offset, duration=duration)
            SR, data = wavfile.read(fname)
            data = minmax_scale(data - data.mean(), feature_range=(-1, 1))
            return data

        file_number = f'Heart Sound {i}.wav'
        d1 = rd_file(f'./Heart Sound/btbwar/{file_number}')
        Audio(d1, rate=SR)

        cs1, f1 = cwt2(d1, nv=12, sr=SR, low_freq=40)
        if cs1 is None or f1 is None:
                print("Failed to compute cwt2 due to memory error.")
                return
                
    # calculate variance of coefficients
        def calc_var(cs, thres):
            c = 20 * np.log10(np.abs(cs))
            c[c < thres] = 0.
            e = np.var(c, axis=0)
            return e / max(e)

        v1 = calc_var(cs1, -30)


        def mask_sig(n, peaks, sr=22050, dur=0.1):
            mask = np.zeros(n)
            subm = int(sr * dur * 0.5)
            if len(peaks > 0):
                for i in range(len(peaks)):
                    mask[max(peaks[i] - subm, 0): min(peaks[i] + subm, n)] = 1
            return mask

    # peak detection + gliding window
        peaks, _ = find_peaks(v1, prominence=0.3)
        m = mask_sig(len(v1), peaks, SR, 0.3)

        def get_mask(vdata, prom=0.2, dur=0.2, sr=22050):
            peaks, _ = find_peaks(vdata, prominence=prom)
            return mask_sig(len(vdata), peaks, sr, dur)

        def get_regions(mask, sr, species, filename):
            regions = scipy.ndimage.find_objects(scipy.ndimage.label(mask)[0])
            regs = []
            for r in regions:
                dur = round((r[0].stop - r[0].start) / sr, 3)
                regs.append([r[0].start, r[0].stop, dur, species, filename])
            return pd.DataFrame(regs, columns=['Start', 'End', 'Duration', 'Species', 'File'])

        mask = get_mask(v1, prom=0.3, dur=0.2, sr=SR)
    
        df = get_regions(mask, SR, 'btbwar', f'{file_number}')
        df.head(6)

        df = df[df.Duration >= 1.0]
        df = df.reset_index(drop=True)
        df

        def img_resize(cs, w=512, h=512, log=True, lthres=-30):
            buf = io.BytesIO()
            if log == True:
                plt.imsave(buf, 20 * np.log10(np.abs(cs)), cmap=CMAP, format='png', vmax=0, vmin=lthres)
            else:
                plt.imsave(buf, np.abs(cs), cmap=CMAP, format='png')
            buf.seek(0)
            img_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    # Parameters:
    #   filename: mp3-file
    #   voices: # of scales per octave
    #   sr: sampling frequency (Hz)
    #   low_freq: low freq cutoff (Hz)
    #   thres: scaleogram threshold (dB)
    #   prom: peak detect prominence
    #   peakdur: peak extension (s)
    #   sigthres: smallest signature detection to process (s)
    #   siglen: length of output signature (s)
    #   img_size: output image size
    #   outdir: output directory

        def scaleo_extract(filename, voices=12, sr=22050, low_freq=40, thres=-30, prom=0.3,
                           peakdur=0.3, sigthres=1, siglen=2, img_size=512, outdir='.'):
            d = rd_file(filename)
            cs, _ = cwt2(d, nv=voices, sr=sr, low_freq=low_freq)  # wavelet transform
            v = calc_var(cs, thres)  # coefficient variance
            peaks, _ = find_peaks(v, prominence=prom)
            m = mask_sig(len(v), peaks, sr=sr, dur=peakdur)  # create signal mask
            df = get_regions(m, sr, filename.split('/')[-2], filename.split('/')[-1])
            df = df[df.Duration >= sigthres]  # filter out insignificant signatures
            df = df.reset_index(drop=True)
            print("scaleo_extract")
            if len(df) > 0:
                for i in range(len(df)):
                    img = img_resize(cs[:, df.Start[i]:df.Start[i] + siglen * sr],
                                     w=img_size, h=img_size, log=True, lthres=thres)
                    fn = df.Species[i] + '-' + filename.split('/')[-1].split('.')[-2] + "-{:03d}.jpg".format(i)
                    cv2.imwrite(outdir + '/' + fn, img)
            return df

        CURRENT_DIR = os.getcwd()

        flist = [f'./Heart Sound/btbwar/{file_number}']

        for filename in flist:
            scaleo_extract(os.path.join(CURRENT_DIR, filename), outdir='./Heart Sound/tmp')

        images = glob.glob('./Heart Sound/tmp' + "/Heart Sound/btbwar*.jpg")
        print("done!")
        plt.figure(figsize=(20, 20))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.subplot(columns + 1, columns, i + 1)
            plt.imshow(mpimg.imread(image))
            plt.axis('off')

        def classification():
            print("classification!")
            global i, word
            command = f"""
            cd ~
            cd project
            export IMAGE_PATH='./Heart Sound/tmp/btbwar-Heart Sound {i}-000.jpg'
            env/bin/python HS_class.py
            """
            
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            output, error = process.communicate()
            
            out = output.decode().split()
            if not out:
                word = "Please try again"
                print(word)
                update_label()
            else:    
                word = out[-1]
                print(word)
                update_label()
            
            
            # parts = output.decode().split()
            # word = parts[-1]
            # update_label()
            # print(word)
           
        
        classification() 
    
    scalogram()

def LS_save_audio():  # consider it a new button
    global current_recording_number, i, scaled_amplitudes
    exists = True
    i = current_recording_number
    while exists:
        if os.path.exists(f"./Lung Sound/btbwar/Lung Sound {i}.wav"):
            i += 1
        else:
            exists = False

    sound_file = wave.open(f"./Lung Sound/btbwar/Lung Sound {i}.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(2)  # 16-bit audio
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(scaled_amplitudes ))
    sound_file.close()

    current_recording_number = i
    print("LS saved completed")

    def scalogram():
        global i

        print("scalogram!")

        def cwt2(data, nv=10, sr=1., low_freq=0.):
            global word
            try:
                data -= np.mean(data)
                n_orig = data.size
                ds = 1 / nv
                dt = 1 / sr
    
            # Pad data symmetrically
                padvalue = n_orig // 2
                x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
                n = x.size
    
            # Define scales
                _, _, wavscales = getDefaultScales(n_orig, ds, sr, low_freq)
                num_scales = wavscales.size
    
            # Frequency vector sampling the Fourier transform of the wavelet
                omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
                omega *= (2 * np.pi) / n
                omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))
    
            # Compute FFT of the (padded) time series
                f = np.fft.fft(x)
                print(f)
    
            # Loop through all the scales and compute wavelet Fourier transform
                psift, freq = waveft(omega, wavscales)
    
            # Inverse transform to obtain the wavelet coefficients.
                cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
                cfs = cwtcfs[:, padvalue:padvalue + n_orig]
                freq = freq * sr
                
    
                return cfs, freq
            except np.core._exceptions._ArrayMemoryError as e:
                word = 'Please try again'
                print(word)
                update_label()
                return None, None

       

        def getDefaultScales(n, ds, sr, low_freq):
            nv = 1 / ds
        # Smallest useful scale (default 2 for Morlet)
            s0 = 2

        # Determine longest useful scale for wavelet
            max_scale = n // (np.sqrt(2) * s0)
            if max_scale <= 1:
                max_scale = n // 2
            max_scale = np.floor(nv * np.log2(max_scale))
            a0 = 2 ** ds
            scales = s0 * a0 ** np.arange(0, max_scale + 1)

        # filter out scales below low_freq
            fourier_factor = 6 / (2 * np.pi)
            frequencies = sr * fourier_factor / scales
            frequencies = frequencies[frequencies >= low_freq]
            scales = scales[0:len(frequencies)]

            return s0, ds, scales

        print("get defult!")

        def waveft(omega, scales):
            num_freq = omega.size
            num_scales = scales.size
            wft = np.zeros([num_scales, num_freq])

            gC = 6
            mul = 2
            for jj, scale in enumerate(scales):
                expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)
                wft[jj,] = mul * np.exp(expnt) * (omega > 0)

            fourier_factor = gC / (2 * np.pi)
            frequencies = fourier_factor / scales

            return wft, frequencies

        DB_RANGE = 100  # dynamic range to show in dB
        SR = 22050
        CMAP = 'magma'

        def show_sigx3(d):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # FFT
            N, T = SR, 1. / SR
            x = np.linspace(0.0, int(N * T), N)
            yf = scipy.fftpack.fft(d * np.hamming(len(d)))
            xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
            axes[0].plot(xf, 20 * np.log10(2.0 / N * np.abs(yf[:N // 2])))
            axes[0].set_ylim(-80, 0)
            axes[0].set_title('FFT')
        # Spectrogram
            f, t, Sxx = signal.spectrogram(d, SR)
            axes[1].pcolormesh(t, f, 20 * np.log10(Sxx), shading='auto', cmap=CMAP, vmax=-60, vmin=-60 - DB_RANGE)
            axes[1].set_title('Spectrogram')
            cs, f = cwt2(d, nv=12, sr=SR, low_freq=40)
            axes[2].imshow(20 * np.log10(np.abs(cs)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-DB_RANGE)
            axes[2].set_title('Scaleogram')
            plt.show()

        SR = 22050  # sample rate in Hz

        def rd_file(fname, offset=0, duration=60):
            #data, _ = librosa.load(fname, sr=SR, mono=True, offset=offset, duration=duration)
            SR, data = wavfile.read(fname)
            data = minmax_scale(data - data.mean(), feature_range=(-1, 1))
            return data

        file_number = f'Lung Sound {i}.wav'
        d1 = rd_file(f'./Lung Sound/btbwar/{file_number}')
        Audio(d1, rate=SR)

        cs1, f1 = cwt2(d1, nv=12, sr=SR, low_freq=40)
        if cs1 is None or f1 is None:
                print("Failed to compute cwt2 due to memory error.")
                return

    # calculate variance of coefficients
        def calc_var(cs, thres):
            c = 20 * np.log10(np.abs(cs))
            c[c < thres] = 0.
            e = np.var(c, axis=0)
            return e / max(e)

        v1 = calc_var(cs1, -30)


        def mask_sig(n, peaks, sr=22050, dur=0.1):
            mask = np.zeros(n)
            subm = int(sr * dur * 0.5)
            if len(peaks > 0):
                for i in range(len(peaks)):
                    mask[max(peaks[i] - subm, 0): min(peaks[i] + subm, n)] = 1
            return mask

    # peak detection + gliding window
        peaks, _ = find_peaks(v1, prominence=0.3)

        m = mask_sig(len(v1), peaks, SR, 0.3)


        def get_mask(vdata, prom=0.2, dur=0.2, sr=22050):
            peaks, _ = find_peaks(vdata, prominence=prom)
            print("get mask!")
            return mask_sig(len(vdata), peaks, sr, dur)

        
        def get_regions(mask, sr, species, filename):
            regions = scipy.ndimage.find_objects(scipy.ndimage.label(mask)[0])
            regs = []
            for r in regions:
                dur = round((r[0].stop - r[0].start) / sr, 3)
                regs.append([r[0].start, r[0].stop, dur, species, filename])
            return pd.DataFrame(regs, columns=['Start', 'End', 'Duration', 'Species', 'File'])
        mask = get_mask(v1, prom=0.3, dur=0.2, sr=SR)
        df = get_regions(mask, SR, 'btbwar', f'{file_number}')
        df.head(6)

        df = df[df.Duration >= 1.0]
        df = df.reset_index(drop=True)
        df

        def img_resize(cs, w=512, h=512, log=True, lthres=-30):
            buf = io.BytesIO()
            if log == True:
                plt.imsave(buf, 20 * np.log10(np.abs(cs)), cmap=CMAP, format='png', vmax=0, vmin=lthres)
            else:
                plt.imsave(buf, np.abs(cs), cmap=CMAP, format='png')
            buf.seek(0)
            img_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        def scaleo_extract(filename, voices=12, sr=22050, low_freq=40, thres=-30, prom=0.3,
                           peakdur=0.3, sigthres=1, siglen=2, img_size=512, outdir='.'):
            d = rd_file(filename)
            cs, _ = cwt2(d, nv=voices, sr=sr, low_freq=low_freq)  # wavelet transform
            v = calc_var(cs, thres)  # coefficient variance
            peaks, _ = find_peaks(v, prominence=prom)
            m = mask_sig(len(v), peaks, sr=sr, dur=peakdur)  # create signal mask
            df = get_regions(m, sr, filename.split('/')[-2], filename.split('/')[-1])
            df = df[df.Duration >= sigthres]  # filter out insignificant signatures
            df = df.reset_index(drop=True)
            print("scaleo extract")
            if len(df) > 0:
                for i in range(len(df)):
                    img = img_resize(cs[:, df.Start[i]:df.Start[i] + siglen * sr],
                                     w=img_size, h=img_size, log=True, lthres=thres)
                    fn = df.Species[i] + '-' + filename.split('/')[-1].split('.')[-2] + "-{:03d}.jpg".format(i)
                    cv2.imwrite(outdir + '/' + fn, img)
            else:
                print("expected error #1")        
            return df

        CURRENT_DIR = os.getcwd()

        flist = [f'./Lung Sound/btbwar/{file_number}']

        for filename in flist:
            scaleo_extract(os.path.join(CURRENT_DIR, filename), outdir='./Lung Sound/tmp')

        images = glob.glob('./Lung Sound/tmp' + "/Lung Sound/btbwar*.jpg")
        print("done!")
        plt.figure(figsize=(20, 20))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.subplot(columns + 1, columns, i + 1)
            plt.imshow(mpimg.imread(image))
            plt.axis('off')
            
        def classification():
            print("classification!")
            global i, word
            command = f"""
            cd ~
            cd project
            export IMAGE_PATH='./Lung Sound/tmp/btbwar-Lung Sound {i}-000.jpg'
            env/bin/python LS_class.py
            """
            
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            output, error = process.communicate()
            
            out = output.decode().split()
            if not out:
                word = "Please try again"
                print(word)
                update_label()
            else:    
                word = out[-1]
                print(word)
                update_label()

        
        classification()     

    scalogram()
    
def play_audio():
    threading.Thread(target=play, args=(audio_frames,)).start()

def play(audio_frames):
    global raw_audio
    # Write the byte data to a temporary wave file
    with wave.open('temp.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(44100)
        wf.writeframes(raw_audio)
    
    # Use the aplay command to play the audio file
    command = "aplay temp.wav"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    
    if error:
        print("Error playing audio:", error.decode())

    
def update_label():
    result.delete("1.0", tk.END)
    result.insert(tk.END, word)
    

# GUI Components


recording = tk.BooleanVar()
recording.set(False)

audio_frames = []

current_recording_number = 1

word = ""

button_record = tk.Button(root, text="Rec HS",font=("Arial",10, "bold"), command=H_record,width=5,height=2, bg="#c46876")
button_record.place(x=10,y=40)

button_record = tk.Button(root, text="Rec LS",font=("Arial",10, "bold"), command=L_record,width=5,height=2, bg="#db7081")
button_record.place(x= 80,y=40)


button_play = tk.Button(root, text="Play", font=("Arial", 10,"bold"), command=play_audio, width=15,height=2, bg="#9a4657")
button_play.place(x=10, y=90)
HS_save = tk.Button(root, text="Heart Sound", font=("Arial", 10,"bold"), command=HS_save_audio, width=15, height=2, bg="#c46876")
HS_save.place(x=10, y=140)

LS_save = tk.Button(root, text="Lung Sound", font=("Arial", 10,"bold"), command=LS_save_audio, width=15, height=2, bg="#db7081")
LS_save.place(x=10, y=190)

entry_duration_label = tk.Label(root, text="SmartSteth", font=("Arial", 15, "bold"), bg="#efbfcf", fg="black")
entry_duration_label.place(x=20,y=10)

classification = tk.Label(root, text="Classification: ", font=("Arial",10, "bold"), bg="#efbfcf", fg="black")
classification.place(x=160, y=210)

result = tk.Text(root, width=20, height=1)
result.place(x=270,y=210)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget().place(x=160,y=10, width=300, height=180)

update_label()

root.mainloop()
