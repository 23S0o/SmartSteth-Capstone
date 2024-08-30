import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import io
import librosa
#import warnings
#warnings.filterwarnings('ignore') # get rid of librosa warnings
import scipy.fftpack
from scipy import signal
from scipy.signal import chirp
import math
from IPython.display import Audio
from sklearn.preprocessing import minmax_scale
import os
from scipy.signal import find_peaks
import glob

# specify your file number
# Get the file number from the user
#file_number = input("Please enter the file number: ")
file_number = 'MS10.mp3'
#file_number = input("Please enter the file name/number: ")
# 327
# 1, 7, 17, 19, 20, 21, 24, 28, 30, 32, 33, 37, 38, 40, 41, 43, 45, 49, 53, 55, 56, 57, 58, 59
# Continous Wavelet Transform with Morlet wavelet
# Original code by Alexander Neergaard, https://github.com/neergaard/CWT
#
# Parameters:
#   data: input data
#   nv: # of voices (scales) per octave
#   sr: sampling frequency (Hz)
#   low_freq: lowest frequency (Hz) of interest (limts longest scale)
def cwt2(data, nv=10, sr=1., low_freq=0.):
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

DB_RANGE = 100 # dynamic range to show in dB
SR = 22050
CMAP = 'magma'

def show_sigx3(d):
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    # FFT
    N, T = SR, 1./SR
    x = np.linspace(0.0, int(N*T), N)
    yf = scipy.fftpack.fft(d*np.hamming(len(d)))
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    axes[0].plot(xf,  20*np.log10(2.0/N *np.abs(yf[:N//2])))
    axes[0].set_ylim(-80,0)
    axes[0].set_title('FFT')
    # Spectrogram
    f, t, Sxx = signal.spectrogram(d, SR)
    axes[1].pcolormesh(t, f, 20*np.log10(Sxx), shading='auto', cmap=CMAP, vmax=-60, vmin=-60-DB_RANGE)
    axes[1].set_title('Spectrogram')
    # CWT
    #cs, f = calc_cwt(d)
    cs, f = cwt2(d, nv=12, sr=SR, low_freq=40)
    axes[2].imshow(20*np.log10(np.abs(cs)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-DB_RANGE)
    axes[2].set_title('Scaleogram')
    plt.show()


SR = 22050 # sample rate in Hz

def rd_file(fname, offset=0, duration=60):
    data, _ = librosa.load(fname, sr=SR, mono=True, offset=offset, duration=duration)
    data = minmax_scale(data-data.mean(), feature_range=(-1,1))
    return data

# load the audio file
#d1 = rd_file('XC139608.wav')
#d1 = rd_file(f'./btbwar/Heart Sound {file_number}.wav')
d1 = rd_file(f'./btbwar/{file_number}')
#d1 = rd_file(f'./btbwar/{file_number}.mp3')
Audio(d1, rate=SR)



cs1, f1 = cwt2(d1, nv=12, sr=SR, low_freq=40)
#plt.figure(figsize = (16,4))
#plt.imshow(20*np.log10(np.abs(cs1)), cmap=CMAP, aspect='auto', norm=None, vmax=0, vmin=-30)
#plt.show()


# calculate variance of coefficients
def calc_var(cs, thres):
    c = 20*np.log10(np.abs(cs))
    c[c < thres] = 0.
    e = np.var(c, axis=0)
    return e / max(e)

#fig, axes = plt.subplots(2, 1, figsize=(16,4))
v1 = calc_var(cs1, -30)
#axes[0].plot(v1)
#plt.show()

def mask_sig(n, peaks, sr=22050, dur=0.1):
    mask = np.zeros(n)
    subm = int(sr*dur*0.5)
    if len(peaks > 0):
        for i in range(len(peaks)):
            mask[max(peaks[i]-subm, 0): min(peaks[i]+subm, n)] = 1
    return mask

#fig, axes = plt.subplots(2, 1, figsize=(16,4))
# peak detection + gliding window
peaks, _ = find_peaks(v1, prominence=0.3)
#axes[0].plot(v1)
#axes[0].plot(peaks, v1[peaks], "x")
m = mask_sig(len(v1), peaks, SR, 0.3)
#axes[1].plot(m)
#plt.show()

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
#df = get_regions(mask, SR, 'btbwar', f'Heart Sound {file_number}.wav')
df = get_regions(mask, SR, 'btbwar', f'{file_number}')
#df = get_regions(mask, SR, 'btbwar', f'{file_number}.mp3')
df.head(6)

df = df[df.Duration >= 1.0]
df = df.reset_index(drop=True)
df

def img_resize(cs, w=512, h=512, log=True, lthres=-30):
    buf = io.BytesIO()
    if log == True:
        plt.imsave(buf, 20*np.log10(np.abs(cs)), cmap=CMAP, format='png', vmax=0, vmin=lthres)
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
    cs, _ = cwt2(d, nv=voices, sr=sr, low_freq=low_freq) # wavelet transform
    v = calc_var(cs, thres) # coefficient variance
    peaks, _ = find_peaks(v, prominence=prom)
    m = mask_sig(len(v), peaks, sr=sr, dur=peakdur) # create signal mask
    df = get_regions(m, sr, filename.split('/')[-2], filename.split('/')[-1])
    df = df[df.Duration >= sigthres] # filter out insignificant signatures
    df = df.reset_index(drop=True)
    if len(df) > 0:
        for i in range(len(df)):
            img = img_resize(cs[:,df.Start[i]:df.Start[i]+siglen*sr],
                             w=img_size, h=img_size, log=True, lthres=thres)
            fn = df.Species[i]+'-'+filename.split('/')[-1].split('.')[-2]+"-{:03d}.jpg".format(i)
            cv2.imwrite(outdir+'/'+fn, img)
    return df

CURRENT_DIR = os.getcwd()

#flist = [f'./btbwar/Heart Sound {file_number}.wav']
flist = [f'./btbwar/{file_number}']
#flist = [f'./btbwar/{file_number}.mp3']

for filename in flist:
    scaleo_extract(os.path.join(CURRENT_DIR, filename), outdir='./tmp')

images = glob.glob('./tmp' + "/btbwar*.jpg")
plt.figure(figsize=(20,20))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.subplot(columns + 1, columns, i + 1)
    plt.imshow(mpimg.imread(image))
    plt.axis('off')