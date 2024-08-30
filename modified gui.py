import tkinter as tk
from tkinter import font as tkFont
from tkinter import filedialog
import sys
import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy import signal



# Create the Tkinter window
root = tk.Tk()
myFont = tkFont.Font(family="Helvetica", size=15, weight="bold")
root.title("SmartSteth - Filters")
root.configure(bg="#b2f0eb")
root.geometry("2000x1200")

ax = plt.subplot(111)  # Create axes object outside the functions
def upload_audio():
    global sample_rate, raw_audio, canvas1, canvas2, selected_audio_file

    selected_audio_file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if selected_audio_file:
        print("Selected audio file:", selected_audio_file)

        # Open the wave file
        wave_file = wave.open(selected_audio_file, "rb")

        # Read the raw audio data
        raw_audio = wave_file.readframes(-1)
        raw_audio = np.frombuffer(raw_audio, dtype=np.int16)  # Convert raw bytes to numpy array
        sample_rate = wave_file.getframerate()

        # Check if the audio has 2 channels
        if wave_file.getnchannels() == 2:
            print("Stereo files are not supported. Use mono files")
            sys.exit(0)


        # Display audio waveform on Canvas
    display_signal()

def display_signal():
    global sample_rate, raw_audio, canvas1
    ax1.clear()
    # Time array for plotting
    time = np.linspace(0, len(raw_audio) / sample_rate, num=len(raw_audio))

    ax1.plot(time, raw_audio, color='#ad305d')
    ax1.set_title(f"Raw audio Signal (Freq: {sample_rate:.2f} Hz)", fontweight='bold')
    ax1.set_xlabel("Time (seconds)", fontweight='bold')
    ax1.set_ylabel("Amplitude", fontweight='bold')
    canvas1.draw()

    display_filtered()

def display_filtered():
    global sample_rate, raw_audio, canvas2, scaled_amplitudes
    ax2.clear()
    # Sample rate and desired cutoff frequencies (in Hz)
    fs = 44100  # Sample rate
    lowcut = 20  # Lower cutoff frequency (in Hz)
    highcut = 200  # Upper cutoff frequency (in Hz)

    # Filter order
    order = 2

    # Generate the filter coefficients
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)

    # Apply the filter to the test signal
    filtered_signal0 = signal.lfilter(b, a, raw_audio)

    lowcut_BRF = 50  # Lower cutoff frequency (in Hz)
    highcut_BRF = 1000  # Upper cutoff frequency (in Hz)
    order_BRF = 2

    c, d = signal.butter(order_BRF, [lowcut_BRF, highcut_BRF], btype='bandstop', fs=fs)

    filtered_signal = signal.lfilter(c, d, filtered_signal0)
    scaled_amplitudes = np.int16(filtered_signal * 10)  # 32767

    time_filtered = np.linspace(0, len(filtered_signal) / sample_rate, num=len(filtered_signal))

    ax2.plot(time_filtered, filtered_signal, color='#ad305d')
    ax2.set_title("Filtered audio Signal", fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontweight='bold')
    ax2.set_ylabel("Amplitude", fontweight='bold')
    canvas2.draw()


def modify_audio():
    global sample_rate, scaled_amplitudes, modified_audio

    # Get start and end times from user input
    start_time = float(start_entry.get())
    end_time = float(end_entry.get())

    # Convert start and end times to frame indices
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    # Replace the audio data within the specified time range with zeros
    modified_audio = scaled_amplitudes.copy()  # Create a copy of the raw audio data
    modified_audio[start_frame:end_frame] = 0

    # Display modified audio waveform on Canvas
    display_modified()

def display_modified():
    global sample_rate, modified_audio, canvas3
    ax3.clear()
    # Time array for plotting
    time = np.linspace(0, len(modified_audio) / sample_rate, num=len(modified_audio))

    ax3.plot(time, modified_audio, color='#ad305d')
    ax3.set_title(f"Modified audio Signal (Freq: {sample_rate:.2f} Hz)", fontweight='bold')
    ax3.set_xlabel("Time (seconds)", fontweight='bold')
    ax3.set_ylabel("Amplitude", fontweight='bold')
    canvas3.draw()

def save_modified():
    global sample_rate, modified_audio
    # Get the filename to save the audio
    save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if save_path:
        # Open the wave file for writing
        with wave.open(save_path, "wb") as wave_file:
            # Set the parameters of the wave file
            wave_file.setnchannels(1)  # Mono audio
            wave_file.setsampwidth(2)  # 16-bit audio
            wave_file.setframerate(sample_rate)
            # Write the audio data to the wave file
            wave_file.writeframes(modified_audio.tobytes())

def save_filtered():
    global sample_rate, scaled_amplitudes
    # Get the filename to save the audio
    save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if save_path:
        # Open the wave file for writing
        with wave.open(save_path, "wb") as wave_file:
            # Set the parameters of the wave file
            wave_file.setnchannels(1)  # Mono audio
            wave_file.setsampwidth(2)  # 16-bit audio
            wave_file.setframerate(sample_rate)
            # Write the audio data to the wave file
            wave_file.writeframes(modified_audio.tobytes())

#current_recording_number = 1

# Create a button for uploading audio
upload_button = tk.Button(root, text="Upload Audio", font=myFont, command=upload_audio, width=15, height=2)
upload_button.place(x=100, y=100)

# Entry widgets for start and end times
start_label = tk.Label(root, text="Start Time (seconds):", font=("Arial", 10))
start_label.place(x=10, y=200)
start_entry = tk.Entry(root, font=("Arial", 15),width=10)
start_entry.place(x=10, y=250)

end_label = tk.Label(root, text="End Time (seconds):", font=("Arial", 10))
end_label.place(x=200, y=200)
end_entry = tk.Entry(root, font=("Arial", 15),width=10)
end_entry.place(x=200, y=250)

# Create a button for modifying audio
modify_button = tk.Button(root, text="Modify Audio", command=modify_audio, width=15, height=2)
modify_button.place(x=400, y=225)

save_m_button = tk.Button(root, text="Save Audio", command=save_modified, width=15, height=2)
save_m_button.place(x=500, y=425)

save_f_button = tk.Button(root, text="Save Audio", command=save_filtered, width=15, height=2)
save_f_button.place(x=500, y=425)

fig1, ax1 = plt.subplots()
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1_widget = canvas1.get_tk_widget().place(x=10, y=570, width=800, height=400)  # Use pack() instead of place()

fig2, ax2 = plt.subplots()
canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2_widget = canvas2.get_tk_widget().place(x=1000, y=120, width=800, height=400)  # Use pack() instead of place()

fig3, ax3 = plt.subplots()
canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3_widget = canvas3.get_tk_widget().place(x=1000, y=570, width=800, height=400)  # Use pack() instead of place()

# Initialize variables
sample_rate = None
raw_audio = None

root.mainloop()