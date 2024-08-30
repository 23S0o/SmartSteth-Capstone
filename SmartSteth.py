import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter.font as tkFont

# Creating the GUI window
root = tk.Tk()
myFont = tkFont.Font(family="Helvetica", size=15, weight="bold")
root.title("SmartSteth")
root.configure(bg="#b2f0eb")
root.geometry("2000x1200")
ax = plt.subplot(111)  # Create axes object outside the functions

def start_recording():
    # Getting the duration from the entry field
    duration_seconds = float(entry_duration.get())
    # Start recording
    threading.Thread(target=record, args=(duration_seconds,)).start()

def record(duration_seconds):
    global recording
    ax.clear()  # Clear previous signal plotted on the canvas
    audio_frames.clear()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    # Specifying the duration of the recording based on the user input
    start = time.time()
    while (time.time() - start) < duration_seconds:
        data = stream.read(1024)
        audio_frames.append(data)

        # Time counter
        passed = time.time() - start
        secs = passed % 60
        mins = passed // 60
        hours = mins // 60
        label.config(text=f"{int(hours):02d}:{int(mins):02d},{int(secs):02d}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # After recording, run the function for displaying the recorded signal
    display_signal()

def display_signal():
    if not audio_frames:
        print("No recorded audio to display.")
        return

    ax.clear()
    # Combining the recorded audio frames to form one signal
    signal = np.frombuffer(b"".join(audio_frames), dtype=np.int16)
    # Specifying the displayed time data
    time_axis = np.linspace(0, len(signal) / 44100, num=len(signal))

    # Plotting the recorded audio
    ax.plot(time_axis, signal, color='#ad305d')
    ax.set_title("Recorded Audio Signal", fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontweight='bold')
    ax.set_ylabel("Amplitude", fontweight='bold')
    canvas.draw()

def HS_save_audio():
    global current_recording_number
    exists = True
    i = current_recording_number
    while exists:
        if os.path.exists(f"Heart Sounds Samples\Heart Sound {i}.wav"):
            i += 1
        else:
            exists = False

    sound_file = wave.open(f"Heart Sounds Samples\Heart Sound {i}.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(2)  # 16-bit audio
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(audio_frames))
    sound_file.close()

    # Clear the audio frames after saving
    audio_frames.clear()
    current_recording_number = i

def LS_save_audio():  # consider it a new button
    global current_recording_number
    exists = True
    i = current_recording_number
    while exists:
        if os.path.exists(f"Lung Sounds Samples\Lung Sound {i}.wav"):
            i += 1
        else:
            exists = False

    sound_file = wave.open(f"Lung Sounds Samples\Lung Sound {i}.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(2)  # 16-bit audio
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(audio_frames))
    sound_file.close()

    audio_frames.clear()
    current_recording_number = i

def play_audio():
    threading.Thread(target=play, args=(audio_frames,)).start()

def play(frames):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)
    stream.write(b"".join(frames))
    stream.stop_stream()
    stream.close()
    audio.terminate()

# GUI Parameters
recording = tk.BooleanVar()
recording.set(False)

audio_frames = []
current_recording_number \
    = 1

# GUI Components
SS_label = tk.Label(root, text="SmartSteth", font=("Gill Sans MT", 40, "bold"), bg="#b2f0eb")
SS_label.place(x=840, y=80)

button_record = tk.Button(root, text="Start Recording", font=("Arial", 17, "bold"), command=start_recording, width=15, height=2)
button_record.place(x=400, y=220)

button_play = tk.Button(root, text="Play", font=("Arial", 17, "bold"), command=play_audio, width=15, height=2)
button_play.place(x=880, y=220)

HS_save = tk.Button(root, text="Heart Sound", font=("Arial", 17, "bold"), command=HS_save_audio, width=15, height=2)
HS_save.place(x=1200, y=220)

LS_save = tk.Button(root, text="Lung Sound", font=("Arial", 17, "bold"), command=LS_save_audio, width=15, height=2)
LS_save.place(x=1530, y=220)

label = tk.Label(root, text="00:00:00", font=("Arial", 17, "bold"), bg="#b2f0eb")
label.place(x=700, y=250)

entry_duration_label = tk.Label(root, text="Duration (seconds):", font=("Arial", 17, "bold"), bg="#b2f0eb")
entry_duration_label.place(x=100, y=220)

entry_duration = tk.Entry(root, font=("Arial", 15),width=20)
entry_duration.place(x=100, y=265)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget().place(x=550, y=370, width=900, height=550)  # Use pack() instead of place()

root.mainloop()