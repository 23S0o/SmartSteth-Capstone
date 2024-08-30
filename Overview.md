# SmartSteth-Capstone

Welcome to SmartSteth Repository, which include all the recorded data, source codes, and Instructions for Our Capstone Project "SmartSteth".

A walk through to the available files:

**1. Test.mlapp**
      It's the initial Matlap GUI code that was used in Capstone one to record using our stethoscope Prototype, display, play, and save the recorded Heart Sounds and Lung          Sounds as .wav file.

____________________________

**2. SmartSteth.py**
      It's the extened GUI code from Matlab to Python, and was used for the data collection stage since we found our methods of recording the audio had better qualit in            python than Matlab. The GUI is also able to Record, Play, and save the heart sounds and lung sounds into their specific folders;

      **To use it:**
      Make sure to to 
      1. Download all the needed libraries.
      2. Save "SmartSteth.py" in a directory with addition to two folders "Heart Sounds Samples" and "Lung Sounds Samples".

____________________________

**3. modified gui.py**
      It was used to filter and modify each recorded audio, it takes any saved .wav file and allow the user to apply the Band Pass filters and Band Reject filter, and further zero out some specified parts of the signal if the user felt they're not needed.

____________________________

**4. Scalogram.py**
      This is the open-source Scalogram code that generate scalogram images from audio files, the original source of this code is:
      https://www.kaggle.com/code/mistag/extracting-bird-song-signatures-with-wavelets

      **To use it:**
      2. Create two folder in the same directory as "Scalogram.py":
            1. "btbwar": save your audio there.
            2. "tmp": where your audio scalogram will be saved
      2. Before running the code, adjust the "file_number = 'MS10.mp3'" based on your intended audio file name and format located on the "btbwar" folder.

____________________________

**5. RPi GUI**
      This is the Raspberry Pi folder format used as the the final GUI, where it allow the user to record, play, save, and get the classification result of the recording.

      **It consists of:**
      - The main code: "SmartSteh24.py"
      - The heart sound classification code "HS_class.py"
      - The Lung sound classification code "LS_class.py"
      - A temporary audio file "temp.wav" for the purpose of playing the recorded sound.
      - "Heart Sound" folder that contain the classification model and labels, and two folders: "btbwar" that includes the saved audio, and "tmp" that stores the generated           scalogram.
      - "Lung Sound" folder that contain the classification model and labels, and the two folders: "btbwar" that includes the saved audio, and "tmp" that stores the                   generated scalogram.
      
      **Note:** if not used whithin the original RPi, and the user is using a new RPi:
      1. Install all the needed libraries.
      2. Create a virtual Environment in the "RPi Gui" directory.
      3. After connecting the connect the stethoscope prototyp, Check the audio device card configration by running the command "arecord -l", and change it accordingly as             instructed on line 48 on "SmartSteth24.py". 

For any Inquiries, e-mail me at 1079869@students.adu.ac.ae, thank you! 

      
