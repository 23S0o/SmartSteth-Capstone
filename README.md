# SmartSteth-Capstone

Welcome to SmartSteth Repository, which includes all the recorded data, source codes, and Instructions for Our Capstone Project "SmartSteth".

A walk through to the available files:

**1. Test.mlapp**
   It's the initial Matlab GUI code that was used in Capstone One to record using our stethoscope Prototype, display, play, and save the recorded Heart Sounds and Lung Sounds as .wav file.
   
________________________________________

**2. SmartSteth.py**
   It's the extended GUI code from Matlab to Python and was used for the data collection stage since we found our methods of recording the audio had better quality in Python than Matlab. The GUI is also able to Record, Play, and save the heart sounds and lung sounds into their specific folders;
   
  **To use it:**
  Make sure to 
  1. Download all the needed libraries.
  2. Save "SmartSteth.py" in a directory with the addition to two folders "Heart Sounds Samples" and "Lung Sounds Samples".
     
________________________________________

**3. modified gui.py**
   It was used to filter and modify each recorded audio, it takes any saved .wav file and allows the user to apply the Band Pass filters and Band Reject filter, and further zero out some specified parts of the signal if the user felt they're not needed.
   
________________________________________

**4. Scalogram.py**
   This is the open-source Scalogram code that generates scalogram images from audio files, the source of this code is: https://www.kaggle.com/code/mistag/extracting-bird-song-signatures-with-wavelets
   
  **To use it:**
  1. Create two folders in the same directory as "Scalogram.py":
        1. "btbwar": save your audio there.
        2. "tmp": where your audio scalogram will be saved
  2. Before running the code, adjust the "file_number = 'MS10.mp3'" based on your intended audio file name and format located in the "btbwar" folder.
     
________________________________________

**5. RPi GUI**
   This is the Raspberry Pi folder format used as the final GUI, which allows the user to record, play, save, and get the classification result of the recording.
  **It consists of:**
  - The main code: "SmartSteh24.py"
  - The heart sound classification code "HS_class.py"
  - The Lung sound classification code "LS_class.py"
  - A temporary audio file "temp.wav" for the purpose of playing the recorded sound.
  - "Heart Sound" folder that contains the classification model and labels, and two folders: "btbwar" that includes the saved audio, and "tmp" that stores the generated scalogram.
  - "Lung Sound" folder that contains the classification model and labels, and the two folders: "btbwar" that includes the saved audio, and "tmp" that stores the generated scalogram.
  
  **Note:** If not used within the original RPi, and the user is using a new RPi:
  1. Install all the needed libraries.
  2. Create a virtual Environment in the "RPi Gui" directory.
  3. After connecting the connect stethoscope prototype, Check the audio device card configuration by running the command "arecord -l", and change it accordingly as instructed on line 48 on "SmartSteth24.py".

________________________________________
     
**6. SmartSteth.txt**
   This file includes a one-drive link that contains the heart sounds and lung sounds recorded by our stethoscope prototype, filtered and modified manually by us, we made it available for any students or researchers interested in implementing any project or research that needs extra data, we're happy to be able to benefit others on the field, and we hope it'll be useful for them.

   
For any Inquiries, e-mail me Samah Osama, at 1079869@students.adu.ac.ae, thank you!



      
