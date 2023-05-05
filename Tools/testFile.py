# %%
import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
import matplotlib.pyplot as plt
from IPython.display import SVG, Audio
from keras.layers import Dropout
from multiprocessing import Process
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
import tensorflow_datasets as tfds
from zipfile import ZipFile
import keras.backend as K
import joblib
import shutil 
import glob
import gdown
from keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %%
# Update: Original Dataset source seems to have been taken down

# Please download the GTZAN Dataset from Kaggle:
# https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

# Drag the zip file into the root of this project! The notebook will handle the rest.

# %%
rootDir = ".."

# %%
# Unzip the dataset ZIP and format directory
if "datasets" not in os.listdir(".."):
    for file in os.listdir(".."):
        if file.endswith(".zip"):
            d0 = os.path.join(rootDir, file)
            with ZipFile(d0, 'r') as zObject:
                zObject.extractall(path="..")

            if "datasets" not in os.listdir(".."):
                for f in os.listdir(f"{rootDir}/Data/"):
                    if f == "genres_original":
                        d1 = os.path.join(f"{rootDir}/Data", f)
                        newPath =  os.path.join(f"{rootDir}/Data/","datasets")
                        os.rename(d1,newPath)
                        shutil.move(newPath, "..") 
                        os.mkdir("../datasets/genres30sec")
                        for wavDir in os.listdir("../datasets"):
                            d2 = os.path.join("../datasets", wavDir)
                            if wavDir != "genres30sec":
                                shutil.move(d2, "../datasets/genres30sec") 
                                      
                shutil.rmtree("../Data")
                # os.remove(f"../{d0}")
                
                # jazz.00054.wav is corrupted. We replace the file with the new one at this link:
                # https://drive.google.com/file/d/14ZV9Wf-6pr32PIRT_ziuvFvh9tQsHpQ4/view
                
                url = "https://drive.google.com/u/0/uc?id=14ZV9Wf-6pr32PIRT_ziuvFvh9tQsHpQ4&export=download"
                output = '../datasets/genres30sec/jazz/jazz.00054.wav'
                gdown.download(url, output, quiet=False)


# %%
# Set to appropriate path if this file is moved
dataDir = f"{rootDir}/datasets"

# %%
# Split 30 sec audio to 3 sec

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
genres = genres.split()

if not os.path.exists(f'{dataDir}/genres3sec'):
  os.makedirs(f'{dataDir}/genres3sec')

  if len(os.listdir(os.path.join(f'{dataDir}/genres3sec'))) == 0:
    for g in genres:
      path_audio = os.path.join(f'{dataDir}/genres3sec',f'{g}')
      os.makedirs(path_audio)
    
  from pydub import AudioSegment
  i = 0
  for g in genres:
    #if len(os.listdir(os.path.join(f'{dataDir}/audio3sec/',f"{g}"))) == 0:
      j=0
      print(f"{g}")
      for filename in os.listdir(os.path.join(f'{dataDir}/genres30sec',f"{g}")):
        song  =  os.path.join(f'{dataDir}/genres30sec/{g}',f'{filename}')
        j = j+1
        for w in range(0,10):
          i = i+1
          #print(i)
          t1 = 3*(w)*1000
          t2 = 3*(w+1)*1000
          newAudio = AudioSegment.from_wav(song)
          new = newAudio[t1:t2]
          new.export(f'{dataDir}/genres3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")

# %%
# Load Example frile for Example of MFCC & MelSpectrogram
y,sr = librosa.load(f'{dataDir}/genres3sec/blues/blues10.wav',duration=3)


# %%
# MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
fig = plt.Figure(figsize=(13, 130))
canvas = FigureCanvas(fig)
p = librosa.display.specshow(mfccs)

# %%
# MelSpectrogram
mels = librosa.feature.melspectrogram(y=y,sr=sr)
fig = plt.Figure(figsize=(13, 130))
canvas = FigureCanvas(fig)
p = librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))

# %%
# MFCCs for 3 Sec Clips

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
genres = genres.split()

  
if not os.path.exists(f'{dataDir}/mfccs3sec'):
    os.makedirs(f'{dataDir}/mfccs3sec')
    os.makedirs(f'{dataDir}/mfccs3sec/train')
    os.makedirs(f'{dataDir}/mfccs3sec/test')

    for g in genres:
        path_train = os.path.join(f'{dataDir}/mfccs3sec/train',f'{g}')
        path_test = os.path.join(f'{dataDir}/mfccs3sec/test',f'{g}')
        os.makedirs(path_train)
        os.makedirs(path_test)



    # Single-Core Implmentation [SLOW]
    
    # for g in genres:
    #     if len(os.listdir(os.path.join(f'{dataDir}/mfccs3sec/train/',f"{g}"))) == 0:
    #         j = 0
    #         print(g)
    #         for filename in os.listdir(os.path.join(f'{dataDir}/genres3sec',f"{g}")):
    #             song  =  os.path.join(f'{dataDir}/genres3sec/{g}',f'{filename}')
    #             j = j+1
    #             y,sr = librosa.load(song,duration=3)
    #             #############################################################################
    #             mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
    #             fig = plt.Figure(figsize=(13, 130))
    #             canvas = FigureCanvas(fig)
    #             p = librosa.display.specshow(mfccs)
    #             #########################################################
    #             plt.savefig(f'{dataDir}/mfccs3sec/train/{g}/{g+str(j)}.png')
    

    # Multi-Core Implmentation (Equal For Loop distribution) [Faster]
    
    # def task(g,dd):
    #     j = 0
    #     for filename in os.listdir(os.path.join(f'{dd}/genres3sec',f"{g}")):
    #         song  =  os.path.join(f'{dd}/genres3sec/{g}',f'{filename}')
    #         j = j+1
    #         y,sr = librosa.load(song,duration=3)
    #         #############################################################################
    #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
    #         fig = plt.Figure(figsize=(13, 130))
    #         canvas = FigureCanvas(fig)
    #         p = librosa.display.specshow(mfccs)
    #         #########################################################
    #         plt.savefig(f'{dataDir}/mfccs3sec/train/{g}/{g+str(j)}.png')
            

    # processes = [Process(target=task, args=(g,dataDir,)) for g in genres]

    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()
    



    # # Multi-Core Implmentation (Batches) [Fastest]
    def task(dd, g,j,filename):
        song  =  os.path.join(f'{dd}/genres3sec/{g}',f'{filename}')
        y,sr = librosa.load(song,duration=3)
        #############################################################################
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
        fig = plt.Figure(figsize=(13, 130))
        canvas = FigureCanvas(fig)
        p = librosa.display.specshow(mfccs)
        #########################################################
        plt.savefig(f'{dd}/mfccs3sec/train/{g}/{g+str(j)}.png')
        plt.close()
        

    batch_size = 32
    counter = 0
    processes = []

    for g in genres:
        print(g)
        
        wavFiles = os.listdir(os.path.join(f'{dataDir}/genres3sec',f"{g}"))
        
        # processes = np.append(processes, joblib.delayed(task)(dataDir,g,j,filename))
        processes = [ joblib.delayed(task)(dataDir,g,itr,wavFiles[itr]) for itr in range(len(wavFiles))  ]
        _ = joblib.Parallel(n_jobs=batch_size)(processes)
            

# %%
# MFCCs for 30 Sec Clips

# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
# genres = genres.split()

# if not os.path.exists(f'{dataDir}'):
#   os.makedirs(f'{dataDir}/')
  
# if not os.path.exists(f'{dataDir}/mfccs30sec'):
#   os.makedirs(f'{dataDir}/mfccs30sec')
#   os.makedirs(f'{dataDir}/mfccs30sec/train')
#   os.makedirs(f'{dataDir}/mfccs30sec/test')

#   for g in genres:
#     path_train = os.path.join(f'{dataDir}/mfccs30sec/train',f'{g}')
#     path_test = os.path.join(f'{dataDir}/mfccs30sec/test',f'{g}')
#     os.makedirs(path_train)
#     os.makedirs(path_test)

# # Multi-Core Implmentation (Batches) Fastest
# def task(g,j,filename):
#     song  =  os.path.join(f'{dataDir}/genres/{g}',f'{filename}')
#     y,sr = librosa.load(song,duration=3)
#     #############################################################################
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
#     fig = plt.Figure(figsize=(13, 130))
#     canvas = FigureCanvas(fig)
#     p = librosa.display.specshow(mfccs)
#     #########################################################
#     plt.savefig(f'{dataDir}/mfccs30sec/train/{g}/{g+str(j)}.png')

# batch_size = 32
# counter = 0
# processes = []

# for g in genres:
#     j = 0
#     print(g)
#     for filename in os.listdir(os.path.join(f'{dataDir}/genres',f"{g}")):
#         if counter <= batch_size:
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
#         else:
#             for process in processes:
#                 process.start()
#             for process in processes:
#                 process.join()
#             counter = 0
#             processes = []
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
            
# for process in processes:
#     process.start()
# for process in processes:
#     process.join()

# %%
# Mel Spectrogram for 3 Sec Clips

# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
# genres = genres.split()

# if not os.path.exists(f'{dataDir}'):
#   os.makedirs(f'{dataDir}/')
  
# if not os.path.exists(f'{dataDir}/spectrograms3sec'):
#   os.makedirs(f'{dataDir}/spectrograms3sec')
#   os.makedirs(f'{dataDir}/spectrograms3sec/train')
#   os.makedirs(f'{dataDir}/spectrograms3sec/test')

#   for g in genres:
#     path_train = os.path.join(f'{dataDir}/spectrograms3sec/train',f'{g}')
#     path_test = os.path.join(f'{dataDir}/spectrograms3sec/test',f'{g}')
#     os.makedirs(path_train)
#     os.makedirs(path_test)

# Parallel (Batches) [Fastest]
# def task(g,j,filename):
#     song  =  os.path.join(f'{dataDir}/audio3sec/{g}',f'{filename}')
#     y,sr = librosa.load(song,duration=3)
#     #############################################################################
#     mels = librosa.feature.melspectrogram(y=y,sr=sr)
#     fig = plt.Figure(figsize=(13, 130))
#     canvas = FigureCanvas(fig)
#     p = librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))
#     #########################################################
#     plt.savefig(f'{dataDir}/spectrograms3sec/train/{g}/{g+str(j)}.png')

# batch_size = 32
# counter = 0
# processes = []

# for g in genres:
#     j = 0
#     print(g)
#     for filename in os.listdir(os.path.join(f'{dataDir}/audio3sec',f"{g}")):
#         if counter <= batch_size:
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
#         else:
#             for process in processes:
#                 process.start()
#             for process in processes:
#                 process.join()
#             counter = 0
#             processes = []
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
            
# for process in processes:
#     process.start()
# for process in processes:
#     process.join()

# %%
# Mel Spectrogram for 30 Sec Clips

# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'
# genres = genres.split()

# if not os.path.exists(f'{dataDir}'):
#   os.makedirs(f'{dataDir}/')
  
# if not os.path.exists(f'{dataDir}/spectrograms30sec'):
#   os.makedirs(f'{dataDir}/spectrograms30sec')
#   os.makedirs(f'{dataDir}/spectrograms30sec/train')
#   os.makedirs(f'{dataDir}/spectrograms30sec/test')

#   for g in genres:
#     path_train = os.path.join(f'{dataDir}/spectrograms30sec/train',f'{g}')
#     path_test = os.path.join(f'{dataDir}/spectrograms30sec/test',f'{g}')
#     os.makedirs(path_train)
#     os.makedirs(path_test)

# Multi-Core Implmentation (Batches) [Fastest]
# def task(g,j,filename):
#     song  =  os.path.join(f'{dataDir}/genres/{g}',f'{filename}')
#     y,sr = librosa.load(song,duration=3)
#     #############################################################################
#     mels = librosa.feature.melspectrogram(y=y,sr=sr)
#     fig = plt.Figure(figsize=(13, 130))
#     canvas = FigureCanvas(fig)
#     p = librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))
#     #########################################################
#     plt.savefig(f'{dataDir}/spectrograms30sec/train/{g}/{g+str(j)}.png')

# batch_size = 32
# counter = 0
# processes = []

# for g in genres:
#     j = 0
#     print(g)
#     for filename in os.listdir(os.path.join(f'{dataDir}/genres',f"{g}")):
#         if counter <= batch_size:
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
#         else:
#             for process in processes:
#                 process.start()
#             for process in processes:
#                 process.join()
#             counter = 0
#             processes = []
#             j = j+1
#             processes = np.append(processes, Process(target=task, args=(g,j,filename)))
#             counter += 1
            
# for process in processes:
#     process.start()
# for process in processes:
#     process.join()


