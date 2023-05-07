import os, sys
import pandas as pd
import pathlib, librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import keras
from random import randint
from scipy.io.wavfile import write
import shutil
from multiprocessing import Process
from PyQt5.QtGui import QPixmap
from  PyQt5.QtWidgets import (QApplication, QLabel, QWidget, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QStackedLayout, QLineEdit,
                             QFormLayout, QGridLayout, QTableWidget,
                             QTableWidgetItem, QMessageBox, QFileDialog,
                             QMainWindow
                             )

GENRES = 'Blues Classical Country Disco HipHop Jazz Metal Pop Reggae Rock'

def audiosToGraph(audio_files_path, save_path, type="ms"):

    if not os.path.exists(audio_files_path):
        print("Error: Path does not exist!")
        return
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    def task(j,filename):
        song  =  os.path.join(audio_files_path,f'{filename}')
        y,sr = librosa.load(song,duration=3)
        #############################################################################
        if type == 'ms':
            mels = librosa.feature.melspectrogram(y=y,sr=sr)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            plt.axis('off')
            p = librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))
        elif type == 'mfcc':
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 13, hop_length=512, n_fft=2048)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            p = librosa.display.specshow(mfccs)

        elif type == 'ms_old':
            mels = librosa.feature.melspectrogram(y=y,sr=sr)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            p = plt.imshow(librosa.power_to_db(mels,ref=np.max))

        else:
            print("Error: Please specify a valid 3rd argument for \"type\"")
            return

        #########################################################
        plt.savefig(os.path.join(f'{save_path}', f'{type+str(j)}.png'))

    batch_size = 32
    counter = 0
    j = 0
    processes = []

    for filename in os.listdir(os.path.join(audio_files_path)):
        if counter <= batch_size:
            j = j+1
            processes = np.append(processes, Process(target=task, args=(j,filename)))
            counter += 1
        else:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            counter = 0
            processes = []
            j = j+1
            processes = np.append(processes, Process(target=task, args=(j,filename)))
            counter += 1
                
    for process in processes:
        process.start()
    for process in processes:
        process.join()

def chopAudio(audio_file_path, save_path):

    if not os.path.exists(audio_file_path):
        print("Error: Path does not exist!")
        return
    
    if audio_file_path[-3:] != 'wav':
        print("Error: File is not of WAV type")
        return

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    batch_size = 16
    counter = 0
    processes = []

    def task(arg):
        t1 = arg * 3 * 1000 #Works in milliseconds
        t2 = (arg+1) * 3 * 1000
        newAudio = AudioSegment.from_wav(audio_file_path)
        newAudio = newAudio[t1:t2]
        newAudio.export(save_path + 'audio_seg_' + str(arg) +'.wav', format="wav") #Exports to a wav file in the current path.

    sample, sample_sr = librosa.load(audio_file_path)
    sample_duration = int(librosa.get_duration(y=sample, sr=sample_sr))

    for i in range(int(sample_duration/3)):
        if counter <= batch_size:
            processes = np.append(processes, Process(target=task, args=(i,)))
            counter += 1
        else:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            counter = 0
            processes = []
            processes = np.append(processes, Process(target=task, args=(i,)))
            counter += 1
                
    for process in processes:
        process.start()
    for process in processes:
        process.join()