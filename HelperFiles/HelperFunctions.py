import os, sys
import pandas as pd
import pathlib, librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import keras
from keras.utils import load_img, img_to_array
from random import randint
import asyncio
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import datetime
from shazamio import Shazam
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

async def shazam(audio_path):
    shazam = Shazam()
    out = await shazam.recognize_song(audio_path)
    return out

def getSongInfo(audio_path):
    loop = asyncio.get_event_loop()
    songInfo = None
    song_title = None
    song_artist = None

    try:
        songInfo = loop.run_until_complete(shazam(audio_path))['track']
        song_title = songInfo["title"]
        song_artist = songInfo ["subtitle"]
    
    except:
        songInfo = "__"
        song_title = "__"
        song_artist = "__"

    return song_title, song_artist

def getDataframe(csv_dir, buttonValue):

    df = pd.read_csv(os.path.join(csv_dir, buttonValue + '.csv'))
    df.fillna('', inplace=True)
    return df

def recordAudio(rec_dir):

    freq = 44100
    duration = 30
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    print('Recording')
    sd.wait()
    print("Finished Recording")

    if not os.path.exists(rec_dir):
        os.mkdir(rec_dir)

    file_count = 0
    for path in os.listdir(rec_dir):
        if os.path.isfile(os.path.join(rec_dir, path)):
            file_count += 1

    file_name = rec_dir + "recording"  + str(file_count) + '.wav'
    write(file_name, freq, recording)

    return file_name

def saveSongInfo(songTitle, songArtist, pred_label, pred_results, csv_dir):

    s1 = np.array([songTitle, songArtist])
    s2 = np.array(pred_results) * 100.0
    songData = [np.append(np.concatenate((s1, s2)), str(datetime.datetime.now()))]
    df = pd.DataFrame(data = songData, 
                columns = ['Title', 'Artist', 
                            'Blues', 'Classical',
                            'Country', 'Disco',
                            'HipHop', 'Jazz',
                            'Metal', 'Pop',
                            'Reggae', 'Rock', 'Time Classified'])

    if not os.path.exists(csv_dir + pred_label + '.csv'):
        df.to_csv(csv_dir + pred_label + '.csv', index=False)

    else:
        loaded_df = pd.read_csv(csv_dir + pred_label + '.csv')
        loaded_df = loaded_df.append(df)
        loaded_df.to_csv(csv_dir + pred_label + '.csv', index=False)


def getResultsImage(img_dir, results, audioTitle, audioArtist) :
    genres = GENRES.split()
    image_name = audioArtist.strip() + '_' + audioTitle.strip() + '.png'
    plt.figure(figsize=(20,7))
    plt.bar(genres, np.array(results))
    plt.title("Inference Results")
    plt.savefig(os.path.join(img_dir, image_name))
    return QPixmap(os.path.join(img_dir, image_name))


def modelPrediction(audio_file):
    genres = GENRES.split()

    audio_seg_dir = "input/audio_segments/"
    spec_dir = "input/specs_segments"
    model = keras.models.load_model("saved/saved_cnn2_3", compile=False)

    predictions = []

    sample, sample_sr = librosa.load(audio_file)
    sample_duration = int(librosa.get_duration(y=sample, sr=sample_sr))

    ########################################################################################3

    for f in os.listdir(audio_seg_dir):
        os.remove(os.path.join(audio_seg_dir, f))

    ### For 3 Sec Segments!
    chopAudio(audio_file, audio_seg_dir)

    # Remove old spec images
    if not os.path.exists(spec_dir):
        os.mkdir(spec_dir)

    for f in os.listdir(spec_dir):
        os.remove(os.path.join(spec_dir, f))

    audiosToGraph(audio_seg_dir, spec_dir, "ms_old")

    for af in os.listdir(spec_dir):
        image_data = load_img(os.path.join(spec_dir, af[:-3] + 'png'),color_mode='rgba',target_size=(288,432))
        image = img_to_array(image_data)
        image = np.reshape(image,(1,288,432,4))
        
        p = model.predict(image/255)
        p = p.reshape((10,))

        predictions.append(p)

    ############################################################################################

    ### For 30 Sec Segments!

    # audio, sr = librosa.load(audio_file)
    # mels = librosa.feature.melspectrogram(y=audio,sr=sr)
    # fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    # p = plt.imshow(librosa.power_to_db(mels,ref=np.max))

    # spec_dir = "input/specs_segment_30"

    # if not os.path.exists(spec_dir):
    #     os.mkdir(spec_dir)

    # plt.savefig(os.path.join(spec_dir, audio_file[:-3] + 'png'))

    # image_data = load_img(os.path.join(spec_dir, audio_file[:-3] + 'png'),color_mode='rgba',target_size=(288,432))
    # image = img_to_array(image_data)
    # image = np.reshape(image,(1,288,432,4))
    
    # p = model.predict(image/255)
    # p = p.reshape((10,))

    # predictions.append(p)

    #########################################################################################3

    avg_preds = [x / len(predictions) for x in np.array(predictions).sum(axis=0)]
    predicted_label = np.argmax(avg_preds)

    return genres[predicted_label], avg_preds

def audiosToGraph(audio_files_path, save_path, type="ms"):

    if not os.path.exists(audio_files_path):
        print("Error: Path does not exist!")
        return

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    def task(j,filename):
        song  =  os.path.join(audio_files_path,f'{filename}')
        y,sr = librosa.load(song,duration=3)
        #############################################################################
        if type == 'ms':
            mels = librosa.feature.melspectrogram(y=y,sr=sr)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
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

    if not os.path.exists(save_path):
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