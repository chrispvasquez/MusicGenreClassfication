import os, sys
import pandas as pd
import pathlib, librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import keras
from random import randint
from HelperFunctions import *
from PyQt5.QtGui import QPixmap
from  PyQt5.QtWidgets import (QApplication, QLabel, QWidget, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QStackedLayout, QLineEdit,
                             QFormLayout, QGridLayout, QTableWidget,
                             QTableWidgetItem, QMessageBox, QFileDialog,
                             QMainWindow
                             )

GENRES = 'Blues Classical Country Disco HipHop Jazz Metal Pop Reggae Rock'

class DataWindow(QWidget):
    def __init__(self, pred_label, results, audio, csv_dir):
        super().__init__()

        self.img_dir = 'input/graphs/'
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        layout = QVBoxLayout()
        self.label = QLabel(self)
        songTitle, songArtist = getSongInfo(audio)

        saveSongInfo(songTitle, songArtist, pred_label, results, csv_dir)

        pixmap = getResultsImage(self.img_dir, results, songTitle, songArtist)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(),
                          pixmap.height())
 
        self.Q_songTitle = QLabel("Song Title: " + songTitle)
        self.Q_songArtist = QLabel("Artist: " + songArtist)
        self.Q_pred_label = QLabel("Predicted Genre: " + pred_label)
 
        layout.addWidget(self.label)
        layout.addWidget(self.Q_songTitle)
        layout.addWidget(self.Q_songArtist)
        layout.addWidget(self.Q_pred_label)
        self.setLayout(layout)
    
    def update(self, pred_label, results, audio, csv_dir):
        
        songTitle, songArtist = getSongInfo(audio)
        
        saveSongInfo(songTitle, songArtist, pred_label, results, csv_dir)
        
        self.Q_songTitle.setText("Song Title: " + songTitle)
        self.Q_songArtist.setText("Artist: " + songArtist)
        self.Q_pred_label.setText("Predicted Genre: " + pred_label)
        
        pixmap = getResultsImage(self.img_dir, results, songTitle, songArtist)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(),
                          pixmap.height())
        
        
        

def createWarningWindow(descrtiption):
    # Create Warning Box for Unpopulated CSV
    warn_msg = QMessageBox()
    warn_msg.setWindowTitle("Warning!")
    warn_msg.setText(descrtiption)
    warn_msg.setIcon(QMessageBox.Critical)

    return warn_msg