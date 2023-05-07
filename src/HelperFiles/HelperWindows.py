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
        self.songTitle, self.songArtist = getSongInfo(audio)

        saveSongInfo(self.songTitle, self.songArtist, pred_label, results, csv_dir)

        self.pixmap = getResultsImage(self.img_dir, results, self.songTitle, self.songArtist)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())
 
        layout.addWidget(self.label)
        layout.addWidget(QLabel("Song Title: " + self.songTitle))
        layout.addWidget(QLabel("Artist: " + self.songArtist))
        layout.addWidget(QLabel("Predicted Genre: " + pred_label))
        self.setLayout(layout)

def createWarningWindow(descrtiption):
    # Create Warning Box for Unpopulated CSV
    warn_msg = QMessageBox()
    warn_msg.setWindowTitle("Warning!")
    warn_msg.setText(descrtiption)
    warn_msg.setIcon(QMessageBox.Critical)

    return warn_msg