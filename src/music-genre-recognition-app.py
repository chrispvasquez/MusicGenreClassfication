import os, sys
import pandas as pd

from  PyQt5.QtWidgets import (QApplication, QLabel, QWidget, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QStackedLayout, QLineEdit,
                             QFormLayout, QGridLayout, QTableWidget,
                             QTableWidgetItem, QMessageBox, QFileDialog
                             )

#####################################
import sys
sys.path.insert(0, 'HelperFiles')
from HelperFunctions import *
from HelperWindows import *
#####################################

class Window(QWidget):
    def __init__(self):
        super().__init__()
        
        self.w2 = None

        # Set Recoridng Directory
        self.rec_dir = 'input/recordings/'
        if not os.path.exists(self.rec_dir):
            os.mkdir(self.rec_dir)

        # Set CSV data location & create if does not exist
        self.csv_dir = "input/classifiedMusicData/"
        if not os.path.exists(self.csv_dir):
            os.mkdir(self.csv_dir)

        # Create Warning Box for Unpopulated CSV
        self.warn_msg = createWarningWindow("No songs of this genre have been classified!")

        # Create a top-level layout
        self.setWindowTitle("Music Genere Classification App")
        topLayout = QVBoxLayout()
        self.setLayout(topLayout)

        # Create Buttons for Inputting Data and Accessing classified genere data

        self.inputMicButton = QPushButton('Microphone') 
        self.inputFolderButton = QPushButton('Folder')
        self.pageButtons = [[QPushButton('Back')],
                            [QPushButton('Blues'), QPushButton('Classical'), QPushButton('Country'), QPushButton('Disco'), QPushButton('HipHop')],
                            [QPushButton('Jazz'), QPushButton('Metal'), QPushButton('Pop'), QPushButton('Reggae'), QPushButton('Rock')],
                           ]

        # Create dictionary to link buttons to pages
        self.pageDict = { 'Back': 0,
                          'Blues' : 1, 'Classical' : 1, 'Country' : 1, 'Disco' : 1, 'HipHop' : 1,
                          'Jazz': 1, 'Metal' : 1, 'Pop' : 1, 'Reggae' : 1, 'Rock' : 1
                        }

        # Create the stacked layout
        self.stackedLayout = QStackedLayout()

        # Connect Buttons to SwitchPage Function
        for i in range(len(self.pageButtons)):
            for j in range(len(self.pageButtons[i])):
                self.pageButtons[i][j].clicked.connect(self.switchPage)
        
        # Create the Main Page
        self.page1 = QWidget()
        self.page1Layout = QVBoxLayout()

        hbox0 = QHBoxLayout()
        for b in self.pageButtons[1]:
            hbox0.addWidget(b)

        hbox1 = QHBoxLayout()
        for b in self.pageButtons[2]:
            hbox1.addWidget(b)

        hbox2 = QHBoxLayout()
        self.inputFolderButton.clicked.connect(self.folderAudioPredict)
        self.inputMicButton.clicked.connect(self.micAudioPredict)
        hbox2.addWidget(self.inputFolderButton)
        hbox2.addWidget(self.inputMicButton)

        self.page1Layout.addLayout(hbox0)
        self.page1Layout.addLayout(hbox1)
        self.page1Layout.addLayout(hbox2)
        self.page1.setLayout(self.page1Layout)
        self.stackedLayout.addWidget(self.page1)

        # Create the csv data page
        self.page2 = QWidget()
        self.page2Layout = QVBoxLayout()

        self.table = QTableWidget()
        self.page2Layout.addWidget(self.table)

        self.page2Layout.addWidget(self.pageButtons[0][0])

        self.page2.setLayout(self.page2Layout)
        self.stackedLayout.addWidget(self.page2)

        # Add the stacked layout to the top-level layout
        topLayout.addLayout(self.stackedLayout)

    def switchPage(self):
        buttonValue = self.sender().text()

        if self.pageDict[buttonValue] == 0:
            self.stackedLayout.setCurrentIndex(0)

        else:
            if(os.path.exists(os.path.join(self.csv_dir, buttonValue + '.csv'))):
                df = getDataframe(self.csv_dir, buttonValue)
                self.table.setRowCount(df.shape[0])
                self.table.setColumnCount(df.shape[1])
                self.table.setHorizontalHeaderLabels(df.columns)

                for row in df.iterrows():
                    values = row[1]
                    for col_index, value in enumerate(values):
                        #if isinstance(value, (float, int)):
                            #value = '{0:0,.0f}'.format(value)
                        tableItem = QTableWidgetItem(str(value))
                        self.table.setItem(row[0], col_index, tableItem)
                        self.table.setColumnWidth(col_index, 200)

                self.stackedLayout.setCurrentIndex(1)

            else:
                self.warn_msg.exec_()

    def folderAudioPredict(self):

        audio_file, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()",
                                                "", "All Files (*);;WAV Files (*.wav)")
        
        if audio_file.split('.')[-1] == "wav":
            predict_label, label_scores = modelPrediction(audio_file)
            if self.w2 is None:
                self.w2 = DataWindow(predict_label, label_scores, audio_file, self.csv_dir)
                self.w2.show()

        elif not audio_file == "":
            createWarningWindow("Incorrect Audio Format! Must be WAV (\".wav\")!").exec_()

    def micAudioPredict(self):


        rec_path = recordAudio(self.rec_dir)
        predict_label, label_scores = modelPrediction(rec_path)
        if self.w2 is None:
            self.w2 = DataWindow(predict_label, label_scores, rec_path, self.csv_dir)
            self.w2.show()


app = QApplication([])
window = Window()
window.show()
sys.exit(app.exec_())

