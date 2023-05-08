MUSIC GENRE CLASSIFICATION
--------------------------------------------

Please install FFMPEG:

$ conda install -c conda-forge ffmpeg

This package is essential for having music title and artist identification

=================================================================

This Project is organized into 3 main folders:


1) ModelNotebooks

2) src

3) Tools


Descriptions:


1) "ModeNotebooks" contains two sub directories for the final model and
    other signifgant models we used for testing, as detailed in the report.
   Many, many notebookes were created for other architectures we had developed. 
   However, we have not included these, as the results are not signifgant, 
   and we do not want to confuse users with the multitude of notebooks that
   would need to be present. Additonally, we re-organized this project's 
   file scheme; therefore, some of the older networks we decied not to include
   would not work.

   Please note that there is a subdirectory within "./ModelNotebooks/FinalModel"
   called "Helper Files". It contains 1 script that were written for common 
   operations that are perfromed in this project, such as audio splitting
   and making graphs. Please do NOT edit the files within here. Feel free
   to look at the scripts to understand what they do, again, do NOT
   modify these unless you know what you are doing.

   Also, the "FinalModel" directory contains a notebook called "FinalModelEvaluator."
   Within this notebook, you will find the results of running the FINAL model with
   the test set.

2) "src" contains the code that is used for the acctual application we developed.
    All necessary files to run the application are included within this project.
    To run the program, simply execute the script "music-genre-recognition-app".
    Ensure you are within the "src" directory when executing. 

3) "Tools" contains a notebook used for formatting the dataset as needed by
   all the notebooks in the "ModelNotebooks" directory. Its worth noting that
   you MUST donwload the dataset yourself from the following link:

   https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

   Given that the orginal source of the dataset is not longer up, as of this
   writing, we ask that users visit Kaggle to download the dataset. Simply,
   drag the downloaded zip file into the ROOT of this project (i.e. the same
   directory that this README file is in). Once done, simply run the notebook
   and it will handle all the necessary dependency installation and dataset 
   preprocessing needed for the model in "ModelNotebooks."