#Config file with some important SYSTEM VARIABLES - to be constant across local repos and branches 
#for reproducibility and easier work (so we have the same paths to files, to other classes, .py files etc.)

import os

#Project root (as config is in main repo folder)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#Path to the database folder
DATABASE_FOLDER = os.path.join(PROJECT_ROOT, "DataBase")

#Path to painintg dataset (for reconstruction)
DATASET_PATH = os.path.join(DATABASE_FOLDER, "WikiArt")

#Name on hugging face (extraxted from link)
# https://www.kaggle.com/datasets/steubk/wikiart
DATASET_NAME = "Artificio/WikiArt"

########################

RANDOM_STATE = 111

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
