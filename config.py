import os

DATA_BIOSENTVEC = "./model"  # from here https://github.com/ncbi-nlp/BioSentVec?tab=readme-ov-file
DATA_DIR = " ./data/"
PROCESSED_DIR = " ./proc_data/"
OUTPUT_DIR = "./out/"
LOINC_PATH = './dataset/LoincTableCore/LoincTableCore.csv'
EXTRA_DF_PATH = './dataset/extra_queries.csv'

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NLTK_DIR = "/mnt/c/nltk_data"