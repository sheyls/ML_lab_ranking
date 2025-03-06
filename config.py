import os

DATA_DIR = " ./data/"
PROCESSED_DIR = " ./proc_data/"
OUTPUT_DIR = "./out/"

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

