import pandas as pd
import numpy as np
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# === 定数設定 ===
INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"
INPUT_TEST_FILE        = "../test.csv"
INPUT_FILE             = "train_normalize_action_tracking-2.csv"
OUTPUT_FILE            = "train_normalize_action_tracking-3.csv"
INTERVAL               = 0.25
TMP_DIR                = "./tmp_results"
LOG_FILE               = "log.txt"

all_data = pd.read_csv(INPUT_FILE)
all_data.sort_values(by=['video_id','mouse_id','action'],ascending=[True,True,True], inplace=True)
all_data['delta_rotate'] = all_data['rotate'].diff()
all_data['delta_rotate'] = (all_data['delta_rotate'] + np.pi) % (2 * np.pi) - np.pi

all_data.to_csv(OUTPUT_FILE)




