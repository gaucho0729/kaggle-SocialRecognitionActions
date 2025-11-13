import pandas as pd
import numpy as np
import os

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "action_output.csv"

test = pd.read_csv(INPUT_TEST_FILE)

pd.set_option('display.max_columns', test.columns.size)
pd.set_option('display.max_rows', len(test))

test.info()
