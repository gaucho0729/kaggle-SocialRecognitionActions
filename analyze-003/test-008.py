import pandas as pd

INPUT_FILE = "../train_tracking/AdaptableSnail/44566106.parquet"
tracking_data = pd.read_parquet(INPUT_FILE)
tracking_data.to_csv('44566106-tracking.csv')

INPUT_FILE = "../train_annotation/AdaptableSnail/44566106.parquet"
anno_data = pd.read_parquet(INPUT_FILE)
anno_data.to_csv('44566106-annontaiton.csv')
