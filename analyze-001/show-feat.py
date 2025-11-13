import pandas as pd

INPUT_FILE = '../train.csv'

pd.options.display.max_rows = 0
pd.options.display.max_columns = 0

train_all = pd.read_csv(INPUT_FILE)
train_all.info()


