import pandas as pd
import numpy as np

train_file = '../train.csv'
test_file = '../test.csv'

train = pd.read_csv(train_file, sep='\t')
print('train.csv:')
print('  row size=', len(train))
train.info()

test = pd.read_csv(test_file, sep='\t')
print('test.csv:')
print('row size=', len(test))
test.info()

print('complete of script!')
