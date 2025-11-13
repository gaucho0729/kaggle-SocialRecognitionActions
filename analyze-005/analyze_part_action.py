import pandas as pd
import numpy as np
import json

# train.csvよりactionの出現を調べる

INPUT_TRAIN_FILE = '../train.csv'
INPUT_TEST_FILE  = '../test.csv'
OUTPUT_FILE = 'action.csv'

def findBodyPart_Action(src_data):
    bodyparts = []
    actions   = []
    for i in range(len(src_data)):
        train = src_data.iloc[i]
        body_parts_tracked = json.loads(train['body_parts_tracked'].replace('""', '"'))
        for j in range(len(body_parts_tracked)):
            body_parts = body_parts_tracked[j]
            bodyparts.append(body_parts)
        if pd.isna(train['behaviors_labeled']) == False:
            behaviors_labeled  = json.loads(train['behaviors_labeled'].replace('""', '"'))
            for i in range(len(behaviors_labeled)):
                act = behaviors_labeled[i].split(',')
                act[2] = act[2].replace("'", "")
                actions.append(act[2])
    return bodyparts, actions

all_data = pd.read_csv(INPUT_TRAIN_FILE)
train_bodyparts, train_actions = findBodyPart_Action(all_data)

all_data = pd.read_csv(INPUT_TEST_FILE)
test_bodyparts, test_actions = findBodyPart_Action(all_data)

only_in_train_bodypart = list(set(train_bodyparts) - set(test_bodyparts))
only_in_train_action   = list(set(train_actions) - set(test_actions))

print('only_in_train_bodypart:')
print(only_in_train_bodypart)

print('only_in_train_action:')
print(only_in_train_action)
