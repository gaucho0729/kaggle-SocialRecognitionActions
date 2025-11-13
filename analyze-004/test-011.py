# action-bodyparts毎に統計データを出す
# 
# 出力形式
# action | bodypart | vector(avg) | vector(min) | vector(max) | vector(std) | 1/4 vector | vector(median) | 3/4 vector | region_w(avg) | region_w(min) | region_w(max) | region_w(std) | 1/4 region_w | region_w(median) | 3/4 region_w | region_h(avg) | region_h(min) | region_h(max) | region_h(std) | 1/4 region_h | region_h(median) | 3/4 region_h |

import pandas as pd
import numpy as np

INPUT_FILE  = 'annotation.csv'
OUTPUT_FILE = 'tracking_action_stat.csv'

def initDataFrame():
    columns = [
        'action','bodypart',
        'vector_avg',  'vector_min',  'vector_max',  'vector_std',  'vector_1_4',  'vector_med',  'vector_3_4',
        'region_w_avg','region_w_min','region_w_max','region_w_std','region_w_1_4','region_w_med','region_w_3_4',
        'region_h_avg','region_h_min','region_h_max','region_h_std','region_h_1_4','region_h_med','region_h_3_4',
        'duration_avg','duration_min','duration_max','duration_std','duration_1_4','duration_med','duration_3_4'
    ]
    df = pd.DataFrame(columns=columns)
    return df

annotation = pd.read_csv(INPUT_FILE)
output = initDataFrame()

actions   = annotation['action'].unique()
bodyparts = annotation['bodypart'].unique()

for act in actions:
    for bp in bodyparts:
        tmp = annotation[annotation['action']==act]
        tmp = tmp[tmp['bodypart']==bp]
        if len(tmp) == 0:
            continue
        vector   = tmp['vector'].values
        region_w = tmp['region_w'].values
        region_h = tmp['region_h'].values
        duration = tmp['duration'].values
        df = pd.DataFrame({
                    'action'      : [act],
                    'bodypart'    : [bp],
                    'vector_avg'  : [np.mean(vector)],  'vector_min'  : [np.min(vector)],  'vector_max'  : [np.max(vector)],  'vector_std'  : [np.std(vector)],  'vector_1_4'  : [np.quantile(vector,0.25)],  'vector_med'  : [np.median(vector)],  'vector_3_4'  : [np.quantile(vector,0.75)],
                    'region_w_avg': [np.mean(region_w)],'region_w_min': [np.min(region_w)],'region_w_max': [np.max(region_w)],'region_w_std': [np.std(region_w)],'region_w_1_4': [np.quantile(region_w,0.25)],'region_w_med': [np.median(region_w)],'region_w_3_4': [np.quantile(region_w,0.75)],
                    'region_h_avg': [np.mean(region_h)],'region_h_min': [np.min(region_h)],'region_h_max': [np.max(region_h)],'region_h_std': [np.std(region_h)],'region_h_1_4': [np.quantile(region_h,0.25)],'region_h_med': [np.median(region_h)],'region_h_3_4': [np.quantile(region_h,0.75)],
                    'duration_avg': [np.mean(duration)],'duration_min': [np.min(duration)],'duration_max': [np.max(duration)],'duration_std': [np.std(duration)],'duration_1_4': [np.quantile(duration,0.25)],'duration_med': [np.median(duration)],'duration_3_4': [np.quantile(duration,0.75)]
                })
        output = pd.concat([output,df], ignore_index=True)
output.to_csv(OUTPUT_FILE, index=False)
