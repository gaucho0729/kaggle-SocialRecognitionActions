import pandas as pd
import os

INPUT_FILE = "../test.csv"
OUTPUT_FILE = "../normalize-test.csv"
ANNOTATION_DIR = "../test_annotation/"
TRACKING_DIR = "../test_tracking/"

mouse_map = {
    1: "mouse1_id",
    2: "mouse2_id",
    3: "mouse3_id",
    4: "mouse4_id",
}

def getMouse(ln, id):
    if id == 1:
        return ln['mouse1_id'],ln['mouse1_sex'],ln['mouse1_age'],ln['mouse1_condition'],ln['mouse1_strain'],ln['mouse1_color']
    if id == 2:
        return ln['mouse2_id'],ln['mouse2_sex'],ln['mouse2_age'],ln['mouse2_condition'],ln['mouse2_strain'],ln['mouse2_color']
    if id == 3:
        return ln['mouse3_id'],ln['mouse3_sex'],ln['mouse3_age'],ln['mouse3_condition'],ln['mouse3_strain'],ln['mouse3_color']
    if id == 4:
        return ln['mouse4_id'],ln['mouse4_sex'],ln['mouse4_age'],ln['mouse4_condition'],ln['mouse4_strain'],ln['mouse4_color']

#video_id
#bodypart
#agent_id
#target_id
#vector
#region_w
#region_h
#duration_qty
#duration

# 正規化された訓練データを生成する
## video_id,agent_id(=mouse_id),target_id,vector,action,region_width,region_height,action_type,action_qty,

output = pd.DataFrame()

i = 0
# 訓練データ読み込み
all_data = pd.read_csv(INPUT_FILE)
print(all_data)
for j in range(len(all_data)):
    ln = all_data.iloc[j]
    lab_id = ln["lab_id"]
    print(i, "/", len(all_data))
    if os.path.isdir(ANNOTATION_DIR + str(ln["lab_id"]))==False:
        i += 1
        continue
    if os.path.isdir(TRACKING_DIR   + str(ln["lab_id"]))==False:
        i += 1
        continue
    if os.path.exists(ANNOTATION_DIR + str(ln["lab_id"]) + "/" + str(ln["video_id"]) + ".parquet") == False:
        i += 1
        continue
    if os.path.exists(TRACKING_DIR   + str(ln["lab_id"]) + "/" + str(ln["video_id"]) + ".parquet") == False:
        i += 1
        continue
    annotation = pd.read_parquet(ANNOTATION_DIR + str(ln["lab_id"]) + "/" + str(ln["video_id"]) + ".parquet")
    tracking   = pd.read_parquet(TRACKING_DIR   + str(ln["lab_id"]) + "/" + str(ln["video_id"]) + ".parquet")
    row = pd.DataFrame()

    actions = annotation['action'].unique()
    for act in actions:
        r = pd.DataFrame()
        annon = annotation[annotation['action']==act].iloc[0]
        trk = tracking[(tracking['video_frame']>=annon['start_frame']) & (tracking['video_frame']<=annon['stop_frame'])]

        bodyparts       = trk['bodypart'].unique()
        video_id        = ln['video_id']
        agent_id,agent_sex,agent_age,agent_condition,agent_strain,agent_color = getMouse(ln, annon['agent_id'])
        target_id,target_sex,target_age,target_condition,target_strain,target_color = getMouse(ln, annon['target_id'])
        duration        = annon['stop_frame'] - annon['start_frame']
        tracking_method = ln['tracking_method']
        for bp in bodyparts:
            bodypart  = bp
            t = trk[trk['bodypart']==bp]
            end = len(t) - 1
            vector = (t['x'].iloc[end] - t['x'].iloc[0])**2 + (t['y'].iloc[end] - t['y'].iloc[0])**2
            region_w = t['x'].max() - t['x'].min()
            region_h = t['y'].max() - t['y'].min()
            duration_qty = len(t)
            r = pd.DataFrame({
                'lab_id'           : [lab_id],
                'video_id'         : [video_id],
                'action'           : [act],
                'agent_id'         : [agent_id],
                'target_id'        : [target_id],
                'bodypart'         : [bodypart],
                'vector'           : [vector],
                'region_w'         : [region_w],
                'region_h'         : [region_h],
                'duration_qty'     : [duration_qty],
                'agent_sex'        : [agent_sex],
                'agent_age'        : [agent_age],
                'agent_condition'  : [agent_condition],
                'agent_strain'     : [agent_strain],
                'agent_color'      : [agent_color],
                'target_sex'       : [target_sex],
                'target_age'       : [target_age],
                'target_condition' : [target_condition],
                'target_strain'    : [target_strain],
                'target_color'     : [target_color],
                'tracking_method'  : [tracking_method],
            })
            row = pd.concat([row, r], ignore_index=True)

    output = pd.concat([output, row], ignore_index=True)

    i += 1

output.to_csv(OUTPUT_FILE)
