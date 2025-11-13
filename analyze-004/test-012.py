import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os

INPUT_FILE = '44566106-tracking.csv'
OUTPUT_FILE = '44566106-body.csv'

train_data = pd.read_csv(INPUT_FILE)

bodyparts = train_data['bodypart'].unique()
mice      = train_data['mouse_id'].unique()
frames    = train_data['video_frame'].unique()

print('frame qty:', len(frames))
print('bodyparts:', bodyparts)

# ==========================================================
# 共通: 出力列の初期化
# ==========================================================
def initDataFrame():
    columns = [
        'lab_id', 'video_id', 'video_frame',
        'mouse_id',
        'nose_x', 'nose_y',
        'neck_x', 'neck_y',
        'ear_right_x', 'ear_right_y',
        'ear_left_x', 'ear_left_y',
        'headpiece_topfrontright_x', 'headpiece_topfrontright_y',
        'headpiece_topfrontleft_x',  'headpiece_topfrontleft_y',
        'headpiece_topbackright_x',  'headpiece_topbackright_y',
        'headpiece_topbackleft_x',   'headpiece_topbackleft_y',
        'headpiece_bottomfrontright_x', 'headpiece_bottomfrontright_y',
        'headpiece_bottomfrontleft_x',  'headpiece_bottomfrontleft_y',
        'headpiece_bottombackright_x',  'headpiece_bottombackright_y',
        'headpiece_bottombackleft_x',   'headpiece_bottombackleft_y',
        'body_center_x', 'body_center_y',
        'lateral_right_x', 'lateral_right_y',
        'lateral_left_x', 'lateral_left_y',
        'tail_base_x', 'tail_base_y',
        'tail_midpoint_x', 'tail_midpoint_y',
        'tail_tip_x', 'tail_tip_y'
    ]
    return pd.DataFrame(columns=columns)

def setDataFrame(video_frame, mouse_id):
    columns = [
        'lab_id', 'video_id', 'video_frame',
        'mouse_id',
        'nose_x', 'nose_y',
        'neck_x', 'neck_y',
        'ear_right_x', 'ear_right_y',
        'ear_left_x', 'ear_left_y',
        'headpiece_topfrontright_x', 'headpiece_topfrontright_y',
        'headpiece_topfrontleft_x',  'headpiece_topfrontleft_y',
        'headpiece_topbackright_x',  'headpiece_topbackright_y',
        'headpiece_topbackleft_x',   'headpiece_topbackleft_y',
        'headpiece_bottomfrontright_x', 'headpiece_bottomfrontright_y',
        'headpiece_bottomfrontleft_x',  'headpiece_bottomfrontleft_y',
        'headpiece_bottombackright_x',  'headpiece_bottombackright_y',
        'headpiece_bottombackleft_x',   'headpiece_bottombackleft_y',
        'body_center_x', 'body_center_y',
        'lateral_right_x', 'lateral_right_y',
        'lateral_left_x', 'lateral_left_y',
        'tail_base_x', 'tail_base_y',
        'tail_midpoint_x', 'tail_midpoint_y',
        'tail_tip_x', 'tail_tip_y'
    ]
    df = pd.DataFrame(columns=columns)
    df['video_frame'] = video_frame
    df['mouse_id']    = mouse_id
    return df


# ==========================================================
# 各フレーム単位の処理関数（並列実行される）
# ==========================================================
def process_frame(frm):
    tmp_output = []
    frm_data = train_data[train_data['video_frame'] == frm]

    for mouse in mice:
        sub = frm_data[frm_data['mouse_id'] == mouse]
        if sub.empty:
            continue

        row = {'video_frame': frm, 'mouse_id':mouse}
#        row = setDataFrame(frm, mouse)
        for bp in bodyparts:
            part_data = sub[sub['bodypart'] == bp]
            if not part_data.empty:
                row[f"{bp}_x"] = part_data['x'].values[0]
                row[f"{bp}_y"] = part_data['y'].values[0]
        tmp_output.append(row)

    if tmp_output:
        return pd.DataFrame(tmp_output)
    else:
        return pd.DataFrame()


# ==========================================================
# 並列実行
# ==========================================================
if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_frame, frames))

    # 結果をまとめる
    output = pd.concat(results, ignore_index=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done. Saved to {OUTPUT_FILE}")
