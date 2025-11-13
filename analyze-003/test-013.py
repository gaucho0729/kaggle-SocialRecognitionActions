import pandas as pd
import numpy as np

INPUT_FILE = '44566106-body.csv'
OUTPUT_FILE = '44566106-direction.csv'

body_axises = pd.read_csv(INPUT_FILE)

output = pd.DataFrame({
    'video_frame' : [],
    'mouse_id'    : [],
    'direction'   : [], 
    'body_rect_w' : [],
    'body_rect_y' : [],
})

# 部位名から座標を安全に取得する関数
def get_point(pivot, mouse_id, part):
    try:
        x = pivot.loc[mouse_id, ('x', part)]
        y = pivot.loc[mouse_id, ('y', part)]
        if np.isnan(x) or np.isnan(y):
            return None
        return np.array([x, y])
    except KeyError:
        return None

def get_orientation(pivot, mouse_id):
    # 基本：nose→tail
    nose = get_point(pivot, mouse_id, 'nose')
    tail = get_point(pivot, mouse_id, 'tail')
    
    # 欠損がある場合の補完
    if nose is None:
        for alt in ['headpiece_topfront', 'headpiece_bottomfront', 'body_center']:
            nose = get_point(pivot, mouse_id, alt)
            if nose is not None:
                break
    if tail is None:
        for alt in ['headpiece_bottomback', 'headpiece_topback', 'body_center']:
            tail = get_point(pivot, mouse_id, alt)
            if tail is not None:
                break

    # どちらも欠損していた場合は None
    if nose is None or tail is None:
        return np.nan

    v = nose - tail
    vx, vy = v[0], v[1]

    # 方向分類
    if vx > 0 and vy < 0:
        return 'up_right'
    elif vx < 0 and vy < 0:
        return 'up_left'
    elif vx > 0 and vy > 0:
        return 'bottom_right'
    elif vx < 0 and vy > 0:
        return 'bottom_left'
    else:
        return np.nan

def getNose(bodyinfo):
    if pd.notna(bodyinfo['nose_x']) | pd.notna(body['nose_y']):
        return bodyinfo['nose_x'],bodyinfo['nose_y']
    if pd.notna(bodyinfo['neck_x']) | pd.notna(body['neck_y']):
        return bodyinfo['neck_x'],bodyinfo['neck_y']
    if pd.notna(bodyinfo['body_center_x']) | pd.notna(body['body_center_y']):
        return bodyinfo['body_center_x'],bodyinfo['body_center_y']
    return np.nan, np.nan

def getTail(bodyinfo):
    if pd.notna(bodyinfo['tail_base_x']) | pd.notna(body['tail_base_y']):
        return bodyinfo['tail_base_x'],bodyinfo['tail_base_y']
    if pd.notna(bodyinfo['tail_midpoint_x']) | pd.notna(body['tail_midpoint_y']):
        return bodyinfo['tail_midpoint_x'],bodyinfo['tail_midpoint_y']
    return np.nan, np.nan

def getBodyRect(bodyinfo):
    x_min = np.nanmin([bodyinfo['nose_x'],
                    bodyinfo['neck_x'],
                    bodyinfo['ear_right_x'],
                    bodyinfo['ear_left_x'],
                    bodyinfo['headpiece_topfrontright_x'],
                    bodyinfo['headpiece_topfrontleft_x'],
                    bodyinfo['headpiece_topbackright_x'],
                    bodyinfo['headpiece_topbackleft_x'],
                    bodyinfo['headpiece_bottomfrontright_x'],
                    bodyinfo['headpiece_bottomfrontleft_x'],
                    bodyinfo['headpiece_bottombackright_x'],
                    bodyinfo['headpiece_bottombackleft_x'],
                    bodyinfo['body_center_x']])
    x_max = np.nanmax([bodyinfo['nose_x'],
                    bodyinfo['neck_x'],
                    bodyinfo['ear_right_x'],
                    bodyinfo['ear_left_x'],
                    bodyinfo['headpiece_topfrontright_x'],
                    bodyinfo['headpiece_topfrontleft_x'],
                    bodyinfo['headpiece_topbackright_x'],
                    bodyinfo['headpiece_topbackleft_x'],
                    bodyinfo['headpiece_bottomfrontright_x'],
                    bodyinfo['headpiece_bottomfrontleft_x'],
                    bodyinfo['headpiece_bottombackright_x'],
                    bodyinfo['headpiece_bottombackleft_x'],
                    bodyinfo['body_center_x']])
    y_min = np.nanmin([bodyinfo['nose_y'],
                    bodyinfo['neck_y'],
                    bodyinfo['ear_right_y'],
                    bodyinfo['ear_left_y'],
                    bodyinfo['headpiece_topfrontright_y'],
                    bodyinfo['headpiece_topfrontleft_y'],
                    bodyinfo['headpiece_topbackright_y'],
                    bodyinfo['headpiece_topbackleft_y'],
                    bodyinfo['headpiece_bottomfrontright_y'],
                    bodyinfo['headpiece_bottomfrontleft_y'],
                    bodyinfo['headpiece_bottombackright_y'],
                    bodyinfo['headpiece_bottombackleft_y'],
                    bodyinfo['body_center_y']])
    y_max = np.nanmax([bodyinfo['nose_y'],
                    bodyinfo['neck_y'],
                    bodyinfo['ear_right_y'],
                    bodyinfo['ear_left_y'],
                    bodyinfo['headpiece_topfrontright_y'],
                    bodyinfo['headpiece_topfrontleft_y'],
                    bodyinfo['headpiece_topbackright_y'],
                    bodyinfo['headpiece_topbackleft_y'],
                    bodyinfo['headpiece_bottomfrontright_y'],
                    bodyinfo['headpiece_bottomfrontleft_y'],
                    bodyinfo['headpiece_bottombackright_y'],
                    bodyinfo['headpiece_bottombackleft_y'],
                    bodyinfo['body_center_y']])
    return x_max - x_min, y_max - y_min


for i in range(len(body_axises)):
    body = body_axises.iloc[i]
    right = False
    up    = False
    nose_x, nose_y = getNose(body)
    tail_x, tail_y = getTail(body)
    if nose_x == np.nan:
        print(i, ': no nose')
        continue
    if tail_x == np.nan:
        print(i, ': no tail')
        continue
    if nose_y > tail_y:
        up = True
    else:
        up = False
    if nose_x > tail_x:
        right = True
    else:
        right = False

    rect_w, rect_h = getBodyRect(body)

    if (right == True) & (up == True):
        direction = 'Up_Right'
    if (right == False) & (up == True):
        direction = 'Up_Left'
    if (right == True) & (up == False):
        direction = 'Down_Right'
    if (right == False) & (up == False):
        direction = 'Down_Left'
    tmp = pd.DataFrame({
        'video_frame' : [body['video_frame']],
        'mouse_id'    : [body['mouse_id']],
        'direction'   : [direction], 
        'body_rect_w' : [rect_w],
        'body_rect_y' : [rect_h],
    })
    output = pd.concat([output, tmp], ignore_index=True)

output.to_csv(OUTPUT_FILE, index=None)
