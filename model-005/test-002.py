import pandas as pd
import numpy as np


def get_distance(df, bp, m1, m2):
#    m1_info = df[(df['bodypart'] == bp) & (df['mouse_id'] == m1)]
#    m2_info = df[(df['bodypart'] == bp) & (df['mouse_id'] == m2)]
    m1_info = df[(df['mouse_id'] == m1)]
    m2_info = df[(df['mouse_id'] == m2)]
    if (len(m1_info) == 0) or (len(m2_info) == 0):
        return np.nan
    m1 = m1_info.iloc[0]
    m2 = m2_info.iloc[0]
    return (m1[f'{bp}_x_cm'] - m2[f'{bp}_x_cm']) ** 2 + (m1[f'{bp}_y_cm'] - m2[f'{bp}_y_cm']) ** 2


# nose1-nose2,nose1-body2,nose1-hip2
# body1-nose2,body1-body2,body1-hip2
# hip1-nose2,hip1-body2,hip1-hip2

# 1-2,1-3,1-4,2-3,2-4,3-4,
#   video_id
#   video_frame
#   nose1-nose2
#   nose1-body2
#   nose1-hip2
#   body1-nose2
#   body1-hip2
#   hip1-hip2
#   nose1-nose3
#   nose1-body3
#   nose1-hip3
#   body1-nose3
#   body1-hip3
#   hip1-hip3
#   nose1-nose4
#   nose1-body4
#   nose1-hip4
#   body1-nose4
#   body1-hip4
#   hip1-hip4
#   nose2-nose3
#   nose2-body3
#   nose2-hip3
#   body2-nose3
#   body2-hip3
#   hip2-hip3
#   nose2-nose4
#   nose2-body4
#   nose2-hip4
#   body2-nose4
#   body2-hip4
#   hip2-hip4
#   nose3-nose4
#   nose3-body4
#   nose3-hip4
#   body3-nose4
#   body3-hip4
#   hip3-hip4
def wrap_up_features(df):
    retval = pd.DataFrame()
    for (video_id, video_frame), group in df.groupby(['video_id','video_frame']):
        nose1_nose2 = get_distance(group, 'nose',       1, 2)
        nose1_body2 = get_distance(group, 'bodycenter', 1, 2)
        nose1_hip2  = get_distance(group, 'hip',        1, 2)
        body1_nose2 = get_distance(group, 'nose',       1, 2)
        body1_body2 = get_distance(group, 'bodycenter', 1, 2)
        body1_hip2  = get_distance(group, 'hip',        1, 2)
        hip1_nose2  = get_distance(group, 'nose',       1, 2)
        hip1_body2  = get_distance(group, 'bodycenter', 1, 2)
        hip1_hip2   = get_distance(group, 'hip',        1, 2)

        nose1_nose3 = get_distance(group, 'nose',       1, 3)
        nose1_body3 = get_distance(group, 'bodycenter', 1, 3)
        nose1_hip3  = get_distance(group, 'hip',        1, 3)
        body1_nose3 = get_distance(group, 'nose',       1, 3)
        body1_body3 = get_distance(group, 'bodycenter', 1, 3)
        body1_hip3  = get_distance(group, 'hip',        1, 3)
        hip1_nose3  = get_distance(group, 'nose',       1, 3)
        hip1_body3  = get_distance(group, 'bodycenter', 1, 3)
        hip1_hip3   = get_distance(group, 'hip',        1, 3)

        nose1_nose4 = get_distance(group, 'nose',       1, 4)
        nose1_body4 = get_distance(group, 'bodycenter', 1, 4)
        nose1_hip4  = get_distance(group, 'hip',        1, 4)
        body1_nose4 = get_distance(group, 'nose',       1, 4)
        body1_body4 = get_distance(group, 'bodycenter', 1, 4)
        body1_hip4  = get_distance(group, 'hip',        1, 4)
        hip1_nose4  = get_distance(group, 'nose',       1, 4)
        hip1_body4  = get_distance(group, 'bodycenter', 1, 4)
        hip1_hip4   = get_distance(group, 'hip',        1, 4)

        nose2_nose3 = get_distance(group, 'nose',       2, 3)
        nose2_body3 = get_distance(group, 'bodycenter', 2, 3)
        nose2_hip3  = get_distance(group, 'hip',        2, 3)
        body2_nose3 = get_distance(group, 'nose',       2, 3)
        body2_body3 = get_distance(group, 'bodycenter', 2, 3)
        body2_hip3  = get_distance(group, 'hip',        2, 3)
        hip2_nose3  = get_distance(group, 'nose',       2, 3)
        hip2_body3  = get_distance(group, 'bodycenter', 2, 3)
        hip2_hip3   = get_distance(group, 'hip',        2, 3)
        nose2_nose4 = get_distance(group, 'nose',       2, 4)
        nose2_body4 = get_distance(group, 'bodycenter', 2, 4)
        nose2_hip4  = get_distance(group, 'hip',        2, 4)
        body2_nose4 = get_distance(group, 'nose',       2, 4)
        body2_body4 = get_distance(group, 'bodycenter', 2, 4)
        body2_hip4  = get_distance(group, 'hip',        2, 4)
        hip2_nose4  = get_distance(group, 'nose',       2, 4)
        hip2_body4  = get_distance(group, 'bodycenter', 2, 4)
        hip2_hip4   = get_distance(group, 'hip',        2, 4)

        nose3_nose4 = get_distance(group, 'nose',       3, 4)
        nose3_body4 = get_distance(group, 'bodycenter', 3, 4)
        nose3_hip4  = get_distance(group, 'hip',        3, 4)
        body3_nose4 = get_distance(group, 'nose',       3, 4)
        body3_body4 = get_distance(group, 'bodycenter', 3, 4)
        body3_hip4  = get_distance(group, 'hip',        3, 4)
        hip3_nose4  = get_distance(group, 'nose',       3, 4)
        hip3_body4  = get_distance(group, 'bodycenter', 3, 4)
        hip3_hip4   = get_distance(group, 'hip',        3, 4)

        tmp = pd.DataFrame({
            'video_id':    [video_id],
            'video_frame': [video_frame],
            'nose1_nose2': [nose1_nose2],   #1- 1
            'nose1_body2': [nose1_body2],   #1- 2
            'nose1_hip2':  [nose1_hip2],    #1- 3
            'body1_body2': [body1_body2],   #1- 4
            'body1_nose2': [body1_nose2],   #1- 5
            'body1_hip2':  [body1_hip2],    #1- 6
            'hip1_body2':  [hip1_body2],    #1- 7
            'hip1_nose2':  [hip1_nose2],    #1- 8
            'hip1_hip2':   [hip1_hip2],     #1- 9
            'nose1_nose3': [nose1_nose3],   #1-10
            'nose1_body3': [nose1_body3],   #1-11
            'nose1_hip3':  [nose1_hip3],    #1-12
            'body1_nose3': [body1_nose3],   #1-13
            'body1_body3': [body1_body3],   #1-14
            'body1_hip3':  [body1_hip3],    #1-15
            'hip1_nose3':  [hip1_nose3],    #1-16
            'hip1_body3':  [hip1_body3],    #1-17
            'hip1_hip3':   [hip1_hip3],     #1-18
            'nose1_nose4': [nose1_nose4],   #1-19
            'nose1_body4': [nose1_body4],   #1-20
            'nose1_hip4':  [nose1_hip4],    #1-21
            'body1_nose4': [body1_nose4],   #1-22
            'body1_body4': [body1_body4],   #1-23
            'body1_hip4':  [body1_hip4],    #1-24
            'hip1_nose4':  [hip1_nose4],    #1-25
            'hip1_body4':  [hip1_body4],    #1-26
            'hip1_hip4':   [hip1_hip4],     #1-27

            'nose2_nose3': [nose2_nose3],   #2-10
            'nose2_body3': [nose2_body3],   #2-11
            'nose2_hip3':  [nose2_hip3],    #2-12
            'body2_nose3': [body2_nose3],   #2-13
            'body2_body3': [body2_body3],   #2-14
            'body2_hip3':  [body2_hip3],    #2-15
            'hip2_nose3':  [hip2_nose3],    #2-16
            'hip2_body3':  [hip2_body3],    #2-17
            'hip2_hip3':   [hip2_hip3],     #2-18
            'nose2_nose4': [nose2_nose4],   #2-19
            'nose2_body4': [nose2_body4],   #2-20
            'nose2_hip4':  [nose2_hip4],    #2-21
            'body2_nose4': [body2_nose4],   #2-22
            'body2_body4': [body2_body4],   #2-23
            'body2_hip4':  [body2_hip4],    #2-24
            'hip2_nose4':  [hip2_nose4],    #2-25
            'hip2_body4':  [hip2_body4],    #2-26
            'hip2_hip4':   [hip2_hip4],     #2-27

            'nose3_nose4': [nose3_nose4],   #3-19
            'nose3_body4': [nose3_body4],   #3-20
            'nose3_hip4':  [nose3_hip4],    #3-21
            'body3_nose4': [body3_nose4],   #3-22
            'body3_body4': [body3_body4],   #3-23
            'body3_hip4':  [body3_hip4],    #3-24
            'hip3_nose4':  [hip3_nose4],    #3-25
            'hip3_body4':  [hip3_body4],    #3-26
            'hip3_hip4':   [hip3_hip4],     #3-27
        })
        retval = pd.concat([retval, tmp])
    return retval

if __name__ == "__main__":
    tracking = pd.read_csv('tracking_shrink.csv')
    tracking_features = wrap_up_features(tracking)
    tracking_features.to_csv('tracking_features.csv')
    print("complete script")
