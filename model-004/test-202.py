import pandas as pd
import numpy as np
from tqdm import tqdm

# カラム名の確認・設定
VID_COL = 'video_id'
FRAME_COL = 'video_frame'
MID_COL = 'mouse_id'

INTERVAL = 0.25

# resuce row data
# trackingデータをinterval毎に間引く
# src_data : original data
# start_time : start time for reducing
# end_time   : end time for reducing
# interval   : inval time for reducting (sec)
def reduce_rows(src_data, start_time, end_time, interval):
    retval = pd.DataFrame()
    mice = src_data['mouse_id'].unique()
    bodyparts = src_data['bodypart'].unique()
    for mouse in mice:
        for bp in bodyparts:
            tmp_df = src_data[(src_data['mouse_id']==mouse) &
                              (src_data['bodypart']==bp)]
            target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

            # necessary for sort by key if using merge_asof
            # merge_asof を使う場合はキーでソートが必要
            sorted = tmp_df.sort_values('video_time').reset_index(drop=True)
            target_df = pd.DataFrame({'video_time': target_times})

            # merge 'nearest' line by direction='nearest
            # direction='nearest' で「最も近い」行を結合
            result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
            retval = pd.concat([retval, result], ignore_index=True)
    return retval

def make_pair_data(tracking_df, annotation_df):
    pair_rows = []
    for vid, df_vid in tqdm(tracking_df.groupby(VID_COL), desc="Processing videos"):
        df_vid = df_vid.sort_values(FRAME_COL)
        frames = df_vid[FRAME_COL].unique()
        
        # 各フレームで順序付きペア(agent→target)を作成
        for frame in frames:
            frame_df = df_vid[df_vid[FRAME_COL] == frame]
            mice = frame_df[MID_COL].unique()
            # 各マウスの行を辞書化
            rows = {mid: row for mid, row in frame_df.set_index(MID_COL).iterrows()}

            for agent in mice:
                for target in mice:
#                    if agent == target:
#                        continue

                    a = rows[agent]
                    t = rows[target]
                    feat_cols = [c for c in df_vid.columns if c not in [VID_COL, FRAME_COL, MID_COL]]

                    # 特徴量をそれぞれ接頭辞 a_ / t_ で付与
                    feats = {}
                    for c in feat_cols:
                        feats[f"a_{c}"] = a[c]
                        feats[f"t_{c}"] = t[c]

                    # 相対特徴（距離や角度）
                    if 'x_cm_body_center' in a and 'y_cm_body_center' in a:
                        dx = t['x_cm_body_center'] - a['x_cm_body_center']
                        dy = t['y_cm_body_center'] - a['y_cm_body_center']
                        feats['dist_agent_target'] = np.sqrt(dx**2 + dy**2)
                        feats['dx_agent_target'] = dx
                        feats['dy_agent_target'] = dy
                        feats['angle_to_target'] = np.degrees(np.arctan2(dy, dx))

                    # ラベル判定
                    mask = (
                        (annotation_df['video_id'] == vid) &
                        (annotation_df['agent_id'] == agent) &
                        (annotation_df['target_id'] == target) &
                        (annotation_df['start_frame'] <= frame) &
                        (annotation_df['stop_frame'] >= frame)
                    )
                    acts = annotation_df.loc[mask, 'action'].unique()
#                    label = acts[0] if len(acts) > 0 else 'none'
                    if len(acts) > 0:
                        label = acts[0]
                    else:
                        mask_2 = (
                            (annotation_df['video_id'] == vid) &
                            (annotation_df['agent_id'] == target) &
                            (annotation_df['target_id'] == agent) &
                            (annotation_df['start_frame'] <= frame) &
                            (annotation_df['stop_frame'] >= frame)
                        )
                        acts_2 = annotation_df.loc[mask_2, 'action'].unique()
                        if len(acts_2) > 0:
                            label = acts_2[0] + "-ed"
                        else:
                            label = 'none'

                    pair_rows.append({
                        'video_id': vid,
                        'frame': frame,
                        'agent_id': agent,
                        'target_id': target,
                        'label': label,
                        **feats
                    })

    pair_df = pd.DataFrame(pair_rows)
    return pair_df


if __name__ == "__main__":
    train_data = pd.read_csv('../train.csv')
    train_row = train_data.iloc[0]
    vid = str(train_row['video_id'])
    lid = str(train_row['lab_id'])
    fps  = train_row['frames_per_second']
    ppcm = train_row['pix_per_cm_approx']

    annotation_path = "../train_annotation/" + lid + "/" + vid + ".parquet"
    annotation = pd.read_parquet(annotation_path)
    annotation['video_id'] = vid
    tracking_path = "../train_tracking/" + lid + "/" + vid + ".parquet"
    tracking = pd.read_parquet(tracking_path)
    tracking['video_id'] = vid
    tracking['video_time'] = tracking['video_frame'] / fps
    start_time = tracking['video_time'].min()
    end_time = tracking['video_time'].max()
    tracking = reduce_rows(tracking, start_time, end_time, INTERVAL)
    pair_df = make_pair_data(tracking, annotation)
    pair_df.to_csv('pair_df.csv')
    annotation.to_csv('annotation-' + vid + ".csv")
    tracking.to_csv('tracking-' + vid + ".csv")
