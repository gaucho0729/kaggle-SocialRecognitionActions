import pandas as pd
import numpy as np

INPUT_FILE_NAME = 'submission-0.csv'
OUTPUT_FILE_NAME = 'submission-2.csv'

# todo: action_remapを自動的に生成するようにする (2025-10-14)
action_remap = {
    0: 'approach',
    1: 'attack',
    2: 'avoid',
    3: 'chase',
    4: 'chaseattack',
    5: 'rear',
    6: 'submit'
}

# todo: solo_actionを自動的に生成するようにする (2025-10-14)
solo_action = [
    'rear'
]

pair_actions = [
    'approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit'
]


def shape_data_for_submit(all_data, actions, mice):
    tmp = pd.DataFrame({
        'video_id'   : [],
        'agent_id'   : [],
        'target_id'  : [],
        'action'     : [],
        'start_frame': [],
        'stop_frame' : [],
    })

    target_group = pd.DataFrame()

    # agent mouse
    for action in actions:
        for mouse in mice:
            row_data = all_data[(all_data['action']       == action) &
                                (all_data['mouse_id']     == mouse)  &
                                (all_data['agent_target'] == True)
                                ]
            if len(row_data) > 0:
                row_df = pd.DataFrame()
                if len(row_data) >= 1:
                    row_df = pd.DataFrame({
                        'video_id'   : row_data['video_id'].iloc[0],
                        'agent_id'   : row_data['mouse_id'].iloc[0],
                        'target_id'  : np.nan,
                        'action'     : [action] * 1,
                        'start_frame': row_data['video_frame'].iloc[0],
                        'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
                    })

                tmp = pd.concat([tmp, row_df], ignore_index=True)

            row_data = all_data[(all_data['action']       == action) &
                                (all_data['mouse_id']     == mouse)  &
                                (all_data['agent_target'] == False)
                                ]
            if len(row_data) == 0:
                continue
            if len(row_data) >= 1:
                row_df = pd.DataFrame({
                    'video_id'   : row_data['video_id'].iloc[0],
                    'agent_id'   : np.nan,
                    'target_id'  : row_data['mouse_id'].iloc[0],
                    'action'     : [action] * 1,
                    'start_frame': row_data['video_frame'].iloc[0],
                    'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
                })
            tmp = pd.concat([tmp, row_df], ignore_index=True)

    tmp['action'] = tmp['action'].map(action_remap)
    # solo_actionのagent_idとtarget_idを一致させる
    for i in range(len(tmp)):
        row = tmp.iloc[i]
        for act in solo_action:
            if row['action'] == act:
                if np.isnan(row['agent_id']):
                    tmp.loc[i, 'agent_id'] = row['target_id']
                if np.isnan(row['target_id']):
                    tmp.loc[i, 'target_id'] = row['agent_id']

    # start_frame順に並び替える
    tmp = tmp.sort_values(by=['start_frame','action'])

    # 1. action + start_time ごとにグループ化
    paired_rows = []
    for (act, start), g in tmp.groupby(['action', 'start_frame']):
        # agent_id のみある行
        agents = g[g['agent_id'].notna() & g['target_id'].isna()].copy()
        # target_id のみある行
        targets = g[g['target_id'].notna() & g['agent_id'].isna()].copy()
        
        used_targets = set()
        for i, agent_row in agents.iterrows():
            # 未使用の target を探す
            available_targets = targets[~targets.index.isin(used_targets)]
            if available_targets.empty:
                continue
            target_row = available_targets.iloc[0]  # 最初のものを対応付け
            used_targets.add(target_row.name)

            # stop_frame は小さい方を採用
            stop_frame = min(agent_row['stop_frame'], target_row['stop_frame'])

            # ペアリング情報を統合
            new_row = agent_row.copy()
            new_row['target_id'] = target_row['target_id']
            new_row['stop_frame'] = stop_frame
            paired_rows.append(new_row)

    # ペアリング結果を DataFrame に
    paired_df = pd.DataFrame(paired_rows)

    print("tmp:")
    tmp.info()

    print("paired_df")
    paired_df.info()

    # 2. 既存データとマージして、ペアがあるものを更新
#    merged = pd.merge(tmp, paired_df, on=['action', 'start_frame', 'agent_id'], how='left', suffixes=('', '_new'))
    merged = pd.merge(tmp, paired_df, on=['start_frame', 'agent_id'], how='left', suffixes=('', '_new'))

    # 既にtarget_idが空欄なら新しいものを採用
    merged['target_id'] = merged['target_id'].combine_first(merged['target_id_new'])
    merged['stop_frame'] = merged['stop_frame_new'].combine_first(merged['stop_frame'])

    # 不要列を削除
    merged = merged[tmp.columns]

    adjusted_stop = []

    for i, row in merged.iterrows():
        vid = row["video_id"]

        # 同じ video 内で agent_id と target_id の次の出現を探す
        next_agent = merged[(merged["video_id"] == vid) &
                        ((merged["agent_id"] == row["agent_id"]) |
                        (merged["agent_id"] == row["target_id"]) 
                        ) &
                        (merged["start_frame"] > row["start_frame"])]
        
        next_target = merged[(merged["video_id"] == vid) &
                        ((merged["target_id"] == row["target_id"]) |
                        (merged["target_id"] == row["agent_id"])
                        ) &
                        (merged["start_frame"] > row["start_frame"])]

        # 次の出現フレーム候補を集める
        next_frames = []
        if not next_agent.empty:
            next_frames.append(next_agent["start_frame"].min())
        if not next_target.empty:
            next_frames.append(next_target["start_frame"].min())

        # stop_frame調整判定
        if next_frames:
            next_min = min(next_frames)
            if row["stop_frame"] > next_min:
                stop_frame = next_min - 1
            else:
                stop_frame = row["stop_frame"]
        else:
            stop_frame = row["stop_frame"]

        adjusted_stop.append(stop_frame)

    merged["stop_frame"] = adjusted_stop

    merged.to_csv('tmp.csv')

    return merged

def build_submission_2(pred_df):
    retval = pd.DataFrame()
    tmp = pred_df.copy()
    tmp['used'] = [False] * len(pred_df)
    video_ids = pred_df['video_id'].unique()
    mice = pred_df['mouse_id'].unique()
#    mice = pred_df['agent_id'].unique()
    for video_id in video_ids:
        print('video_id:', video_id)
        for mouse in mice:
            tmp_df = pred_df[(pred_df['mouse_id']==mouse) & (pred_df['video_id']==video_id)]
#            tmp_df = pred_df[(pred_df['agent_id']==mouse) & (pred_df['video_id']==video_id)]
#            frms = tmp_df['video_frame'].unique()
            actions = tmp_df['pred_action'].unique()
#            actions = tmp_df['action'].unique()
            for action in actions:
                tmp_act = tmp_df[tmp_df['pred_action']==action]
#                tmp_act = tmp_df[tmp_df['action']==action]
                if len(tmp_act)==0:
                    continue
                prev_frame = 0
                start_frame = tmp_act['video_frame'].iloc[0]
                frms = tmp_act['video_frame'].unique()
                for frm in frms:
                    tmp = tmp_act[tmp_act['video_frame']==frm]
                    if tmp['video_frame'].iloc[0] - prev_frame >= 60:
                        if tmp['agent_target'].iloc[0] == True:
                            agent = tmp['mouse_id'].iloc[0]
#                            agent = tmp['agent_id'].iloc[0]
                            target = tmp['target_id'].iloc[0]
                            if prev_frame != 0:
                                row = pd.DataFrame({
                                        'video_id'   : [video_id]    * 1,
                                        'agent_id'   : [agent]       * 1,
                                        'target_id'  : [target]      * 1,
                                        'action'     : [action]      * 1,
                                        'start_frame': [start_frame] * 1,
                                        'stop_frame' : [prev_frame]  * 1
                                })
                                retval = pd.concat([retval, row], ignore_index=True)
#                        else:
#                            target = tmp['mouse_id'].iloc[0]
#                            agent  = tmp['target_id'].iloc[0]
                        start_frame = frm
                    prev_frame = tmp['video_frame'].iloc[0]
    return retval


def build_submission(pred_df):
    retval = pd.DataFrame()
    tmp = pred_df.copy()
    tmp['used'] = [False] * len(pred_df)
    video_ids = pred_df['video_id'].unique()
    frms = pred_df['video_frame'].unique()
    mice = pred_df['mouse_id'].unique()
    for video_id in video_ids:
        actions = pred_df['pred_action'].unique()

        for mouse in mice:
            for action in actions:
                tmp_df = pred_df[(pred_df['video_id'] == video_id) &
                                (pred_df['mouse_id'] == mouse)    &
                                (pred_df['pred_action']==action)]
                if len(tmp_df)==0:
                    continue
                tmp_agent  = tmp_df[tmp_df['agent_target']  == True]
                tmp_target = tmp_df[tmp_df['agent_target'] == False]
                tmp_agent['delta_frame'] = tmp_agent['video_frame'].diff()
                tmp_target['delta_frame']= tmp_target['video_frame'].diff()
                tmp_chunk_agent  = tmp_agent[tmp_agent['delta_frame']>=60]
                tmp_chunk_target = tmp_target[tmp_target['delta_frame']>=60]
                start_frame = tmp_df['video_frame'].iloc[0]
                if len(tmp_agent)==0:
                    continue
                prev_frame = 0
                for j in range(len(tmp_df)):
                    for i in range(len(tmp_chunk_agent)):
                        if tmp_chunk_agent['video_frame'].iloc[i] > tmp_df['video_frame'].iloc[j]:
                            if tmp_df['agent_target'].iloc[i] == True:
                                agent = tmp_df['mouse_id'].iloc[j]
                                target = tmp_df['target_id'].iloc[j]
                                row = pd.DataFrame({
                                        'video_id'   : [tmp_df['video_id'].iloc[j]] * 1,
                                        'agent_id'   : [agent] * 1,
                                        'target_id'  : [target] * 1,
                                        'action'     : [action]      * 1,
                                        'start_frame': [start_frame] * 1,
                                        'stop_frame' : [prev_frame]  * 1
                                })
#                            else:
#                                target = tmp_df['mouse_id'].iloc[j]
#                                agent  = tmp_df['target_id'].iloc[j]

                            start_frame = tmp_chunk_agent['video_frame'].iloc[i]
                            prev_frame = tmp_chunk_agent['video_frame'].iloc[i]
                            retval = pd.concat([retval, row], ignore_index=True)
                            break
                        else:
                            prev_frame = tmp_df['video_frame'].iloc[j]
                    prev_frame = tmp_df['video_frame'].iloc[j]
    return retval                


    # 提出形式に変換
    submission = pd.DataFrame({
        "lab_id": merged["lab_id"],
        "video_frame": merged["video_frame"],
#        "mouse_id": merged["mouse_id_agent"],   # 行動主（agent側マウス）
        "action": merged["pred_action"],
        "agent_id": merged["mouse_id_agent"],
        "target_id": merged["mouse_id_target"],
    })

    return submission

def remove_self_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] == df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def remove_pair_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] != df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def remove_same_start_stop_frame(df):
    cond = (df['start_frame'] == df['stop_frame'])
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

# 🔹 Step 1. 同一ペアの重複・分裂を統合（ポストプロセス）
# 
# 「frame連続かつ同一action」のものを一つの区間にまとめることで、
# 細切れ・重複した行動を整理します。
def merge_overlapping_actions(df):
    """
    同一video_id, agent_id, target_id, actionの重複・隣接をマージ
    """
    merged = []
    for (vid, a, t, act), group in df.groupby(["video_id", "agent_id", "target_id", "action"]):
        group = group.sort_values("start_frame")
        cur_start, cur_stop = None, None
        for _, row in group.iterrows():
            if cur_start is None:
                cur_start, cur_stop = row.start_frame, row.stop_frame
            elif row.start_frame <= cur_stop + 1:
                cur_stop = max(cur_stop, row.stop_frame)
            else:
                merged.append([vid, a, t, act, cur_start, cur_stop])
                cur_start, cur_stop = row.start_frame, row.stop_frame
        merged.append([vid, a, t, act, cur_start, cur_stop])
    return pd.DataFrame(merged, columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])


# 🔹 Step 2. agent/targetの逆転を統一
# 
# 「(1→2, attack) と (2→1, attack)」のようなケースを整理します。
# 一般には、同一frame区間で同じactionが両方向に出ている場合、どちらかを削除または短い方を無効化するのが自然です。
def resolve_reversed_actions(df):
    df = df.sort_values(["video_id", "action", "start_frame"])
    remove_idx = set()
    for (vid, act), group in df.groupby(["video_id", "action"]):
        for i, row_i in group.iterrows():
            for j, row_j in group.iterrows():
                if i >= j: continue
                if row_i.agent_id == row_j.target_id and row_i.target_id == row_j.agent_id:
                    # 時間区間が重なっていれば短い方を削除
                    overlap = min(row_i.stop_frame, row_j.stop_frame) - max(row_i.start_frame, row_j.start_frame)
                    if overlap > 0:
                        len_i = row_i.stop_frame - row_i.start_frame
                        len_j = row_j.stop_frame - row_j.start_frame
                        if len_i < len_j:
                            remove_idx.add(i)
                        else:
                            remove_idx.add(j)
    return df.drop(index=list(remove_idx)).reset_index(drop=True)


# 🔹 Step 3. 行動の矛盾解消（同一区間に複数アクション）
# 
# 同じ (agent_id, target_id) が同一frame区間で複数アクションをしている場合、
# たとえば "attack" と "chase" が重なっていたら、出現頻度の高い方を優先するなどのルールを決めます。
def remove_conflicting_actions(df):
    df = df.sort_values(["video_id", "start_frame"])
    clean = []
    for (vid, a, t), group in df.groupby(["video_id", "agent_id", "target_id"]):
        last_stop = -1
        for _, row in group.iterrows():
            if row.start_frame <= last_stop:
                # 重なっている -> スキップ（またはルールに応じて選択）
                continue
            clean.append(row)
            last_stop = row.stop_frame
    return pd.DataFrame(clean)

def canonical_pair(row):
    """ペアをソートして正規化"""
    a, t = sorted([row['agent_id'], row['target_id']])
    return pd.Series({'p_agent': a, 'p_target': t})

# === 攻撃側（attackを行った方）をagentに統一 ===
def determine_agent_target(sub):
    if len(sub) == 1:
        return sub.iloc[0][['agent_id', 'target_id']]

    attack_rows = sub[sub['action'] == 'attack']
    if len(attack_rows) == 1:
        # 攻撃側をagentに
        row = attack_rows.iloc[0]
        return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
    elif len(attack_rows) == 2:
        # 両方攻撃ならIDが小さい方をagentに
        ids = sorted([attack_rows.iloc[0]['agent_id'], attack_rows.iloc[1]['agent_id']])
        return pd.Series({'agent_id': ids[0], 'target_id': ids[1]})
    else:
        # 攻撃がない場合はp_agent/p_target順
        row = sub.iloc[0]
        return pd.Series({'agent_id': row['p_agent'], 'target_id': row['p_target']})

# === 攻撃側を agent に統一 ===
def unify_attack_direction(sub):
    # 攻撃がない場合は小さい方をagent
    if not (sub['action'] == 'attack').any():
        a, t = sorted([sub['p_agent'].iloc[0], sub['p_target'].iloc[0]])
        return pd.Series({'agent_id': a, 'target_id': t})

    # 攻撃がある場合 → その行のscoreを使って判定
    attacks = sub[sub['action'] == 'attack']
    if len(attacks) == 1:
        row = attacks.iloc[0]
        return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
    else:
        # 複数攻撃あり → scoreが高い方をagent
        top = attacks.loc[attacks['score'].idxmax()]
        return pd.Series({'agent_id': top['agent_id'], 'target_id': top['target_id']})

def clear_contradiction(df):
    # スコアがなければ仮で 1.0
    if "score" not in df.columns:
        df["score"] = 1.0

    # 行動優先順位（上ほど優先度高い）
    priority = ['chase', 'approach', 'attack', 'avoid', 'rear', 'submit']
    priority_map = {a: i for i, a in enumerate(priority)}

    # === 1. agent_id と target_id の逆転重複を整理 ===
    df[['p_agent', 'p_target']] = df.apply(canonical_pair, axis=1)

    # === 2. 同一ペア・重複フレームを統合 ===
    merged = []
    for (vid, a, t), sub in df.groupby(['video_id', 'p_agent', 'p_target']):
        sub = sub.sort_values(['start_frame', 'stop_frame']).reset_index(drop=True)

        # 重複区間を統合（行動優先度を考慮）
        merged_rows = []
        current = sub.iloc[0].copy()
        for _, row in sub.iloc[1:].iterrows():
            overlap = not (row['start_frame'] > current['stop_frame'])
            if overlap:
                # 行動が違う場合：スコア or 優先度で選択
                if row['score'] > current['score']:
                    current['action'] = row['action']
                    current['score'] = row['score']
                elif row['score'] == current['score']:
                    if priority_map[row['action']] < priority_map[current['action']]:
                        current['action'] = row['action']
                # フレーム結合
                current['stop_frame'] = max(current['stop_frame'], row['stop_frame'])
            else:
                merged_rows.append(current)
                current = row.copy()
        merged_rows.append(current)

    clean_df = pd.DataFrame(merged_rows)
    merged_df = pd.DataFrame(merged_rows)

    # 各ペアごとにagent-target方向を決定
    agent_target_df = merged_df.groupby(['video_id', 'p_agent', 'p_target']).apply(unify_attack_direction).reset_index()

    # === 4. 結合して方向統一 ===
    final_df = merged_df.merge(agent_target_df, on=['video_id', 'p_agent', 'p_target'], how='left')

    # === 5. 統一方向に反転 ===
    # （p_agent/p_targetと決定されたagent/targetが逆ならswap）
    swap_mask = (final_df['p_agent'] == final_df['target_id']) & (final_df['p_target'] == final_df['agent_id'])
    final_df.loc[swap_mask, ['agent_id', 'target_id']] = final_df.loc[swap_mask, ['target_id', 'agent_id']].values

    # === 6. 出力整形 ===
    final_df = final_df[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame', 'score']]
    final_df = final_df.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
    final_df.to_csv("submission_cleaned_scored.csv", index=False)

    final_df.index.name = 'row_index'
    print("✅ 出力:", final_df.shape)
    print(final_df.head(10))

    return final_df

def clear_contradiction_2(df):
    # スコア列がなければ仮で1.0
    if "score" not in df.columns:
        df["score"] = 1.0

    # 行動優先順位（スコア同点時に参照）
    priority = ['attack', 'chase', 'approach', 'avoid', 'rear', 'submit']
    priority_map = {a: i for i, a in enumerate(priority)}

    df[['p_agent', 'p_target']] = df.apply(canonical_pair, axis=1)

    # --- 2. 重複・競合統合（スコア＆優先度付き） ---
    merged_rows = []
    for (vid, pa, pt), sub in df.groupby(['video_id', 'p_agent', 'p_target']):
        sub = sub.sort_values(['start_frame', 'stop_frame']).reset_index(drop=True)
        current = sub.iloc[0].copy()

        for _, row in sub.iloc[1:].iterrows():
            overlap = not (row['start_frame'] > current['stop_frame'])
            if overlap:
                # 行動が異なる場合、スコア優先、同点なら優先度
                if row['score'] > current['score']:
                    current['action'] = row['action']
                    current['score'] = row['score']
                elif row['score'] == current['score']:
                    if priority_map[row['action']] < priority_map[current['action']]:
                        current['action'] = row['action']
                # フレーム結合
                current['stop_frame'] = max(current['stop_frame'], row['stop_frame'])
            else:
                merged_rows.append(current)
                current = row.copy()
        merged_rows.append(current)

    merged_df = pd.DataFrame(merged_rows)

    # --- 3. 攻撃側(agent)優先で方向統一 ---
    def unify_attack_direction(sub):
        if not (sub['action'] == 'attack').any():
            # 攻撃なし → p_agent/p_target順
            row = sub.iloc[0]
            return pd.Series({'agent_id': row['p_agent'], 'target_id': row['p_target']})
        attacks = sub[sub['action'] == 'attack']
        # 攻撃が1件 → そのagentを採用
        if len(attacks) == 1:
            row = attacks.iloc[0]
            return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
        # 複数攻撃 → score最大優先、同点ならID小さい方
        top = attacks.loc[attacks['score'].idxmax()]
        return pd.Series({'agent_id': top['agent_id'], 'target_id': top['target_id']})

    agent_target_df = merged_df.groupby(['video_id', 'p_agent', 'p_target']).apply(unify_attack_direction).reset_index()
    # --- 4. 結合して方向統一 ---
    final_df = merged_df.merge(agent_target_df, on=['video_id', 'p_agent', 'p_target'], how='left')

    print('final_df:')
    final_df.info()
    final_df.to_csv('final_df.csv')

    # --- 5. 出力整形 ---
    for col in ['agent_id', 'target_id']:
        if f'{col}_y' in final_df.columns:
            final_df[col] = final_df[f'{col}_y']
        elif f'{col}_x' in final_df.columns:
            final_df[col] = final_df[f'{col}_x']
    final_df = final_df[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
    final_df = final_df.sort_values(['video_id', 'start_frame']).reset_index(drop=True)

    final_df.index.name = 'row_index'
    print("✅ 出力:", final_df.shape)
    print(final_df.head(10))

    return final_df






if __name__ == "__main__":
    all_data = pd.read_csv(INPUT_FILE_NAME)
    all_data.info()
    df = build_submission_2(all_data)
    df['agent_id']  = df['agent_id'].astype('Int64')
    df['target_id'] = df['target_id'].astype('Int64')

    df.to_csv('df.csv')

    df = df.sort_values(by=['start_frame'],ascending=[True])
    df = df.dropna(subset=['target_id'])
    df = df.dropna(subset=['agent_id'])
    df = remove_self_actions(df, pair_actions)
    df = remove_pair_actions(df, solo_action)
    df = remove_same_start_stop_frame(df)
    
    df.to_csv('submission-1.csv')

#    df = merge_overlapping_actions(df)
#    df = resolve_reversed_actions(df)
#    df = remove_conflicting_actions(df)
#    df = df.sort_values(by='start_frame')

    df = clear_contradiction_2(df)

#    all_data['action'] = all_data['pred_action']
#    all_data.info()
#    df = shape_data_for_submit(all_data,all_data['action'].unique(),all_data['mouse_id'].unique())
    df.to_csv(OUTPUT_FILE_NAME)
