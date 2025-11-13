import pandas as pd

INPUT_FILE_NAME = 'submission-1.csv'
OUTPUT_FILE_NAME = 'submission-2.csv'

# --- 入力: submission.csv（統合前データ） ---
df = pd.read_csv(INPUT_FILE_NAME)
df.info()
#df = df.rename(columns={'mouse_id': 'agent_id'})
#df = df.rename(columns={'video_frame': 'start_frame'})

# score列がなければ仮で1.0
if "score" not in df.columns:
    df["score"] = 1.0

# 行動優先順位（スコア同点時の参照）
priority = ['attack', 'chase', 'approach', 'avoid', 'rear', 'submit']
priority_map = {a: i for i, a in enumerate(priority)}

# --- 1. ペア正規化 ---
def canonical_pair(row):
    a, t = sorted([row['agent_id'], row['target_id']])
    return pd.Series({'p_agent': a, 'p_target': t})

df[['p_agent', 'p_target']] = df.apply(canonical_pair, axis=1)

# --- 2. 重複・競合統合 ---
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

# --- 3. 攻撃側優先で agent/target を決定 ---
def unify_attack_direction(sub):
    if not (sub['action'] == 'attack').any():
        # 攻撃なし → p_agent/p_target順
        row = sub.iloc[0]
        return pd.Series({'agent_id': row['p_agent'], 'target_id': row['p_target']})
    attacks = sub[sub['action'] == 'attack']
    if len(attacks) == 1:
        row = attacks.iloc[0]
        return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
    # 複数攻撃 → score最大優先、同点ならID小さい方
    top = attacks.loc[attacks['score'].idxmax()]
    return pd.Series({'agent_id': top['agent_id'], 'target_id': top['target_id']})

agent_target_df = merged_df.groupby(['video_id', 'p_agent', 'p_target']).apply(unify_attack_direction).reset_index()

# --- 4. 結合して方向統一 ---
final_df = merged_df.merge(agent_target_df, on=['video_id', 'p_agent', 'p_target'], how='left')

# --- 5. Kaggle提出列に整形 ---
print('final_df:')
final_df.info()
for col in ['agent_id', 'target_id']:
    if f'{col}_y' in final_df.columns:
        final_df[col] = final_df[f'{col}_y']
    elif f'{col}_x' in final_df.columns:
        final_df[col] = final_df[f'{col}_x']

final_df = final_df[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]

# --- 6. ソートして出力 ---
final_df = final_df.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
final_df.index.name = 'row_id'
final_df.to_csv(OUTPUT_FILE_NAME, index=False)

print("✅ Kaggle提出用クリーンデータ作成完了:", final_df.shape)
print(final_df.head(10))
