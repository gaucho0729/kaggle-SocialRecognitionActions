# XGBoost 学習＋BayesSearchCV＋評価スクリプト（完全版）
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === 設定 ===
DATA_FILE = "pair_dataset_windowed.csv"
MODEL_FILE = "xgb_mouse_behavior_model.pkl"

# === メイン ===
if __name__ == "__main__":
    # === データ読み込み ===
    print("データ読み込み中...")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # 不要列を除外
#    drop_cols = ['video_id', 'frame', 'agent_id', 'target_id']
    drop_cols = ['video_id', 'frame', 'agent_id', 'target_id', 'a_bodypart', 't_bodypart']
    X = df.drop(columns=drop_cols + ['label'])
    y = df['label']
    label = df['label'].unique()
    print(label)

    # 欠損値処理
    X = X.fillna(X.median())

    # === ラベルの分布確認 ===
    print("\nラベル分布:")
    print(y.value_counts())

    # === モデルとパラメータ探索設定 ===
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(y.unique()),
#        eval_metric='mlogloss',
        eval_metric='logloss',
        tree_method='hist',  # GPU利用可なら 'gpu_hist'
        use_label_encoder=False,
        n_job = -1,
        random_state=42
    )

    param_space = {
        'learning_rate': (0.01, 0.3, 'log-uniform'),
        'max_depth': (3, 10),
        'min_child_weight': (1, 10),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'n_estimators': (100, 1000)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=20,                # 時間短縮のため軽め
        cv=3,
        n_jobs=-1,
        scoring='f1_macro',
        verbose=2,
        random_state=42
    )

#    le = LabelEncoder()
#    y = le.fit_transform(df['label'])
    y = df['label']
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]

    le = LabelEncoder()
    y_sample = le.fit_transform(y_sample)

    print("len(X_sample):", len(X_sample))
    print("len(y_sample):", len(y_sample))

    # === 学習 ===
    print("\nBayesSearchCV によるハイパーパラメータ最適化中...")
#    opt.fit(X, y)
    opt.fit(X_sample, y_sample)

    print("\n最良パラメータ:")
    print(opt.best_params_)
    print(f"最良スコア (CV f1_macro): {opt.best_score_:.4f}")

    # === 学習済みモデルを保存 ===
    joblib.dump(opt.best_estimator_, MODEL_FILE)
    print(f"\n✅ モデルを保存しました: {MODEL_FILE}")

    # === 最終学習済みモデルで予測と評価 ===
    y_pred_value = opt.best_estimator_.predict(X)
    y_pred = le.inverse_transform(y_pred_value)
    print("\n=== 訓練データでの性能評価 ===")
    print(classification_report(y, y_pred))

    # 混同行列も確認
    print("\n=== 混同行列 ===")
    print(confusion_matrix(y, y_pred))
