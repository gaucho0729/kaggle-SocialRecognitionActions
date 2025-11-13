# =============================================
# Kaggle向け：敵対的検証＋固定パラメータ＋モデル比較ダッシュボード
# =============================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# =======================
# 0. 固定ハイパーパラメータ
# =======================
LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "random_state": 42
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "use_label_encoder": False
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42
}

MODEL_PARAMS = {
    "lgb": LGB_PARAMS,
    "xgb": XGB_PARAMS,
    "rf": RF_PARAMS
}

# 敵対的検証関数
def run_full_adversarial_workflow(train_df, test_df, target_col, drop_cols=None, model_type="rf", top_k=20, output_dir=None, plot=False, model_params=None):
    # データ結合
    df = pd.concat([train_df, test_df])
    df['is_test'] = [0]*len(train_df) + [1]*len(test_df)
    
    X = df.drop(columns=[target_col, 'is_test'] + (drop_cols or []))
    y = df['is_test']
    
    if model_type == "rf":
        model = RandomForestClassifier(**(model_params or {}))
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented.")
    
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:,1]
    
    auc = roc_auc_score(y, y_pred)
    print(f"AUC for {model_type}: {auc:.3f}")
    
    # 特徴量重要度
    top_features = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_k)
    
    return {"auc": auc, "top_features": top_features}


# =======================
# 1. データ読み込み
# =======================
train_path = "../train.csv"
test_path  = "../test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

target_col = "target"
drop_cols  = ["lab_id", "video_id"]

# =======================
# 2. 複数モデル敵対的検証
# =======================
models = ["lgb", "xgb", "rf"]
output_dir = "reports_dashboard_fixed_params"
top_k = 20
os.makedirs(output_dir, exist_ok=True)

reports = {}
top_features_sets = []

for model_type in models:
    print(f"\n=== Running adversarial validation for {model_type.upper()} ===")
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)

    report = run_full_adversarial_workflow(
        train_df=train,
        test_df=test,
        target_col=target_col,
        drop_cols=drop_cols,
        model_type=model_type,
        top_k=top_k,
        output_dir=model_output_dir,
        plot=False,
        model_params=MODEL_PARAMS[model_type]   # 固定パラメータを渡す
    )
    reports[model_type] = report
    top_features_sets.append(set(report["top_features"]["feature"]))

# =======================
# 3. 共通上位特徴量抽出
# =======================
common_features = set.intersection(*top_features_sets)
common_features = sorted(list(common_features))
pd.DataFrame({"feature": common_features}).to_csv(os.path.join(output_dir, "common_top_features.csv"), index=False)

# =======================
# 4. ダッシュボード用データ作成
# =======================
dashboard = []
for model_type, report in reports.items():
    top_feat = report["top_features"]
    for feat in common_features:
        importance = top_feat[top_feat["feature"]==feat]["importance"].values[0] \
                     if feat in top_feat["feature"].values else 0
        dashboard.append({
            "model": model_type.upper(),
            "feature": feat,
            "importance": importance,
            "AUC": report["AUC"]
        })

dashboard_df = pd.DataFrame(dashboard)

# Heatmap用
heatmap_df = dashboard_df.pivot(index="feature", columns="model", values="importance")
heatmap_df["mean_importance"] = heatmap_df.mean(axis=1)
heatmap_df = heatmap_df.sort_values("mean_importance", ascending=False)
heatmap_df.to_csv(os.path.join(output_dir, "heatmap_common_features.csv"))

# ランキング表用
ranking_df = dashboard_df.sort_values(["AUC", "importance"], ascending=[False, False])
ranking_df.to_csv(os.path.join(output_dir, "ranking_common_features.csv"), index=False)

# =======================
# 5. Notebookダッシュボード表示
# =======================
print("\n=== Model AUC Summary ===")
auc_summary = pd.DataFrame([
    {"model": k.upper(), "AUC": v["AUC"]} for k,v in reports.items()
]).sort_values("AUC", ascending=False)
display(auc_summary)

print("\n=== Heatmap of Common Features ===")
plt.figure(figsize=(8, max(4, len(common_features)*0.5)))
sns.heatmap(heatmap_df[models].astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Common Features Importance Heatmap")
plt.show()

print("\n=== Ranking Table of Common Features ===")
display(ranking_df.head(30))  # 上位30行表示

# =======================
# 6. 共通特徴量ヒストグラム表示
# =======================
for feat in common_features:
    plt.figure(figsize=(6,4))
    plt.hist(train[feat], bins=30, alpha=0.5, label="train")
    plt.hist(test[feat], bins=30, alpha=0.5, label="test")
    plt.title(f"{feat} distribution (common top features)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================
# 7. 次ステップ（モデル学習など）
# =======================
selected_features = common_features
X_train = train[selected_features]
y_train = train[target_col]
X_test  = test[selected_features]

# ここからCVモデル学習やLB提出用の予測を実施可能
