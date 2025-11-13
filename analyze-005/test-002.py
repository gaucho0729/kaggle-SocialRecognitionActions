import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt.space import Real, Integer
from skopt import BayesSearchCV
from enum import Enum
from sklearn.preprocessing import LabelEncoder

# 列挙型の定義
class AlgoType(Enum):
#    RandomForest = 1
#    XGBoost = 2
    LightGBM = 3
#    TensorFlow = 4
#    PyTorch = 5


# RandomForestハイパーパラメータ空間の定義
rf_param_space={
    'n_estimators': Integer(2, 2000), #森の中の木の数
    'max_depth': Integer(1, 32),        #各決定木の最大深さ
    'min_samples_split': Integer(2,32),    # ノードを分割するために必要な最小サンプル
    'min_samples_leaf': Integer(1,32),     # 葉ノードに必要な最小サンプル
    'min_weight_fraction_leaf': Real(0.0, 0.5),  #葉ノードに必要なサンプルの重みの最小合計
    'max_features': Integer(1, 32),
    'bootstrap':  [True, False]
}


# XGBoostハイパーパラメータ空間の定義
xgb_param_space={
    'n_estimators': Integer(100, 1000), #森の中の木の数
    'learning_rate': Real(0.001, 0.5),  #学習率
    'max_depth': Integer(1, 15),        #各決定木の最大深さ
    'min_child_weight': Integer(1,20),  #子ノードに必要な最小サンプル重みの合
    'gamma':    Real(0.0, 1.0),         #ノードの分割に必要な最小損失減少
    'subsample': Real(0.5, 1.0),        #各決定木を構築するために使用されるサンプルの割合
    'colsample_bytree': Real(0.0001,1.0),   #各決定木を構築するために使用される特徴量の割合
    'colsample_bylevel': Real(0.0001,1.0),   #各レベルの決定木を構築するために使用される特徴量の割合
    'colsample_bynode': Real(0.0001, 1.0), #各ノードの分割に使用される特徴量の割合
    'reg_alpha': Real(0.01, 1.0),         #L1正則化項の重み
    'reg_lambda': Real(0.01, 1.0),           #L2正則化項の重み
#    'scale_pos_weight': Real(0.1, 4.0),      #正例と負例の不均衡を補正するための重み
#    'base_score': Real(0.1, 1.0),           #すべての観測値に対する初期予測確率
}


# LightGBMハイパーパラメータ空間の定義
lgbm_param_space={
    'num_iterations': Integer(50,2000),   #ブースティング回数
    'learning_rate': Real(0.01, 0.3),   #学習率
    'num_leaves': Integer(20,512),    #一つの木の最大葉数
    'max_depth': Integer(5,32),     #木の最大深さ
    'lambda_l1': Real(0.0, 1.0),    #L1正則化の強度
    'lambda_l2': Real(0.0, 1.0),    #L2正則化の強度
}


def predictForEva(type, trainX, trainY, test):
    '''
    if type == AlgoType.RandomForest:
        model = RandomForestClassifier()
        # ベイズ最適化の設定
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=rf_param_space,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
    elif type == AlgoType.XGBoost:
#    if type == AlgoType.XGBoost:
        model = xgb.XGBClassifier()
        # ベイズ最適化の設定
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=xgb_param_space,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
    elif type == AlgoType.LightGBM:
    '''
    if type == AlgoType.LightGBM:
        model = lgb.LGBMClassifier(verbose=-1)
        # ベイズ最適化の設定
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=lgbm_param_space,
            n_iter=50,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
    else:
        return (None, None)

    opt.fit(trainX, trainY)
    predY = opt.best_estimator_.predict(test)
    return (predY, opt)


def predict(opt, test):
    return opt.best_estimator_.predict(test)

le = LabelEncoder()

X = pd.read_csv('X_train.csv')
y_df = pd.read_csv('y_train.csv')
y = y_df.pop("0")
y = le.fit_transform(y)
# カラム名のクリーニング
X.columns = [
    re.sub(r'[^A-Za-z0-9_]+', '_', c)  # 英数字・アンダースコア以外をアンダースコアに置換
    for c in X.columns
]

X_test = pd.read_csv('X_test.csv')
y_test_df = pd.read_csv('y_test.csv')
y_test = y_test_df.pop("0")
y_test = le.fit_transform(y_test)
# カラム名のクリーニング
X_test.columns = [
    re.sub(r'[^A-Za-z0-9_]+', '_', c)  # 英数字・アンダースコア以外をアンダースコアに置換
    for c in X.columns
]

le = LabelEncoder()
y = le.fit_transform(y)
y_test = le.fit_transform(y_test)

for algo in AlgoType:
    y_pred, opt = predictForEva(algo, X, y, X_test)
    print(algo, 'acc=', accuracy_score(y_test, y_pred))
    print(algo, 'MAE=', mean_absolute_error(y_test, y_pred))
    print(algo, 'MSE=', mean_squared_error(y_test, y_pred))
