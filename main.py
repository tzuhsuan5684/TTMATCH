#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 多任務分類模型：Predict actionId / pointId / serverGetPoint
------------------------------------------------------------
評估指標：
- actionId：Macro F1
- pointId：Macro F1
- serverGetPoint：AUC-ROC
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

# =========================================================
# 1️⃣ 讀取資料
# =========================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(f"✅ Train shape: {train.shape}")
print(f"✅ Test shape: {test.shape}")

# =========================================================
# 2️⃣ 修正 -1 類別問題
# =========================================================
for col in ["actionId", "pointId", "serverGetPoint"]:
    if (train[col] == -1).any():
        max_label = train[col].max()
        print(f"⚠️ {col} 含有 -1，將其替換為 {max_label + 1}")
        train[col] = train[col].replace(-1, max_label + 1)

# =========================================================
# 3️⃣ 特徵與標籤分離
# =========================================================
target_cols = ["actionId", "pointId", "serverGetPoint"]
drop_cols = ["rally_uid", "rally_id"]
feature_cols = [c for c in train.columns if c not in target_cols + drop_cols]

X = train[feature_cols]
y_action = train["actionId"]
y_point = train["pointId"]
y_server = train["serverGetPoint"]
X_test = test[feature_cols]

# =========================================================
# 4️⃣ 資料切分
# =========================================================
X_train, X_valid, y_action_train, y_action_valid = train_test_split(
    X, y_action, test_size=0.2, random_state=42, stratify=y_action
)
_, _, y_point_train, y_point_valid = train_test_split(
    X, y_point, test_size=0.2, random_state=42, stratify=y_point
)
_, _, y_server_train, y_server_valid = train_test_split(
    X, y_server, test_size=0.2, random_state=42, stratify=y_server
)

# =========================================================
# 5️⃣ XGBoost 訓練函式
# =========================================================
def train_xgb(X_train, y_train, X_valid, y_valid, objective, num_class=None):
    params = {
        "objective": objective,
        "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
        "random_state": 42,
        "tree_method": "hist"
    }
    if num_class is not None:
        params["num_class"] = num_class

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    return model

# =========================================================
# 6️⃣ 三個模型訓練
# =========================================================
print("🚀 訓練 actionId 模型中...")
model_action = train_xgb(X_train, y_action_train, X_valid, y_action_valid,
                         objective="multi:softmax", num_class=y_action.nunique())

print("🚀 訓練 pointId 模型中...")
model_point = train_xgb(X_train, y_point_train, X_valid, y_point_valid,
                        objective="multi:softmax", num_class=y_point.nunique())

print("🚀 訓練 serverGetPoint 模型中...")
model_server = train_xgb(X_train, y_server_train, X_valid, y_server_valid,
                         objective="binary:logistic")

# =========================================================
# 7️⃣ 模型評估
# =========================================================
pred_action = model_action.predict(X_valid)
pred_point = model_point.predict(X_valid)
pred_server = model_server.predict_proba(X_valid)[:, 1]

f1_action = f1_score(y_action_valid, pred_action, average="macro")
f1_point = f1_score(y_point_valid, pred_point, average="macro")
auc_server = roc_auc_score(y_server_valid, pred_server)

print("\n📊 Validation Results:")
print(f"actionId Macro F1: {f1_action:.4f}")
print(f"pointId  Macro F1: {f1_point:.4f}")
print(f"serverGetPoint AUC: {auc_server:.4f}")

score=0.4*f1_action+0.4*f1_point+0.2*auc_server
print(f"綜合評分: {score:.4f}")

# =========================================================
# 8️⃣ 測試集預測
# =========================================================
print("\n🧮 產生測試預測中...")
pred_action_test = model_action.predict(X_test)
pred_point_test = model_point.predict(X_test)
pred_server_test = model_server.predict(X_test)

# =========================================================
# 9️⃣ 將映射回 -1
# =========================================================
def revert_negative(pred, original_train_col):
    """將 max+1 類別轉回 -1"""
    max_label = original_train_col.max()
    pred = pd.Series(pred)
    pred[pred == max_label + 1] = -1
    return pred.values

pred_action_test = revert_negative(pred_action_test, train["actionId"])
pred_point_test = revert_negative(pred_point_test, train["pointId"])
pred_server_test = revert_negative(pred_server_test, train["serverGetPoint"])

# =========================================================
# 🔟 輸出 submission.csv
# =========================================================
submission = pd.DataFrame({
    "rally_uid": test["rally_uid"],
    "serverGetPoint": pred_server_test,
    "pointId": pred_point_test,
    "actionId": pred_action_test
})

submission.to_csv("submission.csv", index=False)
print("\n✅ 已輸出 submission.csv")
print(submission.head())
