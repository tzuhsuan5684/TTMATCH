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
# 假設 train.csv 和 test.csv 存在於同目錄
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
except FileNotFoundError:
    print("❌ 錯誤：找不到 train.csv 或 test.csv。請確保檔案在正確的路徑。")
    exit()


print(f"✅ Train shape: {train.shape}")
print(f"✅ Test shape: {test.shape}")

# 🌟 MODIFICATION (1/5):
# 競賽要求對每個 "rally" 進行一次預測。
# 我們假設 test.csv 中的每一行是回合中的一次擊球。
# 我們需要使用每個 rally_uid 的 "最後一筆" 資料來預測 "下一次" 的擊球。
test_last_shot = test.groupby('rally_uid').tail(1).copy()
print(f"✅ Test (last shots) shape: {test_last_shot.shape}")


# =========================================================
# 2️⃣ 修正 -1 類別問題
# =========================================================
# 儲存 -1 標籤的原始最大值，以便後續還原
original_max_labels = {}

for col in ["actionId", "pointId", "serverGetPoint"]:
    if col in train.columns and (train[col] == -1).any():
        max_label = train[col].max()
        original_max_labels[col] = max_label + 1 # 儲存 replacement value
        
        print(f"⚠️ {col} 含有 -1，將其替換為 {max_label + 1}")
        train[col] = train[col].replace(-1, max_label + 1)

# =========================================================
# 3️⃣ 特徵與標籤分離
# =========================================================
target_cols = ["actionId", "pointId", "serverGetPoint"]
# rally_id 可能是 rally_uid 的另一種 key，先移除
drop_cols = ["rally_uid", "rally_id"] 
feature_cols = [c for c in train.columns if c not in target_cols + drop_cols and c in test.columns]

X = train[feature_cols]
y_action = train["actionId"]
y_point = train["pointId"]
y_server = train["serverGetPoint"]

# 🌟 MODIFICATION (2/5):
# X_test 必須使用 'test_last_shot' DataFrame，
# 這樣我們才能為每個 rally_uid 僅預測一次。
X_test = test_last_shot[feature_cols]

# =========================================================
# 🧠 5.5️⃣ 特徵選取（Feature Selection）
# =========================================================
# (這部分邏輯保留不變，但請注意：
#  這裡是 "僅" 根據 y_action 選特徵，然後用於三個模型。
#  未來優化方向：可以為三個 target 各自選取一組最佳特徵。)

def select_features(X, y, top_k=30):
    """
    使用 XGBoost 先訓練一輪，選出最重要的前 K 個特徵。
    同時排除方差過低的無效特徵。
    """
    # 1️⃣ 移除方差過低的特徵
    selector = VarianceThreshold(threshold=0.0)
    # 確保 X 中沒有 NaN，否則 fit_transform 會出錯
    X_filled = X.fillna(0) 
    X_var = selector.fit_transform(X_filled)
    selected_cols = X.columns[selector.get_support()]

    # 2️⃣ 以 XGBoost 訓練快速重要度模型
    model_tmp = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    model_tmp.fit(X_var, y)

    # 3️⃣ 根據特徵重要度排序
    importances = model_tmp.feature_importances_
    importance_df = pd.DataFrame({
        "feature": selected_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n🔥 特徵重要度前十名 (基於 actionId)：")
    print(importance_df.head(10))

    # 4️⃣ 選出最重要的前 top_k 特徵
    top_features = importance_df.head(top_k)["feature"].tolist()
    return X[top_features], top_features

# 執行特徵選取
print("🧩 進行特徵選取中...")
X_selected, top_features = select_features(X, y_action, top_k=40)

# 🌟 MODIFICATION: 確保 X_test_selected 也使用 fillna(0)
X_test_selected = X_test[top_features].fillna(0) 

# 更新 train/valid 分割
X_train, X_valid, y_action_train, y_action_valid = train_test_split(
    X_selected, y_action, test_size=0.2, random_state=42, stratify=y_action
)
_, _, y_point_train, y_point_valid = train_test_split(
    X_selected, y_point, test_size=0.2, random_state=42, stratify=y_point
)
_, _, y_server_train, y_server_valid = train_test_split(
    X_selected, y_server, test_size=0.2, random_state=42, stratify=y_server
)

# 更新 X_test 使用同樣特徵
X_test = X_test_selected
print(f"✅ 使用前 {len(top_features)} 個重要特徵進行訓練")

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
        "tree_method": "hist",
        "early_stopping_rounds": 20 # 新增 early stopping
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
# 檢查 serverGetPoint 是否有 > 2 個類別 (例如 -1 被替換後)
if y_server.nunique() > 2:
    print("⚠️ serverGetPoint 發現多於2個類別，使用 multi:softmax")
    model_server = train_xgb(X_train, y_server_train, X_valid, y_server_valid,
                            objective="multi:softmax", num_class=y_server.nunique())
else:
    model_server = train_xgb(X_train, y_server_train, X_valid, y_server_valid,
                            objective="binary:logistic")

# =========================================================
# 7️⃣ 模型評估
# =========================================================
pred_action = model_action.predict(X_valid)
pred_point = model_point.predict(X_valid)

# 根據 serverGetPoint 的類別數決定如何評估
if y_server.nunique() > 2:
    # 多分類的 AUC (One-vs-Rest)
    pred_server_proba = model_server.predict_proba(X_valid)
    auc_server = roc_auc_score(y_server_valid, pred_server_proba, multi_class="ovr")
else:
    # 二分類 AUC
    pred_server_proba = model_server.predict_proba(X_valid)[:, 1]
    auc_server = roc_auc_score(y_server_valid, pred_server_proba)


f1_action = f1_score(y_action_valid, pred_action, average="macro")
f1_point = f1_score(y_point_valid, pred_point, average="macro")

print("\n📊 Validation Results:")
print(f"actionId Macro F1: {f1_action:.4f}")
print(f"pointId  Macro F1: {f1_point:.4f}")
print(f"serverGetPoint AUC: {auc_server:.4f}")

score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server
print(f"綜合評分: {score:.4f}")

# =========================================================
# 8️⃣ 測試集預測
# =========================================================
print("\n🧮 產生測試預測中...")
pred_action_test = model_action.predict(X_test)
pred_point_test = model_point.predict(X_test)

# 🌟 MODIFICATION (3/5):
# 為了提交 AUC，我們需要 "機率" 而不是 "類別" (0/1)
# 並且要處理多分類或二分類的情況
if y_server.nunique() > 2:
    # 如果 serverGetPoint 是多分類 (0, 1, 2)
    # 我們需要預測類別，因為 -1 (即 2) 需要被還原
    pred_server_test_labels = model_server.predict(X_test)
else:
    # 如果是二分類 (0, 1)
    # 提交機率
    pred_server_test_proba = model_server.predict_proba(X_test)[:, 1]

# =========================================================
# 9️⃣ 將映射回 -1
# =========================================================
def revert_negative(pred, col_name, original_max_labels_dict):
    """將 max+1 類別轉回 -1"""
    if col_name in original_max_labels_dict:
        replacement_val = original_max_labels_dict[col_name]
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred # 如果沒有 -1，原樣返回

pred_action_test = revert_negative(pred_action_test, "actionId", original_max_labels)
pred_point_test = revert_negative(pred_point_test, "pointId", original_max_labels)

# 🌟 MODIFICATION (4/5): 
# 根據 serverGetPoint 的類別數決定如何處理
if y_server.nunique() > 2:
    pred_server_final = revert_negative(pred_server_test_labels, "serverGetPoint", original_max_labels)
else:
    pred_server_final = pred_server_test_proba # 直接使用機率

# =========================================================
# 🔟 輸出 submission.csv
# =========================================================
# 🌟 MODIFICATION (5/5):
# 1. 'rally_uid' 必須來自 test_last_shot，以確保 row 數量正確
# 2. 'serverGetPoint' 應使用我們最終處理過的 pred_server_final
submission = pd.DataFrame({
    "rally_uid": test_last_shot["rally_uid"],
    "serverGetPoint": pred_server_final,
    "pointId": pred_point_test,
    "actionId": pred_action_test
})

# 確保欄位順序與 sample_submission 一致
sample_sub = pd.read_csv("sample_submission.csv")
submission = submission[sample_sub.columns]

submission.to_csv("submission.csv", index=False)
print("\n✅ 已輸出 submission.csv")
print(f"Submission shape: {submission.shape}")
print(submission.head())
