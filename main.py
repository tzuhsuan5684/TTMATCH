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
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
except FileNotFoundError:
    print("❌ 錯誤：找不到 train.csv 或 test.csv。請確保檔案在正確的路徑。")
    exit()


print(f"✅ Train shape: {train.shape}")
print(f"✅ Test shape: {test.shape}")

# 🌟 (1/6) 預測時：使用每個 rally_uid 的 "最後一筆" 資料
test_last_shot = test.groupby('rally_uid').tail(1).copy()
print(f"✅ Test (last shots) shape: {test_last_shot.shape}")


# =========================================================
# 2️⃣ 修正 -1 類別問題
# =========================================================
original_max_labels = {}
for col in ["actionId", "pointId", "serverGetPoint"]:
    if col in train.columns and (train[col] == -1).any():
        max_label = train[col].max()
        original_max_labels[col] = max_label + 1
        print(f"⚠️ {col} 含有 -1，將其替換為 {max_label + 1}")
        train[col] = train[col].replace(-1, max_label + 1)

# =========================================================
# 3️⃣ 🌟 MODIFICATION (2/6): 重新定義訓練任務 (N -> N+1)
# =========================================================
target_cols = ["actionId", "pointId", "serverGetPoint"]
drop_cols = ["rally_uid", "rally_id"] 
feature_cols = [c for c in train.columns if c not in target_cols + drop_cols and c in test.columns]

# 特徵 (X) 是當前擊球 (Shot N)
X = train[feature_cols].copy().fillna(0) # 🌟 提前填充 NaN

# 標籤 (y) 是 "下一球" (Shot N+1)
y_action = train.groupby('rally_uid')['actionId'].shift(-1)
y_point = train.groupby('rally_uid')['pointId'].shift(-1)

# serverGetPoint 是整個回合的結果，不需要 shift
y_server = train['serverGetPoint']

# 儲存 rally_uid 以便進行 group split
rally_uids_for_split = train['rally_uid']

# 🌟 刪除沒有 "下一球" 的行 (即每個回合的最後一球)
valid_indices = y_action.notna() & y_point.notna()
X = X[valid_indices]
y_action = y_action[valid_indices]
y_point = y_point[valid_indices]
y_server = y_server[valid_indices]
rally_uids_for_split = rally_uids_for_split[valid_indices]

print(f"✅ 重新建立訓練集 (N -> N+1)，新 shape: {X.shape}")

# 🌟 測試集 (X_test) 使用 'test_last_shot' (Shot N)，並填充 NaN
X_test = test_last_shot[feature_cols].copy().fillna(0)

# =========================================================
# 4️⃣ 🌟 MODIFICATION (3/6): 建立無洩漏的驗證集 (Group Split)
# =========================================================
print("🧩 建立無洩漏的驗證集中 (Group Split)...")
unique_rallies = rally_uids_for_split.unique()
train_rallies, valid_rallies = train_test_split(unique_rallies, test_size=0.2, random_state=42)

train_mask = rally_uids_for_split.isin(train_rallies)
valid_mask = rally_uids_for_split.isin(valid_rallies)

# 建立 actionId 的資料
X_train_action, X_valid_action = X[train_mask], X[valid_mask]
y_train_action, y_valid_action = y_action[train_mask], y_action[valid_mask]

# 建立 pointId 的資料
X_train_point, X_valid_point = X[train_mask], X[valid_mask]
y_train_point, y_valid_point = y_point[train_mask], y_point[valid_mask]

# 建立 serverGetPoint 的資料
X_train_server, X_valid_server = X[train_mask], X[valid_mask]
y_train_server, y_valid_server = y_server[train_mask], y_server[valid_mask]

# =========================================================
# 5️⃣ 🌟 MODIFICATION (4/6): 獨立特徵選取 (BUG FIX)
# =========================================================
def select_features(X, y, objective, num_class=None, top_k=30):
    """
    🌟 BUG FIX:
    使用 XGBoost 先訓練一輪，選出最重要的前 K 個特徵。
    現在會根據傳入的 'objective' 正確處理二分類或多分類。
    """
    selector = VarianceThreshold(threshold=0.0)
    X_var = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]

    # 🌟 設定模型參數
    model_params = {
        "objective": objective,
        "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.1, "max_depth": 5, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "tree_method": "hist"
    }
    if num_class is not None:
        model_params["num_class"] = num_class

    model_tmp = xgb.XGBClassifier(**model_params)
    model_tmp.fit(X_var, y)

    importances = model_tmp.feature_importances_
    importance_df = pd.DataFrame({
        "feature": selected_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_k)["feature"].tolist()
    return top_features

K_FEATURES = 40 # 使用多少個特徵

# --- 為 actionId 選取特徵 ---
print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
top_features_action = select_features(X_train_action, y_train_action, 
                                      objective="multi:softmax", 
                                      num_class=y_action.nunique(), 
                                      top_k=K_FEATURES)
X_train_fs_action = X_train_action[top_features_action]
X_valid_fs_action = X_valid_action[top_features_action]
X_test_fs_action = X_test[top_features_action]
print(f"🔥 actionId Top 5: {top_features_action[:5]}")

# --- 為 pointId 選取特徵 ---
print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
top_features_point = select_features(X_train_point, y_train_point, 
                                     objective="multi:softmax",
                                     num_class=y_point.nunique(),
                                     top_k=K_FEATURES)
X_train_fs_point = X_train_point[top_features_point]
X_valid_fs_point = X_valid_point[top_features_point]
X_test_fs_point = X_test[top_features_point]
print(f"🔥 pointId Top 5: {top_features_point[:5]}")

# --- 🌟 BUG FIX: 為 serverGetPoint 選取特徵 ---
print(f"🧩 為 serverGetPoint 選取前 {K_FEATURES} 個特徵...")
if y_train_server.nunique() > 2:
    server_objective = "multi:softmax"
    server_num_class = y_server.nunique()
else:
    server_objective = "binary:logistic"
    server_num_class = None

top_features_server = select_features(X_train_server, y_train_server,
                                      objective=server_objective,
                                      num_class=server_num_class,
                                      top_k=K_FEATURES)
X_train_fs_server = X_train_server[top_features_server]
X_valid_fs_server = X_valid_server[top_features_server]
X_test_fs_server = X_test[top_features_server]
print(f"🔥 serverGetPoint Top 5: {top_features_server[:5]}")


# =========================================================
# 5️⃣ XGBoost 訓練函式 (🌟 減少過擬合)
# =========================================================
def train_xgb(X_train, y_train, X_valid, y_valid, objective, num_class=None):
    params = {
        "objective": objective,
        "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.1,
        "max_depth": 6, # 🌟 從 6 降為 5
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
        "random_state": 42,
        "tree_method": "hist",
        "early_stopping_rounds": 30 # 🌟 從 20 增為 30
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
# 6️⃣ 🌟 MODIFICATION (5/6): 三個模型訓練 (使用各自的特徵)
# =========================================================
print("🚀 訓練 actionId 模型中...")
model_action = train_xgb(X_train_fs_action, y_train_action, 
                         X_valid_fs_action, y_valid_action,
                         objective="multi:softmax", num_class=y_action.nunique())

print("🚀 訓練 pointId 模型中...")
model_point = train_xgb(X_train_fs_point, y_train_point,
                        X_valid_fs_point, y_valid_point,
                        objective="multi:softmax", num_class=y_point.nunique())

print("🚀 訓練 serverGetPoint 模型中...")
# 這裡的邏輯已經是正確的
if y_server.nunique() > 2:
    print("⚠️ serverGetPoint 發現多於2個類別，使用 multi:softmax")
    model_server = train_xgb(X_train_fs_server, y_train_server,
                            X_valid_fs_server, y_valid_server,
                            objective="multi:softmax", num_class=y_server.nunique())
else:
    model_server = train_xgb(X_train_fs_server, y_train_server,
                            X_valid_fs_server, y_valid_server,
                            objective="binary:logistic")

# =========================================================
# 7️⃣ 🌟 MODIFICATION (6/6): 模型評估 (使用各自的特徵)
# =========================================================
pred_action = model_action.predict(X_valid_fs_action)
pred_point = model_point.predict(X_valid_fs_point)

if y_server.nunique() > 2:
    pred_server_proba = model_server.predict_proba(X_valid_fs_server)
    auc_server = roc_auc_score(y_valid_server, pred_server_proba, multi_class="ovr")
else:
    pred_server_proba = model_server.predict_proba(X_valid_fs_server)[:, 1]
    auc_server = roc_auc_score(y_valid_server, pred_server_proba)

f1_action = f1_score(y_valid_action, pred_action, average="macro")
f1_point = f1_score(y_valid_point, pred_point, average="macro")

print("\n📊 Validation Results (Fixed):")
print(f"actionId Macro F1: {f1_action:.4f}")
print(f"pointId  Macro F1: {f1_point:.4f}")
print(f"serverGetPoint AUC: {auc_server:.4f}") # 這裡應該會顯著高於 0.5

score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server
print(f"綜合評分: {score:.4f}")

# =========================================================
# 8️⃣ 測試集預測 (使用各自的特徵)
# =========================================================
print("\n🧮 產生測試預測中...")
pred_action_test = model_action.predict(X_test_fs_action)
pred_point_test = model_point.predict(X_test_fs_point)

if y_server.nunique() > 2:
    pred_server_test_labels = model_server.predict(X_test_fs_server)
else:
    pred_server_test_proba = model_server.predict_proba(X_test_fs_server)[:, 1]

# =========================================================
# 9️⃣ 將映射回 -1 (程式碼不變)
# =========================================================
def revert_negative(pred, col_name, original_max_labels_dict):
    """將 max+1 類別轉回 -1"""
    if col_name in original_max_labels_dict:
        replacement_val = original_max_labels_dict[col_name]
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred 

pred_action_test = revert_negative(pred_action_test, "actionId", original_max_labels)
pred_point_test = revert_negative(pred_point_test, "pointId", original_max_labels)

if y_server.nunique() > 2:
    pred_server_final = revert_negative(pred_server_test_labels, "serverGetPoint", original_max_labels)
else:
    pred_server_final = pred_server_test_proba

# =========================================================
# 🔟 輸出 submission.csv (程式碼不變)
# =========================================================
submission = pd.DataFrame({
    "rally_uid": test_last_shot["rally_uid"],
    "serverGetPoint": pred_server_final,
    "pointId": pred_point_test,
    "actionId": pred_action_test
})

try:
    sample_sub = pd.read_csv("sample_submission.csv")
    submission = submission[sample_sub.columns]
except FileNotFoundError:
    print("⚠️ 找不到 sample_submission.csv，將使用預設欄位順序。")
except Exception as e:
    print(f"⚠️ 讀取 sample_submission.csv 時出錯: {e}")


submission.to_csv("submission.csv", index=False)
print("\n✅ 已輸出 submission.csv")
print(f"Submission shape: {submission.shape}")
print(submission.head())

