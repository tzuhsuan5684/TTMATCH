#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 單任務分類模型 (actionId)：僅預測 actionId
---------------------------------------------------------------------
🌟 來源：
- 從 v2 多任務模型重構而來，專注於 actionId 預測。
- 保留了 v2 的滯後特徵 (prev_1, prev_2, prev_3) 和 score_diff 特徵。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import sys

# =========================================================
# 1️⃣ 資料讀取 (無變動)
# =========================================================
def load_data(train_path="train.csv", test_path="test.csv"):
    """讀取訓練集和測試集"""
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"✅ Train shape: {train.shape}")
        print(f"✅ Test shape: {test.shape}")
        return train, test
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到 {train_path} 或 {test_path}。請確保檔案在正確的路徑。")
        sys.exit(1)

# =========================================================
# 2️⃣ 特徵工程 (無變動)
# =========================================================
def create_features(df):
    """為 train 和 test 數據集建立新的序列特徵 (滯後特徵)"""
    df_new = df.copy()
    
    # 確保資料按回合和拍數排序
    df_new = df_new.sort_values(by=['rally_uid', 'strickNumber'])
    
    # 1. 滯後特徵 (Lag Features)
    lag_cols = ['actionId', 'pointId', 'spinId', 'strengthId', 'positionId']
    
    print(f"  > 正在建立 N-1, N-2, N-3 滯後特徵...")
    for col in lag_cols:
        for n in [1, 2, 3]:
            df_new[f'prev_{n}_{col}'] = df_new.groupby('rally_uid')[col].shift(n)

    # 2. 情境特徵 (Context Features) - 分數
    df_new['score_diff'] = df_new['scoreSelf'] - df_new['scoreOther']

    # 填充 shift() 產生的 NaNs
    fill_cols = [col for col in df_new.columns if 'prev_' in col]
    df_new[fill_cols] = df_new[fill_cols].fillna(-1) 

    return df_new

# =========================================================
# 3️⃣ 預處理 (簡化版)
# =========================================================
def preprocess(train_df, test_df):
    """
    1. 修正 actionId 的 -1 類別問題
    2. 取得測試集最後一筆資料
    """
    # 預測時：使用每個 rally_uid 的 "最後一筆" 資料
    test_last_shot = test_df.groupby('rally_uid').tail(1).copy()
    print(f"✅ Test (last shots) shape: {test_last_shot.shape}")

    # 修正 -1 類別 (僅針對 actionId)
    original_max_label = None
    if (train_df["actionId"] == -1).any():
        max_label = train_df["actionId"].max()
        original_max_label = max_label + 1
        print(f"⚠️ actionId 含有 -1，將其替換為 {original_max_label}")
        train_df["actionId"] = train_df["actionId"].replace(-1, original_max_label)
    
    return train_df, test_last_shot, original_max_label

# =========================================================
# 4️⃣ 建立訓練任務 (N -> N+1) (簡化版)
# =========================================================
def create_training_data(train_df, feature_cols):
    """
    重新定義訓練任務 (N -> N+1)
    - 特徵 (X) 是當前擊球 (Shot N)
    - 標籤 (y) 是 "下一球" 的 actionId (Shot N+1)
    """
    # 特徵 (X) 是當前擊球 (Shot N)
    X = train_df[feature_cols].copy().fillna(0)

    # 標籤 (y) 是 "下一球" (Shot N+1) 的 actionId
    y = train_df.groupby('rally_uid')['actionId'].shift(-1)
    
    # 儲存 rally_uid 以便進行 group split
    rally_uids_for_split = train_df['rally_uid']

    # 刪除沒有 "下一球" 的行 (即每個回合的最後一球)
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    rally_uids_for_split = rally_uids_for_split[valid_indices]

    print(f"✅ 重新建立訓練集 (N -> N+1)，新 shape: {X.shape}")
    
    return X, y.astype(int), rally_uids_for_split

# =========================================================
# 5️⃣ 建立無洩漏的驗證集 (Group Split) (簡化版)
# =========================================================
def create_group_split(X, y, rally_uids):
    """
    使用 Group Split 建立無洩漏的驗證集
    """
    print("🧩 建立無洩漏的驗證集中 (Group Split)...")
    unique_rallies = rally_uids.unique()
    train_rallies, valid_rallies = train_test_split(unique_rallies, test_size=0.2, random_state=42)

    train_mask = rally_uids.isin(train_rallies)
    valid_mask = rally_uids.isin(valid_rallies)

    X_train, X_valid = X[train_mask], X[valid_mask]
    y_train, y_valid = y[train_mask], y[valid_mask]
    
    return X_train, X_valid, y_train, y_valid

# =========================================================
# 6️⃣ 特徵選取 (無變動的核心函式)
# =========================================================
def select_features(X, y, objective, num_class=None, top_k=30):
    """
    使用 XGBoost 先訓練一輪，選出最重要的前 K 個特徵。
    """
    selector = VarianceThreshold(threshold=0.0)
    X_var = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    
    X_var = pd.DataFrame(X_var, columns=selected_cols, index=X.index)

    model_params = {
        "objective": objective,
        "eval_metric": "mlogloss",
        "learning_rate": 0.1, "max_depth": 5, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "tree_method": "hist",
        "num_class": num_class
    }

    model_tmp = xgb.XGBClassifier(**model_params)
    model_tmp.fit(X_var, y)

    importances = model_tmp.feature_importances_
    importance_df = pd.DataFrame({
        "feature": selected_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_k)["feature"].tolist()
    return top_features

# =========================================================
# 7️⃣ XGBoost 訓練函式 (無變動)
# =========================================================
def train_xgb(X_train, y_train, X_valid, y_valid, objective, num_class):
    """訓練 XGBoost 模型的通用函式"""
    params = {
        "objective": objective,
        "eval_metric": "mlogloss",
        "learning_rate": 0.1,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 200,
        "random_state": 42,
        "tree_method": "hist",
        "early_stopping_rounds": 30,
        "num_class": num_class
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    return model

# =========================================================
# 8️⃣ 標籤還原 (無變動的核心函式)
# =========================================================
def revert_negative(pred, replacement_val):
    """將 max+1 類別轉回 -1"""
    if replacement_val is not None:
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

# =========================================================
# 9️⃣ 輸出 submission.csv (修改版)
# =========================================================
def save_submission(test_last_shot, pred_action, sample_path="sample_submission.csv", output_path="submission_actionId.csv"):
    """
    讀取 sample_submission，僅更新 actionId 欄位後儲存
    """
    try:
        submission = pd.read_csv(sample_path)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到 {sample_path}。無法建立提交檔案。")
        return
        
    # 確保 rally_uid 對齊
    submission = submission.set_index('rally_uid')
    
    # 建立一個包含預測結果的 Series，並以 rally_uid 為索引
    pred_df = pd.DataFrame({
        "rally_uid": test_last_shot["rally_uid"],
        "actionId": pred_action
    }).set_index('rally_uid')

    # 更新 actionId 欄位
    submission['actionId'].update(pred_df['actionId'])
    submission['actionId'] = submission['actionId'].astype(int)

    # 恢復索引並儲存
    submission.reset_index(inplace=True)
    submission.to_csv(output_path, index=False)
    print(f"\n✅ 已輸出 {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

# =========================================================
# 🚀 主執行流程
# =========================================================
def main():
    # --- 參數設定 ---
    K_FEATURES = 40
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"
    SUBMISSION_PATH = "submission_actionId.csv"

    # --- 1. 讀取資料 ---
    train, test = load_data(TRAIN_PATH, TEST_PATH)

    # --- 2. 特徵工程 ---
    print("⚙️ 正在為 train 建立滯後特徵...")
    train = create_features(train)
    print("⚙️ 正在為 test 建立滯後特徵...")
    test = create_features(test)

    # --- 3. 預處理 ---
    target_cols = ["actionId", "pointId", "serverGetPoint"]
    drop_cols = ["rally_uid", "rally_id"] 
    feature_cols = [c for c in train.columns if c not in target_cols + drop_cols and c in test.columns]
    print(f"✅ 使用 {len(feature_cols)} 個特徵進行訓練。")
    
    train, test_last_shot, original_max_label = preprocess(train, test)

    # --- 4. 建立 N -> N+1 訓練資料 ---
    X, y, rally_uids_for_split = create_training_data(train, feature_cols)
    X_test = test_last_shot[feature_cols].copy().fillna(0)

    # --- 5. 建立 Group Split ---
    X_train, X_valid, y_train, y_valid = create_group_split(X, y, rally_uids_for_split)
    
    num_class = y.nunique()

    # --- 6. 特徵選取 ---
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features = select_features(X_train, y_train, 
                                     objective="multi:softmax", 
                                     num_class=num_class, 
                                     top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features[:5]}")
    
    X_train_fs = X_train[top_features]
    X_valid_fs = X_valid[top_features]
    X_test_fs = X_test[top_features]

    # --- 7. 訓練模型 ---
    print("\n🚀 訓練 actionId 模型中...")
    model = train_xgb(X_train_fs, y_train, X_valid_fs, y_valid,
                      objective="multi:softmax", num_class=num_class)

    # --- 8. 評估模型 ---
    print("\n📊 Validation Results:")
    pred_valid = model.predict(X_valid_fs)
    f1_action = f1_score(y_valid, pred_valid, average="macro")
    print(f"actionId Macro F1: {f1_action:.4f}")

    # --- 9. 產生預測 ---
    print("\n🧮 產生測試預測中...")
    pred_test = model.predict(X_test_fs)
    pred_test_reverted = revert_negative(pred_test, original_max_label)
    
    # --- 10. 儲存提交檔案 ---
    save_submission(test_last_shot, pred_test_reverted, SAMPLE_SUB_PATH, SUBMISSION_PATH)

if __name__ == "__main__":
    main()