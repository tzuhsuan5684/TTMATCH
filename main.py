#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 多任務分類模型 (重構版 v2)：Predict actionId / pointId / serverGetPoint
---------------------------------------------------------------------
🌟 v2 更新：
- 新增 `create_features` 函式。
- 加入 3 拍的滯後特徵 (prev_1, prev_2, prev_3)。
- 加入 `score_diff` 情境特徵。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import sys

# =========================================================
# 1️⃣ 資料讀取
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
# 🌟 2️⃣ 特徵工程 (NEW)
# =========================================================
def create_features(df):
    """為 train 和 test 數據集建立新的序列特徵 (滯後特徵)"""
    df_new = df.copy()
    
    # 確保資料按回合和拍數排序
    df_new = df_new.sort_values(by=['rally_uid', 'strickNumber'])
    
    # 1. 滯後特徵 (Lag Features)
    # 選擇要滯後的欄位
    lag_cols = ['actionId', 'pointId', 'spinId', 'strengthId', 'positionId']
    
    # 建立 N-1, N-2, N-3 的滯後特徵
    print(f"  > 正在建立 N-1, N-2, N-3 滯後特徵...")
    for col in lag_cols:
        for n in [1, 2, 3]:
            # .shift(n) 獲取 (N-n) 拍的數據
            df_new[f'prev_{n}_{col}'] = df_new.groupby('rally_uid')[col].shift(n)

    # 2. 情境特徵 (Context Features) - 分數
    df_new['score_diff'] = df_new['scoreSelf'] - df_new['scoreOther']

    # 🌟 填充 shift() 產生的 NaNs
    # 用 -1 填充，以區別於 0 (0 可能是一個有效的 ID)
    fill_cols = [col for col in df_new.columns if 'prev_' in col]
    df_new[fill_cols] = df_new[fill_cols].fillna(-1) 

    return df_new

# =========================================================
# 3️⃣ 預處理
# =========================================================
def preprocess(train_df, test_df):
    """
    1. 修正 -1 類別問題
    2. 取得測試集最後一筆資料
    """
    # 🌟 (1/6) 預測時：使用每個 rally_uid 的 "最後一筆" 資料
    test_last_shot = test_df.groupby('rally_uid').tail(1).copy()
    print(f"✅ Test (last shots) shape: {test_last_shot.shape}")

    # 修正 -1 類別
    original_max_labels = {}
    for col in ["actionId", "pointId", "serverGetPoint"]:
        if col in train_df.columns and (train_df[col] == -1).any():
            max_label = train_df[col].max()
            original_max_labels[col] = max_label + 1
            print(f"⚠️ {col} 含有 -1，將其替換為 {max_label + 1}")
            train_df[col] = train_df[col].replace(-1, max_label + 1)
    
    return train_df, test_last_shot, original_max_labels

# =========================================================
# 4️⃣ 建立訓練任務 (N -> N+1)
# =========================================================
def create_training_data(train_df, feature_cols):
    """
    重新定義訓練任務 (N -> N+1)
    - 特徵 (X) 是當前擊球 (Shot N)
    - 標籤 (y) 是 "下一球" (Shot N+1)
    """
    # 特徵 (X) 是當前擊球 (Shot N)
    X = train_df[feature_cols].copy().fillna(0) # 🌟 提前填充 NaN

    # 標籤 (y) 是 "下一球" (Shot N+1)
    y_action = train_df.groupby('rally_uid')['actionId'].shift(-1)
    y_point = train_df.groupby('rally_uid')['pointId'].shift(-1)
    y_server = train_df['serverGetPoint'] # serverGetPoint 是回合結果，不需 shift

    # 儲存 rally_uid 以便進行 group split
    rally_uids_for_split = train_df['rally_uid']

    # 🌟 刪除沒有 "下一球" 的行 (即每個回合的最後一球)
    valid_indices = y_action.notna() & y_point.notna()
    X = X[valid_indices]
    y_action = y_action[valid_indices]
    y_point = y_point[valid_indices]
    y_server = y_server[valid_indices]
    rally_uids_for_split = rally_uids_for_split[valid_indices]

    print(f"✅ 重新建立訓練集 (N -> N+1)，新 shape: {X.shape}")
    
    return X, y_action, y_point, y_server, rally_uids_for_split

# =========================================================
# 5️⃣ 建立無洩漏的驗證集 (Group Split)
# =========================================================
def create_group_split(X, y_action, y_point, y_server, rally_uids):
    """
    使用 Group Split 建立無洩漏的驗證集
    """
    print("🧩 建立無洩漏的驗證集中 (Group Split)...")
    unique_rallies = rally_uids.unique()
    train_rallies, valid_rallies = train_test_split(unique_rallies, test_size=0.2, random_state=42)

    train_mask = rally_uids.isin(train_rallies)
    valid_mask = rally_uids.isin(valid_rallies)

    # 建立 train/valid 資料集
    data = {
        'action': (X[train_mask], X[valid_mask], y_action[train_mask], y_action[valid_mask]),
        'point': (X[train_mask], X[valid_mask], y_point[train_mask], y_point[valid_mask]),
        'server': (X[train_mask], X[valid_mask], y_server[train_mask], y_server[valid_mask])
    }
    
    return data, (y_action, y_point, y_server) # 回傳 y_all 以便計算 nunique

# =========================================================
# 6️⃣ 獨立特徵選取
# =========================================================
def select_features(X, y, objective, num_class=None, top_k=30):
    """
    使用 XGBoost 先訓練一輪，選出最重要的前 K 個特徵。
    """
    selector = VarianceThreshold(threshold=0.0)
    X_var = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    
    # 確保 X_var 是 DataFrame
    X_var = pd.DataFrame(X_var, columns=selected_cols, index=X.index)

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

def apply_feature_selection(split_data, y_all, X_test, K_FEATURES):
    """
    為三個目標分別進行特徵選取
    """
    X_train_action, X_valid_action, y_train_action, _ = split_data['action']
    X_train_point, X_valid_point, y_train_point, _ = split_data['point']
    X_train_server, X_valid_server, y_train_server, _ = split_data['server']
    
    y_action_all, y_point_all, y_server_all = y_all

    # --- 為 actionId 選取特徵 ---
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features_action = select_features(X_train_action, y_train_action, 
                                          objective="multi:softmax", 
                                          num_class=y_action_all.nunique(), 
                                          top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features_action[:5]}")

    # --- 為 pointId 選取特徵 ---
    print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
    top_features_point = select_features(X_train_point, y_train_point, 
                                         objective="multi:softmax",
                                         num_class=y_point_all.nunique(),
                                         top_k=K_FEATURES)
    print(f"🔥 pointId Top 5: {top_features_point[:5]}")

    # --- 為 serverGetPoint 選取特徵 ---
    print(f"🧩 為 serverGetPoint 選取前 {K_FEATURES} 個特徵...")
    if y_train_server.nunique() > 2:
        server_objective = "multi:softmax"
        server_num_class = y_server_all.nunique()
    else:
        server_objective = "binary:logistic"
        server_num_class = None

    top_features_server = select_features(X_train_server, y_train_server,
                                          objective=server_objective,
                                          num_class=server_num_class,
                                          top_k=K_FEATURES)
    print(f"🔥 serverGetPoint Top 5: {top_features_server[:5]}")

    # 建立最終的特徵集
    fs_data = {
        'action': (X_train_action[top_features_action], X_valid_action[top_features_action], X_test[top_features_action]),
        'point': (X_train_point[top_features_point], X_valid_point[top_features_point], X_test[top_features_point]),
        'server': (X_train_server[top_features_server], X_valid_server[top_features_server], X_test[top_features_server])
    }
    
    return fs_data

# =========================================================
# 7️⃣ XGBoost 訓練函式
# =========================================================
def train_xgb(X_train, y_train, X_valid, y_valid, objective, num_class=None):
    """訓練 XGBoost 模型的通用函式"""
    params = {
        "objective": objective,
        "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.05,
        "max_depth": 9, # 遵照前一版修正 (6 -> 5)
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_estimators": 100,
        "random_state": 42,
        "tree_method": "hist",
        "early_stopping_rounds": 30 # 遵照前一版修正 (20 -> 30)
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

def train_xgb_with_search(X_train, X_valid, y_train, y_valid, num_class, top_features, objective="multi:softmax", n_iter=25):
    """
    用 RandomizedSearchCV + class_weight + early stopping 訓練 XGBoost 多分類模型
    """
    X_train_fs = X_train[top_features]
    X_valid_fs = X_valid[top_features]

    X_search = pd.concat([X_train_fs, X_valid_fs])
    y_search = pd.concat([y_train, y_valid])

    test_fold = np.zeros(len(X_search))
    test_fold[:len(X_train_fs)] = -1
    ps = PredefinedSplit(test_fold)

    search_weights = compute_sample_weight(class_weight='balanced', y=y_search)

    fit_params = {
        "eval_set": [(X_valid_fs, y_valid)],
        "verbose": False
    }
    if xgb.__version__ >= "2.0.0":
        valid_weights = compute_sample_weight(class_weight='balanced', y=y_valid)
        fit_params["sample_weight_eval_set"] = [valid_weights]

    param_dist = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300, 400],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }

    base_model = xgb.XGBClassifier(
        objective=objective,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        num_class=num_class,
        early_stopping_rounds=30
    )

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=ps,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    rand_search.fit(
        X_search,
        y_search,
        sample_weight=search_weights,
        **fit_params
    )

    print(f"✅ {objective} 最佳參數: {rand_search.best_params_}")
    print(f"✅ {objective} 最佳 F1 Macro (Val): {rand_search.best_score_:.4f}")

    return rand_search.best_estimator_

def select_features_xgb(X, y, num_class, top_k=40, objective="multi:softmax"):
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

def revert_negative_pointid(pred, replacement_val):
    """將 max+1 類別轉回 -1（for pointId）"""
    if replacement_val is not None:
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

# =========================================================
# 8️⃣ 三個模型訓練
# =========================================================
def train_all_models(fs_data, split_data, y_all):
    """
    使用各自選取的特徵集訓練三個模型
    """
    models = {}
    
    # 取得標籤
    _, _, y_train_action, y_valid_action = split_data['action']
    _, _, y_train_point, y_valid_point = split_data['point']
    _, _, y_train_server, y_valid_server = split_data['server']
    
    # 取得特徵
    X_train_fs_action, X_valid_fs_action, _ = fs_data['action']
    X_train_fs_point, X_valid_fs_point, _ = fs_data['point']
    X_train_fs_server, X_valid_fs_server, _ = fs_data['server']
    
    y_action_all, y_point_all, y_server_all = y_all

    print("🚀 訓練 actionId 模型中...")
    models['action'] = train_xgb(X_train_fs_action, y_train_action, 
                                 X_valid_fs_action, y_valid_action,
                                 objective="multi:softmax", num_class=y_action_all.nunique())

    print("🚀 訓練 pointId 模型中...")
    models['point'] = train_xgb(X_train_fs_point, y_train_point,
                                X_valid_fs_point, y_valid_point,
                                objective="multi:softmax", num_class=y_point_all.nunique())

    print("🚀 訓練 serverGetPoint 模型中...")
    if y_server_all.nunique() > 2:
        print("⚠️ serverGetPoint 發現多於2個類別，使用 multi:softmax")
        models['server'] = train_xgb(X_train_fs_server, y_train_server,
                                      X_valid_fs_server, y_valid_server,
                                      objective="multi:softmax", num_class=y_server_all.nunique())
    else:
        models['server'] = train_xgb(X_train_fs_server, y_train_server,
                                      X_valid_fs_server, y_valid_server,
                                      objective="binary:logistic")
                                      
    return models

# =========================================================
# 9️⃣ 模型評估
# =========================================================
def evaluate_models(models, fs_data, split_data, y_all):
    """
    在驗證集上評估模型
    """
    # 取得標籤
    _, _, _, y_valid_action = split_data['action']
    _, _, _, y_valid_point = split_data['point']
    _, _, _, y_valid_server = split_data['server']
    
    # 取得特徵
    _, X_valid_fs_action, _ = fs_data['action']
    _, X_valid_fs_point, _ = fs_data['point']
    _, X_valid_fs_server, _ = fs_data['server']

    y_server_all = y_all[2]
    
    # 預測
    pred_action = models['action'].predict(X_valid_fs_action)
    pred_point = models['point'].predict(X_valid_fs_point)

    if y_server_all.nunique() > 2:
        pred_server_proba = models['server'].predict_proba(X_valid_fs_server)
        auc_server = roc_auc_score(y_valid_server, pred_server_proba, multi_class="ovr")
    else:
        pred_server_proba = models['server'].predict_proba(X_valid_fs_server)[:, 1]
        auc_server = roc_auc_score(y_valid_server, pred_server_proba)

    f1_action = f1_score(y_valid_action, pred_action, average="macro")
    f1_point = f1_score(y_valid_point, pred_point, average="macro")

    print("\n📊 Validation Results (Fixed):")
    print(f"actionId Macro F1: {f1_action:.4f}")
    print(f"pointId  Macro F1: {f1_point:.4f}")
    print(f"serverGetPoint AUC: {auc_server:.4f}")

    score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server
    print(f"綜合評分: {score:.4f}")

# =========================================================
# 🔟 測試集預測 & 標籤還原
# =========================================================
def revert_negative(pred, col_name, original_max_labels_dict):
    """將 max+1 類別轉回 -1"""
    if col_name in original_max_labels_dict:
        replacement_val = original_max_labels_dict[col_name]
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

def generate_predictions(models, fs_data, y_all, original_max_labels):
    """
    產生測試集預測並還原 -1 標籤
    """
    print("\n🧮 產生測試預測中...")
    
    _, _, X_test_fs_action = fs_data['action']
    _, _, X_test_fs_point = fs_data['point']
    _, _, X_test_fs_server = fs_data['server']
    
    y_server_all = y_all[2]

    # 預測
    pred_action_test = models['action'].predict(X_test_fs_action)
    pred_point_test = models['point'].predict(X_test_fs_point)

    if y_server_all.nunique() > 2:
        pred_server_test_labels = models['server'].predict(X_test_fs_server)
    else:
        pred_server_test_proba = models['server'].predict_proba(X_test_fs_server)[:, 1]

    # 還原 -1
    pred_action_test = revert_negative(pred_action_test, "actionId", original_max_labels)
    pred_point_test = revert_negative(pred_point_test, "pointId", original_max_labels)

    if y_server_all.nunique() > 2:
        pred_server_final = revert_negative(pred_server_test_labels, "serverGetPoint", original_max_labels)
    else:
        pred_server_final = pred_server_test_proba # 機率不用還原
        
    return pred_action_test, pred_point_test, pred_server_final

# =========================================================
# 1️⃣1️⃣ 輸出 submission.csv
# =========================================================
def save_submission(test_last_shot, pred_action, pred_point, pred_server, 
                    sample_path="sample_submission.csv", output_path="submission.csv"):
    """
    儲存提交檔案
    """
    submission = pd.DataFrame({
        "rally_uid": test_last_shot["rally_uid"],
        "serverGetPoint": pred_server,
        "pointId": pred_point,
        "actionId": pred_action
    })

    try:
        sample_sub = pd.read_csv(sample_path)
        submission = submission[sample_sub.columns]
    except FileNotFoundError:
        print(f"⚠️ 找不到 {sample_path}，將使用預設欄位順序。")
    except Exception as e:
        print(f"⚠️ 讀取 {sample_path} 時出錯: {e}")

    submission.to_csv(output_path, index=False)
    print(f"\n✅ 已輸出 {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

# =========================================================
# 🚀 主執行流程
# =========================================================
def main():
    # --- 參數設定 ---
    K_FEATURES = 20
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"
    SUBMISSION_PATH = "submission.csv"

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

    train, test_last_shot, original_max_labels = preprocess(train, test)

    # --- 4. 建立 N -> N+1 訓練資料 ---
    X, y_action, y_point, y_server, rally_uids_for_split = create_training_data(train, feature_cols)
    X_test = test_last_shot[feature_cols].copy().fillna(0)

    # --- 5. 建立 Group Split ---
    split_data, y_all = create_group_split(X, y_action, y_point, y_server, rally_uids_for_split)

    # --- 5.1 actionId split ---
    X_train_action, X_valid_action, y_train_action, y_valid_action = split_data['action']
    num_class_action = y_all[0].nunique()
    print(f"✅ actionId 類別數量: {num_class_action}")

    # --- 6. 特徵選取 (for actionId) ---
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features_action = select_features_xgb(X_train_action, y_train_action, num_class_action, top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features_action[:5]}")

    # --- 7. 訓練 actionId 模型 (RandomizedSearchCV) ---
    print("🚀 訓練 actionId 模型 (RandomizedSearchCV)...")
    actionid_model = train_xgb_with_search(X_train_action, X_valid_action, y_train_action, y_valid_action, num_class_action, top_features_action)

    # --- 5.1 pointId split ---
    X_train_point, X_valid_point, y_train_point, y_valid_point = split_data['point']
    num_class_point = y_all[1].nunique()
    print(f"✅ pointId 類別數量: {num_class_point}")

    # --- 6. 特徵選取 (for pointId) ---
    print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
    top_features_point = select_features_xgb(X_train_point, y_train_point, num_class_point, top_k=K_FEATURES)
    print(f"🔥 pointId Top 5: {top_features_point[:5]}")

    # --- 7. 訓練 pointId 模型 (RandomizedSearchCV) ---
    print("🚀 訓練 pointId 模型 (RandomizedSearchCV)...")
    pointid_model = train_xgb_with_search(X_train_point, X_valid_point, y_train_point, y_valid_point, num_class_point, top_features_point)

    # --- 8. serverGetPoint 用原本流程（或同樣流程，視需求） ---
    # 用原本 main.py 的流程
    fs_data = apply_feature_selection(split_data, y_all, X_test, K_FEATURES)
    models = train_all_models(fs_data, split_data, y_all)

    # --- 9. 評估模型 ---
    evaluate_models(models, fs_data, split_data, y_all)

    # --- 10. 產生預測 ---
    # actionId/serverGetPoint 用原本流程
    pred_action, _, pred_server = generate_predictions(models, fs_data, y_all, original_max_labels)

    # pointId 用新模型
    X_test_point_fs = X_test[top_features_point]
    pred_point_test = pointid_model.predict(X_test_point_fs)
    pred_point_test = revert_negative_pointid(pred_point_test, original_max_labels.get("pointId", None))

    # --- 11. 儲存提交檔案 ---
    save_submission(test_last_shot, pred_action, pred_point_test, pred_server,
                    SAMPLE_SUB_PATH, SUBMISSION_PATH)

if __name__ == "__main__":
    main()

