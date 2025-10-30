#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 多任務分類模型 (重構版 v2.6) - 主執行檔
---------------------------------------------------------------------
🌟 v2.6 更新：
- 新增 `log_experiment_results` 函式，將每次運行的分數和參數記錄到 'experiment_log.csv'。
- 修正 `main()` 函式中的邏輯，確保「評估」和「預測」使用的是同一組模型。
- 移除 `evaluate_models`、`generate_predictions`、`train_all_models`、`apply_feature_selection`
  等函式，將其簡化並整合到 `main()` 流程中，以確保邏輯一致性。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import csv  # 🌟 新增
import os   # 🌟 新增
from datetime import datetime # 🌟 新增
import json # 🌟 新增 (用於記錄字典)


# 從 data_processing.py 匯入所有資料處理函式
from data_processing import (
    load_data,
    create_features,
    preprocess,
    create_training_data,
    create_group_split
)

# =========================================================
# 6️⃣ 獨立特徵選取
# =========================================================
def select_features(X, y, objective, num_class=None, top_k=30):
    """使用 XGBoost 先訓練一輪，選出最重要的前 K 個特徵。"""
    selector = VarianceThreshold(threshold=0.0)
    X_var = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    X_var = pd.DataFrame(X_var, columns=selected_cols, index=X.index)

    model_params = {
        "objective": objective, "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.1, "max_depth": 5, "n_estimators": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "tree_method": "hist"
    }
    if num_class is not None:
        model_params["num_class"] = num_class

    model_tmp = xgb.XGBClassifier(**model_params)
    model_tmp.fit(X_var, y)

    importances = model_tmp.feature_importances_
    importance_df = pd.DataFrame({"feature": selected_cols, "importance": importances}).sort_values("importance", ascending=False)
    return importance_df.head(top_k)["feature"].tolist()

def select_features_xgb(X, y, num_class, top_k=40, objective="multi:softmax"):
    """與 select_features 類似，但為 RandomizedSearch 流程客製化"""
    selector = VarianceThreshold(threshold=0.0)
    X_var = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    X_var = pd.DataFrame(X_var, columns=selected_cols, index=X.index)

    model_params = {
        "objective": objective, "eval_metric": "mlogloss", "learning_rate": 0.1, 
        "max_depth": 5, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "tree_method": "hist", "num_class": num_class
    }
    model_tmp = xgb.XGBClassifier(**model_params)
    model_tmp.fit(X_var, y)
    importances = model_tmp.feature_importances_
    importance_df = pd.DataFrame({"feature": selected_cols, "importance": importances}).sort_values("importance", ascending=False)
    return importance_df.head(top_k)["feature"].tolist()


# =========================================================
# 7️⃣ XGBoost 訓練函式
# =========================================================
def train_xgb(X_train, y_train, X_valid, y_valid, objective, num_class=None):
    """訓練 XGBoost 模型的通用函式（用於 serverGetPoint）"""
    params = {
        "objective": objective, "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        "learning_rate": 0.05, "max_depth": 9, "subsample": 0.9,
        "colsample_bytree": 0.9, "n_estimators": 100, "random_state": 42,
        "tree_method": "hist", "early_stopping_rounds": 30
    }
    if num_class is not None:
        params["num_class"] = num_class

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model

def train_xgb_with_search(X_train, X_valid, y_train, y_valid, num_class, top_features, 
                            objective="multi:softmax", n_iter=25,
                            custom_weight_adjustments=None): # 🌟 新增參數
    """
    用 RandomizedSearchCV + class_weight + early stopping 訓練 XGBoost
    (🌟 還原 class_weight 並加入微調功能)
    """
    X_train_fs, X_valid_fs = X_train[top_features], X_valid[top_features]
    X_search, y_search = pd.concat([X_train_fs, X_valid_fs]), pd.concat([y_train, y_valid])
    
    ps = PredefinedSplit([-1] * len(X_train_fs) + [0] * len(X_valid_fs))
    
    # --- 🌟 1. 計算基礎 'balanced' 權重 ---
    print("  > 正在使用 'balanced' 自動權重")
    search_weights = compute_sample_weight(class_weight='balanced', y=y_search)

    # --- 🌟 2. 根據 custom_weight_adjustments 進行微調 ---
    if custom_weight_adjustments:
        print(f"  > 正在微調權重: {custom_weight_adjustments}")
        # 建立一個 DataFrame 以便快速映射標籤
        temp_weights_df = pd.DataFrame({'label': y_search, 'weight': search_weights})
        for label, multiplier in custom_weight_adjustments.items():
            temp_weights_df.loc[temp_weights_df['label'] == label, 'weight'] *= multiplier
        search_weights = temp_weights_df['weight'].values

    fit_params = {"eval_set": [(X_valid_fs, y_valid)], "verbose": False}
    
    # --- 🌟 3. 同樣邏輯應用於驗證集權重 ---
    if xgb.__version__ >= "2.0.0":
        valid_weights = compute_sample_weight(class_weight='balanced', y=y_valid)
        
        if custom_weight_adjustments:
            temp_valid_weights_df = pd.DataFrame({'label': y_valid, 'weight': valid_weights})
            for label, multiplier in custom_weight_adjustments.items():
                temp_valid_weights_df.loc[temp_valid_weights_df['label'] == label, 'weight'] *= multiplier
            valid_weights = temp_valid_weights_df['weight'].values

        fit_params["sample_weight_eval_set"] = [valid_weights]

    param_dist = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2], 'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300, 400], 'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9], 'gamma': [0, 0.1, 0.2]
    }
    base_model = xgb.XGBClassifier(objective=objective, eval_metric="mlogloss", random_state=42, tree_method="hist", num_class=num_class, early_stopping_rounds=30)
    
    rand_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=n_iter, scoring='f1_macro', cv=ps, n_jobs=-1, verbose=2, random_state=42)
    
    rand_search.fit(
        X_search,
        y_search,
        # --- 🌟 使用微調後的權重 ---
        sample_weight=search_weights,
        **fit_params
    )
    
    print(f"✅ {objective} 最佳參數: {rand_search.best_params_}")
    print(f"✅ {objective} 最佳 F1 Macro (Val): {rand_search.best_score_:.4f}")
    return rand_search.best_estimator_

# =========================================================
# 1️⃣1️⃣ 輸出 submission.csv
# =========================================================
def revert_negative(pred, col_name, original_max_labels_dict):
    """將 max+1 類別轉回 -1"""
    if col_name in original_max_labels_dict:
        replacement_val = original_max_labels_dict[col_name]
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

def revert_negative_pointid(pred, replacement_val):
    """將 max+1 類別轉回 -1（for pointId）"""
    if replacement_val is not None:
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

def save_submission(test_last_shot, pred_action, pred_point, pred_server, sample_path, output_path):
    """儲存提交檔案"""
    submission = pd.DataFrame({"rally_uid": test_last_shot["rally_uid"], "serverGetPoint": pred_server, "pointId": pred_point, "actionId": pred_action})
    try:
        sample_sub = pd.read_csv(sample_path)
        submission = submission[sample_sub.columns]
    except FileNotFoundError:
        print(f"⚠️ 找不到 {sample_path}，將使用預設欄位順序。")
    except Exception as e:
        print(f"⚠️ 讀取 {sample_path} 時出錯: {e}")
    submission.to_csv(output_path, index=False)
    print(f"\n✅ 已輸出 {output_path}\nSubmission shape: {submission.shape}\n{submission.head()}")

# =========================================================
# 1️⃣2️⃣ 實驗結果紀錄 (🌟 NEW 🌟)
# =========================================================
def log_experiment_results(log_path, results_dict):
    """將單次實驗結果 (字典) 附加到 CSV 檔案中"""
    try:
        # 確保字典中的值是可序列化的 (例如, 轉換 None 和 dict)
        loggable_dict = {}
        for key, value in results_dict.items():
            if value is None:
                loggable_dict[key] = "None"
            elif isinstance(value, dict):
                 loggable_dict[key] = json.dumps(value) # 將字典轉為 JSON 字串
            else:
                loggable_dict[key] = value

        fieldnames = loggable_dict.keys()
        file_exists = os.path.isfile(log_path)

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader() # 如果檔案不存在，寫入標頭
            
            writer.writerow(loggable_dict)
        print(f"✅ 實驗結果已紀錄至 {log_path}")
    except Exception as e:
        print(f"⚠️ 紀錄實驗結果時發生錯誤: {e}")

# =========================================================
# 🚀 主執行流程 (🌟 重構 🌟)
# =========================================================
def main():
    # --- 參數設定 ---
    K_FEATURES = 20
    N_ITER_SEARCH = 25 # 🌟 方便紀錄 RandomizedSearch 的迭代次數
    LOG_FILE = "experiment_log.csv" # 🌟 紀錄檔案
    
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SAMPLE_SUB_PATH = "sample_submission.csv"
    SUBMISSION_PATH = "submission.csv"

    # --- 1. 讀取 & 2. 特徵工程 ---
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    print("⚙️ 正在為 train 建立特徵...")
    train = create_features(train)
    print("⚙️ 正在為 test 建立特徵...")
    test = create_features(test)

    # --- 欄位對齊 ---
    print("🔄 正在對齊 Train 和 Test 的欄位...")
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    missing_in_test = list(train_cols - test_cols)
    if missing_in_test:
        for c in missing_in_test:
            if c.startswith('type_'): test[c] = 0
    missing_in_train = list(test_cols - train_cols)
    if missing_in_train:
        for c in missing_in_train:
            if c.startswith('type_'): train[c] = 0
    common_cols = [col for col in train.columns if col in test.columns]
    test = test[common_cols]
    train = train[common_cols + list(train_cols - test_cols)]
    
    # --- 3. 預處理 ---
    target_cols = ["actionId", "pointId", "serverGetPoint"]
    drop_cols = ["rally_uid", "rally_id", "match", "numberGame"]
    feature_cols = [c for c in train.columns if c not in target_cols + drop_cols and c in test.columns]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    
    print(f"✅ 使用 {len(feature_cols)} 個特徵進行訓練。 ('sex' 欄位已保留)")
    
    train, test_last_shot, original_max_labels = preprocess(train, test)

    # --- 4. 建立 N -> N+1 訓練資料 & 5. Group Split ---
    X, y_action, y_point, y_server, rally_uids_for_split = create_training_data(train, feature_cols)
    X_test = test_last_shot[feature_cols].copy().fillna(0)
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    split_data, y_all = create_group_split(X, y_action, y_point, y_server, rally_uids_for_split)
    y_action_all, y_point_all, y_server_all = y_all

    # --- 6 & 7. actionId 模型訓練 (RandomizedSearchCV) ---
    X_train_action, X_valid_action, y_train_action, y_valid_action = split_data['action']
    num_class_action = y_action_all.nunique()
    print(f"✅ actionId 類別數量: {num_class_action}")
    
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features_action = select_features_xgb(X_train_action, y_train_action, num_class_action, top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features_action[:5]}")

    action_weight_adjustments = { 19: 0.8}
    # action_weight_adjustments = None

    print("🚀 訓練 actionId 模型 (RandomizedSearchCV)...")
    actionid_model = train_xgb_with_search(
        X_train_action, X_valid_action, y_train_action, y_valid_action, 
        num_class_action, top_features_action,
        n_iter=N_ITER_SEARCH,
        custom_weight_adjustments=action_weight_adjustments
    )

    # --- 6 & 7. pointId 模型訓練 (RandomizedSearchCV) ---
    X_train_point, X_valid_point, y_train_point, y_valid_point = split_data['point']
    num_class_point = y_point_all.nunique()
    print(f"✅ pointId 類別數量: {num_class_point}")
    
    print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
    top_features_point = select_features_xgb(X_train_point, y_train_point, num_class_point, top_k=K_FEATURES)
    print(f"🔥 pointId Top 5: {top_features_point[:5]}")
    
    # point_weight_adjustments = { 5: 0.7 }
    point_weight_adjustments = None

    print("🚀 訓練 pointId 模型 (RandomizedSearchCV)...")
    pointid_model = train_xgb_with_search(
        X_train_point, X_valid_point, y_train_point, y_valid_point, 
        num_class_point, top_features_point,
        n_iter=N_ITER_SEARCH,
        custom_weight_adjustments=point_weight_adjustments
    )

    # --- 8. 僅訓練 serverGetPoint 模型 ---
    print("🚀 訓練 serverGetPoint 模型中...")
    X_train_server, X_valid_server, y_train_server, y_valid_server = split_data['server']
    
    server_objective = "binary:logistic" if y_server_all.nunique() <= 2 else "multi:softmax"
    server_num_class = y_server_all.nunique() if y_server_all.nunique() > 2 else None
    
    top_features_server = select_features(X_train_server, y_train_server, 
                                          server_objective, server_num_class, 
                                          top_k=K_FEATURES)
    
    X_train_fs_server = X_train_server[top_features_server]
    X_valid_fs_server = X_valid_server[top_features_server]

    server_model = train_xgb(X_train_fs_server, y_train_server, X_valid_fs_server, y_valid_server, 
                             server_objective, server_num_class)

    # --- 9. 評估模型 (使用 *新* 模型) 並紀錄 ---
    print("\n📊 評估 *最終* 模型...")
    
    X_valid_fs_action = X_valid_action[top_features_action]
    X_valid_fs_point = X_valid_point[top_features_point]

    pred_action_val = actionid_model.predict(X_valid_fs_action)
    pred_point_val = pointid_model.predict(X_valid_fs_point)
    
    if y_server_all.nunique() > 2:
        pred_server_proba_val = server_model.predict_proba(X_valid_fs_server)
        auc_server = roc_auc_score(y_valid_server, pred_server_proba_val, multi_class="ovr")
    else:
        pred_server_proba_val = server_model.predict_proba(X_valid_fs_server)[:, 1]
        auc_server = roc_auc_score(y_valid_server, pred_server_proba_val)

    f1_action = f1_score(y_valid_action, pred_action_val, average="macro")
    f1_point = f1_score(y_valid_point, pred_point_val, average="macro")
    weighted_score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server
    
    print(f"actionId Macro F1: {f1_action:.4f}")
    print(f"pointId  Macro F1: {f1_point:.4f}")
    print(f"serverGetPoint AUC: {auc_server:.4f}")
    print(f"綜合評分: {weighted_score:.4f}")

    # --- 9.5 紀錄實驗結果 (🌟 NEW 🌟) ---
    results_to_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "f1_action": f"{f1_action:.4f}",
        "f1_point": f"{f1_point:.4f}",
        "auc_server": f"{auc_server:.4f}",
        "weighted_score": f"{weighted_score:.4f}",
        "weighted_score": f"{weighted_score:4f}",
        "K_FEATURES": K_FEATURES,
        "n_iter_search": N_ITER_SEARCH,
        "action_weights_adj": json.dumps(action_weight_adjustments),
        "point_weights_adj": json.dumps(point_weight_adjustments)
    }
    log_experiment_results(LOG_FILE, results_to_log)

    # --- 10. 產生預測 (使用 *新* 模型) ---
    print("\n🧮 產生測試預測中...")
    X_test_fs_action = X_test.reindex(columns=X_train_action.columns, fill_value=0)[top_features_action]
    X_test_fs_point = X_test.reindex(columns=X_train_point.columns, fill_value=0)[top_features_point]
    X_test_fs_server = X_test.reindex(columns=X_train_server.columns, fill_value=0)[top_features_server]

    pred_action_test_raw = actionid_model.predict(X_test_fs_action)
    pred_point_test_raw = pointid_model.predict(X_test_fs_point)

    if y_server_all.nunique() > 2:
        pred_server_test_raw = server_model.predict(X_test_fs_server)
        pred_server = revert_negative(pred_server_test_raw, "serverGetPoint", original_max_labels)
    else:
        pred_server = server_model.predict_proba(X_test_fs_server)[:, 1] # 機率

    # 還原 -1
    pred_action = revert_negative(pred_action_test_raw, "actionId", original_max_labels)
    pred_point = revert_negative_pointid(pred_point_test_raw, original_max_labels.get("pointId"))

    # --- 11. 儲存提交檔案 ---
    save_submission(test_last_shot, pred_action, pred_point, pred_server, SAMPLE_SUB_PATH, SUBMISSION_PATH)

if __name__ == "__main__":
    main()

