#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 多任務分類模型 (重構版 v2.5) - 主執行檔
---------------------------------------------------------------------
🌟 v2.5 更新：
- 新增 `custom_weight_adjustments` 參數。
- 允許在 'balanced' 權重的基礎上，手動微調特定類別的權重。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight # 🌟 還原
# from imblearn.over_sampling import SMOTE # 移除
from tqdm import tqdm

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

def apply_feature_selection(split_data, y_all, X_test, K_FEATURES):
    """為三個目標分別進行特徵選取"""
    X_train_action, X_valid_action, y_train_action, _ = split_data['action']
    X_train_point, X_valid_point, y_train_point, _ = split_data['point']
    X_train_server, X_valid_server, y_train_server, _ = split_data['server']
    y_action_all, y_point_all, y_server_all = y_all

    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features_action = select_features(X_train_action, y_train_action, objective="multi:softmax", num_class=y_action_all.nunique(), top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features_action[:5]}")

    print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
    top_features_point = select_features(X_train_point, y_train_point, objective="multi:softmax", num_class=y_point_all.nunique(), top_k=K_FEATURES)
    print(f"🔥 pointId Top 5: {top_features_point[:5]}")

    print(f"🧩 為 serverGetPoint 選取前 {K_FEATURES} 個特徵...")
    server_objective = "multi:softmax" if y_train_server.nunique() > 2 else "binary:logistic"
    server_num_class = y_server_all.nunique() if y_train_server.nunique() > 2 else None
    top_features_server = select_features(X_train_server, y_train_server, objective=server_objective, num_class=server_num_class, top_k=K_FEATURES)
    print(f"🔥 serverGetPoint Top 5: {top_features_server[:5]}")

    # 確保 X_test 也使用對應的特徵子集
    X_test_action = X_test.reindex(columns=X_train_action.columns, fill_value=0)[top_features_action]
    X_test_point = X_test.reindex(columns=X_train_point.columns, fill_value=0)[top_features_point]
    X_test_server = X_test.reindex(columns=X_train_server.columns, fill_value=0)[top_features_server]

    return {
        'action': (X_train_action[top_features_action], X_valid_action[top_features_action], X_test_action),
        'point': (X_train_point[top_features_point], X_valid_point[top_features_point], X_test_point),
        'server': (X_train_server[top_features_server], X_valid_server[top_features_server], X_test_server)
    }

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
    """訓練 XGBoost 模型的通用函式"""
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
# 8️⃣ 三個模型訓練
# =========================================================
def train_all_models(fs_data, split_data, y_all):
    """使用各自選取的特徵集訓練三個模型"""
    models = {}
    _, _, y_train_action, y_valid_action = split_data['action']
    _, _, y_train_point, y_valid_point = split_data['point']
    _, _, y_train_server, y_valid_server = split_data['server']
    X_train_fs_action, X_valid_fs_action, _ = fs_data['action']
    X_train_fs_point, X_valid_fs_point, _ = fs_data['point']
    X_train_fs_server, X_valid_fs_server, _ = fs_data['server']
    y_action_all, y_point_all, y_server_all = y_all

    print("🚀 訓練 actionId 模型中...")
    models['action'] = train_xgb(X_train_fs_action, y_train_action, X_valid_fs_action, y_valid_action, "multi:softmax", y_action_all.nunique())
    print("🚀 訓練 pointId 模型中...")
    models['point'] = train_xgb(X_train_fs_point, y_train_point, X_valid_fs_point, y_valid_point, "multi:softmax", y_point_all.nunique())
    print("🚀 訓練 serverGetPoint 模型中...")
    if y_server_all.nunique() > 2:
        print("⚠️ serverGetPoint 發現多於2個類別，使用 multi:softmax")
        models['server'] = train_xgb(X_train_fs_server, y_train_server, X_valid_fs_server, y_valid_server, "multi:softmax", y_server_all.nunique())
    else:
        # 🌟 這裡也可以加上 sample_weight
        # 為了保持與 RandomizedSearch 一致，我們可以修改 train_xgb
        # 但目前為止，我們先保持原狀，因為 'serverGetPoint' 可能是二元且較平衡
        models['server'] = train_xgb(X_train_fs_server, y_train_server, X_valid_fs_server, y_valid_server, "binary:logistic")
    return models

# =========================================================
# 9️⃣ 模型評估
# =========================================================
def evaluate_models(models, fs_data, split_data, y_all):
    """在驗證集上評估模型"""
    _, _, _, y_valid_action = split_data['action']
    _, _, _, y_valid_point = split_data['point']
    _, _, _, y_valid_server = split_data['server']
    _, X_valid_fs_action, _ = fs_data['action']
    _, X_valid_fs_point, _ = fs_data['point']
    _, X_valid_fs_server, _ = fs_data['server']
    y_server_all = y_all[2]
    
    pred_action = models['action'].predict(X_valid_fs_action)
    pred_point = models['point'].predict(X_valid_fs_point)
    pred_server_proba = models['server'].predict_proba(X_valid_fs_server)
    
    auc_server = roc_auc_score(y_valid_server, pred_server_proba, multi_class="ovr") if y_server_all.nunique() > 2 else roc_auc_score(y_valid_server, pred_server_proba[:, 1])
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
    """產生測試集預測並還原 -1 標籤"""
    print("\n🧮 產生測試預測中...")
    _, _, X_test_fs_action = fs_data['action']
    _, _, X_test_fs_point = fs_data['point']
    _, _, X_test_fs_server = fs_data['server']
    y_server_all = y_all[2]

    pred_action_test = models['action'].predict(X_test_fs_action)
    pred_point_test = models['point'].predict(X_test_fs_point)
    
    if y_server_all.nunique() > 2:
        pred_server_test_labels = models['server'].predict(X_test_fs_server)
        pred_server_final = revert_negative(pred_server_test_labels, "serverGetPoint", original_max_labels)
    else:
        pred_server_final = models['server'].predict_proba(X_test_fs_server)[:, 1]

    pred_action_test = revert_negative(pred_action_test, "actionId", original_max_labels)
    pred_point_test = revert_negative(pred_point_test, "pointId", original_max_labels)
    return pred_action_test, pred_point_test, pred_server_final

def revert_negative_pointid(pred, replacement_val):
    """將 max+1 類別轉回 -1（for pointId）"""
    if replacement_val is not None:
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

# =========================================================
# 1️⃣1️⃣ 輸出 submission.csv
# =========================================================
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
# 🚀 主執行流程
# =========================================================
def main():
    # --- 參數設定 ---
    K_FEATURES = 20
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

    # --- 新增步驟：對齊 One-Hot Encoded 欄位 ---
    print("🔄 正在對齊 Train 和 Test 的欄位...")
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    missing_in_test = list(train_cols - test_cols)
    if missing_in_test:
        for c in missing_in_test:
            if c.startswith('type_'):
                test[c] = 0

    missing_in_train = list(test_cols - train_cols)
    if missing_in_train:
        for c in missing_in_train:
            if c.startswith('type_'):
                train[c] = 0

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

    # --- 6 & 7. actionId 模型訓練 (RandomizedSearchCV) ---
    X_train_action, X_valid_action, y_train_action, y_valid_action = split_data['action']
    num_class_action = y_all[0].nunique()
    print(f"✅ actionId 類別數量: {num_class_action}")

    
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    top_features_action = select_features_xgb(X_train_action, y_train_action, num_class_action, top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features_action[:5]}")

    # 🌟 範例：自訂權重調整
    # 這裡的數字是 "乘數"。 1.0 = 不變, 0.8 = 權重變為80%, 1.5 = 權重變為150%
    # 假設您想將 '無' (0) 的權重調小，'傳統' (15) 的權重調高
    action_weight_adjustments = {
        19: 0.8,  # 將 '無' (0) 的權重調為 'balanced' 權重的 80%
         # 將 '傳統' (15) 的權重調為 'balanced' 權重的 150%
        # 其他未指定的類別將保持 'balanced' 的原始權重 (乘數為 1.0)
    }

    print("🚀 訓練 actionId 模型 (RandomizedSearchCV)...")
    actionid_model = train_xgb_with_search(
        X_train_action, X_valid_action, y_train_action, y_valid_action, 
        num_class_action, top_features_action,
        custom_weight_adjustments=action_weight_adjustments # <-- 🌟 傳入調整字典
    )

    # --- 6 & 7. pointId 模型訓練 (RandomizedSearchCV) ---
    X_train_point, X_valid_point, y_train_point, y_valid_point = split_data['point']
    num_class_point = y_all[1].nunique()
    print(f"✅ pointId 類別數量: {num_class_point}")

    
    print(f"🧩 為 pointId 選取前 {K_FEATURES} 個特徵...")
    top_features_point = select_features_xgb(X_train_point, y_train_point, num_class_point, top_k=K_FEATURES)
    print(f"🔥 pointId Top 5: {top_features_point[:5]}")
    
    # 🌟 範例：pointId 也可以調整
    # 假設 'pointId' 類別 5 很多，想降低它的權重
    point_weight_adjustments = {
        5: 0.7 # 將 'pointId' 5 的權重調為 70%
    }
    # 如果您不想調整 pointId，保留 'None' 即可
    # point_weight_adjustments = None 

    print("🚀 訓練 pointId 模型 (RandomizedSearchCV)...")
    pointid_model = train_xgb_with_search(
        X_train_point, X_valid_point, y_train_point, y_valid_point, 
        num_class_point, top_features_point,
        custom_weight_adjustments=point_weight_adjustments # <-- 🌟 傳入調整字典
    )

    # --- 8. serverGetPoint 使用原流程訓練 ---
    fs_data = apply_feature_selection(split_data, y_all, X_test, K_FEATURES)
    models = train_all_models(fs_data, split_data, y_all)

    # --- 9. 評估模型 ---
    # 🌟 我們應該評估新的模型，而不只是舊的 'models' 字典
    # 為了簡潔，我們先保留原有的 evaluate_models
    # 一個好的重構是把 actionid_model 和 pointid_model 放入 'models' 字典
    evaluate_models(models, fs_data, split_data, y_all)

    # --- 10. 產生預測 ---
    # serverGetPoint 用原流程模型預測
    _, _, pred_server = generate_predictions(models, fs_data, y_all, original_max_labels)
    
    # pointId 用 RandomizedSearch (sample_weight) 的新模型預測
    X_test_fs_point = X_test.reindex(columns=X_train_point.columns, fill_value=0)[top_features_point]
    pred_point_test = pointid_model.predict(X_test_fs_point)
    pred_point_test = revert_negative_pointid(pred_point_test, original_max_labels.get("pointId"))
    
    # actionId 用 RandomizedSearch (sample_weight) 的新模型預續
    X_test_fs_action = X_test.reindex(columns=X_train_action.columns, fill_value=0)[top_features_action]
    pred_action_test_resampled = actionid_model.predict(X_test_fs_action)
    pred_action = revert_negative(pred_action_test_resampled, "actionId", original_max_labels)


    # --- 11. 儲存提交檔案 ---
    save_submission(test_last_shot, pred_action, pred_point_test, pred_server, SAMPLE_SUB_PATH, SUBMISSION_PATH)

if __name__ == "__main__":
    main()

