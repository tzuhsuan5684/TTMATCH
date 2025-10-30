#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 單任務分類模型 (actionId)：僅預測 actionId
---------------------------------------------------------------------
🌟 來源：
- 從 v2 多任務模型重構而來，專注於 actionId 預測。
- 保留了 v2 的滯後特徵 (prev_1, prev_2, prev_3) 和 score_diff 特徵。

⭐ v3 更新：
- 整合 RandomizedSearchCV 進行超參數調優。
- 保留 Group Split (使用 PredefinedSplit)。
- 整合 class_weight ('balanced') 處理不平衡問題。
- 整合 Early Stopping 提升搜尋效率。

🐞 Debug 筆記：
1.  **[最可能的錯誤] UnicodeEncodeError**：
    如果你的作業系統 (特別是 Windows) 的
    console（命令提示字元）預設編碼不是 UTF-8，
    執行 `print("✅ 搜尋完成!")` 
    這類包含中文的指令時，可能會引發 `UnicodeEncodeError`。

    **解決方法**：
    在執行此腳本前，先在你的終端機設定環境變數：
    - (Windows CMD): `set PYTHONIOENCODING=utf-8`
    - (Windows PowerShell): `$env:PYTHONIOENCODING = "utf-8"`
    - (Linux/macOS): `export PYTHONIOENCODING=utf-8`
    然後再執行 `python your_script_name.py`

2.  **[清除] 移除未使用的套件**：
    移除了 `from tqdm import tqdm`，因為 `RandomizedSearchCV(verbose=2)` 
    已經提供了足夠的進度顯示。
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
# from tqdm import tqdm  # (Debug) 移除未使用的 import
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
    
    print(f"   > 正在建立 N-1, N-2, N-3 滯後特徵...")
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
# 3️⃣ 預處理 (無變動)
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
# 4️⃣ 建立訓練任務 (N -> N+1) (無變動)
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
# 5️⃣ 建立無洩漏的驗證集 (Group Split) (無變動)
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
# 6️⃣ 特徵選取 (無變動)
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
# 7️⃣ 標籤還原 
# =========================================================
def revert_negative(pred, replacement_val):
    """將 max+1 類別轉回 -1"""
    if replacement_val is not None:
        pred = pd.Series(pred)
        pred[pred == replacement_val] = -1
        return pred.values
    return pred

# =========================================================
# 8️⃣ 輸出 submission.csv 
# =========================================================
def save_submission(test_last_shot, pred_action, sample_path="sample_submission.csv", output_path="submission.csv"):
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
    N_ITER_SEARCH = 25 # RandomizedSearch 的搜尋次數
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
    
    train, test_last_shot, original_max_label = preprocess(train, test)

    # --- 4. 建立 N -> N+1 訓練資料 ---
    X, y, rally_uids_for_split = create_training_data(train, feature_cols)
    X_test = test_last_shot[feature_cols].copy().fillna(0)

    # --- 5. 建立 Group Split ---
    X_train, X_valid, y_train, y_valid = create_group_split(X, y, rally_uids_for_split)
    num_class = y.nunique()

    # --- 6. 特徵選取 ---
    print(f"🧩 為 actionId 選取前 {K_FEATURES} 個特徵...")
    # 注意：特徵選取在 X_train 上進行，以避免過擬合
    top_features = select_features(X_train, y_train, 
                                      objective="multi:softmax", 
                                      num_class=num_class, 
                                      top_k=K_FEATURES)
    print(f"🔥 actionId Top 5: {top_features[:5]}")
    
    X_train_fs = X_train[top_features]
    X_valid_fs = X_valid[top_features]
    X_test_fs = X_test[top_features]

    # =========================================================
    # ⭐ 7. 設定超參數搜尋 (RandomizedSearchCV)
    # =========================================================
    print("\n🚀 設定超參數搜尋 (RandomizedSearchCV)...")

    # 7a. 將訓練集和驗證集合併，以符合 PredefinedSplit 的要求
    X_search = pd.concat([X_train_fs, X_valid_fs])
    y_search = pd.concat([y_train, y_valid])

    # 7b. 建立 PredefinedSplit
    # -1 代表訓練集, 0 代表驗證集
    test_fold = np.zeros(len(X_search))
    test_fold[:len(X_train_fs)] = -1
    ps = PredefinedSplit(test_fold)

    # 7c. 為搜尋資料計算 'balanced' 權重
    print("   > 正在計算 'balanced' 類別權重 (for Search)...")
    search_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_search
    )
    
    # 7d. 為 early stopping 準備 fit_params
    # **** DEBUG FIX ****
    # 'early_stopping_rounds' 是 XGBClassifier 的 *constructor* 參數，
    # 而不是 .fit() 方法的參數 (在 scikit-learn 流程中)。
    # 我們將把它移至 base_model 的 constructor 中。
    # .fit() 只需要 eval_set 即可觸發 early stopping。
    fit_params = {
        # "early_stopping_rounds": 30, # <-- 錯誤的放置位置
        "eval_set": [(X_valid_fs, y_valid)],
        "verbose": False
    }

    # 檢查 XGBoost 版本是否支援 eval_sample_weight
    if xgb.__version__ >= "2.0.0":
        print("   > 偵測到 XGBoost >= 2.0.0，啟用 eval_sample_weight。")
        valid_weights = compute_sample_weight(class_weight='balanced', y=y_valid)
        # **** DEBUG FIX ****
        # 參數名稱應為 'sample_weight_eval_set' 而不是 'eval_sample_weight'
        fit_params["sample_weight_eval_set"] = [valid_weights]
    else:
        print(f"   > 警告: XGBoost 版本 ({xgb.__version__}) 過舊，無法使用 eval_sample_weight。")

    # 7e. 定義參數網格 (param_distributions)
    param_dist = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 200, 300, 400],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2] # 新增 gamma 參數
    }

    # 7f. 建立基本模型
    base_model = xgb.XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        num_class=num_class,
        early_stopping_rounds=30  # <-- DEBUG FIX: 參數應在此處
    )

    # 7g. 建立 RandomizedSearchCV 物件
    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,  # 搜尋次數
        scoring='f1_macro',    # 我們的目標指標
        cv=ps,                 # 使用我們自訂的 (Train/Valid) 切分
        n_jobs=-1,             # 使用所有 CPU 核心
        verbose=2,             # 顯示搜尋進度
        random_state=42
    )

    # =========================================================
    # ⭐ 8. 執行搜尋與評估
    # =========================================================
    print("\n🚀 開始執行超參數搜尋...")
    
    # 將 search_weights 傳遞給 fit
    rand_search.fit(
        X_search, 
        y_search, 
        sample_weight=search_weights,
        **fit_params
    )

    print("\n📊 搜尋完成!")
    print(f"✅ 最佳參數: {rand_search.best_params_}")
    print(f"✅ 最佳 F1 Macro (Val): {rand_search.best_score_:.4f}")

    # 取得最佳模型
    model = rand_search.best_estimator_
    

    # =========================================================
    # 9. 產生預測
    # =========================================================
    print("\n🧮 產生測試預測中...")
    pred_test = model.predict(X_test_fs)
    pred_test_reverted = revert_negative(pred_test, original_max_label)
    
    # 儲存
    save_submission(test_last_shot, pred_test_reverted, SAMPLE_SUB_PATH, SUBMISSION_PATH)
    
    print("\n🎉 流程執行完畢。")

if __name__ == "__main__":
    main()


