# data_processing.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏓 資料處理模組
---------------------------------------------------------------------
🌟 v2.1 更新：
- 新增 `action_type` 特徵，並進行 One-Hot Encoding。
- 將 `action_type` 也加入滯後特徵的計算。
"""

import pandas as pd
import sys
from sklearn.model_selection import train_test_split

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
# 2️⃣ 特徵工程 (🌟 更新)
# =========================================================
def create_features(df):
    """為 train 和 test 數據集建立新的序列特徵和 action type 特徵"""
    df_new = df.copy()

    # 1. 建立 Action Type 特徵 (NEW)
    print("  > 正在建立 Action Type 特徵...")
    action_type_map = {
        0: 'Zero', 1: 'Attack', 2: 'Attack', 3: 'Attack', 4: 'Attack',
        5: 'Attack', 6: 'Attack', 7: 'Attack', 8: 'Control', 9: 'Control',
        10: 'Control', 11: 'Control', 12: 'Defensive', 13: 'Defensive',
        14: 'Defensive', 15: 'Serve', 16: 'Serve', 17: 'Serve', 18: 'Serve',
        -1: 'EndPoint'
    }
    # 將 actionId 映射到 action_type
    df_new['action_type'] = df_new['actionId'].map(action_type_map)

    # 確保資料按回合和拍數排序
    df_new = df_new.sort_values(by=['rally_uid', 'strickNumber'])
    
    # 2. 滯後特徵 (Lag Features)
    print(f"  > 正在建立 N-1, N-2, N-3 滯後特徵...")
    # 🌟 將 action_type 也加入滯後列表
    lag_cols = ['actionId', 'pointId', 'spinId', 'strengthId', 'positionId', 'action_type']
    for col in lag_cols:
        for n in [1, 2, 3]:
            df_new[f'prev_{n}_{col}'] = df_new.groupby('rally_uid')[col].shift(n)

    # 3. 情境特徵 (Context Features) - 分數
    df_new['score_diff'] = df_new['scoreSelf'] - df_new['scoreOther']

    # 4. 填充 NaNs (🌟 更新)
    # 數值型特徵用 -1 填充
    num_fill_cols = [col for col in df_new.columns if 'prev_' in col and 'action_type' not in col]
    df_new[num_fill_cols] = df_new[num_fill_cols].fillna(-1)
    # 類別型特徵用 'None' 填充，代表沒有前一拍
    cat_fill_cols = [col for col in df_new.columns if 'prev_' in col and 'action_type' in col]
    df_new[cat_fill_cols] = df_new[cat_fill_cols].fillna('None')

    # 5. One-Hot Encoding for Action Type (NEW)
    print("  > 正在對 Action Type 進行 One-Hot Encoding...")
    type_cols = ['action_type'] + cat_fill_cols
    # 使用 get_dummies 進行轉換，並加上前綴以區分，同時將原本的類別欄位移除
    df_new = pd.get_dummies(df_new, columns=type_cols, prefix='type')

    return df_new

# =========================================================
# 3️⃣ 預處理 (無變動)
# =========================================================
def preprocess(train_df, test_df):
    """
    1. 修正 -1 類別問題
    2. 取得測試集最後一筆資料
    """
    test_last_shot = test_df.groupby('rally_uid').tail(1).copy()
    print(f"✅ Test (last shots) shape: {test_last_shot.shape}")

    original_max_labels = {}
    for col in ["actionId", "pointId", "serverGetPoint"]:
        if col in train_df.columns and (train_df[col] == -1).any():
            max_label = train_df[col].max()
            original_max_labels[col] = max_label + 1
            print(f"⚠️ {col} 含有 -1，將其替換為 {max_label + 1}")
            train_df[col] = train_df[col].replace(-1, max_label + 1)
    
    return train_df, test_last_shot, original_max_labels

# =========================================================
# 4️⃣ 建立訓練任務 (N -> N+1) (無變動)
# =========================================================
def create_training_data(train_df, feature_cols):
    """
    重新定義訓練任務 (N -> N+1)
    """
    X = train_df[feature_cols].copy().fillna(0)
    y_action = train_df.groupby('rally_uid')['actionId'].shift(-1)
    y_point = train_df.groupby('rally_uid')['pointId'].shift(-1)
    y_server = train_df['serverGetPoint']
    rally_uids_for_split = train_df['rally_uid']

    valid_indices = y_action.notna() & y_point.notna()
    X = X[valid_indices]
    y_action = y_action[valid_indices]
    y_point = y_point[valid_indices]
    y_server = y_server[valid_indices]
    rally_uids_for_split = rally_uids_for_split[valid_indices]

    print(f"✅ 重新建立訓練集 (N -> N+1)，新 shape: {X.shape}")
    
    return X, y_action, y_point, y_server, rally_uids_for_split

# =========================================================
# 5️⃣ 建立無洩漏的驗證集 (Group Split) (無變動)
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

    data = {
        'action': (X[train_mask], X[valid_mask], y_action[train_mask], y_action[valid_mask]),
        'point': (X[train_mask], X[valid_mask], y_point[train_mask], y_point[valid_mask]),
        'server': (X[train_mask], X[valid_mask], y_server[train_mask], y_server[valid_mask])
    }
    
    return data, (y_action, y_point, y_server)