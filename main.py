import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb  # 確保 LightGBM 已導入
import category_encoders as ce
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_labels, filename, title='Confusion matrix'):
    """ 繪製並儲存混淆矩陣圖 """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train_binary_lgb(name, X, y, X_valid, y_valid):
    """ 使用 LightGBM 訓練二元分類模型 """
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='logloss',
        learning_rate=0.1,
        random_state=42,
        n_estimators=100,
        num_leaves=64,
        max_depth=6,
        verbose=-1  # 關閉繁瑣的日誌
    )

    model.fit(X, y)
    y_server_pred = model.predict(X_valid)

    score = roc_auc_score(y_valid, y_server_pred)
    plot_confusion_matrix(y_valid, y_server_pred, class_labels=[0, 1],
                          filename=f'confusion_matrix_{name}.png',
                          title=f'Confusion Matrix for {name}')
    print(f"Validation ROC AUC Score for {name}: {score:.4f}")

    return model, score

def trainLGB(name, X, y, X_valid, y_valid, objective='multiclass'):
    """ 使用 LightGBM 訓練多分類模型（帶有類別權重） """
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = y.map(class_weight_dict)

    model = lgb.LGBMClassifier(
        objective=objective,
        metric='multi_logloss',
        learning_rate=0.03,
        num_iterations=200,
        num_class=len(y.unique()),
        random_state=42,
        n_estimators=100,
        num_leaves=64,
        max_depth=6,
        verbose=-1  # 關閉繁瑣的日誌
    )

    # LightGBM 的 fit 方法也支援 sample_weight
    model.fit(X, y, sample_weight=sample_weights)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred, average='weighted')
    print(f"Validation F1 Score for {name}: {f1:.4f}")
    plot_confusion_matrix(y_valid, y_pred, class_labels=sorted(y.unique()),
                          filename=f'confusion_matrix_{name}.png',
                          title=f'Confusion Matrix for {name}')
    return model, f1


def main():
    # 載入資料
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_new = train.copy()
    test_new = test.copy()
    
    for col in ['actionId', 'pointId']:
        train_new[col] = train_new[col]+1
        test_new[col] = test_new[col]+1
    
    # 特徵工程
    lag_cols = ['actionId', 'pointId', 'spinId', 'strengthId', 'positionId']
    for col in lag_cols:
        for n in [1, 2, 3]:
            train_new[f'prev_{n}_{col}'] = train_new.groupby('rally_uid')[col].shift(n)
            test_new[f'prev_{n}_{col}'] = test_new.groupby('rally_uid')[col].shift(n)
    
    # Target encoding for strengthId, spinId, handId (using actionId as target)
    y_action_te = train_new.groupby('rally_uid')['actionId'].shift(-1)
    valid_te_idx = y_action_te.notna()
    for col in ['strengthId', 'spinId', 'handId']:
        if col in train_new.columns:
            encoder = ce.TargetEncoder(cols=[col])
            train_new.loc[valid_te_idx, f'{col}_te'] = encoder.fit_transform(
                train_new.loc[valid_te_idx, col], y_action_te[valid_te_idx]
            ).values
            test_new[f'{col}_te'] = encoder.transform(test_new[col]).values
    
    # 前處理
    target_cols = ["actionId", "pointId", "serverGetPoint"]
    drop_cols = ["rally_uid", "rally_id", "match", "numberGame", "serverGetPoint"]
    
    y_action = train_new.groupby('rally_uid')['actionId'].shift(-1)
    y_point = train_new.groupby('rally_uid')['pointId'].shift(-1)
    y_server = train_new['serverGetPoint']
    rally_uids = train_new['rally_uid']
    X = train_new.drop(columns=drop_cols)

    valid_indices = y_action.notna() & y_point.notna()
    X = X[valid_indices]
    y_action = y_action[valid_indices]
    y_point = y_point[valid_indices]
    y_server = y_server[valid_indices]
    rally_uids = rally_uids[valid_indices]

    # **[LGBM 優化]**：將類別特徵轉換為 'category' dtype
    # 這樣 LightGBM 就能自動將它們視為類別特徵
    cat_cols = ['sex'] + lag_cols + [f'prev_{n}_{col}' for col in lag_cols for n in [1, 2, 3]]
    
    # 確保所有 cat_cols 都在 X 的欄位中
    cat_cols = [col for col in cat_cols if col in X.columns]
    
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # 驗證集切分
    unique_rallies = rally_uids.unique()
    train_rallies, valid_rallies = train_test_split(unique_rallies, test_size=0.2, random_state=42)

    train_mask = rally_uids.isin(train_rallies)
    valid_mask = rally_uids.isin(valid_rallies)
    X_train, X_valid = X[train_mask], X[valid_mask]
    y_action_train, y_action_valid = y_action[train_mask], y_action[valid_mask]
    y_point_train, y_point_valid = y_point[train_mask], y_point[valid_mask]
    y_server_train, y_server_valid = y_server[train_mask], y_server[valid_mask]

    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}")

    # **[LGBM]**：呼叫 LightGBM 函數
    model_action, f1_action = trainLGB('actionId', X_train, y_action_train, X_valid, y_action_valid)
    model_point, f1_point = trainLGB('pointId', X_train, y_point_train, X_valid, y_point_valid)
    model_server, roc_server = train_binary_lgb('serverGetPoint', X_train, y_server_train, X_valid, y_server_valid)
    
    print(f"Validation score: {0.4*f1_action + 0.4*f1_point + 0.2*roc_server:.4f}")

    # 測試集預測
    X_test_raw = test_new.drop(columns=["rally_uid", "rally_id", "match", "numberGame"])
    last_shot_indices = test_new.groupby('rally_uid').tail(1).index
    X_test = X_test_raw.loc[last_shot_indices]

    # **[LGBM 優化]**：同樣轉換測試集的 'category' dtype
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
            
    pred_action = model_action.predict(X_test)
    pred_point = model_point.predict(X_test)
    pred_server = model_server.predict(X_test)

    test_last_shot = test_new.groupby('rally_uid').tail(1)['rally_uid'].values
    submission = pd.DataFrame({"rally_uid": test_last_shot, "serverGetPoint": pred_server, "pointId": pred_point - 1, "actionId": pred_action-1})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()