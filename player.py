#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# =========================================================
# 1️⃣ 讀取資料
# =========================================================
df = pd.read_csv("train.csv")

df.columns = [
    "rally_uid", "sex", "match_id", "game_number", "rally_id", "stroke_number",
    "score_self", "score_other", "server_get_point",
    "player_id", "player_other_id", "server_id", "serve_number",
    "stroke_id", "hand_id", "strength_id", "spin_id",
    "point_id", "action_id", "let", "position_id"
]

# =========================================================
# 2️⃣ 建立 player-level 特徵表
# =========================================================
def get_mode(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

player_features = (
    df.groupby("player_id")
    .agg(
        match_count=("match_id", "nunique"),
        rally_count=("rally_uid", "count"),
        avg_strength=("strength_id", "mean"),
        avg_spin=("spin_id", "mean"),
        avg_stroke=("stroke_number", "mean"),
        win_rate=("server_get_point", "mean"),
        fav_action=("action_id", get_mode),
        fav_point=("point_id", get_mode)
    )
    .reset_index()
)

print("✅ Player features summary:")
print(player_features.head())

# =========================================================
# 3️⃣ 將主方與對手的選手特徵 merge 回主表
# =========================================================
df = df.merge(player_features.add_prefix("self_"), left_on="player_id", right_on="self_player_id", how="left")
df = df.merge(player_features.add_prefix("opp_"), left_on="player_other_id", right_on="opp_player_id", how="left")

# 移除重複的ID欄位
df = df.drop(columns=["self_player_id", "opp_player_id"])

print(f"✅ Dataset shape after merge: {df.shape}")

# =========================================================
# 4️⃣ 建立訓練資料
# =========================================================
target = "server_get_point"   # 這裡也可以改成 action_id 或 point_id
feature_cols = [
    # 原始比賽特徵
    "sex", "stroke_number", "strength_id", "spin_id", "position_id",
    # 主方選手特徵
    "self_avg_strength", "self_avg_spin", "self_avg_stroke",
    "self_win_rate", "self_match_count", "self_fav_action", "self_fav_point",
    # 對手選手特徵
    "opp_avg_strength", "opp_avg_spin", "opp_avg_stroke",
    "opp_win_rate", "opp_match_count", "opp_fav_action", "opp_fav_point",
]

X = df[feature_cols].fillna(0)
y = df[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =========================================================
# 5️⃣ 訓練 XGBoost
# =========================================================
params = dict(
    objective="binary:logistic",
    eval_metric="logloss",  # 改為 logloss，macro F1 需自算
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

y_pred = model.predict(X_valid)
macro_f1 = f1_score(y_valid, y_pred, average="macro")
print(f"🎯 Validation Macro F1 Score: {macro_f1:.4f}")

# =========================================================
# 6️⃣ 檢查特徵重要性
# =========================================================
import matplotlib.pyplot as plt
import xgboost as xgb
xgb.plot_importance(model, max_num_features=15, importance_type="gain")
plt.title("Feature Importance (Top 15)")
plt.show()
