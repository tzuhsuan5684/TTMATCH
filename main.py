import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf # ❇️ 新增
import tensorflow.keras.backend as K # ❇️ 新增
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import F1Score, AUC
import pickle

import wandb
from wandb.integration.keras import WandbModelCheckpoint
from tensorflow.keras.callbacks import LambdaCallback 

# ❇️ --- 新增混淆矩陣需要的函式庫 ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# ❇️ --- 結束 ---

# =========================================================
# 1️⃣ 讀取資料 (‼️ 已修正：移除數值特徵處理)
# =========================================================
CSV_PATH = "train.csv"
MAX_SEQ_LEN = 10  # padding 上限

df = pd.read_csv(CSV_PATH)

# 1. 定義目標欄位
target_cols = ['actionId', 'pointId', 'serverGetPoint']

# 2. 定義所有要"作為特徵"的欄位 (!! 關鍵：包含 target_cols)
# (您可以自行增減)
# ‼️ 假設所有特徵均為分類
categorical_features = [
    'sex', 'actionId', 'pointId', 'serverGetPoint', # 目標欄位 (同時也是特徵)
    'handId', 'strengthId', 'positionId', 'let',
    'PlayerId', 'PlayerServed', 'server', 'set', 'game',
    'strickNumber', 'scoreSelf', 'scoreOther' # 假設 'strickNum' 等也視為分類
]

# 過濾掉 'train.csv' 中可能不存在的欄位
categorical_features = [col for col in categorical_features if col in df.columns]

# 總特徵列表 (按此順序)
feature_cols = categorical_features # 僅包含分類特徵

# 3. 記住 target 欄位在總特徵列表中的 "索引"
# 這將用於第 2 節中切分 Y
target_indices = [feature_cols.index(col) for col in target_cols]

# 4. Label encoding (針對所有分類特徵)
encoders = {}
for col in categorical_features:
    if col in df.columns: # 再次檢查，確保欄位存在
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

# 5. 特徵標準化 (已刪除)

# 6. 建立一個只包含"目標"解碼器的字典，以便稍後反解
target_encoders = {col: encoders[col] for col in target_cols if col in encoders}


# =========================================================
# 2️⃣ 建立序列資料 (‼️ 已修正：使用新的 feature_cols)
# =========================================================
X_list, Y_list = [], []
for _, g in df.groupby(['match', 'rally_id']):
    g = g.sort_values('strickNumber')
    
    # 提取所有 "feature_cols" 定義的特徵
    all_feats_array = g[feature_cols].values
    
    if len(g) > 1:
        for i in range(1, len(g)):
            # 特徵: t=0 到 t=i-1 的所有特徵
            X_list.append(all_feats_array[:i]) 
            # 標籤: t=i 的 'actionId', 'pointId', 'serverGetPoint'
            Y_list.append(all_feats_array[i, target_indices]) 

# padding 對齊序列長度
X = pad_sequences(X_list, maxlen=MAX_SEQ_LEN, dtype='float32', padding='pre', truncating='pre')
Y = np.array(Y_list)

# 轉換標籤 (這部分程式碼完全不用動)
y_action, y_point, y_server = Y[:, 0], Y[:, 1], Y[:, 2]
num_action = len(target_encoders['actionId'].classes_)
num_point = len(target_encoders['pointId'].classes_)
num_server = len(target_encoders['serverGetPoint'].classes_)

y_action = to_categorical(y_action, num_action)
y_point = to_categorical(y_point, num_point)
y_server = to_categorical(y_server, num_server)

# 分割資料 (這部分程式碼完全不用動)
X_train, X_val, y_action_train, y_action_val, y_point_train, y_point_val, y_server_train, y_server_val = \
    train_test_split(X, y_action, y_point, y_server, test_size=0.2, random_state=42)

print(f"訓練樣本數: {X_train.shape[0]}")
print(f"驗證樣本數: {X_val.shape[0]}")
print(f"序列維度 (MAX_SEQ_LEN, num_features): ({X_train.shape[1]}, {X_train.shape[2]})")


# =========================================================
# 3️⃣ 模型設計（input_dim 會自動更新）
# =========================================================
input_dim = X.shape[2] # 特徵維度現在會自動更新 (變多了)
inputs = Input(shape=(MAX_SEQ_LEN, input_dim))
x = Masking(mask_value=0.0)(inputs)
x = LSTM(128)(x)
x = Dropout(0.3)(x)

# 三個輸出 head (這部分程式碼完全不用動)
action_out = Dense(num_action, activation='softmax', name='actionId')(x)
point_out = Dense(num_point, activation='softmax', name='pointId')(x)
server_out = Dense(num_server, activation='softmax', name='serverGetPoint')(x)

model = Model(inputs=inputs, outputs=[action_out, point_out, server_out])
model.summary() # 建議檢查 summary，確認模型架構

EPOCH=10
BATCH_SIZE=64
DROPOUT=0.3
learning_rate=1e-3


wandb.init(
    project="TTMATCH-prediction", 
    job_type="train",
    config={
        "max_seq_len": MAX_SEQ_LEN,
        "lstm_units": 128,
        "dropout": DROPOUT,
        "learning_rate": learning_rate,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "loss_action": "focal_loss", # ❇️ (可選) 在 config 中註記
        "loss_point": "focal_loss"  # ❇️ (可選) 在 config 中註記
    }
)

# =========================================================
# ❇️ 4️⃣-1：定義 Focal Loss
# =========================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Categorical Focal Loss.
    """
    def focal_loss(y_true, y_pred):
        # Clip predictions to avoid log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate Categorical Crossentropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate the modulating factor (1-y_pred)^gamma
        modulating_factor = K.pow(1.0 - y_pred, gamma)
        
        # Apply alpha
        alpha_factor = alpha
        
        # Calculate the final focal loss
        # We multiply by y_true to select only the loss for the true class
        focal_loss = alpha_factor * modulating_factor * cross_entropy
        
        # Sum over classes, mean over batch
        return K.mean(K.sum(focal_loss, axis=-1))

    return focal_loss


# =========================================================
# 4️⃣-2：編譯與訓練 (❇️ 已修改 loss)
# =========================================================
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss={
        # ❇️ 修改：使用 Focal Loss，並傳入超參數
        'actionId': categorical_focal_loss(gamma=2.0, alpha=0.25),
        'pointId': categorical_focal_loss(gamma=2.0, alpha=0.25),
        # ❇️ 不變：serverGetPoint 仍使用標準交叉熵
        'serverGetPoint': 'categorical_crossentropy'
    },
    metrics={
        'actionId': ['accuracy', F1Score(average='macro', name='f1_action')],
        'pointId': ['accuracy', F1Score(average='macro', name='f1_point')],
        'serverGetPoint': ['accuracy', AUC(name='auc_server')]
    }
)

wandb_callbacks = [
    LambdaCallback(on_epoch_end=lambda epoch, logs: wandb.log(logs)),
    WandbModelCheckpoint(
        filepath=f"wandb-run-{wandb.run.id}-best-model.keras", 
        monitor='val_loss', 
        save_best_only=True,
        mode='min'
    )
]

history = model.fit(
    X_train,
    {'actionId': y_action_train, 'pointId': y_point_train, 'serverGetPoint': y_server_train},
    validation_data=(X_val, {'actionId': y_action_val, 'pointId': y_point_val, 'serverGetPoint': y_server_val}),
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=wandb_callbacks  # ✅ 別忘了這一行！
)

# =========================================================
# ❇️ 5️⃣-1：計算並記錄混淆矩陣 (❇️ 新增區塊)
# =========================================================
print("\n--- 正在計算驗證集的預測結果 (用於混淆矩陣)... ---")
preds_val = model.predict(X_val)
pred_action_probs, pred_point_probs, pred_server_probs = preds_val

# 將 one-hot (y_val) 轉回類別
true_action_labels = np.argmax(y_action_val, axis=1)
true_point_labels = np.argmax(y_point_val, axis=1)
true_server_labels = np.argmax(y_server_val, axis=1)

# 將預測機率 (preds_val) 轉為類別
pred_action_labels = np.argmax(pred_action_probs, axis=1)
pred_point_labels = np.argmax(pred_point_probs, axis=1)
pred_server_labels = np.argmax(pred_server_probs, axis=1)

# 取得類別名稱 (用於 W&B 繪圖)
action_classes = target_encoders['actionId'].classes_
point_classes = target_encoders['pointId'].classes_
server_classes = target_encoders['serverGetPoint'].classes_

# --- 1. 記錄到 W&B (使用 wandb.plot.confusion_matrix) ---
print("--- 正在記錄混淆矩陣至 W&B (wandb.plot)... ---")
wandb.log({
    "conf_mat_action": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_action_labels,
        preds=pred_action_labels,
        class_names=action_classes,
        title="CM ActionID (Val)"
    ),
    "conf_mat_point": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_point_labels,
        preds=pred_point_labels,
        class_names=point_classes,
        title="CM PointID (Val)"
    ),
    "conf_mat_server": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_server_labels,
        preds=pred_server_labels,
        class_names=server_classes,
        title="CM ServerGetPoint (Val)"
    )
})

# --- 2. 繪製熱圖 (Heatmap) 並記錄為圖片 (wandb.Image) ---
# (這種圖在 W&B 報告中通常更清晰)
print("--- 正在記錄混淆矩陣至 W&B (Seaborn Heatmap)... ---")

def plot_cm_heatmap(cm, class_names, title):
    """一個輔助函式，用 Seaborn 繪製熱圖"""
    plt.figure(figsize=(12, 10))
    # 檢查類別數量，如果太多，就不顯示 x/y 標籤，避免擁擠
    show_ticks = len(class_names) <= 50 
    
    sns.heatmap(
        cm, 
        annot=True, # 顯示數字
        fmt='d',    # 整數格式
        cmap='Blues', 
        xticklabels=class_names if show_ticks else False, 
        yticklabels=class_names if show_ticks else False
    )
    plt.title(title, fontsize=16)
    plt.ylabel('Actual (True Label)', fontsize=12)
    plt.xlabel('Predicted (Model Label)', fontsize=12)
    
    plt.tight_layout()
    # 傳回 plt 物件以便 W&B 記錄
    return plt

# 計算 Sklearn 的 CM
cm_action = confusion_matrix(true_action_labels, pred_action_labels)
cm_point = confusion_matrix(true_point_labels, pred_point_labels)
cm_server = confusion_matrix(true_server_labels, pred_server_labels)

# 繪製並記錄 ActionID
plt_action = plot_cm_heatmap(cm_action, action_classes, "Heatmap - ActionID (Val)")
wandb.log({"heatmap_action": wandb.Image(plt_action)})
plt.close() # 關閉 plt 以免佔用記憶體

# 繪製並記錄 PointID
plt_point = plot_cm_heatmap(cm_point, point_classes, "Heatmap - PointID (Val)")
wandb.log({"heatmap_point": wandb.Image(plt_point)})
plt.close()

# 繪製並記錄 ServerGetPoint
plt_server = plot_cm_heatmap(cm_server, server_classes, "Heatmap - ServerGetPoint (Val)")
wandb.log({"heatmap_server": wandb.Image(plt_server)})
plt.close()

print("--- 混淆矩陣記錄完畢 ---")


# =========================================================
# 5️⃣-2：推論測試 與 顯示最終指標 (‼️ 已修正)
# =========================================================

print("\n--- 推論測試 (範例) ---")
# (我們可以使用剛才計算 'preds_val' 的第一個樣本)
pred_action = pred_action_labels[0]
pred_point = pred_point_labels[0]
pred_server = pred_server_labels[0]

# 使用 target_encoders 來反解
print("Predicted actionId:", target_encoders['actionId'].inverse_transform([pred_action])[0])
print("Predicted pointId:", target_encoders['pointId'].inverse_transform([pred_point])[0])
print("Predicted ServerGetPoint:", target_encoders['serverGetPoint'].inverse_transform([pred_server])[0])


print("\n--- 最終模型評估 (Validation Set) ---")
final_epoch_metrics = history.history

# 提取 F1-Score
# 注意：Keras 會自動加上 'val_' 前缀，以及輸出層的名稱 (e.g., 'val_actionId_f1_action')
f1_action = final_epoch_metrics['val_actionId_f1_action'][-1]
f1_point = final_epoch_metrics['val_pointId_f1_point'][-1]
print(f"Final Validation F1-Score (actionId): {f1_action:.4f}")
print(f"Final Validation F1-Score (pointId):  {f1_point:.4f}")

# KAUC
# 注意：Keras 會自動加上 'val_' 前缀，以及輸出層的名稱 (e.g., 'val_serverGetPoint_auc_server')
auc_server = final_epoch_metrics['val_serverGetPoint_auc_server'][-1]
print(f"Final Validation AUC (serverGetPoint): {auc_server:.4f}")
print(f"Final Score: {(0.4 * f1_action + 0.4 * f1_point + 0.2 *auc_server):.4f}")

wandb.summary["final_val_f1_action"] = f1_action
wandb.summary["final_val_f1_point"] = f1_point
wandb.summary["final_val_auc_server"] = auc_server

wandb.finish()

print("\n--- 儲存模型與 Preprocessor ---")

# 1. 儲存模型
MODEL_SAVE_PATH = "my_lstm_model.keras"
model.save(MODEL_SAVE_PATH)
print(f"模型已儲存至: {MODEL_SAVE_PATH}")

# 2. 儲存 Encoders 和 feature_cols 列表
# (!! 關鍵：我們把訓練時最終使用的 feature_cols 列表一起存起來 !!)
PREPROCESSOR_PATH = "preprocessor.pkl"
data_to_save = {
    'encoders': encoders,
    'feature_cols': feature_cols  # <--- ❇️ 儲存這個重要的列表
}
with open(PREPROCESSOR_PATH, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"Encoders 和 feature_cols 已儲存至: {PREPROCESSOR_PATH}")

print("儲存完畢，訓練腳本結束。")
