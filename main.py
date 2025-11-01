import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import F1Score, AUC

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
    'actionId', 'pointId', 'serverGetPoint', # 目標欄位 (同時也是特徵)
    'handId', 'strengthId', 'positionId', 'let',
    'PlayerId', 'PlayerServed', 'server', 'set', 'game',
    'strickNum', 'scoreSelf', 'scoreBlue' # 假設 'strickNum' 等也視為分類
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
    g = g.sort_values('strickNum')
    
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

# =========================================================
# 4️⃣ 編譯與訓練 (這部分程式碼完全不用動)
# =========================================================
model.compile(
    optimizer=Adam(1e-3),
    loss={
        'actionId': 'categorical_crossentropy',
        'pointId': 'categorical_crossentropy',
        'serverGetPoint': 'categorical_crossentropy'
    },
    metrics={
        'actionId': ['accuracy', F1Score(average='macro', name='f1_action')],
        'pointId': ['accuracy', F1Score(average='macro', name='f1_point')],
        'serverGetPoint': ['accuracy', AUC(name='auc_server')]
    }
)

history = model.fit(
    X_train,
    {'actionId': y_action_train, 'pointId': y_point_train, 'serverGetPoint': y_server_train},
    validation_data=(X_val, {'actionId': y_action_val, 'pointId': y_point_val, 'serverGetPoint': y_server_val}),
    epochs=20,
    batch_size=64,
    verbose=1
)

# =========================================================
# 5️⃣ 推論測試 與 顯示最終指標 (‼️ 已修正)
# =========================================================

print("\n--- 推論測試 (範例) ---")
preds = model.predict(X_val[:1])
pred_action = np.argmax(preds[0], axis=1)[0]
pred_point = np.argmax(preds[1], axis=1)[0]
pred_server = np.argmax(preds[2], axis=1)[0]

# 使用 target_encoders 來反解
print("Predicted actionId:", target_encoders['actionId'].inverse_transform([pred_action])[0])
print("Predicted pointId:", target_encoders['pointId'].inverse_transform([pred_point])[0])
print("Predicted ServerGetPoint:", target_encoders['serverGetPoint'].inverse_transform([pred_server])[0])


print("\n--- 最終模型評估 (Validation Set) ---")
final_epoch_metrics = history.history

# 提取 F1-Score
# 注意：Keras 會自動加上 'val_' 前綴，以及輸出層的名稱 (e.g., 'val_actionId_f1_action')
f1_action = final_epoch_metrics['val_actionId_f1_action'][-1]
f1_point = final_epoch_metrics['val_pointId_f1_point'][-1]
print(f"Final Validation F1-Score (actionId): {f1_action:.4f}")
print(f"Final Validation F1-Score (pointId):  {f1_point:.4f}")

# KAUC
# 注意：Keras 會自動加上 'val_' 前綴，以及輸出層的名稱 (e.g., 'val_serverGetPoint_auc_server')
auc_server = final_epoch_metrics['val_serverGetPoint_auc_server'][-1]
print(f"Final Validation AUC (serverGetPoint): {auc_server:.4f}")


