import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# =========================================================
# ❇️ 0. 定義 Focal Loss (載入模型時必須)
# =========================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        modulating_factor = K.pow(1.0 - y_pred, gamma)
        alpha_factor = alpha
        focal_loss = alpha_factor * modulating_factor * cross_entropy
        return K.mean(K.sum(focal_loss, axis=-1))
    return focal_loss

# =========================================================
# 1. 定義常數 (MAX_SEQ_LEN 必須與訓練時一致)
# =========================================================
MAX_SEQ_LEN = 10 
target_cols = ['actionId', 'pointId', 'serverGetPoint']

# 檔案路徑
TEST_CSV_PATH = "test.csv"
SAMPLE_SUBMISSION_PATH = "sample_submission.csv"
MODEL_PATH = "my_lstm_model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl" # ❇️ 讀取新的 pkl 檔案
OUTPUT_PATH = "submission.csv"

# =========================================================
# ❇️ 2. 載入模型和 Preprocessor (!! 已修改 !!)
# =========================================================
print("載入模型中...")
model = load_model(
    MODEL_PATH, 
    custom_objects={'focal_loss': categorical_focal_loss()}
)
print("模型載入完畢。")

print("載入 Preprocessor (Encoders 和 feature_cols) 中...")
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor_data = pickle.load(f)

encoders = preprocessor_data['encoders']
feature_cols = preprocessor_data['feature_cols'] # ❇️ 關鍵：載入訓練時的特徵列表

print("Preprocessor 載入完畢。")
print(f"模型期望 {len(feature_cols)} 個特徵: {feature_cols}")

# =========================================================
# ❇️ 3. 讀取並預處理 Test Data (!! 已修改 !!)
# =========================================================
test_df = pd.read_csv(TEST_CSV_PATH)
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# 3.1 補全 test.csv 缺少的欄位
# (例如 'actionId', 'pointId' 在 test.csv 中不存在，我們必須補上)
# 我們使用一個固定的預設值 'UNKNOWN' 來填充
DEFAULT_PLACEHOLDER = 'UNKNOWN' 

for col in feature_cols:
    if col not in test_df.columns:
        print(f"警告: test.csv 缺少欄位 '{col}'。將使用 '{DEFAULT_PLACEHOLDER}' 填充。")
        test_df[col] = DEFAULT_PLACEHOLDER

print("欄位補全完畢。")

# 3.2 Label Encoding (!! 關鍵：處理未知標籤 !!)
for col in feature_cols: # ❇️ 遍歷固定的 feature_cols
    if col in encoders:
        # print(f"處理欄位: {col}")
        le = encoders[col]
        
        # 取得所有已知的標籤
        known_classes = set(le.classes_)
        
        # 檢查我們的預設值 'UNKNOWN' 是否在訓練類別中
        if DEFAULT_PLACEHOLDER not in known_classes:
            # 如果不在，動態地將它添加到 LabelEncoder 中
            le.classes_ = np.append(le.classes_, DEFAULT_PLACEHOLDER)
            known_classes.add(DEFAULT_PLACEHOLDER)
            print(f"已將 '{DEFAULT_PLACEHOLDER}' 添加到 '{col}' 的編碼器中。")

        # 找出 test set 中所有 *其他* 未知標籤
        unknown_in_test = set(test_df[col].unique()) - known_classes
        
        # 決定一個預設值 (例如訓練集的第一個類別)
        default_class_for_others = le.classes_[0]

        def transform_value(x):
            if x in known_classes:
                return x
            else:
                return default_class_for_others # 替換所有其他未知值
        
        if unknown_in_test:
            print(f"警告: 欄位 '{col}' 發現未知標籤 {unknown_in_test}。將替換為 '{default_class_for_others}'。")
            test_df[col] = test_df[col].apply(transform_value)
        
        # 進行轉換
        test_df[col] = le.transform(test_df[col])
    else:
        print(f"錯誤：在儲存的 encoders 中找不到欄位 '{col}' 的編碼器。")


print("Test data 預處理完畢。")

# =========================================================
# 4. 進行推論 (此部分無需修改)
# =========================================================
print("開始進行推論...")

# 建立一個 'target_encoders' 子字典，方便反解
target_encoders = {col: encoders[col] for col in target_cols if col in encoders}

# 將 test_df 按照 rally_uid 分組
test_groups = test_df.groupby('rally_uid')

results = []

for rally_uid in tqdm(submission_df['rally_uid']):
    try:
        g = test_groups.get_group(rally_uid)
        g = g.sort_values('strickNumber')
        
        # ❇️ 關鍵：這裡現在會從 g 中選取 'feature_cols' (11個)
        rally_features = g[feature_cols].values 
        
        X_test_rally = pad_sequences(
            [rally_features], 
            maxlen=MAX_SEQ_LEN, 
            dtype='float32', 
            padding='pre', 
            truncating='pre'
        )
        
        # ❇️ X_test_rally.shape 現在應該是 (1, 10, 11)
        preds = model.predict(X_test_rally, verbose=0)
        
        pred_action_idx = np.argmax(preds[0], axis=1)[0]
        pred_point_idx = np.argmax(preds[1], axis=1)[0]
        pred_server_idx = np.argmax(preds[2], axis=1)[0]
        
        pred_action_label = target_encoders['actionId'].inverse_transform([pred_action_idx])[0]
        pred_point_label = target_encoders['pointId'].inverse_transform([pred_point_idx])[0]
        pred_server_label = target_encoders['serverGetPoint'].inverse_transform([pred_server_idx])[0]
        
        results.append({
            'rally_uid': rally_uid,
            'actionId': pred_action_label,
            'pointId': pred_point_label,
            'serverGetPoint': pred_server_label
        })
        
    except KeyError:
        print(f"警告: 在 test.csv 中找不到 rally_uid {rally_uid}。將使用 None 填充。")
        results.append({
            'rally_uid': rally_uid,
            'actionId': None,
            'pointId': None,
            'serverGetPoint': None
        })

print("推論完成。")

# =========================================================
# 5. 產生並儲存 Submission 檔案 (此部分無需修改)
# =========================================================
output_df = pd.DataFrame(results)
output_df = output_df[['rally_uid', 'serverGetPoint', 'pointId', 'actionId']]
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"預測結果已儲存至: {OUTPUT_PATH}")
print(output_df.head())