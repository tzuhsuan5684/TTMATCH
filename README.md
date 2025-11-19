# Table Tennis Rally Prediction — LightGBM Pipeline

針對桌球回合同步預測三個目標變數，使用 LightGBM 建立多模型 pipeline，並產出可直接提交的 `submission.csv`。

---

## 1. 問題與任務說明

比賽需要對每一個 rally（來回）預測三個目標：

- **`actionId`（多分類）**：預測「下一拍動作類型」
- **`pointId`（多分類）**：預測「下一拍落點區域」
- **`serverGetPoint`（二分類）**：預測「該回合最後是否由發球方得分」

本程式將三個任務拆成三個獨立模型來訓練與預測，最後組合成一份提交檔。

---

## 2. 方法與整體流程

### 2.1 資料前處理（Data Preprocessing）

1. **讀取資料**
   - 讀取 `train.csv`、`test.csv` 至 DataFrame。
   - 複製為 `train_new`、`test_new` 供後續修改。

2. **調整標籤編碼**
   - 將 `actionId`、`pointId` 整體加 1，避免使用 0 作為類別編碼，利於後續編碼與處理。

3. **對齊「下一拍」標籤**
   - 以 `rally_uid` 分組，將 `actionId`、`pointId` 進行 **向後位移一拍（shift -1）**，作為要預測的「下一拍」標籤。
   - 排除在位移後產生缺失值（通常是每個 rally 的最後一拍）的樣本。

4. **目標欄位與特徵欄位**
   - 目標：
     - `y_action`：下一拍的 `actionId`
     - `y_point`：下一拍的 `pointId`
     - `y_server`：該回合的 `serverGetPoint`
   - 移除與預測無關或身分性質欄位，例如：
     - `rally_uid`, `rally_id`, `match`, `numberGame`, `serverGetPoint` 等，作為特徵的排除欄位。

---

### 2.2 特徵工程（Feature Engineering）

1. **時序 Lag 特徵**
   - 針對具代表性的欄位（例如 `actionId`, `pointId`, `spinId`, `strengthId`, `positionId`），以 `rally_uid` 分組，建立過去 **1、2、3 拍**的歷史特徵（`prev_1_*`, `prev_2_*`, `prev_3_*`）。
   - 這些特徵讓模型能掌握球來回的節奏與模式。

2. **Target Encoding**
   - 對部分類別欄位（例如 `strengthId`, `spinId`, `handId`）進行 target encoding。
   - 使用 **下一拍的 `actionId`** 作為 target，讓這些特徵反映「此類型擊球通常導致什麼樣的下一拍動作」。

3. **類別型欄位標註**
   - 將包含離散值的欄位（如 `sex`、原始欄位與各種 lag 特徵）標註為 `category` 型別，讓 LightGBM 自動以類別方式處理。

---

### 2.3 訓練與驗證策略（Training & Validation）

1. **Rally 等級切分（避免資料洩漏）**
   - 先取得所有 `rally_uid` 的唯一值，隨機切分為訓練集與驗證集的 rally 清單（例如 8:2）。
   - 之後再依 `rally_uid` 是否在清單內，產生 train/valid 的 row mask。
   - 這樣可以確保：同一個 rally 不會同時出現在訓練與驗證中，避免時間與內容洩漏。

2. **不平衡問題處理**
   - 使用 `class_weight` 計算類別權重，轉成 `sample_weight` 傳入 LightGBM 訓練，使少數類別能被較公平地學習。

3. **模型架構**
   - 共訓練 **三個獨立 LightGBM 模型**：
     - `actionId`：多分類模型
     - `pointId`：多分類模型
     - `serverGetPoint`：二分類模型
   - 多分類模型使用多類別交叉熵（multi_logloss）作為優化目標，二分類模型使用 binary logloss。

4. **評估指標**
   - `actionId`：加權 F1 分數（weighted F1）
   - `pointId`：加權 F1 分數（weighted F1）
   - `serverGetPoint`：ROC AUC 分數
   - 並計算一個加權總分：
     - `0.4 * F1_action + 0.4 * F1_point + 0.2 * ROC_AUC_server`
   - 訓練過程同時輸出各任務的混淆矩陣圖檔，協助分析各類別的預測情形。

---

### 2.4 推論與提交檔產生（Inference & Submission）

1. **測試集特徵處理**
   - 在 `test_new` 上進行與訓練資料相同的前處理與特徵工程（lag 特徵、target encoding、欄位型態設定等）。

2. **只取每個 rally 的最後一拍進行預測**
   - 針對測試集，對每個 `rally_uid` 取出最後一筆紀錄，作為該 rally 的代表樣本。
   - 對這些最後一拍的特徵餵入三個模型進行預測。

3. **還原 label 編碼與輸出**
   - 由於訓練時 `actionId`、`pointId` 有做 +1 平移，因此在輸出時需再 -1 還原回原始編碼。
   - 將 `rally_uid`、`serverGetPoint`、`pointId`、`actionId` 組成 DataFrame，輸出成 `submission.csv`，格式符合競賽要求。

---

## 3. 操作說明（How to Run）

### 3.1 環境需求

需要 Python 3.x，並安裝下列套件（可使用 `pip`）：

- `pandas`
- `numpy`
- `lightgbm`
- `xgboost`
- `category_encoders`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `seaborn`

安裝範例：

```bash
pip install pandas numpy lightgbm xgboost category_encoders scikit-learn tqdm matplotlib seaborn
```

---

### 3.2 檔案結構

請將程式與資料整理如下：

```text
.
├── main.py        # 內含使用 LightGBM 訓練與推論的完整程式
└── README.md
```

若你的主程式檔名不同，請自行調整以下執行指令中的檔名。

---

### 3.3 執行步驟

1. 確認 `train.csv`、`test.csv` 與 `main.py` 位於同一資料夾。
2. 在該資料夾開啟終端機或命令列。
3. 執行：

   ```bash
   python main.py
   ```

4. 程式執行流程會自動完成：
   - 資料前處理與特徵工程
   - Rally-level 切分訓練／驗證
   - 三個 LightGBM 模型訓練與評估
   - 混淆矩陣圖檔輸出
   - 測試集推論與 `submission.csv` 生成

---

## 4. 輸出檔案說明

### 4.1 `submission.csv`

- 欄位說明：

  | 欄位名稱        | 說明                            |
  |-----------------|---------------------------------|
  | `rally_uid`     | 回合同一識別碼                 |
  | `serverGetPoint` | 模型預測之發球方是否得分 (0/1) |
  | `pointId`       | 模型預測之下一拍落點類別       |
  | `actionId`      | 模型預測之下一拍動作類別       |

- 可直接上傳至比賽平台作為預測結果。

### 4.2 混淆矩陣圖檔

程式會輸出三張 PNG 圖檔（檔名依任務命名），內容為：

- 真實標籤 vs 預測標籤的混淆矩陣
- 有助於檢視模型對不同類別的預測強弱與錯誤型態

---

## 5. 總結

本專案的重點特色：

- 使用 **時序特徵（lag features）** 搭配 rally 序列結構
- 將三個預測目標拆成三個 LightGBM 模型，便於獨立調整與分析
- 使用 **class weight / sample weight** 緩解資料不平衡問題
- 採用 **rally-level 訓練／驗證切分**，避免同一回合資訊洩漏
- 自動產生 `submission.csv` 與混淆矩陣圖檔，方便提交與檢視結果

只要準備好 `train.csv`、`test.csv`，安裝必要套件並執行 `python main.py`，即可完整重現本方法的訓練與推論流程。
