# 人工智慧加分作業：FER2013 情緒分類（Keras）

三個實驗，比較「原始資料 / 影像增強 / GAN 補資料」對情緒分類的影響。

## 資料夾內容
- `archive/`：資料集 FER2013（train / test，各 7 類，48×48 灰階）
- `common.py`：共用的資料讀取、CNN 模型、評估與畫圖
- `train_01_baseline.py`：任務 1，原始資料直接訓練
- `train_02_augmentation.py`：任務 2，影像增強後補平衡再訓練
- `train_03_gan.py`：任務 3，條件式 GAN 生成假臉補平衡再訓練
- `requirements.txt`：套件版本
- `outputs/`：已跑出的結果（模型 .keras/.h5、訓練曲線、混淆矩陣、GAN 範例圖、result.json、log）

## 安裝與執行
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train_01_baseline.py       # 約 30 分鐘（CPU）
python train_02_augmentation.py   # 約 50 分鐘
python train_03_gan.py            # 約 1.5~2 小時（GAN 約 30 分鐘 + 分類器）
```
第一次執行會把圖片讀成 `outputs/dataset_cache.npz`，之後直接讀快取。

## 結果（測試集）
| 實驗 | 準確率 | macro-F1 | disgust F1 |
|---|---|---|---|
| 1 原始資料 | 59.6% | 0.493 | 0.00 |
| 2 影像增強 | 60.8% | 0.581 | 0.59 |
| 3 GAN | 59.0% | 0.571 | 0.57 |

類別順序：angry, disgust, fear, happy, sad, surprise, neutral（與老師的 emotion_detection_model.h5 相同）。

## 網頁 Demo（GitHub Pages）
`docs/` 是純靜態網頁，用 TensorFlow.js 直接在瀏覽器執行三個訓練好的模型（上傳圖片或開攝影機即時辨識），並展示測試集結果與圖表。

- 本機預覽：`cd docs && python3 -m http.server 8000`，開 http://localhost:8000
- 重新匯出模型權重（重訓後）：`pip install h5py numpy && python export_web.py`
- 部署：推上 GitHub 後，到 Settings → Pages → Source 選 `Deploy from a branch`，Branch 選 `main`、資料夾選 `/docs`
