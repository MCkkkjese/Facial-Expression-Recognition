"""
============================================================
  任務 1：原始資料、不做任何處理，直接訓練情緒分類模型（基準線）
============================================================
  資料集 FER2013 類別嚴重不平衡（happy 7215 張 vs disgust 436 張，約 16 倍）。
  這個實驗當作「對照組」，後面兩個實驗的進步幅度都跟它比。

  執行：python train_01_baseline.py [epochs]
"""

import sys                                   # sys：讀取命令列參數（epoch 數）
from common import (load_dataset, class_counts, train_classifier, evaluate)  # 共用函式

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30   # 預設最多訓練 30 輪（EarlyStopping 會提早停）

(x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_dataset()   # 載入資料，不做任何增強或平衡
print("訓練集各類數量：", class_counts(y_tr))                # 印出不平衡的程度
print("驗證集各類數量：", class_counts(y_val))
print("測試集各類數量：", class_counts(y_te))

model, history = train_classifier("01_baseline", x_tr, y_tr, x_val, y_val, epochs=epochs)
evaluate("01_baseline", model, history, x_te, y_te, class_counts(y_tr))
