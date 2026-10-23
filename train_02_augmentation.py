"""
============================================================
  任務 2：影像增強（Data Augmentation）平衡各類數量後再訓練
============================================================
  做法：對數量不足的類別，隨機抽原圖 → 做隨機旋轉／平移／縮放／左右翻轉／
        亮度與對比變化 → 產生「新」圖片，直到每個類別都和最大類別（happy）一樣多。
  注意：
  - 只增強「訓練集」。驗證集與測試集保持原樣，否則評估結果不可信。
  - 增強不改變臉的表情（不做會破壞表情的變形，例如上下顛倒）。

  執行：python train_02_augmentation.py [epochs]
"""

import sys
import numpy as np
import keras
from keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (load_dataset, class_counts, to_float, train_classifier, evaluate,
                    CLASS_NAMES, NUM_CLASSES, OUT_DIR, SEED)

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

# 增強管線：每次呼叫（training=True）都會對每張圖抽新的隨機參數
augmenter = keras.Sequential([
    layers.RandomFlip("horizontal"),                         # 左右翻轉：臉左右對稱，表情不變
    layers.RandomRotation(0.08, fill_mode="nearest"),        # 旋轉 ±0.08 圈（約 ±29°）
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),  # 上下左右平移最多 10%
    layers.RandomZoom(0.15, fill_mode="nearest"),            # 放大縮小最多 15%
    layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),    # 亮度 ±20%
    layers.RandomContrast(0.2),                              # 對比 ±20%
])


def augment_to_target(x, y, target):
    """
    把每個類別補到 target 張：不足的部分從該類原圖隨機抽樣（可重複）後做增強。
    回傳新的 (x, y)（uint8），原圖全部保留，只是額外加上增強圖。
    """
    rng = np.random.RandomState(SEED)
    xs, ys = [x], [y]                                        # 先放入全部原圖
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        need = target - len(idx)                             # 還差幾張
        if need <= 0:                                        # 已經夠多（最大類別）就不用補
            continue
        pick = rng.choice(idx, need, replace=True)           # 隨機抽原圖當素材
        new = []
        for s in range(0, need, 512):                        # 分批處理避免一次吃太多記憶體
            batch = to_float(x[pick[s:s + 512]])
            out = augmenter(batch, training=True).numpy()    # training=True 才會真的做隨機變換
            new.append(np.clip(out * 255.0, 0, 255).astype(np.uint8))
        xs.append(np.concatenate(new)); ys.append(np.full(need, c, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


(x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_dataset()
print("增強前各類數量：", class_counts(y_tr))
target = max(class_counts(y_tr))                             # 目標 = 最大類別的數量
x_aug, y_aug = augment_to_target(x_tr, y_tr, target)
print("增強後各類數量：", class_counts(y_aug))

# 存一張範例圖：每列一個類別，第 1 欄是原圖，後面是增強後的樣子（方便寫報告）
fig, axes = plt.subplots(NUM_CLASSES, 8, figsize=(10, 9))
for c in range(NUM_CLASSES):
    src = x_tr[np.where(y_tr == c)[0][0]]                    # 該類第一張原圖
    axes[c, 0].imshow(src[..., 0], cmap="gray"); axes[c, 0].set_ylabel(CLASS_NAMES[c], fontsize=8)
    for k in range(1, 8):
        a = augmenter(to_float(src[None]), training=True).numpy()[0, ..., 0]
        axes[c, k].imshow(a, cmap="gray", vmin=0, vmax=1)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_title("original", fontsize=8)
fig.suptitle("Augmentation examples"); fig.tight_layout()
fig.savefig(OUT_DIR / "02_augmentation_examples.png", dpi=120); plt.close(fig)

model, history = train_classifier("02_augmentation", x_aug, y_aug, x_val, y_val, epochs=epochs)
evaluate("02_augmentation", model, history, x_te, y_te, class_counts(y_aug))
