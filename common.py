"""
============================================================
  共用模組：資料讀取、CNN 模型、評估與畫圖
  三個實驗（原始 / 影像增強 / GAN）都會 import 這個檔案，
  這樣三個實驗唯一的差別只有「訓練資料」，結果才能公平比較。
============================================================
"""

import json                                  # json：把實驗結果存成檔案，最後比較用
from pathlib import Path                     # pathlib：用 __file__ 組路徑，不受執行目錄影響
import numpy as np                           # numpy：影像以陣列形式處理
from PIL import Image                        # Pillow：讀取 jpg 圖片
import matplotlib                            # matplotlib：畫訓練曲線與混淆矩陣
matplotlib.use("Agg")                        # Agg = 只存檔不開視窗，在終端機跑也不會卡住
import matplotlib.pyplot as plt              # pyplot：實際的畫圖介面
import keras                                 # Keras 3：作業指定使用的深度學習套件
from keras import layers                     # layers：建立網路層

# ------------------------------------------------------------
# 路徑與常數
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent                       # 本程式所在資料夾 emotion_training
DATA_DIR = HERE / "archive"                                  # 資料集：與程式放在同一層的 archive 資料夾（train / test）
OUT_DIR = HERE / "outputs"                                   # 所有輸出（模型、圖、結果）都放這裡
OUT_DIR.mkdir(exist_ok=True)                                 # 資料夾不存在就建立
CACHE_FILE = OUT_DIR / "dataset_cache.npz"                   # 第一次讀完圖片後存成 npz，之後秒讀

# 類別順序與老師的 emotion_detection_model.h5 / 03 程式一致，訓練出的模型可以直接替換使用
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # 索引 0~6
NUM_CLASSES = len(CLASS_NAMES)                               # 7 類情緒
IMG_SIZE = 48                                                # FER2013 圖片大小 48x48 灰階
SEED = 42                                                    # 固定亂數種子，三個實驗切出同樣的驗證集

keras.utils.set_random_seed(SEED)                            # 同時固定 python / numpy / TF 的亂數種子


# ------------------------------------------------------------
# 資料讀取
# ------------------------------------------------------------
def _load_split(split):
    """讀取 archive/<split>/<class>/*.jpg，回傳 (影像 uint8 N×48×48×1, 標籤 N)"""
    xs, ys = [], []                                          # 影像清單、標籤清單
    for label, name in enumerate(CLASS_NAMES):               # 依固定順序走過 7 個類別資料夾
        for path in sorted((DATA_DIR / split / name).glob("*.jpg")):  # sorted 讓順序固定
            img = Image.open(path).convert("L")              # 轉成單通道灰階
            if img.size != (IMG_SIZE, IMG_SIZE):             # 保險：尺寸不對就縮放
                img = img.resize((IMG_SIZE, IMG_SIZE))
            xs.append(np.asarray(img, dtype=np.uint8))       # 存成 uint8 省記憶體
            ys.append(label)                                 # 資料夾順序就是標籤
    x = np.stack(xs)[..., None]                              # (N,48,48) → (N,48,48,1) 加上通道維度
    return x, np.asarray(ys, dtype=np.int64)


def load_dataset(val_ratio=0.1):
    """
    回傳 (x_train, y_train), (x_val, y_val), (x_test, y_test)
    - x 是 uint8（0~255），要餵模型前才轉成 0~1 的 float
    - 驗證集從 train 裡「每個類別各切 10%」，三個實驗用同一份（SEED 固定）
    - test 完全不參與訓練與調參，只在最後評估一次
    """
    if CACHE_FILE.exists():                                  # 有快取就直接讀
        d = np.load(CACHE_FILE)
        x_tr, y_tr, x_te, y_te = d["x_tr"], d["y_tr"], d["x_te"], d["y_te"]
    else:                                                    # 沒有就讀圖並建立快取
        print("第一次執行，讀取圖片中（之後會用快取）...")
        x_tr, y_tr = _load_split("train")
        x_te, y_te = _load_split("test")
        np.savez_compressed(CACHE_FILE, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)

    rng = np.random.RandomState(SEED)                        # 專用亂數產生器，不影響其他地方
    val_idx = []                                             # 被選去當驗證集的索引
    for c in range(NUM_CLASSES):                             # 每類各自抽 10%，保留原始類別比例
        idx = np.where(y_tr == c)[0]
        val_idx.extend(rng.choice(idx, int(len(idx) * val_ratio), replace=False))
    val_mask = np.zeros(len(y_tr), dtype=bool)               # 布林遮罩：True = 驗證集
    val_mask[val_idx] = True
    return ((x_tr[~val_mask], y_tr[~val_mask]),
            (x_tr[val_mask], y_tr[val_mask]),
            (x_te, y_te))


def to_float(x):
    """uint8 0~255 → float32 0~1（神經網路對小範圍數值比較好訓練）"""
    return x.astype("float32") / 255.0


def class_counts(y):
    """回傳每個類別的樣本數（長度 7 的 list）"""
    return np.bincount(y, minlength=NUM_CLASSES).tolist()


# ------------------------------------------------------------
# 分類模型（三個實驗共用同一個架構）
# ------------------------------------------------------------
def build_classifier():
    """
    簡單的 CNN：4 個卷積區塊（Conv→BN→ReLU）+ 全連接層。
    - BatchNormalization：讓訓練更穩定、收斂更快
    - Dropout：隨機丟掉部分神經元，降低過擬合
    輸入 48×48×1，輸出 7 類機率（softmax），與老師的模型介面相同
    """
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 1))               # 輸入：48×48 灰階
    x = inp
    for filters, n_conv in ((32, 1), (64, 1), (128, 2), (256, 2)):  # (通道數, 卷積層數)；前面解析度高的區塊少放一層以節省 CPU 運算
        for _ in range(n_conv):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)   # 3×3 卷積
            x = layers.BatchNormalization()(x)               # 標準化
            x = layers.Activation("relu")(x)                 # 非線性
        x = layers.MaxPooling2D()(x)                         # 長寬減半：48→24→12→6→3
        x = layers.Dropout(0.25)(x)                          # 每個區塊後丟掉 25%
    x = layers.Flatten()(x)                                  # 3×3×256 攤平成一維
    x = layers.Dense(256, activation="relu")(x)              # 全連接層整合特徵
    x = layers.Dropout(0.5)(x)                               # 全連接層最容易過擬合，丟 50%
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)  # 7 類機率
    return keras.Model(inp, out)


def train_classifier(name, x_tr, y_tr, x_val, y_val, epochs=30, batch_size=128):
    """
    訓練分類器並回傳 (model, history)。
    三個實驗使用完全相同的設定（優化器、epoch、callback），差別只有 x_tr / y_tr。
    - EarlyStopping：驗證 loss 連續 8 輪沒進步就停，並還原最佳權重
    - ReduceLROnPlateau：驗證 loss 卡住就把學習率減半
    """
    keras.utils.set_random_seed(SEED)                        # 每個實驗從同樣的亂數起點開始
    model = build_classifier()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),     # Adam：最常用的優化器
                  loss="sparse_categorical_crossentropy",    # 標籤是整數 0~6，用 sparse 版本
                  metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(to_float(x_tr), y_tr,
                        validation_data=(to_float(x_val), y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=2)      # verbose=2：每個 epoch 印一行
    model.save(OUT_DIR / f"{name}.keras")                    # 新格式
    model.save(OUT_DIR / f"{name}.h5")                       # 舊格式，和老師的 .h5 一樣可直接載入
    return model, history


# ------------------------------------------------------------
# 評估與畫圖
# ------------------------------------------------------------
def evaluate(name, model, history, x_test, y_test, train_counts):
    """
    在 test 集評估：整體準確率、每類的 precision / recall / F1、macro-F1、混淆矩陣。
    結果存成 outputs/<name>_result.json，並畫出訓練曲線與混淆矩陣圖。
    macro-F1 = 各類 F1 的平均，每一類權重相同，能反映少數類別（disgust）有沒有學好；
    只看 accuracy 會被大類別（happy）掩蓋。
    """
    probs = model.predict(to_float(x_test), verbose=0)       # (N,7) 每張圖的各類機率
    pred = probs.argmax(axis=1)                              # 機率最大的類別就是預測
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)     # 混淆矩陣：列 = 真實，欄 = 預測
    for t, p in zip(y_test, pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)                           # 預測對的數量
    precision = tp / np.maximum(cm.sum(axis=0), 1)           # 預測為某類中，真的是該類的比例
    recall = tp / np.maximum(cm.sum(axis=1), 1)              # 某類真實樣本中，被抓出來的比例
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    result = {
        "name": name,
        "train_counts": dict(zip(CLASS_NAMES, train_counts)),
        "test_accuracy": float((pred == y_test).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i])} for i, c in enumerate(CLASS_NAMES)},
        "epochs_run": len(history.history["loss"]),
    }
    (OUT_DIR / f"{name}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # --- 訓練曲線：看有沒有過擬合（train 一直升、val 不升） ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["loss"], label="train"); ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train"); ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(name); fig.tight_layout(); fig.savefig(OUT_DIR / f"{name}_curves.png", dpi=120); plt.close(fig)

    # --- 混淆矩陣（每列正規化成比例，數字 = 該類真實樣本被判成各類的比例） ---
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name}\nacc={result['test_accuracy']:.3f}  macro-F1={result['macro_f1']:.3f}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT_DIR / f"{name}_confusion.png", dpi=120); plt.close(fig)

    print(f"\n[{name}] test acc = {result['test_accuracy']:.4f}   macro-F1 = {result['macro_f1']:.4f}")
    for c in CLASS_NAMES:
        m = result["per_class"][c]
        print(f"  {c:9s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    return result
