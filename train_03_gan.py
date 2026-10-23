"""
============================================================
  任務 3：用條件式 GAN（cDCGAN）生成假臉補足少數類別，再訓練情緒分類模型
============================================================
  GAN 原理：兩個網路互相對抗
    生成器 G：輸入「隨機雜訊 + 想要的情緒標籤」→ 輸出一張 48×48 假臉
    判別器 D：輸入「圖片 + 情緒標籤」→ 判斷是真圖還是 G 生成的
    G 想騙過 D，D 想抓出 G，兩者一起進步，G 就越畫越像真的。
  「條件式」(conditional) = 把情緒標籤也餵給 G 和 D，這樣才能指定要生成哪一種表情。

  流程：
    1. 只用「訓練集」訓練 cGAN（驗證／測試集不能給 GAN 看，避免資料洩漏）
    2. 對數量不足的類別生成假圖，補到和最大類別一樣多
    3. 真圖 + 假圖 一起訓練分類器（架構、超參數與實驗 1、2 完全相同）
    4. 驗證與測試仍然只用真圖

  執行：python train_03_gan.py [gan_steps] [classifier_epochs]
"""

import sys
import time
import numpy as np
import tensorflow as tf                                      # 自訂訓練迴圈需要 GradientTape
import keras
from keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (load_dataset, class_counts, train_classifier, evaluate,
                    CLASS_NAMES, NUM_CLASSES, IMG_SIZE, OUT_DIR, SEED)

GAN_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000   # GAN 訓練步數（每步 = 一個 batch）
CLF_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30    # 分類器最多訓練幾輪
BATCH = 64                                                    # GAN 每個 batch 的圖片數
LATENT = 100                                                  # 雜訊向量長度

keras.utils.set_random_seed(SEED)


# ------------------------------------------------------------
# 生成器：雜訊 + 標籤 → 假臉
# ------------------------------------------------------------
def build_generator():
    z = keras.Input((LATENT,))                                # 隨機雜訊
    lab = keras.Input((1,), dtype="int32")                    # 想生成的情緒（0~6）
    emb = layers.Flatten()(layers.Embedding(NUM_CLASSES, 50)(lab))  # 標籤 → 50 維向量
    x = layers.Concatenate()([z, emb])                        # 雜訊與標籤接在一起
    x = layers.Dense(6 * 6 * 256, use_bias=False)(x)          # 先變成小尺寸特徵圖
    x = layers.Reshape((6, 6, 256))(x)
    for filters in (128, 64):                                 # 轉置卷積：每層把長寬放大 2 倍 6→12→24
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh")(x)  # 24→48，tanh 輸出 -1~1
    return keras.Model([z, lab], out, name="generator")


# ------------------------------------------------------------
# 判別器：圖片 + 標籤 → 真/假分數
# ------------------------------------------------------------
def build_discriminator():
    img = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    lab = keras.Input((1,), dtype="int32")
    emb = layers.Flatten()(layers.Embedding(NUM_CLASSES, 50)(lab))
    emb = layers.Dense(IMG_SIZE * IMG_SIZE)(emb)              # 標籤變成跟圖片一樣大的一個通道
    emb = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(emb)
    x = layers.Concatenate()([img, emb])                      # 圖片與標籤圖疊成 2 通道
    for filters in (64, 128, 256):                            # 步幅 2 的卷積：48→24→12→6
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)                            # Dropout 防止 D 太強而壓死 G
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)                                  # 不加 sigmoid，直接輸出 logit（損失函數會處理）
    return keras.Model([img, lab], out, name="discriminator")


G, D = build_generator(), build_discriminator()
g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)               # GAN 常用設定：lr 2e-4、beta_1 0.5
d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
bce = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real, labels):
    """一步訓練：先更新 D，再更新 G"""
    n = tf.shape(real)[0]
    z = tf.random.normal((n, LATENT))
    with tf.GradientTape() as d_tape:
        fake = G([z, labels], training=True)
        real_logit = D([real, labels], training=True)
        fake_logit = D([fake, labels], training=True)
        # D 希望：真圖→1（用 0.9 的 label smoothing 避免 D 過度自信），假圖→0
        d_loss = bce(tf.ones_like(real_logit) * 0.9, real_logit) + bce(tf.zeros_like(fake_logit), fake_logit)
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, D.trainable_variables), D.trainable_variables))

    z = tf.random.normal((n, LATENT))
    with tf.GradientTape() as g_tape:
        fake = G([z, labels], training=True)
        fake_logit = D([fake, labels], training=True)
        g_loss = bce(tf.ones_like(fake_logit), fake_logit)    # G 希望 D 把假圖當成真的（→1）
    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, G.trainable_variables), G.trainable_variables))
    return d_loss, g_loss


def save_samples(step, per_class=10):
    """每個類別生成 10 張存成圖，肉眼檢查 GAN 畫得好不好"""
    fixed_z = np.random.RandomState(0).normal(size=(NUM_CLASSES * per_class, LATENT)).astype("float32")
    lab = np.repeat(np.arange(NUM_CLASSES), per_class)[:, None].astype("int32")
    imgs = G.predict([fixed_z, lab], verbose=0)[..., 0] * 0.5 + 0.5    # -1~1 → 0~1
    fig, axes = plt.subplots(NUM_CLASSES, per_class, figsize=(per_class, NUM_CLASSES))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    for c in range(NUM_CLASSES):
        axes[c, 0].text(-4, 24, CLASS_NAMES[c], ha="right", va="center", fontsize=8)
    fig.suptitle(f"cGAN samples @ step {step}"); fig.tight_layout()
    fig.savefig(OUT_DIR / f"03_gan_samples_step{step}.png", dpi=110); plt.close(fig)


# ------------------------------------------------------------
# 1. 訓練 GAN（只用訓練集）
# ------------------------------------------------------------
(x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_dataset()
print("原始訓練集各類數量：", class_counts(y_tr))

x_gan = (x_tr.astype("float32") - 127.5) / 127.5              # 0~255 → -1~1（對應 G 的 tanh 輸出）
idx_by_class = [np.where(y_tr == c)[0] for c in range(NUM_CLASSES)]
rng = np.random.RandomState(SEED)

t0 = time.time()
for step in range(1, GAN_STEPS + 1):
    # 類別平衡取樣：每個樣本的類別是均勻亂數抽的，disgust 也會被充分看到（否則 436 張很快被淹沒）
    labels = rng.randint(0, NUM_CLASSES, BATCH)
    batch_idx = np.array([rng.choice(idx_by_class[c]) for c in labels])
    real = tf.constant(x_gan[batch_idx])
    d_loss, g_loss = train_step(real, tf.constant(labels[:, None].astype("int32")))
    if step % 200 == 0:
        print(f"step {step}/{GAN_STEPS}  D={float(d_loss):.3f}  G={float(g_loss):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if step % 2000 == 0 or step == GAN_STEPS:
        save_samples(step)
G.save(OUT_DIR / "03_generator.keras")

# ------------------------------------------------------------
# 2. 用 G 補足少數類別
# ------------------------------------------------------------
target = max(class_counts(y_tr))
xs, ys = [x_tr], [y_tr]
for c in range(NUM_CLASSES):
    need = target - len(idx_by_class[c])
    if need <= 0:
        continue
    z = np.random.RandomState(100 + c).normal(size=(need, LATENT)).astype("float32")
    lab = np.full((need, 1), c, dtype="int32")
    fake = G.predict([z, lab], batch_size=256, verbose=0)     # -1~1
    xs.append(np.clip(fake * 127.5 + 127.5, 0, 255).astype(np.uint8))   # 轉回 0~255 uint8
    ys.append(np.full(need, c, dtype=np.int64))
x_mix, y_mix = np.concatenate(xs), np.concatenate(ys)
print("真圖 + GAN 假圖 各類數量：", class_counts(y_mix))

# ------------------------------------------------------------
# 3. 訓練分類器（驗證、測試仍是真圖）
# ------------------------------------------------------------
model, history = train_classifier("03_gan", x_mix, y_mix, x_val, y_val, epochs=CLF_EPOCHS)
evaluate("03_gan", model, history, x_te, y_te, class_counts(y_mix))
