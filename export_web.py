"""
把 outputs/*.keras 的權重匯出給網頁（TensorFlow.js）使用。
- 直接讀 .keras（zip）裡的 model.weights.h5，不需要安裝 TensorFlow
- 把 BatchNormalization 合併進前一層卷積（推論時數學上等價），網頁端模型更小更快
執行：python export_web.py   （需要 h5py、numpy）
"""
import io, json, zipfile
from pathlib import Path
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "docs" / "models"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-3                                   # Keras BatchNormalization 預設 epsilon
NAMES = ["01_baseline", "02_augmentation", "03_gan"]


def name(prefix, i):
    return prefix if i == 0 else f"{prefix}_{i}"


def export(model):
    z = zipfile.ZipFile(HERE / "outputs" / f"{model}.keras")
    f = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
    tensors = []                             # 依順序：conv1 w,b, conv2 w,b ..., dense w,b, dense w,b
    for i in range(6):
        w = f[f"layers/{name('conv2d', i)}/vars/0"][()]
        g, b, m, v = (f[f"layers/{name('batch_normalization', i)}/vars/{k}"][()] for k in range(4))
        s = g / np.sqrt(v + EPS)
        tensors += [w * s, b - m * s]        # BN 合併：w' = w*s，bias' = beta - mean*s
    for n in ("dense", "dense_1"):
        tensors += [f[f"layers/{n}/vars/0"][()], f[f"layers/{n}/vars/1"][()]]
    tensors = [t.astype("float32") for t in tensors]
    (OUT / f"{model}.bin").write_bytes(b"".join(t.tobytes() for t in tensors))
    return [list(t.shape) for t in tensors]


shapes = None
for n in NAMES:
    shapes = export(n)
    print("exported", n)
(OUT / "manifest.json").write_text(json.dumps({"shapes": shapes, "models": NAMES}))
