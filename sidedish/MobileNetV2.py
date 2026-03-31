# !pip install -q huggingface_hub tensorflow matplotlib pillow scikit-learn seaborn

import os
import zipfile
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from huggingface_hub import hf_hub_download
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version:", tf.__version__)

# 한글 폰트 설정
import subprocess
subprocess.run(["apt-get", "install", "-y", "-q", "fonts-nanum"], capture_output=True)
fm._load_fontmanager(try_read_cache=False)

nanum_path = None
for f in fm.findSystemFonts():
    if "NanumGothic" in f and f.endswith(".ttf") and "Coding" not in f and "Eco" not in f:
        nanum_path = f
        break
if nanum_path is None:
    for f in fm.findSystemFonts():
        if "Nanum" in f and f.endswith(".ttf"):
            nanum_path = f
            break

if nanum_path:
    fm.fontManager.addfont(nanum_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=nanum_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

# 데이터 다운로드 및 압축 해제
EXTRACT_DIR = "/content/kfood_data"
zip_path = hf_hub_download(repo_id="hayul419/smart2", filename="kfood_27class_700.zip", repo_type="dataset")

if os.path.exists(EXTRACT_DIR):
    import shutil
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

# 데이터 경로 탐지
DATA_DIR = None
for candidate in ["/content/kfood_data/kfood_27class_700", "/content/kfood_data/sampled_final", "/content/kfood_data"]:
    if os.path.exists(candidate):
        if len([d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]) >= 2:
            DATA_DIR = candidate
            break

if DATA_DIR is None:
    raise FileNotFoundError("데이터 폴더를 찾지 못했습니다.")

# 하이퍼파라미터
MODEL_DIR        = "/content/kfood_mobilenetv2_output"
IMG_SIZE         = (160, 160)
BATCH_SIZE       = 32
SEED             = 42
INITIAL_EPOCHS   = 12
FINE_TUNE_EPOCHS = 12
INITIAL_LR       = 1e-3
FINE_TUNE_LR     = 1e-5
IMAGE_EXTS       = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# 클래스 목록 및 데이터 수 확인
class_names  = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
num_classes  = len(class_names)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

for cls in class_names:
    n = len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(IMAGE_EXTS)])
    print(f"  {cls}: {n}")

# 클래스별 stratified split (70/15/15)
all_paths, all_labels = [], []
for cls in class_names:
    files = sorted([os.path.join(DATA_DIR, cls, f) for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(IMAGE_EXTS)])
    random.shuffle(files)
    for f in files:
        all_paths.append(f)
        all_labels.append(class_to_idx[cls])

train_paths, train_labels = [], []
val_paths,   val_labels   = [], []
test_paths,  test_labels  = [], []

for cls in class_names:
    idx   = class_to_idx[cls]
    files = [p for p, l in zip(all_paths, all_labels) if l == idx]
    n     = len(files)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))
    test_paths  += files[:n_test];               test_labels  += [idx] * n_test
    val_paths   += files[n_test:n_test + n_val]; val_labels   += [idx] * n_val
    train_paths += files[n_test + n_val:];       train_labels += [idx] * (n - n_test - n_val)

print(f"train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}")

# 데이터셋 구성
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32), label

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds   = make_dataset(val_paths,   val_labels)
test_ds  = make_dataset(test_paths,  test_labels)

# 샘플 이미지 확인
plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i].numpy())])
        plt.axis("off")
plt.tight_layout(); plt.show()

# 데이터 증강
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# 모델 구성
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

inputs  = layers.Input(shape=IMG_SIZE + (3,))
x       = data_augmentation(inputs)
x       = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(INITIAL_LR), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.summary()

# 콜백
best_model_path = os.path.join(MODEL_DIR, "best_mobilenetv2.keras")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ModelCheckpoint(filepath=best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# 1차 학습
history_initial = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=callbacks)
initial_epochs_done = len(history_initial.history["accuracy"])
print(f"1차 학습 완료: {initial_epochs_done} epochs")

# Fine-tuning (상위 30% 레이어 해동)
base_model.trainable = True
for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LR), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs_done + FINE_TUNE_EPOCHS, initial_epoch=initial_epochs_done, callbacks=callbacks)
print(f"Fine-tuning 완료: {len(history_fine.history['accuracy'])} epochs")

# 학습 곡선
acc      = history_initial.history["accuracy"]     + history_fine.history["accuracy"]
val_acc  = history_initial.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss     = history_initial.history["loss"]         + history_fine.history["loss"]
val_loss = history_initial.history["val_loss"]     + history_fine.history["val_loss"]

epochs_range = range(1, len(acc) + 1)
fine_start   = initial_epochs_done + 1

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train Accuracy", color="royalblue")
plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="tomato")
plt.axvline(x=fine_start, color="gray", linestyle="--", linewidth=1.2, label="Fine-tune 시작")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Train vs Validation Accuracy")
plt.ylim(0, 1); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss", color="royalblue")
plt.plot(epochs_range, val_loss, label="Validation Loss", color="tomato")
plt.axvline(x=fine_start, color="gray", linestyle="--", linewidth=1.2, label="Fine-tune 시작")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Validation Loss")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 평가
best_model = tf.keras.models.load_model(best_model_path)
_, val_acc_eval  = best_model.evaluate(val_ds,  verbose=1)
_, test_acc_eval = best_model.evaluate(test_ds, verbose=1)
print(f"Validation Accuracy : {val_acc_eval:.4f}")
print(f"Test Accuracy       : {test_acc_eval:.4f}")

# Classification report
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()

# 모델 저장
best_model.save(os.path.join(MODEL_DIR, "kfood_mobilenetv2.keras"))
print("Keras 모델 저장 완료")

with open(os.path.join(MODEL_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(class_names) + "\n")
print("labels.txt 저장 완료")

# TFLite 변환
tflite_path = os.path.join(MODEL_DIR, "kfood_mobilenetv2.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
with open(tflite_path, "wb") as f:
    f.write(converter.convert())
print("TFLite 저장 완료")

tflite_fp16_path = os.path.join(MODEL_DIR, "kfood_mobilenetv2_fp16.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with open(tflite_fp16_path, "wb") as f:
    f.write(converter.convert())
print("Float16 TFLite 저장 완료")

# 저장 파일 목록
for fname in sorted(os.listdir(MODEL_DIR)):
    fpath = os.path.join(MODEL_DIR, fname)
    print(f"  {fname} - {os.path.getsize(fpath) / (1024*1024):.2f} MB")

# fp16 다운로드
from google.colab import files
files.download(tflite_fp16_path)

# 단일 이미지 예측
from tensorflow.keras.utils import load_img, img_to_array
uploaded = files.upload()
img = load_img(list(uploaded.keys())[0], target_size=IMG_SIZE)
x   = np.expand_dims(img_to_array(img), axis=0)
pred = best_model.predict(x, verbose=0)[0]
print(f"\n예측 결과: {class_names[np.argmax(pred)]}")
print(f"신뢰도   : {float(pred[np.argmax(pred)]):.4f}")
print("\nTop-5 예측:")
for idx in np.argsort(pred)[::-1][:5]:
    print(f"  {class_names[idx]}: {pred[idx]:.4f}")