"""
Real vs AI-generated (fake) face classifier
Binary classification with a small CNN (2 Conv layers) using Keras.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 15
DATA = "data"

# ---------- 1. Load data ----------
def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        f"{DATA}/{split}",
        labels="inferred",
        label_mode="binary",       # fake=0, real=1 (alphabetical order)
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=42,
    )

train_ds = load("train", True)
val_ds = load("valid", False)
test_ds = load("test", False)
class_names = train_ds.class_names   # save now: prefetch() removes this attribute
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ---------- 2. Model (only 2 Conv layers) ----------
model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1.0 / 255),
    layers.RandomFlip("horizontal"),          # light augmentation

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),    # binary output
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- 3. Train ----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop])

model.save("fake_face_cnn.keras")   # saved right after training

# ---------- 4. Evaluate ----------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.3f}")

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

# Confusion matrix
y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel()
y_pred = (model.predict(test_ds).ravel() > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.savefig("confusion_matrix.png"); plt.show()
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------- 5. Predict on a new image ----------

def predict_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    p = float(model.predict(arr)[0][0])
    label = class_names[int(p > 0.5)]
    print(f"{path}: {label} (score={p:.2f})")

predict_image("00276TOPP4.jpg")