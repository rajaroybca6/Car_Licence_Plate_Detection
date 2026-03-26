"""
train_ocr.py — Train the character OCR model
=============================================
Steps:
  1. Generates synthetic character images  (or loads your own dataset)
  2. Trains a CNN classifier
  3. Exports a frozen TF protobuf (.pb) ready for OCR class

Usage:
    python train_ocr.py
    python train_ocr.py --data-dir my_chars/ --epochs 30
"""

import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 36 classes
IMG_SIZE = 128       # model input size


# ------------------------------------------------------------------
# 1.  Synthetic data generator
# ------------------------------------------------------------------

def generate_synthetic_data(output_dir: str,
                             samples_per_char: int = 500,
                             img_size: int = IMG_SIZE):
    """
    Render each character in CHARS with random fonts / sizes / positions
    and save as PNG images under output_dir/<CHAR>/<CHAR>_N.png
    """
    print(f"[DATA] Generating synthetic data → {output_dir}")

    # Try to load system fonts; fall back to PIL default
    candidate_fonts = [
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf", "cour.ttf", "times.ttf",
    ]
    available_fonts = [f for f in candidate_fonts if os.path.exists(f)]

    for char in CHARS:
        char_dir = os.path.join(output_dir, char)
        os.makedirs(char_dir, exist_ok=True)

        for i in range(samples_per_char):
            bg_val = random.randint(180, 255)
            img = Image.new("RGB", (img_size, img_size),
                            color=(bg_val, bg_val, bg_val))
            draw = ImageDraw.Draw(img)

            font_size = random.randint(70, 100)
            if available_fonts:
                try:
                    font = ImageFont.truetype(
                        random.choice(available_fonts), font_size
                    )
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()

            # Random position with slight noise
            x = random.randint(8, 25)
            y = random.randint(8, 25)
            fg_val = random.randint(0, 60)
            draw.text((x, y), char, fill=(fg_val, fg_val, fg_val), font=font)

            # Optional: add mild rotation
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, fillcolor=(bg_val, bg_val, bg_val))

            img.save(os.path.join(char_dir, f"{char}_{i:04d}.png"))

    print(f"[DATA] Done. Generated {len(CHARS) * samples_per_char} images.")


# ------------------------------------------------------------------
# 2.  Dataset loader
# ------------------------------------------------------------------

def load_dataset(data_dir: str, img_size: int = IMG_SIZE):
    """
    Load images from data_dir/<LABEL>/*.png|jpg
    Returns (X, y) numpy arrays.
    """
    X, y = [], []
    label_to_idx = {c: i for i, c in enumerate(CHARS)}

    for char in CHARS:
        char_dir = os.path.join(data_dir, char)
        if not os.path.isdir(char_dir):
            continue
        for fname in os.listdir(char_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(char_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size),
                             interpolation=cv2.INTER_CUBIC)
            img = img.astype("float32") / 255.0
            X.append(img)
            y.append(label_to_idx[char])

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    print(f"[DATA] Loaded {len(X)} images, {len(CHARS)} classes.")
    return X, y


# ------------------------------------------------------------------
# 3.  Model definition
# ------------------------------------------------------------------

def build_model(num_classes: int = 36, img_size: int = IMG_SIZE):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu",
                      input_shape=(img_size, img_size, 3), name="input"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Head
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name="final_result"),
    ])
    return model


# ------------------------------------------------------------------
# 4.  Freeze and export
# ------------------------------------------------------------------

def freeze_and_save(keras_model, output_pb: str):
    """
    Convert a Keras model to a frozen TF1-compatible .pb graph.
    The OCR class expects input node  'import/input'
    and output node 'import/final_result'.
    """
    saved_dir = "saved_model_tmp"
    keras_model.save(saved_dir)

    # Reload as TF2 SavedModel and freeze
    loaded = tf.saved_model.load(saved_dir)
    infer  = loaded.signatures["serving_default"]

    frozen_func = convert_variables_to_constants_v2(infer)
    frozen_graph = frozen_func.graph.as_graph_def()

    os.makedirs(os.path.dirname(output_pb), exist_ok=True)
    with open(output_pb, "wb") as f:
        f.write(frozen_graph.SerializeToString())

    print(f"[MODEL] Frozen graph saved → {output_pb}")

    # Clean up temp dir
    import shutil
    shutil.rmtree(saved_dir, ignore_errors=True)


def save_labels(output_txt: str):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as f:
        for c in CHARS:
            f.write(c + "\n")
    print(f"[MODEL] Labels saved → {output_txt}")


# ------------------------------------------------------------------
# 5.  CLI / entry point
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train OCR character classifier")
    p.add_argument("--data-dir",   default="training_data",
                   help="Root folder with per-class sub-dirs (default: training_data)")
    p.add_argument("--generate",   action="store_true",
                   help="Generate synthetic training data before training")
    p.add_argument("--samples",    type=int, default=500,
                   help="Synthetic samples per character (default: 500)")
    p.add_argument("--epochs",     type=int, default=25,
                   help="Training epochs (default: 25)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size (default: 64)")
    p.add_argument("--model-out",  default="model/binary_128_0.50_ver3.pb",
                   help="Output frozen model path")
    p.add_argument("--labels-out", default="model/binary_128_0.50_labels_ver2.txt",
                   help="Output labels path")
    return p.parse_args()


def main():
    args = parse_args()

    # --- (optional) generate synthetic data ---
    if args.generate:
        generate_synthetic_data(args.data_dir, samples_per_char=args.samples)

    # --- load dataset ---
    X, y = load_dataset(args.data_dir)
    if len(X) == 0:
        print("[ERROR] No images found. Run with --generate or provide your own dataset.")
        return

    # One-hot encode labels
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CHARS))

    # Train / val split (90 / 10)
    split = int(len(X) * 0.9)
    idx   = np.random.permutation(len(X))
    X_train, y_train = X[idx[:split]], y_cat[idx[:split]]
    X_val,   y_val   = X[idx[split:]], y_cat[idx[split:]]

    print(f"[TRAIN] train={len(X_train)}  val={len(X_val)}")

    # --- build + compile model ---
    model = build_model(num_classes=len(CHARS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # --- callbacks ---
    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=1),
        callbacks.ModelCheckpoint("best_ocr_model.keras",
                                  monitor="val_accuracy",
                                  save_best_only=True, verbose=1),
    ]

    # --- train ---
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1
    )

    # --- export ---
    freeze_and_save(model, args.model_out)
    save_labels(args.labels_out)
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()