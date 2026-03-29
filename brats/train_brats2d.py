import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_BASE = "/content/drive/MyDrive/Bachelor/Models/unet_edema"
RESULTS_BASE = "/content/drive/MyDrive/Bachelor/Results/unet_edema"

def set_low_power():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

def ensure_dirs():
    os.makedirs(MODEL_BASE, exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/histories", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/metrics", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/plots", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/configs", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/predictions", exist_ok=True)

def conv_block(x, filters, activation):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation)
    return x

def activation_layer(x, activation):
    act = activation.lower()
    if act == "relu":
        return layers.ReLU()(x)
    if act == "leakyrelu":
        return layers.LeakyReLU(0.1)(x)
    if act == "elu":
        return layers.ELU()(x)
    if act == "gelu":
        return tf.keras.activations.gelu(x)
    if act in ["silu", "swish"]:
        return tf.keras.activations.swish(x)
    if act == "mish":
        return x * tf.math.tanh(tf.math.softplus(x))
    return layers.ReLU()(x)

def build_unet(img_size=128, activation="relu"):
    inputs = layers.Input((img_size, img_size, 1))

    c1 = conv_block(inputs, 16, activation)
    p1 = layers.MaxPool2D()(c1)

    c2 = conv_block(p1, 32, activation)
    p2 = layers.MaxPool2D()(c2)

    c3 = conv_block(p2, 64, activation)
    p3 = layers.MaxPool2D()(c3)

    c4 = conv_block(p3, 128, activation)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = conv_block(u5, 64, activation)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = conv_block(u6, 32, activation)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = conv_block(u7, 16, activation)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)
    return models.Model(inputs=inputs, outputs=outputs)

def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * inter + eps) / (denom + eps)

def load_split(split_dir):
    img_dir = os.path.join(split_dir, "images")
    msk_dir = os.path.join(split_dir, "masks")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    X = np.stack([np.load(os.path.join(img_dir, f)) for f in img_files], axis=0).astype(np.float32)
    Y = np.stack([np.load(os.path.join(msk_dir, f)) for f in img_files], axis=0).astype(np.float32)
    return X, Y

def save_history(history, activation, img_size):
    hist_df = pd.DataFrame(history.history)
    hist_path = f"{RESULTS_BASE}/histories/history_{activation}_{img_size}.csv"
    hist_df.to_csv(hist_path, index=False)

def save_plots(history, activation, img_size):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {activation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_BASE}/plots/loss_{activation}_{img_size}.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["dice_coef"], label="train_dice")
    plt.plot(history.history["val_dice_coef"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"Dice - {activation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_BASE}/plots/dice_{activation}_{img_size}.png")
    plt.close()

def save_summary(history, activation, img_size, epochs, batch_size, elapsed_sec):
    summary_path = f"{RESULTS_BASE}/metrics/summary.csv"

    row = {
        "activation": activation,
        "img_size": img_size,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]),
        "batch_size": batch_size,
        "best_val_dice": float(np.max(history.history["val_dice_coef"])),
        "best_epoch": int(np.argmax(history.history["val_dice_coef"]) + 1),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_dice": float(history.history["dice_coef"][-1]),
        "final_val_dice": float(history.history["val_dice_coef"][-1]),
        "training_time_sec": float(elapsed_sec),
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_path, index=False)

def save_config(data_dir, img_size, activation, epochs, batch_size):
    config = {
        "data_dir": data_dir,
        "img_size": img_size,
        "activation": activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_dir": MODEL_BASE,
        "results_dir": RESULTS_BASE,
    }
    with open(f"{RESULTS_BASE}/configs/config_{activation}_{img_size}.json", "w") as f:
        json.dump(config, f, indent=2)

def main(data_dir="brats2d", img_size=128, activation="relu", epochs=5, batch_size=1):
    set_low_power()
    ensure_dirs()

    Xtr, Ytr = load_split(os.path.join(data_dir, "train"))
    Xva, Yva = load_split(os.path.join(data_dir, "val"))

    print("Loaded:", Xtr.shape, Ytr.shape, Xva.shape, Yva.shape)

    model = build_unet(img_size=img_size, activation=activation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[dice_coef],
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{MODEL_BASE}/unet_edema_{activation}_{img_size}.keras",
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )

    start_time = time.time()

    history = model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, es],
        verbose=1,
    )

    elapsed_sec = time.time() - start_time

    save_history(history, activation, img_size)
    save_plots(history, activation, img_size)
    save_summary(history, activation, img_size, epochs, batch_size, elapsed_sec)
    save_config(data_dir, img_size, activation, epochs, batch_size)

    print(f"Saved history, plots, summary, and config for {activation}.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="brats2d")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    main(**vars(args))