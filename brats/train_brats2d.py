import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.spatial.distance import directed_hausdorff

MODEL_BASE = "/content/drive/MyDrive/Bachelor/Models/unet_edema"
RESULTS_BASE = "/content/drive/MyDrive/Bachelor/Results/unet_edema"


def set_low_power():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def ensure_dirs():
    os.makedirs(MODEL_BASE, exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/histories", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/metrics", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/plots/dice", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/plots/loss", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/configs", exist_ok=True)
    os.makedirs(f"{RESULTS_BASE}/predictions", exist_ok=True)


def activation_layer(x, activation):
    act = activation.lower()

    if act == "relu":
        return layers.ReLU()(x)

    if act == "leakyrelu":
        return layers.LeakyReLU(negative_slope=0.1)(x)

    if act == "elu":
        return layers.ELU()(x)

    if act == "gelu":
        return layers.Activation("gelu")(x)

    if act in ["silu", "swish"]:
        return layers.Activation("swish")(x)

    if act == "mish":
        return layers.Lambda(
            lambda t: t * tf.math.tanh(tf.math.softplus(t))
        )(x)

    return layers.ReLU()(x)


def conv_block(x, filters, activation):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation)

    return x


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


def precision_metric(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    return (tp + eps) / (tp + fp + eps)


def specificity_metric(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    return (tn + eps) / (tn + fp + eps)


def iou_metric(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter

    return (inter + eps) / (union + eps)


def clean_mask_to_edema(mask, edema_labels=1):
    if isinstance(edema_labels, (list, tuple, set)):
        edema_labels = list(edema_labels)
        cleaned = np.isin(mask, edema_labels).astype(np.float32)
    else:
        cleaned = (mask == edema_labels).astype(np.float32)

    return cleaned


def ensure_channel_dim(arr):
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)
    elif arr.ndim == 4 and arr.shape[-1] != 1:
        raise ValueError(f"Expected last channel = 1, got shape {arr.shape}")
    return arr


def validate_image_mask_alignment(img_files, msk_dir):
    missing_masks = []
    for f in img_files:
        mask_path = os.path.join(msk_dir, f)
        if not os.path.exists(mask_path):
            missing_masks.append(f)

    if missing_masks:
        raise FileNotFoundError(
            "The following mask files are missing:\n" + "\n".join(missing_masks[:20])
        )


def load_split(split_dir, edema_labels=1):
    img_dir = os.path.join(split_dir, "images")
    msk_dir = os.path.join(split_dir, "masks")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not os.path.isdir(msk_dir):
        raise FileNotFoundError(f"Masks directory not found: {msk_dir}")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
    if not img_files:
        raise ValueError(f"No .npy image files found in: {img_dir}")

    validate_image_mask_alignment(img_files, msk_dir)

    X_list = []
    Y_list = []

    global_raw_values = set()
    global_clean_values = set()

    positive_pixel_total = 0
    total_pixel_total = 0
    empty_mask_count = 0

    for f in img_files:
        img_path = os.path.join(img_dir, f)
        msk_path = os.path.join(msk_dir, f)

        img = np.load(img_path).astype(np.float32)
        raw_mask = np.load(msk_path)

        raw_unique = np.unique(raw_mask)
        cleaned_mask = clean_mask_to_edema(raw_mask, edema_labels=edema_labels)
        clean_unique = np.unique(cleaned_mask)

        global_raw_values.update(raw_unique.tolist())
        global_clean_values.update(clean_unique.tolist())

        positive_pixels = int(np.sum(cleaned_mask))
        total_pixels = int(cleaned_mask.size)
        total_pixel_total += total_pixels
        positive_pixel_total += positive_pixels

        if positive_pixels == 0:
            empty_mask_count += 1

        X_list.append(img)
        Y_list.append(cleaned_mask)

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)

    X = ensure_channel_dim(X)
    Y = ensure_channel_dim(Y)

    split_name = os.path.basename(split_dir.rstrip("/"))
    overall_ratio = positive_pixel_total / max(total_pixel_total, 1)

    print(f"\n[{split_name.upper()}] Loaded {len(img_files)} samples")
    print(f"[{split_name.upper()}] X shape: {X.shape}")
    print(f"[{split_name.upper()}] Y shape: {Y.shape}")
    print(f"[{split_name.upper()}] Raw mask labels found: {sorted(global_raw_values)}")
    print(f"[{split_name.upper()}] Clean mask labels kept: {sorted(global_clean_values)}")
    print(f"[{split_name.upper()}] Empty edema masks: {empty_mask_count}/{len(img_files)}")
    print(f"[{split_name.upper()}] Overall edema pixel ratio: {overall_ratio:.6f}")

    return X, Y


def np_binarize(arr, threshold=0.5):
    return (arr > threshold).astype(np.uint8)


def np_dice(y_true, y_pred, eps=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    inter = np.sum(y_true * y_pred)
    denom = np.sum(y_true) + np.sum(y_pred)

    return float((2.0 * inter + eps) / (denom + eps))


def np_precision(y_true, y_pred, eps=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    tp = np.sum(y_true * y_pred)
    fp = np.sum((1.0 - y_true) * y_pred)

    return float((tp + eps) / (tp + fp + eps))


def np_specificity(y_true, y_pred, eps=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    tn = np.sum((1.0 - y_true) * (1.0 - y_pred))
    fp = np.sum((1.0 - y_true) * y_pred)

    return float((tn + eps) / (tn + fp + eps))


def np_iou(y_true, y_pred, eps=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter

    return float((inter + eps) / (union + eps))


def np_hausdorff(y_true, y_pred):
    """
    Symmetric Hausdorff distance on foreground pixel coordinates.
    Uses image diagonal when one mask is empty and the other is not.
    Returns 0 when both masks are empty.
    """
    y_true_pts = np.argwhere(y_true > 0)
    y_pred_pts = np.argwhere(y_pred > 0)

    if len(y_true_pts) == 0 and len(y_pred_pts) == 0:
        return 0.0

    if len(y_true_pts) == 0 or len(y_pred_pts) == 0:
        h, w = y_true.shape[:2]
        return float(np.sqrt(h ** 2 + w ** 2))

    hd_forward = directed_hausdorff(y_true_pts, y_pred_pts)[0]
    hd_backward = directed_hausdorff(y_pred_pts, y_true_pts)[0]

    return float(max(hd_forward, hd_backward))


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true_bin = np_binarize(y_true, threshold=0.5)
    y_pred_bin = np_binarize(y_prob, threshold=threshold)

    rows = []

    for i in range(len(y_true_bin)):
        gt = np.squeeze(y_true_bin[i])
        pr = np.squeeze(y_pred_bin[i])

        row = {
            "sample_idx": i,
            "dice": np_dice(gt, pr),
            "precision": np_precision(gt, pr),
            "specificity": np_specificity(gt, pr),
            "iou": np_iou(gt, pr),
            "hausdorff": np_hausdorff(gt, pr),
            "gt_positive_pixels": int(np.sum(gt)),
            "pred_positive_pixels": int(np.sum(pr)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    summary = {
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std(ddof=0)),
        "precision_mean": float(df["precision"].mean()),
        "precision_std": float(df["precision"].std(ddof=0)),
        "specificity_mean": float(df["specificity"].mean()),
        "specificity_std": float(df["specificity"].std(ddof=0)),
        "iou_mean": float(df["iou"].mean()),
        "iou_std": float(df["iou"].std(ddof=0)),
        "hausdorff_mean": float(df["hausdorff"].mean()),
        "hausdorff_std": float(df["hausdorff"].std(ddof=0)),
        "hausdorff_max": float(df["hausdorff"].max()),
    }

    return df, summary


def save_history(history, activation, img_size, timestamp):
    hist_df = pd.DataFrame(history.history)
    hist_path = (
        f"{RESULTS_BASE}/histories/"
        f"history_unet_edema_{activation}_{img_size}_{timestamp}.csv"
    )
    hist_df.to_csv(hist_path, index=False)


def save_plots(history, activation, img_size, timestamp):
    loss_plot_path = (
        f"{RESULTS_BASE}/plots/loss/"
        f"loss_unet_edema_{activation}_{img_size}_{timestamp}.png"
    )
    dice_plot_path = (
        f"{RESULTS_BASE}/plots/dice/"
        f"dice_unet_edema_{activation}_{img_size}_{timestamp}.png"
    )

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {activation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure()
    plt.plot(history.history["dice_coef"], label="train_dice")
    plt.plot(history.history["val_dice_coef"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"Dice - {activation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_plot_path)
    plt.close()


def save_evaluation_metrics(
    val_sample_df,
    eval_summary,
    activation,
    img_size,
    timestamp,
):
    sample_metrics_path = (
        f"{RESULTS_BASE}/metrics/"
        f"val_metrics_unet_edema_{activation}_{img_size}_{timestamp}.csv"
    )

    run_eval_path = (
        f"{RESULTS_BASE}/metrics/"
        f"eval_summary_unet_edema_{activation}_{img_size}_{timestamp}.csv"
    )

    val_sample_df.to_csv(sample_metrics_path, index=False)
    pd.DataFrame([eval_summary]).to_csv(run_eval_path, index=False)


def save_summary(
    history,
    activation,
    img_size,
    epochs,
    batch_size,
    elapsed_sec,
    timestamp,
    edema_labels,
    eval_summary,
):
    cumulative_summary_path = (
        f"{RESULTS_BASE}/metrics/summary_unet_edema_experiments.csv"
    )
    latest_summary_path = (
        f"{RESULTS_BASE}/metrics/summary_unet_edema_latest.csv"
    )
    run_summary_path = (
        f"{RESULTS_BASE}/metrics/"
        f"summary_unet_edema_{activation}_{img_size}_{timestamp}.csv"
    )

    if isinstance(edema_labels, (list, tuple, set)):
        edema_labels_str = ",".join(map(str, edema_labels))
    else:
        edema_labels_str = str(edema_labels)

    row = {
        "timestamp": timestamp,
        "activation": activation,
        "img_size": img_size,
        "edema_labels_kept": edema_labels_str,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history["loss"]),
        "batch_size": batch_size,
        "best_val_dice": float(np.max(history.history["val_dice_coef"])),
        "best_epoch": int(np.argmax(history.history["val_dice_coef"]) + 1),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_dice": float(history.history["dice_coef"][-1]),
        "final_val_dice": float(history.history["val_dice_coef"][-1]),
        "final_train_precision": float(history.history["precision_metric"][-1]),
        "final_val_precision": float(history.history["val_precision_metric"][-1]),
        "final_train_specificity": float(history.history["specificity_metric"][-1]),
        "final_val_specificity": float(history.history["val_specificity_metric"][-1]),
        "final_train_iou": float(history.history["iou_metric"][-1]),
        "final_val_iou": float(history.history["val_iou_metric"][-1]),
        "eval_val_dice_mean": eval_summary["dice_mean"],
        "eval_val_precision_mean": eval_summary["precision_mean"],
        "eval_val_specificity_mean": eval_summary["specificity_mean"],
        "eval_val_iou_mean": eval_summary["iou_mean"],
        "eval_val_hausdorff_mean": eval_summary["hausdorff_mean"],
        "eval_val_hausdorff_max": eval_summary["hausdorff_max"],
        "training_time_sec": float(elapsed_sec),
    }

    pd.DataFrame([row]).to_csv(run_summary_path, index=False)

    if os.path.exists(cumulative_summary_path):
        df_all = pd.read_csv(cumulative_summary_path)
        df_all = pd.concat([df_all, pd.DataFrame([row])], ignore_index=True)
    else:
        df_all = pd.DataFrame([row])

    df_all.to_csv(cumulative_summary_path, index=False)

    if os.path.exists(latest_summary_path):
        df_latest = pd.read_csv(latest_summary_path)
        df_latest = df_latest[
            ~(
                (df_latest["activation"] == activation) &
                (df_latest["img_size"] == img_size)
            )
        ]
        df_latest = pd.concat([df_latest, pd.DataFrame([row])], ignore_index=True)
    else:
        df_latest = pd.DataFrame([row])

    df_latest = df_latest.sort_values(
        by=["activation", "img_size", "timestamp"]
    ).reset_index(drop=True)

    df_latest.to_csv(latest_summary_path, index=False)


def save_config(data_dir, img_size, activation, epochs, batch_size, timestamp, edema_labels):
    config = {
        "timestamp": timestamp,
        "data_dir": data_dir,
        "img_size": img_size,
        "activation": activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "edema_labels_kept": edema_labels if isinstance(edema_labels, list) else [edema_labels],
        "model_dir": MODEL_BASE,
        "results_dir": RESULTS_BASE,
    }

    config_path = (
        f"{RESULTS_BASE}/configs/"
        f"config_unet_edema_{activation}_{img_size}_{timestamp}.json"
    )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def parse_edema_labels(edema_labels):
    if isinstance(edema_labels, int):
        return edema_labels

    if isinstance(edema_labels, str):
        edema_labels = edema_labels.strip()
        if "," in edema_labels:
            labels = [int(x.strip()) for x in edema_labels.split(",") if x.strip() != ""]
            return labels
        return int(edema_labels)

    if isinstance(edema_labels, (list, tuple, set)):
        return [int(x) for x in edema_labels]

    raise ValueError(f"Unsupported edema_labels format: {edema_labels}")


def main(
    data_dir="brats2d",
    img_size=128,
    activation="relu",
    epochs=5,
    batch_size=1,
    edema_labels=1,
):
    set_low_power()
    ensure_dirs()

    edema_labels = parse_edema_labels(edema_labels)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    Xtr, Ytr = load_split(
        os.path.join(data_dir, "train"),
        edema_labels=edema_labels,
    )
    Xva, Yva = load_split(
        os.path.join(data_dir, "val"),
        edema_labels=edema_labels,
    )

    print("\nLoaded:")
    print("Train:", Xtr.shape, Ytr.shape)
    print("Val  :", Xva.shape, Yva.shape)

    model = build_unet(img_size=img_size, activation=activation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            dice_coef,
            precision_metric,
            specificity_metric,
            iou_metric,
        ],
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{MODEL_BASE}/unet_edema_{activation}_{img_size}_{timestamp}.keras",
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

    print("\nRunning validation prediction for saved metrics...")
    y_val_prob = model.predict(Xva, batch_size=batch_size, verbose=1)

    val_sample_df, eval_summary = evaluate_predictions(
        y_true=Yva,
        y_prob=y_val_prob,
        threshold=0.5,
    )

    print("\nValidation evaluation summary:")
    for k, v in eval_summary.items():
        print(f"{k}: {v:.6f}")

    save_history(history, activation, img_size, timestamp)
    save_plots(history, activation, img_size, timestamp)
    save_evaluation_metrics(
        val_sample_df=val_sample_df,
        eval_summary=eval_summary,
        activation=activation,
        img_size=img_size,
        timestamp=timestamp,
    )
    save_summary(
        history=history,
        activation=activation,
        img_size=img_size,
        epochs=epochs,
        batch_size=batch_size,
        elapsed_sec=elapsed_sec,
        timestamp=timestamp,
        edema_labels=edema_labels,
        eval_summary=eval_summary,
    )
    save_config(
        data_dir=data_dir,
        img_size=img_size,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        timestamp=timestamp,
        edema_labels=edema_labels,
    )

    print(f"\nSaved model, history, plots, evaluation metrics, summaries, and config for {activation}.")
    print(f"Run timestamp: {timestamp}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="brats2d")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--edema_labels", default="1")

    args = ap.parse_args()
    main(**vars(args))