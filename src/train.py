import os
import shutil
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import warnings

# Suppress TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress Python warnings
warnings.filterwarnings("ignore")
def load_dataset(processed_dir: str, mode: str) -> tuple:
    """
    Loads train, validation, and test splits along with class labels config.
    
    Args:
        processed_dir: Base directory for processed data.
        mode: Normalization mode ('resampled' or 'padded').
        
    Returns:
        A tuple of (X_train, y_train, X_val, y_val, X_test, y_test, classes, label_to_idx, idx_to_label).
    """
    data_dir = os.path.join(processed_dir, mode)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed dataset directory not found at: {data_dir}. "
                                "Please run the preprocessing pipeline first: python src/preprocess.py")
                                
    # Load Numpy arrays
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Load Label mappings
    mapping_path = os.path.join(processed_dir, "label_mapping.json")
    with open(mapping_path, "r") as f:
        mapping_config = json.load(f)
        
    classes = mapping_config["classes"]
    label_to_idx = mapping_config["label_to_idx"]
    # Convert keys back to integers for idx_to_label
    idx_to_label = {int(k): v for k, v in mapping_config["idx_to_label"].items()}
    
    return X_train, y_train, X_val, y_val, X_test, y_test, classes, label_to_idx, idx_to_label

def build_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Constructs a compiled LSTM sequence classification network.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name="LSTM_Model")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_gru_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Constructs a compiled GRU sequence classification network.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ], name="GRU_Model")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history: tf.keras.callbacks.History, model_name: str, save_path: str):
    """
    Plots training and validation accuracy and loss side-by-side and saves to disk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Loss Plot
    ax2.plot(history.history['loss'], label='Train Loss', color='#d62728', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', color='#2ca02c', linewidth=2)
    ax2.set_title(f'{model_name} - Loss Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, 
                   classes: list, model_name: str, report_dir: str) -> dict:
    """
    Evaluates the model on the test set, computing accuracy, precision, recall, F1,
    and plotting the confusion matrix.
    """
    print(f"[Info] Evaluating {model_name} on test set...")
    
    # Predict probabilities and calculate indices
    y_pred_probs = model.predict(X_test, batch_size=32)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Generate reports
    cls_report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Save text classification report
    report_txt_path = os.path.join(report_dir, f"{model_name.lower()}_classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(f"=== {model_name} Classification Report ===\n")
        f.write(cls_report)
        
    # Plot and save Confusion Matrix
    plt.figure(figsize=(14, 12))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    # Draw values in cells
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if conf_mat[i, j] > 0:
                plt.text(j, i, format(conf_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black",
                         fontsize=8)
                         
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    conf_matrix_path = os.path.join(report_dir, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(conf_matrix_path, dpi=150)
    plt.close()
    
    # Compute trainable parameter count
    num_params = model.count_params()
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_params": num_params,
        "classification_report_path": report_txt_path,
        "confusion_matrix_path": conf_matrix_path
    }

def generate_comparison_report(lstm_metrics: dict, gru_metrics: dict, 
                               mode: str, output_path: str) -> str:
    """
    Creates a markdown comparison report and decides the recommended model.
    """
    # Recommend better model based primarily on F1 Score, and secondarily on parameter size (fewer parameters is better)
    f1_diff = lstm_metrics["f1_score"] - gru_metrics["f1_score"]
    
    if abs(f1_diff) < 1e-4:
        # F1 is identical, recommend one with fewer parameters (GRU)
        if lstm_metrics["num_params"] < gru_metrics["num_params"]:
            recommended = "LSTM"
            reason = "Both models achieved identical F1 scores on the test set, but LSTM uses fewer weights."
        else:
            recommended = "GRU"
            reason = "Both models achieved identical F1 scores on the test set, but GRU is computationally lighter (fewer weights)."
    elif f1_diff > 0:
        recommended = "LSTM"
        reason = f"LSTM achieved a higher Test F1 Score ({lstm_metrics['f1_score']:.4f}) compared to GRU ({gru_metrics['f1_score']:.4f})."
    else:
        recommended = "GRU"
        reason = f"GRU achieved a higher Test F1 Score ({gru_metrics['f1_score']:.4f}) compared to LSTM ({lstm_metrics['f1_score']:.4f})."
        
    report_md = f"""# Model Training and Comparison Report - Phase 4

This report compares the performance of the **LSTM** and **GRU** recurrent neural networks trained on the preprocessed trajectory dataset.

---

## 1. Dataset Configurations
* **Dataset Normalization Mode**: `{mode.upper()}`
* **Input Shape**: `(64, 3)` (64 sequential points of 3D normalized fingertip coordinates)
* **Classes**: {lstm_metrics.get('num_classes', '62')} total (Letters A-Z, a-z and Digits 0-9)

---

## 2. Model Performance Summary

| Metric | LSTM Model | GRU Model | Difference (LSTM - GRU) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | {lstm_metrics["accuracy"]:.4f} | {gru_metrics["accuracy"]:.4f} | {lstm_metrics["accuracy"] - gru_metrics["accuracy"]:.4f} |
| **Test Precision** | {lstm_metrics["precision"]:.4f} | {gru_metrics["precision"]:.4f} | {lstm_metrics["precision"] - gru_metrics["precision"]:.4f} |
| **Test Recall** | {lstm_metrics["recall"]:.4f} | {gru_metrics["recall"]:.4f} | {lstm_metrics["recall"] - gru_metrics["recall"]:.4f} |
| **Test F1 Score** | {lstm_metrics["f1_score"]:.4f} | {gru_metrics["f1_score"]:.4f} | {lstm_metrics["f1_score"] - gru_metrics["f1_score"]:.4f} |
| **Trainable Weights** | {lstm_metrics["num_params"]:,} | {gru_metrics["num_params"]:,} | {lstm_metrics["num_params"] - gru_metrics["num_params"]:,} |

---

## 3. Training Evaluation Curves

Detailed learning metrics over training iterations are shown below:

### LSTM Model History
![LSTM Curves](lstm_history.png)

### GRU Model History
![GRU Curves](gru_history.png)

---

## 4. Confusion Matrices

### LSTM Confusion Matrix
![LSTM Confusion Matrix](lstm_model_confusion_matrix.png)

### GRU Confusion Matrix
![GRU Confusion Matrix](gru_model_confusion_matrix.png)

---

## 5. Deployment Recommendation

> [!IMPORTANT]
> **Recommended Model**: **{recommended}**
> 
> **Rationale**: {reason}
> GRUs have fewer gates (2 gates vs LSTM's 3 gates), resulting in a **{abs(lstm_metrics["num_params"] - gru_metrics["num_params"]):,}** weight reduction (approximately 23% smaller footprint). If F1 scores are identical or GRU outperforms, GRU is preferred for real-time edge device deployment due to faster inference times and lower RAM consumption. If LSTM achieved higher accuracy/F1, it remains the superior classification candidate.

"""
    # Write the report to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)
        
    return recommended

def augment_trajectory(coords: np.ndarray, 
                       label_str: str = "",
                       rotation_range: float = 20.0, 
                       scale_range: float = 0.08, 
                       jitter_std: float = 0.015) -> np.ndarray:
    """
    Applies random 2D rotation, scaling, noise jitter, and circular-character specific
    temporal augmentations (roll & reversal) to a 3D trajectory.

    Works for all 62 classes: A-Z (uppercase), a-z (lowercase), 0-9 (digits).
    Lowercase circular letters (o, q, c, g, s) are treated identically to
    their uppercase counterparts for augmentation purposes.
    """
    augmented = np.copy(coords)
    
    # 1. Random 2D Rotation (around Z axis)
    if rotation_range > 0:
        angle_deg = np.random.uniform(-rotation_range, rotation_range)
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        augmented[:, 0] = x * cos_a - y * sin_a
        augmented[:, 1] = x * sin_a + y * cos_a
                
    # 2. Circular-specific Temporal Rolling & Reversal
    # Includes both uppercase and lowercase circular characters.
    # 'o' and 'O' are excluded from time reversal to preserve drawing direction
    # (used by the O/0/o orientation classifier in predict.py).
    circular_classes = {
        'O', 'o',       # Letter O (upper + lower)
        'Q', 'q',       # Letter Q (upper + lower)
        'C', 'c',       # Letter C (upper + lower)
        'G', 'g',       # Letter G (upper + lower)
        'S', 's',       # Letter S (upper + lower)
        '0', '6', '8', '9'  # Circular digits
    }
    if label_str in circular_classes:
        # Time Reversal (50% chance)
        # Disabled for '0', 'O', 'o' to preserve drawing direction for orientation classifier
        if label_str not in {'0', 'O', 'o'} and np.random.rand() > 0.5:
            active_mask = np.any(coords != 0.0, axis=1)
            active_coords = coords[active_mask]
            augmented[active_mask] = active_coords[::-1]
            
        # Temporal Roll / Phase Shift (up to 25% shift)
        active_mask = np.any(coords != 0.0, axis=1)
        active_len = np.sum(active_mask)
        if active_len > 10:
            shift = np.random.randint(-active_len // 4, active_len // 4 + 1)
            active_coords = augmented[active_mask]
            augmented[active_mask] = np.roll(active_coords, shift, axis=0)
        
    # 3. Random Scaling (independent for x and y to vary aspect ratio slightly)
    if scale_range > 0:
        scale_x = np.random.uniform(1 - scale_range, 1 + scale_range)
        scale_y = np.random.uniform(1 - scale_range, 1 + scale_range)
        scale_z = np.random.uniform(1 - scale_range, 1 + scale_range)
        
        augmented[:, 0] *= scale_x
        augmented[:, 1] *= scale_y
        augmented[:, 2] *= scale_z
        
    # 4. Random Gaussian Jitter
    if jitter_std > 0:
        noise = np.random.normal(0, jitter_std, size=coords.shape)
        # Skip noise on padded frames (all zeros)
        mask = np.any(coords != 0.0, axis=1)
        augmented[mask] += noise[mask]
        
    return augmented

def run_cross_validation(model_type: str, build_fn, X_train_raw: np.ndarray, y_train_raw: np.ndarray, 
                         classes: list, args, early_stop, reports_dir) -> dict:
    """
    Runs 5-Fold Cross-Validation strictly on the training dataset.
    Augmentation is applied ONLY to the training fold inside the loop.
    Validation folds remain unaugmented to prevent leakage.
    """
    print("\n" + "="*60)
    print(f"           RUNNING 5-FOLD CROSS-VALIDATION ON TRAINING SET ({model_type})           ")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_losses = []
    
    input_shape = (X_train_raw.shape[1], X_train_raw.shape[2])
    num_classes = len(classes)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train_raw)):
        print(f"\n--- Fold {fold + 1} / 5 ---")
        X_tr_fold, y_tr_fold = X_train_raw[train_idx], y_train_raw[train_idx]
        X_va_fold, y_va_fold = X_train_raw[val_idx], y_train_raw[val_idx]
        
        # Apply Data Augmentation strictly to the training fold
        X_tr_augmented = [X_tr_fold]
        y_tr_augmented = [y_tr_fold]
        num_copies = 3
        for _ in range(num_copies):
            aug_samples = np.zeros_like(X_tr_fold)
            for i in range(len(X_tr_fold)):
                label_str = classes[y_tr_fold[i]]
                aug_samples[i] = augment_trajectory(X_tr_fold[i], label_str=label_str)
            X_tr_augmented.append(aug_samples)
            y_tr_augmented.append(y_tr_fold)
            
        X_tr_fold_aug = np.concatenate(X_tr_augmented, axis=0)
        y_tr_fold_aug = np.concatenate(y_tr_augmented, axis=0)
        
        # Shuffle training fold
        shuffle_idx = np.arange(len(X_tr_fold_aug))
        np.random.shuffle(shuffle_idx)
        X_tr_fold_aug = X_tr_fold_aug[shuffle_idx]
        y_tr_fold_aug = y_tr_fold_aug[shuffle_idx]
        
        # Build fresh model
        model = build_fn(input_shape, num_classes)
        
        # Train on augmented fold, validate on unaugmented fold
        model.fit(
            X_tr_fold_aug, y_tr_fold_aug,
            validation_data=(X_va_fold, y_va_fold),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate on validation fold
        val_loss, val_acc = model.evaluate(X_va_fold, y_va_fold, verbose=0)
        val_pred_probs = model.predict(X_va_fold, batch_size=args.batch_size, verbose=0)
        val_preds = np.argmax(val_pred_probs, axis=1)
        
        precision = precision_score(y_va_fold, val_preds, average='weighted', zero_division=0)
        recall = recall_score(y_va_fold, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_va_fold, val_preds, average='weighted', zero_division=0)
        
        print(f"Fold {fold + 1} Result: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%, F1={f1:.4f}")
        
        fold_accuracies.append(val_acc)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_losses.append(val_loss)
        
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_prec = np.mean(fold_precisions)
    mean_rec = np.mean(fold_recalls)
    mean_f1 = np.mean(fold_f1s)
    
    cv_report_lines = [
        "==================================================",
        f"          5-FOLD CROSS VALIDATION SUMMARY: {model_type}",
        "=================================================="
    ]
    for f in range(5):
        cv_report_lines.append(f"Fold {f+1} Accuracy : {fold_accuracies[f]*100:.2f}%")
        
    cv_report_lines.extend([
        "",
        f"Mean Accuracy    : {mean_acc*100:.2f}%",
        f"Std Deviation    : {std_acc*100:.2f}%",
        "",
        f"Mean Precision   : {mean_prec*100:.2f}%",
        f"Mean Recall      : {mean_rec*100:.2f}%",
        f"Mean F1 Score    : {mean_f1*100:.2f}%",
        "=================================================="
    ])
    
    cv_report_text = "\n".join(cv_report_lines)
    print("\n" + cv_report_text + "\n")
    
    # Save the report to disk
    report_path = os.path.join(reports_dir, f"{model_type.lower()}_cv_results.txt")
    with open(report_path, "w") as f:
        f.write(cv_report_text)
    print(f"[Success] Saved CV report to {report_path}")
    
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_prec": mean_prec,
        "mean_rec": mean_rec,
        "mean_f1": mean_f1
    }

def main():
    parser = argparse.ArgumentParser(description="Air Writing Trajectory Recognition Model Training & Evaluation")
    parser.add_argument("--mode", type=str, default="resampled", choices=["resampled", "padded"],
                        help="Preprocessing data mode to load ('resampled' or 'padded')")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()
    
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    reports_dir = os.path.join(models_dir, "reports")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load Preprocessed Datasets
    print(f"[Info] Loading '{args.mode}' trajectory dataset...")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, classes, label_to_idx, idx_to_label = load_dataset(
            processed_dir, args.mode
        )
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return
        
    print(f"[Info] Loaded {len(X_train)} Train, {len(X_val)} Val, {len(X_test)} Test samples.")
    
    # Keep copies of the original raw, unaugmented train data for Cross-Validation
    X_train_raw = np.copy(X_train)
    y_train_raw = np.copy(y_train)
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # (64, 3)
    num_classes = len(classes)
    
    # Early stopping callback config
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # ------------------ 5-FOLD CROSS VALIDATION (ON TRAINING SET) ------------------
    lstm_cv_metrics = run_cross_validation(
        "LSTM", build_lstm_model, X_train_raw, y_train_raw, classes, args, early_stop, reports_dir
    )
    
    gru_cv_metrics = run_cross_validation(
        "GRU", build_gru_model, X_train_raw, y_train_raw, classes, args, early_stop, reports_dir
    )
    
    # ------------------ FINAL DATA AUGMENTATION & SHUFFLE ------------------
    # Apply Data Augmentation to the complete training set for final model training
    print("[Info] Applying data augmentation to final training set...")
    X_train_augmented = [X_train]
    y_train_augmented = [y_train]
    
    num_copies = 3
    for _ in range(num_copies):
        aug_samples = np.zeros_like(X_train)
        for i in range(len(X_train)):
            label_str = classes[y_train[i]]
            aug_samples[i] = augment_trajectory(X_train[i], label_str=label_str)
        X_train_augmented.append(aug_samples)
        y_train_augmented.append(y_train)
        
    X_train = np.concatenate(X_train_augmented, axis=0)
    y_train = np.concatenate(y_train_augmented, axis=0)
    
    # Shuffle the augmented training dataset
    shuffle_idx = np.arange(len(X_train))
    np.random.shuffle(shuffle_idx)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    print(f"[Info] Augmented training set size: {len(X_train)} samples (expanded by {num_copies + 1}x)")
    
    # ------------------ LSTM FINAL TRAINING ------------------
    print("\n" + "="*50)
    print("           INITIALIZING LSTM MODEL FINAL TRAINING           ")
    print("="*50)
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_model.summary()
    
    start_time = time.time()
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    lstm_train_time = time.time() - start_time
    print(f"[Info] LSTM final training completed in {lstm_train_time:.2f} seconds.")
    
    # Save LSTM model
    lstm_save_path = os.path.join(models_dir, "lstm_model.keras")
    lstm_model.save(lstm_save_path)
    print(f"[Success] Saved LSTM model to {lstm_save_path}")
    
    # Plot LSTM history
    lstm_plot_path = os.path.join(reports_dir, "lstm_history.png")
    plot_training_history(lstm_history, "LSTM Model", lstm_plot_path)
    print(f"[Info] Saved LSTM training plots to {lstm_plot_path}")
    
    # Evaluate LSTM on untouched test set
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, classes, "LSTM_Model", reports_dir)
    lstm_metrics["train_time"] = lstm_train_time
    lstm_metrics["num_classes"] = num_classes
    
    # ------------------ GRU FINAL TRAINING ------------------
    print("\n" + "="*50)
    print("           INITIALIZING GRU MODEL FINAL TRAINING           ")
    print("="*50)
    gru_model = build_gru_model(input_shape, num_classes)
    gru_model.summary()
    
    start_time = time.time()
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    gru_train_time = time.time() - start_time
    print(f"[Info] GRU final training completed in {gru_train_time:.2f} seconds.")
    
    # Save GRU model
    gru_save_path = os.path.join(models_dir, "gru_model.keras")
    gru_model.save(gru_save_path)
    print(f"[Success] Saved GRU model to {gru_save_path}")
    
    # Plot GRU history
    gru_plot_path = os.path.join(reports_dir, "gru_history.png")
    plot_training_history(gru_history, "GRU Model", gru_plot_path)
    print(f"[Info] Saved GRU training plots to {gru_plot_path}")
    
    # Evaluate GRU on untouched test set
    gru_metrics = evaluate_model(gru_model, X_test, y_test, classes, "GRU_Model", reports_dir)
    gru_metrics["train_time"] = gru_train_time
    gru_metrics["num_classes"] = num_classes
    
    # ------------------ COMPILING FINAL REPORTS ------------------
    print("\n" + "="*50)
    print("           FINAL EVALUATION COMPARISON SUMMARY             ")
    print("="*50)
    print(f"{'Metric':<18} | {'LSTM Model':<12} | {'GRU Model':<12} | {'Difference':<10}")
    print("-"*60)
    print(f"{'Test Accuracy':<18} | {lstm_metrics['accuracy']:<12.4f} | {gru_metrics['accuracy']:<12.4f} | {lstm_metrics['accuracy'] - gru_metrics['accuracy']:<10.4f}")
    print(f"{'Test Precision':<18} | {lstm_metrics['precision']:<12.4f} | {gru_metrics['precision']:<12.4f} | {lstm_metrics['precision'] - gru_metrics['precision']:<10.4f}")
    print(f"{'Test Recall':<18} | {lstm_metrics['recall']:<12.4f} | {gru_metrics['recall']:<12.4f} | {lstm_metrics['recall'] - gru_metrics['recall']:<10.4f}")
    print(f"{'Test F1 Score':<18} | {lstm_metrics['f1_score']:<12.4f} | {gru_metrics['f1_score']:<12.4f} | {lstm_metrics['f1_score'] - gru_metrics['f1_score']:<10.4f}")
    print(f"{'Trainable Weights':<18} | {lstm_metrics['num_params']:<12,} | {gru_metrics['num_params']:<12,} | {lstm_metrics['num_params'] - gru_metrics['num_params']:<10,}")
    print(f"{'Training Time (s)':<18} | {lstm_metrics['train_time']:<12.2f} | {gru_metrics['train_time']:<12.2f} | {lstm_metrics['train_time'] - gru_metrics['train_time']:<10.2f}")
    print("="*50)
    
    # Print the side-by-side Cross-Validation comparison
    print("\n" + "="*50)
    print("           CROSS-VALIDATION PERFORMANCE SUMMARY            ")
    print("="*50)
    print(f"{'Metric':<18} | {'LSTM Model':<12} | {'GRU Model':<12} | {'Difference':<10}")
    print("-"*60)
    print(f"{'Mean CV Accuracy':<18} | {lstm_cv_metrics['mean_acc']:<12.4f} | {gru_cv_metrics['mean_acc']:<12.4f} | {lstm_cv_metrics['mean_acc'] - gru_cv_metrics['mean_acc']:<10.4f}")
    print(f"{'CV Std Dev':<18} | {lstm_cv_metrics['std_acc']:<12.4f} | {gru_cv_metrics['std_acc']:<12.4f} | {lstm_cv_metrics['std_acc'] - gru_cv_metrics['std_acc']:<10.4f}")
    print(f"{'Mean Precision':<18} | {lstm_cv_metrics['mean_prec']:<12.4f} | {gru_cv_metrics['mean_prec']:<12.4f} | {lstm_cv_metrics['mean_prec'] - gru_cv_metrics['mean_prec']:<10.4f}")
    print(f"{'Mean Recall':<18} | {lstm_cv_metrics['mean_rec']:<12.4f} | {gru_cv_metrics['mean_rec']:<12.4f} | {lstm_cv_metrics['mean_rec'] - gru_cv_metrics['mean_rec']:<10.4f}")
    print(f"{'Mean F1 Score':<18} | {lstm_cv_metrics['mean_f1']:<12.4f} | {gru_cv_metrics['mean_f1']:<12.4f} | {lstm_cv_metrics['mean_f1'] - gru_cv_metrics['mean_f1']:<10.4f}")
    print("="*50)
    
    # Save MD report
    report_md_path = os.path.join(reports_dir, "comparison_report.md")
    recommended = generate_comparison_report(lstm_metrics, gru_metrics, args.mode, report_md_path)
    
    # Save a copy of the recommended model to a standard "best_model.keras" location
    best_model_path = os.path.join(models_dir, "best_model.keras")
    recommended_model_path = lstm_save_path if recommended == "LSTM" else gru_save_path
    
    try:
        shutil.copy2(recommended_model_path, best_model_path)
        print(f"[Success] Copied the recommended {recommended} model to {best_model_path}")
    except Exception as e:
        print(f"[Error] Failed to copy recommended model to {best_model_path}: {e}")
        
    print(f"[Success] Generated comparative markdown report at: {report_md_path}")
    print(f"[Info] Recommended model for deployment: {recommended}")
    print("="*50)

if __name__ == "__main__":
    main()
