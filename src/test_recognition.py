"""
test_recognition.py
====================
Standalone evaluation script for the Air-Writing Recognition System.

Supports all 62 classes: A-Z (uppercase), a-z (lowercase), 0-9 (digits).

Usage:
    python src/test_recognition.py --model models/gru_model.keras
    python src/test_recognition.py --model models/lstm_model.keras --mode padded
    python src/test_recognition.py --model models/gru_model.keras --data-dir data/raw

Outputs:
    - Per-class Precision, Recall, F1-score table (printed to console)
    - Overall weighted Precision, Recall, F1 and Accuracy
    - Classification report saved to models/reports/test_62class_report.txt
    - 62x62 Confusion Matrix saved as PNG to models/reports/test_62class_confusion_matrix.png
    - Commonly confused pair analysis (O/o/0, I/l/1, C/c, S/s, Z/z, etc.)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

# Ensure src directory is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_single_trajectory


# ---------------------------------------------------------------------------
# Commonly confused character pairs (for highlighted reporting)
# ---------------------------------------------------------------------------
CONFUSED_PAIRS = [
    ('O', 'o', '0'),    # Circle group
    ('I', 'l', '1'),    # Vertical stroke group
    ('C', 'c'),         # Open arc group
    ('S', 's'),         # S-shape group
    ('Z', 'z'),         # Z-shape group
    ('W', 'w'),         # W-shape group
    ('X', 'x'),         # X-shape group
    ('V', 'v'),         # V-shape group
    ('U', 'u'),         # U-shape group
    ('P', 'p'),         # P-shape group
    ('K', 'k'),         # K-shape group
    ('B', 'b'),         # B-shape group
    ('D', 'd'),         # D-shape group
    ('G', 'g'),         # G-shape group
    ('Q', 'q'),         # Q-shape group
]


def load_recognition_system(model_path: str, mapping_path: str):
    """
    Loads the trained model and label mapping configuration.

    Returns:
        model: Loaded Keras model.
        classes: List of class label strings.
        idx_to_label: Dict mapping index -> label.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Label mapping not found: {mapping_path}")

    print(f"[Info] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[Info] Loading label mapping from: {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping_config = json.load(f)

    classes = mapping_config["classes"]
    idx_to_label = {int(k): v for k, v in mapping_config["idx_to_label"].items()}
    label_to_idx = mapping_config["label_to_idx"]
    return model, classes, idx_to_label, label_to_idx


def load_test_samples_from_raw(raw_dir: str, classes: list, label_to_idx: dict,
                                target_len: int = 64, smooth_window: int = 3) -> tuple:
    """
    Loads and preprocesses all .npy files from the raw data directory.
    Used when preprocessed numpy splits are not available.

    Returns:
        X: np.ndarray of shape (N, target_len, 3)
        y_true: np.ndarray of shape (N,) with integer class indices
        y_labels: List of true label strings (for display)
    """
    X_list = []
    y_list = []
    y_labels = []

    print(f"[Info] Loading raw samples from: {raw_dir}")
    for label in classes:
        label_dir = os.path.join(raw_dir, label)
        if not os.path.exists(label_dir):
            print(f"[Warning] No directory for class '{label}'")
            continue

        npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"[Warning] No samples found for class '{label}'")
            continue

        for fname in npy_files:
            fpath = os.path.join(label_dir, fname)
            try:
                raw = np.load(fpath)
                if len(raw) < 8:
                    continue
                preprocessed = preprocess_single_trajectory(
                    raw, target_len=target_len, smooth_window=smooth_window, mode='resample'
                )
                X_list.append(preprocessed)
                y_list.append(label_to_idx[label])
                y_labels.append(label)
            except Exception as e:
                print(f"[Warning] Skipped {fpath}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y_true = np.array(y_list, dtype=np.int64)
    return X, y_true, y_labels


def load_test_samples_from_processed(processed_dir: str, mode: str) -> tuple:
    """
    Loads the preprocessed test split (.npy files).

    Returns:
        X_test, y_test, classes, idx_to_label
    """
    split_dir = os.path.join(processed_dir, mode)
    X_test = np.load(os.path.join(split_dir, "X_test.npy"))
    y_test = np.load(os.path.join(split_dir, "y_test.npy"))
    mapping_path = os.path.join(processed_dir, "label_mapping.json")
    with open(mapping_path, "r") as f:
        mapping_config = json.load(f)
    classes = mapping_config["classes"]
    idx_to_label = {int(k): v for k, v in mapping_config["idx_to_label"].items()}
    label_to_idx = mapping_config["label_to_idx"]
    return X_test, y_test, classes, idx_to_label, label_to_idx


def plot_confusion_matrix(conf_mat: np.ndarray, classes: list, save_path: str,
                          title: str = "62-Class Confusion Matrix"):
    """
    Plots and saves a confusion matrix with readable class labels.
    """
    n = len(classes)
    # Use larger figure for 62 classes
    fig_size = max(20, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    tick_marks = np.arange(n)
    font_size = max(5, 10 - n // 15)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90, fontsize=font_size)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=font_size)

    # Draw values in cells (only if count > 0 and matrix is not too large)
    thresh = conf_mat.max() / 2.0
    if n <= 62:
        cell_font = max(4, 8 - n // 12)
        for i in range(n):
            for j in range(n):
                if conf_mat[i, j] > 0:
                    ax.text(j, i, str(conf_mat[i, j]),
                            ha="center", va="center",
                            color="white" if conf_mat[i, j] > thresh else "black",
                            fontsize=cell_font)

    ax.set_ylabel('True Class', fontsize=12)
    ax.set_xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Info] Confusion matrix saved to: {save_path}")


def analyze_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                           classes: list, label_to_idx: dict) -> str:
    """
    Analyzes confusion between commonly confused character pairs.
    Returns a formatted report string.
    """
    lines = [
        "\n" + "=" * 70,
        "   COMMONLY CONFUSED CHARACTER PAIR ANALYSIS",
        "=" * 70,
        f"{'Pair':<20} {'True→Pred Confusions':>25} {'Note':<20}",
        "-" * 70,
    ]

    for pair in CONFUSED_PAIRS:
        # Filter pair to only include classes in this model
        pair_in_model = [c for c in pair if c in label_to_idx]
        if len(pair_in_model) < 2:
            continue

        pair_indices = {c: label_to_idx[c] for c in pair_in_model}
        pair_str = " / ".join(pair_in_model)

        confusions = []
        for true_char, true_idx in pair_indices.items():
            true_mask = (y_true == true_idx)
            if true_mask.sum() == 0:
                continue
            pred_in_pair = y_pred[true_mask]
            for pred_char, pred_idx in pair_indices.items():
                if pred_char == true_char:
                    continue
                count = np.sum(pred_in_pair == pred_idx)
                if count > 0:
                    total = true_mask.sum()
                    confusions.append(f"{true_char}→{pred_char}: {count}/{total} ({count/total*100:.0f}%)")

        if confusions:
            conf_str = ", ".join(confusions)
            note = "⚠️ Confused" if confusions else "✓ Clean"
        else:
            conf_str = "No confusions"
            note = "✓ Clean"

        lines.append(f"{pair_str:<20} {conf_str:<35} {note}")

    lines.append("=" * 70)
    return "\n".join(lines)


def print_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             classes: list) -> str:
    """
    Prints a formatted per-class metrics table grouped by category.
    Returns the full table as a string.
    """
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    lines = [
        "\n" + "=" * 75,
        "   PER-CLASS METRICS — 62 CLASS RECOGNITION REPORT",
        "=" * 75,
        f"{'Class':<8} {'Category':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Samples':>8}",
        "-" * 75,
    ]

    # Print uppercase letters
    lines.append("  --- UPPERCASE LETTERS (A-Z) ---")
    for i, cls in enumerate(classes):
        if cls.isupper() and cls.isalpha():
            lines.append(f"  {cls:<6} {'Uppercase':<12} {prec[i]:>9.3f}  {rec[i]:>9.3f}  {f1[i]:>9.3f}  {support[i]:>7}")

    # Print lowercase letters
    lines.append("  --- LOWERCASE LETTERS (a-z) ---")
    for i, cls in enumerate(classes):
        if cls.islower() and cls.isalpha():
            lines.append(f"  {cls:<6} {'Lowercase':<12} {prec[i]:>9.3f}  {rec[i]:>9.3f}  {f1[i]:>9.3f}  {support[i]:>7}")

    # Print digits
    lines.append("  --- DIGITS (0-9) ---")
    for i, cls in enumerate(classes):
        if cls.isdigit():
            lines.append(f"  {cls:<6} {'Digit':<12} {prec[i]:>9.3f}  {rec[i]:>9.3f}  {f1[i]:>9.3f}  {support[i]:>7}")

    lines.append("=" * 75)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="62-Class Air-Writing Recognition Evaluation Script"
    )
    parser.add_argument("--model", type=str, default="models/best_model.keras",
                        help="Path to trained Keras model (.keras) (defaults to the recommended model)")
    parser.add_argument("--mode", type=str, default="resampled",
                        choices=["resampled", "padded"],
                        help="Which processed split to use for evaluation")
    parser.add_argument("--data-dir", type=str, default="",
                        help="If set, load raw samples directly from this directory "
                             "instead of using preprocessed splits. "
                             "Example: data/raw")
    parser.add_argument("--mapping", type=str,
                        default=os.path.join("data", "processed", "label_mapping.json"),
                        help="Path to label_mapping.json")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model inference")
    args = parser.parse_args()

    reports_dir = os.path.join("models", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    processed_dir = os.path.join("data", "processed")

    # 1. Load model and label mapping
    try:
        model, classes, idx_to_label, label_to_idx = load_recognition_system(
            args.model, args.mapping
        )
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    num_classes = len(classes)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"[Info] Model: {model_name} | Classes: {num_classes} | Mode: {args.mode}")

    # 2. Load test samples
    if args.data_dir:
        print(f"[Info] Loading ALL samples from raw directory: {args.data_dir}")
        X_test, y_test, y_labels = load_test_samples_from_raw(
            args.data_dir, classes, label_to_idx
        )
        source = "raw"
    else:
        print(f"[Info] Loading preprocessed test split ({args.mode})...")
        try:
            X_test, y_test, classes_loaded, idx_to_label_loaded, label_to_idx_loaded = \
                load_test_samples_from_processed(processed_dir, args.mode)
            # Use loaded mapping (may differ from model if run separately)
            classes = classes_loaded
            idx_to_label = idx_to_label_loaded
            label_to_idx = label_to_idx_loaded
            source = "processed"
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            print("[Info] Run: python src/preprocess.py to generate processed splits.")
            return

    if len(X_test) == 0:
        print("[Error] No test samples found. Please collect data first.")
        return

    print(f"[Info] Loaded {len(X_test)} test samples from {source} data.")

    # 3. Run inference
    print(f"[Info] Running inference (batch_size={args.batch_size})...")
    y_pred_probs = model.predict(X_test, batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Compute overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("           OVERALL EVALUATION METRICS")
    print("=" * 60)
    print(f"  Total Test Samples  : {len(X_test)}")
    print(f"  Number of Classes   : {num_classes}")
    print(f"  Test Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Weighted Precision  : {precision:.4f}")
    print(f"  Weighted Recall     : {recall:.4f}")
    print(f"  Weighted F1 Score   : {f1:.4f}")
    print("=" * 60)

    # 5. Per-class metrics table
    per_class_table = print_per_class_metrics(y_test, y_pred, classes)
    print(per_class_table)

    # 6. Full sklearn classification report
    cls_report = classification_report(
        y_test, y_pred, target_names=classes,
        labels=list(range(len(classes))), zero_division=0
    )

    # 7. Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))

    # 8. Confused pair analysis
    confused_pairs_report = analyze_confused_pairs(y_test, y_pred, classes, label_to_idx)
    print(confused_pairs_report)

    # 9. Save reports
    report_txt_path = os.path.join(reports_dir, f"test_{num_classes}class_report_{model_name}.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== {model_name.upper()} — {num_classes}-Class Recognition Evaluation Report ===\n\n")
        f.write(f"Source: {source} data\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Total Test Samples: {len(X_test)}\n\n")
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"  Accuracy  : {accuracy:.4f}\n")
        f.write(f"  Precision : {precision:.4f}\n")
        f.write(f"  Recall    : {recall:.4f}\n")
        f.write(f"  F1 Score  : {f1:.4f}\n\n")
        f.write("=== PER-CLASS CLASSIFICATION REPORT ===\n")
        f.write(cls_report)
        f.write("\n")
        f.write(per_class_table)
        f.write("\n")
        f.write(confused_pairs_report)
        f.write("\n")

    print(f"\n[Success] Full report saved to: {report_txt_path}")

    # 10. Plot and save confusion matrix
    conf_mat_path = os.path.join(reports_dir, f"test_{num_classes}class_confusion_matrix_{model_name}.png")
    plot_confusion_matrix(
        conf_mat, classes, conf_mat_path,
        title=f"{model_name.upper()} — {num_classes}-Class Confusion Matrix ({source} test set)"
    )

    print(f"[Success] All evaluation outputs saved to: {reports_dir}/")
    print("\n[Done] Testing complete.")


if __name__ == "__main__":
    main()
