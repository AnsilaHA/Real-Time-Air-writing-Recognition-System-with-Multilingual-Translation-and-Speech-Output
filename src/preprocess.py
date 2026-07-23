import os
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any


def get_folder_name(label: str) -> str:
    """
    Returns the filesystem folder name for a given class label.

    Windows NTFS is case-insensitive, so 'A/' and 'a/' refer to the same directory.
    To avoid collision, lowercase letters use a 'lc_' prefix:
        'a' -> folder 'lc_a'
        'b' -> folder 'lc_b'
        ... etc.
    Uppercase letters and digits keep their natural folder names:
        'A' -> folder 'A'
        '0' -> folder '0'

    Args:
        label: Class label string (e.g. 'a', 'A', '5').

    Returns:
        Folder name string (e.g. 'lc_a', 'A', '5').
    """
    if label.isalpha() and label.islower():
        return f"lc_{label}"
    return label


def get_label_from_folder(folder_name: str) -> str:
    """
    Reverse of get_folder_name — converts a folder name back to a class label.

    Args:
        folder_name: Directory name (e.g. 'lc_a', 'A', '5').

    Returns:
        Class label string (e.g. 'a', 'A', '5').
    """
    if folder_name.startswith("lc_") and len(folder_name) == 4:
        return folder_name[3:]  # 'lc_a' -> 'a'
    return folder_name


def discover_classes(raw_dir: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Auto-discovers available classes by scanning subdirectories in raw_dir.
    Returns classes in the canonical order: A-Z (uppercase), a-z (lowercase), 0-9 (digits).
    This order is deterministic and does NOT depend on filesystem sort order,
    ensuring consistent label indices across machines and OS versions.

    Folder naming convention (Windows-safe):
        Uppercase letters: A/, B/, ... Z/          (natural names)
        Lowercase letters: lc_a/, lc_b/, ... lc_z/ (prefixed to avoid case collision)
        Digits:            0/, 1/, ... 9/           (natural names)

    Args:
        raw_dir: Path to the raw dataset root directory (e.g. 'data/raw/').

    Returns:
        classes: Ordered list of class label strings (e.g. ['A', ..., 'a', ..., '0', ...]).
        label_to_idx: Dict mapping label string -> integer index.
        idx_to_label: Dict mapping integer index -> label string.
    """
    # Define canonical class ordering: uppercase A-Z, lowercase a-z, digits 0-9
    canonical_order = (
        [chr(i) for i in range(ord('A'), ord('Z') + 1)] +   # A-Z
        [chr(i) for i in range(ord('a'), ord('z') + 1)] +   # a-z
        [str(i) for i in range(10)]                          # 0-9
    )

    # Discover which classes actually exist as subdirectories
    if os.path.exists(raw_dir):
        existing_dirs = set(os.listdir(raw_dir))
    else:
        existing_dirs = set()

    # Keep only classes whose corresponding folder exists, preserving canonical order
    classes = [c for c in canonical_order if get_folder_name(c) in existing_dirs]

    if not classes:
        # Fallback: if raw_dir doesn't exist yet (e.g. during import), use full 62-class list
        classes = canonical_order

    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for idx, label in enumerate(classes)}
    return classes, label_to_idx, idx_to_label


# Module-level constants — populated at import time using the raw dataset directory.
# These are used by preprocess_single_trajectory (called from predict.py) which does NOT
# need class lists, so this is safe. Full class discovery happens in main().
CLASSES, LABEL_TO_IDX, IDX_TO_LABEL = discover_classes(os.path.join("data", "raw"))


def load_raw_dataset(raw_dir: str, classes: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Scans the raw data directories and loads the .npy files for all classes.
    Filters out corrupted or extremely short trajectories.
    Uses get_folder_name() to resolve Windows-safe directory names.

    Args:
        raw_dir: Path to raw dataset root directory.
        classes: Ordered list of class label strings (from discover_classes).

    Returns:
        A dictionary mapping label string to list of NumPy arrays of shape (N, 5).
    """
    dataset = {label: [] for label in classes}
    total_loaded = 0
    total_skipped = 0

    print("Loading raw trajectory files...")
    for label in classes:
        folder = get_folder_name(label)  # e.g. 'a' -> 'lc_a'
        label_dir = os.path.join(raw_dir, folder)
        if not os.path.exists(label_dir):
            print(f"[Warning] Directory for label '{label}' (folder '{folder}') does not exist at {label_dir}")
            continue

        # Get all .npy files in directory
        npy_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        for file in npy_files:
            file_path = os.path.join(label_dir, file)
            try:
                # Shape is (N, 5) with columns [pixel_x, pixel_y, norm_x, norm_y, norm_z]
                data = np.load(file_path)

                # Validation: discard sequences that are too short (under 8 frames)
                if len(data) < 8:
                    print(f"[Warning] Skipping too short sample ({len(data)} frames): {file_path}")
                    total_skipped += 1
                    continue

                dataset[label].append(data)
                total_loaded += 1
            except Exception as e:
                print(f"[Error] Error loading {file_path}: {e}")
                total_skipped += 1

    print(f"Successfully loaded {total_loaded} samples. Skipped {total_skipped} corrupt/short samples.")
    return dataset

def smooth_trajectory(coords: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Applies a moving average filter to smooth coordinates and remove high-frequency jitter.
    Maintains boundary integrity by dynamically shrinking the window at the edges.
    
    Args:
        coords: numpy array of shape (N, D)
        window_size: Odd integer specifying the filter window size.
        
    Returns:
        Smoothed numpy array of shape (N, D).
    """
    N, D = coords.shape
    smoothed = np.copy(coords)
    half_w = window_size // 2
    
    for i in range(N):
        start = max(0, i - half_w)
        end = min(N, i + half_w + 1)
        smoothed[i] = np.mean(coords[start:end], axis=0)
        
    return smoothed

def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Performs spatial normalization:
    1. Centering: Center the trajectory coordinates at the origin (0, 0, 0) by subtracting the centroid.
    2. Scaling: Scale the coordinates by the maximum dimension of the bounding box
       to ensure size invariance while preserving the aspect ratio.
       
    Args:
        coords: numpy array of shape (N, 3) representing [x, y, z].
        
    Returns:
        Spatially normalized numpy array of shape (N, 3).
    """
    # 1. Centering
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    
    # 2. Scaling (preserve aspect ratio using max of width/height)
    x_coords = centered[:, 0]
    y_coords = centered[:, 1]
    w = np.max(x_coords) - np.min(x_coords)
    h = np.max(y_coords) - np.min(y_coords)
    
    scale_factor = max(w, h)
    if scale_factor < 1e-6:
        scale_factor = 1.0  # Avoid division by zero for single-point lines
        
    # Scale all dimensions (including z depth) by the scale factor
    normalized = centered / scale_factor
    return normalized

def resample_trajectory(coords: np.ndarray, target_len: int = 64) -> np.ndarray:
    """
    Resamples a variable-length trajectory of length N to a fixed length L
    using temporal linear interpolation.
    
    Args:
        coords: numpy array of shape (N, D)
        target_len: Integer specifying target sequence length (L).
        
    Returns:
        Resampled numpy array of shape (target_len, D).
    """
    N, D = coords.shape
    if N == target_len:
        return coords
        
    # Build old and new time indices
    old_t = np.arange(N)
    new_t = np.linspace(0, N - 1, target_len)
    
    resampled = np.zeros((target_len, D))
    for d in range(D):
        resampled[:, d] = np.interp(new_t, old_t, coords[:, d])
        
    return resampled

def pad_truncate_trajectory(coords: np.ndarray, target_len: int = 64, pad_val: float = 0.0) -> np.ndarray:
    """
    Applies padding or truncation as an alternative temporal normalization technique.
    Short sequences are padded with pad_val at the end.
    Long sequences are truncated to target_len.
    
    Args:
        coords: numpy array of shape (N, D)
        target_len: Integer specifying target sequence length.
        pad_val: The value used to pad short sequences.
        
    Returns:
        Numpy array of shape (target_len, D).
    """
    N, D = coords.shape
    if N >= target_len:
        # Truncate from the end
        return coords[:target_len]
    else:
        # Pad with pad_val
        padded = np.full((target_len, D), pad_val, dtype=np.float32)
        padded[:N] = coords
        return padded

def preprocess_single_trajectory(raw_data: np.ndarray, 
                                 target_len: int = 64, 
                                 smooth_window: int = 3,
                                 mode: str = 'resample') -> np.ndarray:
    """
    Runs a raw trajectory sample through the preprocessing steps.
    
    Args:
        raw_data: Raw NumPy array of shape (N, 5) from data collector.
        target_len: Target sequence length.
        smooth_window: Moving average smoothing window size.
        mode: Temporal normalization mode ('resample' or 'pad').
        
    Returns:
        Preprocessed NumPy array of shape (target_len, 3) representing [x, y, z].
    """
    # 1. Slice: extract only the normalized coordinates [norm_x, norm_y, norm_z]
    # In raw_data, indices are: pixel_x (0), pixel_y (1), norm_x (2), norm_y (3), norm_z (4)
    coords = raw_data[:, 2:5]
    
    # 2. Smooth: apply low-pass moving average filter
    smoothed = smooth_trajectory(coords, window_size=smooth_window)
    
    # 3. Normalize: center and scale preserving aspect ratio
    normalized = normalize_coordinates(smoothed)
    
    # 4. Temporal Normalization: resample or pad/truncate
    if mode == 'resample':
        final = resample_trajectory(normalized, target_len=target_len)
    elif mode == 'pad':
        final = pad_truncate_trajectory(normalized, target_len=target_len)
    else:
        raise ValueError(f"Unknown temporal normalization mode: {mode}")
        
    return final

def split_and_shuffle_dataset(X: np.ndarray, y: np.ndarray,
                              num_classes: int,
                              train_ratio: float = 0.70,
                              val_ratio: float = 0.15,
                              random_seed: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                              Tuple[np.ndarray, np.ndarray],
                                                              Tuple[np.ndarray, np.ndarray]]:
    """
    Performs a stratified split on the dataset to create Train, Val, and Test sets.
    Assures class distribution is identical across sets.

    Args:
        X: Coordinates array of shape (M, L, 3)
        y: Labels array of shape (M,)
        num_classes: Total number of classes.
        train_ratio: Proportion of train data.
        val_ratio: Proportion of validation data.
        random_seed: Random state seed.

    Returns:
        A nested tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    np.random.seed(random_seed)

    train_idx_list, val_idx_list, test_idx_list = [], [], []

    # Group indices by class label to perform stratified splitting
    for class_idx in range(num_classes):
        class_indices = np.where(y == class_idx)[0]
        # Shuffle class-specific indices
        np.random.shuffle(class_indices)

        n_samples = len(class_indices)
        n_train = int(train_ratio * n_samples)
        n_val = int(val_ratio * n_samples)

        train_idx_list.extend(class_indices[:n_train])
        val_idx_list.extend(class_indices[n_train:n_train + n_val])
        test_idx_list.extend(class_indices[n_train + n_val:])

    # Convert lists to NumPy arrays
    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    # Shuffle each set so classes are mixed in training
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def create_comparison_visualization(raw_traj: np.ndarray, 
                                    processed_traj: np.ndarray, 
                                    label: str, 
                                    output_path: str):
    """
    Generates a high-quality BGR image comparison of the trajectory before and after preprocessing.
    Saves the image using OpenCV.
    
    Args:
        raw_traj: Raw numpy array (N, 5).
        processed_traj: Preprocessed numpy array (L, 3).
        label: The text label of the trajectory (e.g. 'A').
        output_path: Output file path.
    """
    # Canvas properties
    canvas_w = 800
    canvas_h = 400
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Draw Background Dividers
    cv2.line(canvas, (400, 0), (400, 400), (50, 50, 50), 2)
    
    # Text headers
    cv2.putText(canvas, f"Label: {label} (RAW - N={len(raw_traj)})", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Label: {label} (PREPROCESSED - L={len(processed_traj)})", (420, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Grid lines to visually indicate scale
    # Left side (Raw space is normalized 0-1)
    # Right side (Processed space is centered at 0, scaled to [-0.5, 0.5])
    
    # Extract Raw normalized x, y
    raw_coords = raw_traj[:, 2:4]
    
    # Extract Processed x, y
    proc_coords = processed_traj[:, 0:2]
    
    # Map raw coordinates to left panel (box size 300x300, offset x: 50 to 350, y: 70 to 370)
    # Since raw coordinates are in range [0, 1] relative to camera (norm_x, norm_y)
    # We want to center them in the 300x300 viewport
    raw_centered = raw_coords - np.mean(raw_coords, axis=0)
    raw_max_dim = max(np.max(raw_centered[:, 0]) - np.min(raw_centered[:, 0]), 
                      np.max(raw_centered[:, 1]) - np.min(raw_centered[:, 1]), 1e-5)
    raw_scaled = raw_centered / raw_max_dim
    
    def map_to_panel(pts, center_x, center_y, size=240):
        # pts are in [-0.5, 0.5] range
        px_pts = []
        for pt in pts:
            px_x = int(center_x + pt[0] * size)
            # Invert y since pixel coordinate 0 is top
            px_y = int(center_y + pt[1] * size)
            px_pts.append((px_x, px_y))
        return px_pts
        
    left_pixels = map_to_panel(raw_scaled, center_x=200, center_y=220)
    right_pixels = map_to_panel(proc_coords, center_x=600, center_y=220)
    
    # Draw left panel (Raw)
    # Draw points as tiny dots and connect them
    for i in range(1, len(left_pixels)):
        cv2.line(canvas, left_pixels[i-1], left_pixels[i], (100, 100, 180), 2, cv2.LINE_AA)
    for pt in left_pixels:
        cv2.circle(canvas, pt, 4, (50, 50, 255), -1)
        
    # Draw right panel (Preprocessed)
    for i in range(1, len(right_pixels)):
        cv2.line(canvas, right_pixels[i-1], right_pixels[i], (255, 180, 0), 2, cv2.LINE_AA)
    for pt in right_pixels:
        cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
        
    # Save the output image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

def compute_completeness_stats(dataset: Dict[str, List[np.ndarray]],
                               label_to_idx: Dict[str, int],
                               target_length: int = 64,
                               smoothing_window: int = 3) -> Dict[str, Dict]:
    """
    Computes per-class geometric feature statistics (p5 and p95 percentiles) over all samples
    for each class. These statistics are used by predict.py to validate character completeness.

    Args:
        dataset: Dict mapping label -> list of raw (N,5) trajectory arrays.
        label_to_idx: Dict mapping label string -> index.
        target_length: Target sequence length for preprocessing.
        smoothing_window: Smoothing window for preprocessing.

    Returns:
        Dict mapping label_str -> {feature_name -> {"p5": float, "p95": float}}.
    """
    # Feature extraction logic (mirrors predict.py extract_geometric_features)
    def extract_features(preprocessed_traj: np.ndarray) -> dict:
        pts = preprocessed_traj[:, :2]
        L = len(pts)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)
        w_dim = float(xmax - xmin)
        h_dim = float(ymax - ymin)
        bbox_diagonal = float(np.sqrt(w_dim**2 + h_dim**2))
        bbox_area = float(w_dim * h_dim)
        aspect_ratio = float(w_dim / max(h_dim, 1e-6))
        diffs = np.diff(pts, axis=0)
        step_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        path_length = float(np.sum(step_lengths))
        disp_vec = pts[-1] - pts[0]
        start_end_dist = float(np.sqrt(np.sum(disp_vec**2)))
        dpr = float(start_end_dist / max(path_length, 1e-6))
        loop_closure = float(start_end_dist / max(bbox_diagonal, 1e-6))
        dx = diffs[:, 0]
        dy = diffs[:, 1]
        sign_threshold = 0.01
        def count_sign_changes(arr):
            changes = 0
            prev_sign = 0
            for val in arr:
                if abs(val) > sign_threshold:
                    current_sign = np.sign(val)
                    if prev_sign != 0 and current_sign != prev_sign:
                        changes += 1
                    prev_sign = current_sign
            return changes
        dir_changes_x = count_sign_changes(dx)
        dir_changes_y = count_sign_changes(dy)
        total_dir_changes = dir_changes_x + dir_changes_y
        cumulative_angle = 0.0
        for i in range(L - 2):
            v1 = pts[i+1] - pts[i]
            v2 = pts[i+2] - pts[i+1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                cumulative_angle += np.arccos(cos_theta)
        q1 = np.any((x_coords >= 0) & (y_coords >= 0))
        q2 = np.any((x_coords < 0) & (y_coords >= 0))
        q3 = np.any((x_coords < 0) & (y_coords < 0))
        q4 = np.any((x_coords >= 0) & (y_coords < 0))
        quadrant_coverage = int(q1) + int(q2) + int(q3) + int(q4)
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        intersections = 0
        for i in range(L - 1):
            for j in range(i + 2, L - 1):
                if intersect(pts[i], pts[i+1], pts[j], pts[j+1]):
                    intersections += 1
        return {
            "path_length": path_length, "aspect_ratio": aspect_ratio,
            "dpr": dpr, "start_end_dist": start_end_dist,
            "loop_closure": loop_closure, "dir_changes": float(total_dir_changes),
            "cumulative_angle": float(cumulative_angle),
            "quadrant_coverage": float(quadrant_coverage),
            "intersections": float(intersections),
            "bbox_diagonal": bbox_diagonal, "bbox_area": bbox_area
        }

    completeness_stats = {}
    for label, samples in dataset.items():
        if len(samples) < 2:
            print(f"[Warning] Skipping completeness stats for '{label}': only {len(samples)} sample(s).")
            continue
        feature_lists: Dict[str, List[float]] = {}
        for raw_sample in samples:
            try:
                preprocessed = preprocess_single_trajectory(
                    raw_sample, target_len=target_length,
                    smooth_window=smoothing_window, mode='resample'
                )
                feats = extract_features(preprocessed)
                for k, v in feats.items():
                    feature_lists.setdefault(k, []).append(v)
            except Exception:
                continue
        class_stats = {}
        for feat_name, values in feature_lists.items():
            arr = np.array(values)
            class_stats[feat_name] = {
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }
        completeness_stats[label] = class_stats
    return completeness_stats


def main():
    raw_data_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    target_length = 64
    smoothing_window = 3

    # --- Step 0: Auto-discover classes from raw data directory ---
    classes, label_to_idx, idx_to_label = discover_classes(raw_data_dir)
    num_classes = len(classes)
    print(f"[Info] Discovered {num_classes} classes: {classes}")

    # Initialize output directory structures
    resampled_out_dir = os.path.join(processed_dir, "resampled")
    padded_out_dir = os.path.join(processed_dir, "padded")
    vis_dir = os.path.join(processed_dir, "visualizations")

    os.makedirs(resampled_out_dir, exist_ok=True)
    os.makedirs(padded_out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load dataset
    dataset = load_raw_dataset(raw_data_dir, classes)

    # Containers for processed data
    X_resampled_list = []
    X_padded_list = []
    y_list = []

    # Select a few samples for saving before/after visualizations
    vis_targets = {'A': True, '5': True, 'M': True, 'S': True, '9': True,
                   'a': True, 'm': True, 's': True}

    print("\nProcessing trajectories...")
    for label, samples in dataset.items():
        if len(samples) == 0:
            print(f"[Warning] No samples found for class '{label}'")
            continue

        class_idx = label_to_idx[label]
        for idx, raw_sample in enumerate(samples):
            # Process sample using Resampling mode
            resampled_traj = preprocess_single_trajectory(
                raw_sample,
                target_len=target_length,
                smooth_window=smoothing_window,
                mode='resample'
            )

            # Process sample using Padding/Truncation mode
            padded_traj = preprocess_single_trajectory(
                raw_sample,
                target_len=target_length,
                smooth_window=smoothing_window,
                mode='pad'
            )

            X_resampled_list.append(resampled_traj)
            X_padded_list.append(padded_traj)
            y_list.append(class_idx)

            # Save visual comparisons for targets (first sample of chosen classes)
            if label in vis_targets and vis_targets[label]:
                vis_path = os.path.join(vis_dir, f"{label}_comparison.png")
                create_comparison_visualization(raw_sample, resampled_traj, label, vis_path)
                print(f"[Info] Saved visual comparison for '{label}' to {vis_path}")
                vis_targets[label] = False  # Only visualize the first sample

    if len(X_resampled_list) == 0:
        print("\u274c Error: No samples loaded or processed. Preprocessing aborted.")
        return

    X_resampled = np.array(X_resampled_list, dtype=np.float32)
    X_padded = np.array(X_padded_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\nTotal Dataset sizes:")
    print(f"X_resampled shape: {X_resampled.shape}")
    print(f"X_padded shape:    {X_padded.shape}")
    print(f"y shape:           {y.shape}")

    # ------------------ Stratified Split and Saving ------------------
    # Split resampled data
    (X_tr_res, y_tr_res), (X_va_res, y_va_res), (X_te_res, y_te_res) = split_and_shuffle_dataset(
        X_resampled, y, num_classes=num_classes, train_ratio=0.70, val_ratio=0.15
    )

    # Split padded data
    (X_tr_pad, y_tr_pad), (X_va_pad, y_va_pad), (X_te_pad, y_te_pad) = split_and_shuffle_dataset(
        X_padded, y, num_classes=num_classes, train_ratio=0.70, val_ratio=0.15
    )

    # Save resampled dataset splits
    np.save(os.path.join(resampled_out_dir, "X_train.npy"), X_tr_res)
    np.save(os.path.join(resampled_out_dir, "y_train.npy"), y_tr_res)
    np.save(os.path.join(resampled_out_dir, "X_val.npy"), X_va_res)
    np.save(os.path.join(resampled_out_dir, "y_val.npy"), y_va_res)
    np.save(os.path.join(resampled_out_dir, "X_test.npy"), X_te_res)
    np.save(os.path.join(resampled_out_dir, "y_test.npy"), y_te_res)

    # Save padded dataset splits
    np.save(os.path.join(padded_out_dir, "X_train.npy"), X_tr_pad)
    np.save(os.path.join(padded_out_dir, "y_train.npy"), y_tr_pad)
    np.save(os.path.join(padded_out_dir, "X_val.npy"), X_va_pad)
    np.save(os.path.join(padded_out_dir, "y_val.npy"), y_va_pad)
    np.save(os.path.join(padded_out_dir, "X_test.npy"), X_te_pad)
    np.save(os.path.join(padded_out_dir, "y_test.npy"), y_te_pad)

    # Save class label index mapping configuration
    mapping_config = {
        "classes": classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()}
    }
    mapping_path = os.path.join(processed_dir, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping_config, f, indent=4)

    print(f"\nSaved label mapping to: {mapping_path}")

    # ------------------ Completeness Stats Generation ------------------
    print("\n[Info] Computing per-class completeness statistics (p5/p95 geometric features)...")
    completeness_stats = compute_completeness_stats(
        dataset, label_to_idx, target_length=target_length, smoothing_window=smoothing_window
    )
    stats_path = os.path.join(processed_dir, "completeness_stats.json")
    with open(stats_path, "w") as f:
        json.dump(completeness_stats, f, indent=4)
    print(f"[Info] Saved completeness stats to: {stats_path}")

    print("\nPreprocessed Dataset Summaries:")
    print("="*60)
    print(f" RESAMPLED (Interpolated to {target_length} points):")
    print(f"  |- Train: X={X_tr_res.shape}, y={y_tr_res.shape}")
    print(f"  |- Val:   X={X_va_res.shape}, y={y_va_res.shape}")
    print(f"  |- Test:  X={X_te_res.shape}, y={y_te_res.shape}")
    print("-"*60)
    print(f" PADDED (Padded/truncated to {target_length} points):")
    print(f"  |- Train: X={X_tr_pad.shape}, y={y_tr_pad.shape}")
    print(f"  |- Val:   X={X_va_pad.shape}, y={y_va_pad.shape}")
    print(f"  |- Test:  X={X_te_pad.shape}, y={y_te_pad.shape}")
    print("="*60)
    print("Data Preprocessing Completed Successfully!")


if __name__ == "__main__":
    main()
