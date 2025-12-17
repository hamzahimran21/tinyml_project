"""
Radar Data Loader for Wave Detection
Loads energy values from sensor CSV files
"""

import numpy as np
import pandas as pd
from typing import Tuple
import os

# Configuration
WINDOW_SIZE = 16  # Number of consecutive frames to analyze
NUM_CLASSES = 2   # 0=no_presence, 1=waving

CLASS_NAMES = ['no_presence', 'waving']

def load_csv_data(filepath: str) -> np.ndarray:
    """Load energy values from CSV file"""
    df = pd.read_csv(filepath)
    return df['presence_energy'].values

def create_windows(data: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Create sliding windows from sequential data"""
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to 0-1 range"""
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    return data

def load_dataset(data_dir: str = '.') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare dataset from CSV files

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Find CSV files
    no_presence_file = None
    waving_file = None

    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            filepath = os.path.join(data_dir, f)
            df = pd.read_csv(filepath)
            energies = df['presence_energy'].values

            # Determine class by variance
            variance = np.var(energies)
            if variance < 1000:  # Low variance = no presence
                no_presence_file = filepath
            else:  # High variance = waving
                waving_file = filepath

    if no_presence_file is None or waving_file is None:
        raise ValueError("Need both no_presence and waving CSV files")

    print(f"No presence data: {no_presence_file}")
    print(f"Waving data: {waving_file}")

    # Load data
    no_presence_data = load_csv_data(no_presence_file)
    waving_data = load_csv_data(waving_file)

    print(f"No presence: {len(no_presence_data)} frames, variance={np.var(no_presence_data):.1f}")
    print(f"Waving: {len(waving_data)} frames, variance={np.var(waving_data):.1f}")

    # Create windows
    X_no_presence = create_windows(no_presence_data, WINDOW_SIZE)
    X_waving = create_windows(waving_data, WINDOW_SIZE)

    # Create labels
    y_no_presence = np.zeros(len(X_no_presence), dtype=np.int32)
    y_waving = np.ones(len(X_waving), dtype=np.int32)

    # Combine
    X = np.concatenate([X_no_presence, X_waving], axis=0)
    y = np.concatenate([y_no_presence, y_waving], axis=0)

    # Normalize globally
    global_min = X.min()
    global_max = X.max()
    X = (X - global_min) / (global_max - global_min)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nDataset created:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Input shape: {X_train.shape[1:]}")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"\nClass distribution (train): {np.bincount(y_train)}")
    print(f"Class distribution (test): {np.bincount(y_test)}")
