"""
TinyML Wave Detection Model Training
Target: ARM Cortex-M7 (ATSAMS70Q21)

Simple model for detecting hand waving based on radar energy patterns
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_dataset, WINDOW_SIZE, NUM_CLASSES, CLASS_NAMES

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Model configuration
INPUT_SHAPE = (WINDOW_SIZE,)
MODEL_NAME = 'radar_wave_model'


def create_model() -> keras.Model:
    """
    Simple dense model for wave detection
    Input: window of energy values
    Output: binary classification (no_presence/waving)
    """
    model = keras.Sequential([
        layers.InputLayer(input_shape=INPUT_SHAPE),

        # Small dense network
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=MODEL_NAME)

    return model


def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray,
                epochs: int = 100) -> keras.callbacks.History:
    """Train the model"""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        )
    ]

    print("\nModel Summary:")
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """Evaluate model performance"""

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Per class accuracy
    print("\nPer-class Results:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).mean()
            print(f"  {name}: {class_acc*100:.1f}%")

    return {'loss': loss, 'accuracy': accuracy}


def save_model(model: keras.Model, path: str = 'models'):
    """Save model"""
    os.makedirs(path, exist_ok=True)

    keras_path = os.path.join(path, f'{model.name}.keras')
    model.save(keras_path)
    print(f"\nModel saved to: {keras_path}")


def main():
    print("=" * 60)
    print("TINYML WAVE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Input Shape: {INPUT_SHAPE}")
    print(f"Classes: {CLASS_NAMES}")

    # Load real sensor data
    print("\n[1/3] Loading radar sensor data...")
    X_train, X_test, y_train, y_test = load_dataset()

    # Create model
    print("\n[2/3] Creating model...")
    model = create_model()

    # Print model size estimate
    total_params = model.count_params()
    print(f"Total parameters: {total_params}")
    print(f"Estimated size (int8): {total_params} bytes (~{total_params/1024:.1f} KB)")

    # Train
    print("\n[3/3] Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test)

    # Save
    save_model(model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"Model Parameters: {total_params}")
    print("\nNext: Run export_model.py to quantize for MCU")

    return model, history, eval_results


if __name__ == '__main__':
    main()
