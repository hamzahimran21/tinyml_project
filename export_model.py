"""
TinyML Model Export and Quantization
Converts trained Keras model to TensorFlow Lite (int8) and C header

Target: ARM Cortex-M7 with TensorFlow Lite Micro
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_loader import load_dataset, WINDOW_SIZE

MODEL_PATH = 'models/radar_wave_model.keras'
OUTPUT_DIR = 'models'
FIRMWARE_DIR = 'firmware'


def representative_dataset_gen():
    """Generator for calibration data during quantization"""
    X_train, _, _, _ = load_dataset()

    for i in range(min(100, len(X_train))):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]


def convert_to_tflite_float(model: keras.Model, output_path: str) -> int:
    """Convert model to TFLite (float32)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"Float32 TFLite model: {size:,} bytes ({size/1024:.1f} KB)")
    return size


def convert_to_tflite_int8(model: keras.Model, output_path: str) -> int:
    """Convert model to fully quantized TFLite (int8)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size = len(tflite_model)
    print(f"Int8 TFLite model: {size:,} bytes ({size/1024:.1f} KB)")
    return size


def convert_to_c_array(tflite_path: str, output_path: str, array_name: str = 'g_model'):
    """Convert TFLite model to C header file"""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    c_code = f"""/*
 * TinyML Wave Detection Model
 * Auto-generated from {os.path.basename(tflite_path)}
 *
 * Model size: {len(model_data):,} bytes
 * Input: {WINDOW_SIZE} float values (energy window)
 * Output: 2 classes (no_presence, waving)
 */

#ifndef RADAR_MODEL_DATA_H
#define RADAR_MODEL_DATA_H

#include <stdint.h>

#define RADAR_MODEL_SIZE {len(model_data)}

/* Model input/output dimensions */
#define MODEL_INPUT_SIZE     {WINDOW_SIZE}
#define MODEL_OUTPUT_CLASSES 2

/* Class indices */
#define CLASS_NO_PRESENCE  0
#define CLASS_WAVING       1

alignas(16) const uint8_t {array_name}[{len(model_data)}] = {{
"""

    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        c_code += f"    {hex_values},\n"

    c_code += f"""}};\n
const unsigned int {array_name}_len = {len(model_data)};

#endif /* RADAR_MODEL_DATA_H */
"""

    with open(output_path, 'w') as f:
        f.write(c_code)

    print(f"C header generated: {output_path}")


def verify_tflite_model(tflite_path: str):
    """Verify the quantized model"""
    print("\nVerifying TFLite model...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input:  shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"Output: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")

    # Test with real data
    X_train, X_test, y_train, y_test = load_dataset()

    correct = 0
    total = len(X_test)

    for i in range(total):
        if input_details[0]['dtype'] == np.int8:
            scale = input_details[0]['quantization'][0]
            zero_point = input_details[0]['quantization'][1]
            input_data = ((X_test[i:i+1] / scale) + zero_point).astype(np.int8)
        else:
            input_data = X_test[i:i+1].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)

        if predicted == y_test[i]:
            correct += 1

    accuracy = correct / total
    print(f"TFLite Accuracy: {accuracy*100:.1f}% ({correct}/{total})")

    return accuracy


def analyze_model_for_mcu(tflite_path: str):
    """Analyze model requirements for MCU"""
    with open(tflite_path, 'rb') as f:
        model_size = len(f.read())

    print("\n" + "=" * 60)
    print("MCU DEPLOYMENT ANALYSIS")
    print("=" * 60)
    print(f"Model Size (Flash):     {model_size:,} bytes ({model_size/1024:.1f} KB)")

    flash_total = 2 * 1024 * 1024
    ram_total = 384 * 1024

    print(f"\n--- ATSAMS70Q21 Resource Usage ---")
    print(f"Flash: {model_size/1024:.1f} KB / {flash_total/1024:.0f} KB ({model_size/flash_total*100:.3f}%)")

    return {'model_size': model_size}


def main():
    print("=" * 60)
    print("TINYML MODEL EXPORT")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIRMWARE_DIR, exist_ok=True)

    print(f"\nLoading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {model.name}")
    model.summary()

    # Convert to TFLite float32
    print("\n[1/4] Converting to TFLite (float32)...")
    float_path = os.path.join(OUTPUT_DIR, 'radar_model_float.tflite')
    convert_to_tflite_float(model, float_path)

    # Convert to TFLite int8
    print("\n[2/4] Converting to TFLite (int8)...")
    int8_path = os.path.join(OUTPUT_DIR, 'radar_model_int8.tflite')
    convert_to_tflite_int8(model, int8_path)

    # Generate C header
    print("\n[3/4] Generating C header...")
    c_header_path = os.path.join(FIRMWARE_DIR, 'radar_model_data.h')
    convert_to_c_array(int8_path, c_header_path)

    # Verify
    print("\n[4/4] Verifying model...")
    verify_tflite_model(int8_path)
    analyze_model_for_mcu(int8_path)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"TFLite int8: {int8_path}")
    print(f"C header:    {c_header_path}")


if __name__ == '__main__':
    main()
