/*
 * TinyML Wave Detection - Inference API
 * Target: ARM Cortex-M7 (ATSAMS70Q21)
 *
 * Detects hand waving gestures from radar energy patterns
 */

#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

/* Model configuration */
#define ML_WINDOW_SIZE       16  /* Number of energy values per inference */
#define ML_NUM_CLASSES       2

/* Class definitions */
typedef enum {
    ML_CLASS_NO_PRESENCE = 0,
    ML_CLASS_WAVING = 1
} ml_class_t;

/* Inference result structure */
typedef struct {
    ml_class_t predicted_class;
    float confidence;
    float class_scores[ML_NUM_CLASSES];
    uint32_t inference_time_us;
    bool valid;
} ml_result_t;

/* Error codes */
typedef enum {
    ML_OK = 0,
    ML_ERROR_NOT_INITIALIZED,
    ML_ERROR_ALLOCATION_FAILED,
    ML_ERROR_INVOKE_FAILED,
    ML_ERROR_INVALID_INPUT
} ml_error_t;

/*
 * Initialize the TFLite Micro interpreter
 * Must be called once before any inference
 */
ml_error_t ml_init(void);

/*
 * Run inference on energy window
 * Input: array of ML_WINDOW_SIZE energy values (float, normalized 0-1)
 * Output: result structure with predicted class and confidence
 */
ml_error_t ml_inference(const float* energy_window, ml_result_t* result);

/*
 * Get class name string
 */
const char* ml_get_class_name(ml_class_t class_id);

/*
 * Get model size in bytes
 */
uint32_t ml_get_model_size(void);

/*
 * Check if ML module is initialized
 */
bool ml_is_initialized(void);

#endif /* TFLITE_INFERENCE_H */
