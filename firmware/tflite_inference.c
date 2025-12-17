/*
 * TinyML Wave Detection - Inference Implementation
 * Target: ARM Cortex-M7 (ATSAMS70Q21)
 *
 * Memory requirements:
 *   - Model (Flash): ~3 KB
 *   - Tensor Arena (RAM): ~4 KB
 */

#include "tflite_inference.h"
#include "radar_model_data.h"

/* TensorFlow Lite Micro headers */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* For timing measurements */
#ifdef __ARM_ARCH
#include "core_cm7.h"
#define GET_CYCLE_COUNT() DWT->CYCCNT
#define CYCLES_TO_US(c) ((c) / 300)  /* 300 MHz clock */
#else
#define GET_CYCLE_COUNT() 0
#define CYCLES_TO_US(c) 0
#endif

/* Tensor arena - small for this simple model */
#define TENSOR_ARENA_SIZE (4 * 1024)  /* 4 KB */

static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

/* TFLite Micro objects */
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static bool initialized = false;

/* Class names */
static const char* class_names[ML_NUM_CLASSES] = {
    "no_presence",
    "waving"
};

/* Quantization parameters */
static float input_scale = 1.0f;
static int32_t input_zero_point = 0;
static float output_scale = 1.0f;
static int32_t output_zero_point = 0;


ml_error_t ml_init(void)
{
    if (initialized) {
        return ML_OK;
    }

    /* Load the model */
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return ML_ERROR_ALLOCATION_FAILED;
    }

    /* Create op resolver with required operations */
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    /* Create interpreter */
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    /* Allocate tensors */
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return ML_ERROR_ALLOCATION_FAILED;
    }

    /* Get input and output tensors */
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    /* Get quantization parameters from model */
    if (input_tensor->quantization.type == kTfLiteAffineQuantization) {
        input_scale = input_tensor->params.scale;
        input_zero_point = input_tensor->params.zero_point;
    }
    if (output_tensor->quantization.type == kTfLiteAffineQuantization) {
        output_scale = output_tensor->params.scale;
        output_zero_point = output_tensor->params.zero_point;
    }

    initialized = true;
    return ML_OK;
}


ml_error_t ml_inference(const float* energy_window, ml_result_t* result)
{
    if (!initialized) {
        return ML_ERROR_NOT_INITIALIZED;
    }

    if (!energy_window || !result) {
        return ML_ERROR_INVALID_INPUT;
    }

    uint32_t start_cycles = GET_CYCLE_COUNT();

    /* Quantize input: float (0-1) to int8 */
    int8_t* input_data = input_tensor->data.int8;

    for (int i = 0; i < ML_WINDOW_SIZE; i++) {
        int32_t quantized = (int32_t)(energy_window[i] / input_scale) + input_zero_point;
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        input_data[i] = (int8_t)quantized;
    }

    /* Run inference */
    if (interpreter->Invoke() != kTfLiteOk) {
        result->valid = false;
        return ML_ERROR_INVOKE_FAILED;
    }

    /* Process output */
    int8_t* output_data = output_tensor->data.int8;
    int best_class = 0;
    int8_t best_score = output_data[0];
    float sum_exp = 0.0f;

    for (int i = 0; i < ML_NUM_CLASSES; i++) {
        float score = (output_data[i] - output_zero_point) * output_scale;
        result->class_scores[i] = score;

        if (output_data[i] > best_score) {
            best_score = output_data[i];
            best_class = i;
        }
    }

    /* Apply softmax */
    for (int i = 0; i < ML_NUM_CLASSES; i++) {
        sum_exp += expf(result->class_scores[i]);
    }
    for (int i = 0; i < ML_NUM_CLASSES; i++) {
        result->class_scores[i] = expf(result->class_scores[i]) / sum_exp;
    }

    uint32_t end_cycles = GET_CYCLE_COUNT();

    result->predicted_class = (ml_class_t)best_class;
    result->confidence = result->class_scores[best_class];
    result->inference_time_us = CYCLES_TO_US(end_cycles - start_cycles);
    result->valid = true;

    return ML_OK;
}


const char* ml_get_class_name(ml_class_t class_id)
{
    if (class_id < ML_NUM_CLASSES) {
        return class_names[class_id];
    }
    return "unknown";
}


uint32_t ml_get_model_size(void)
{
    return RADAR_MODEL_SIZE;
}


bool ml_is_initialized(void)
{
    return initialized;
}
