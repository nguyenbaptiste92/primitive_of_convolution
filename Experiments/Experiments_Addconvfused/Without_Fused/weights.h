#include "nnom.h"

#define ADD_CONV2D_KERNEL_0 {25, -62, 16, 52, 25, 45, -68, -14, 71, 47, 3, 1, -10, -7, 29, -43, 41, -63, -39, 23, 26, -47, -51, 36, -60, -2, -71, 3, -11, -33, 69, -21, 39, 2, -11, -4, 41, -55, -53, 56, 39, 6, -59, 10, 32, -29, -62, -1, -44, 72, -29, 65, -63, 66, 68, 2, 69, -53, 62, -43, -19, -41, 69, -34, -25, 16, -63, 60, -68, 57, 2, -68, 43, -3, 36, 47, -66, 56, 26, -43, 64, 35, 9, 45, -53, -31, 29, -11, 57, -70, -63, 72, 18, 4, -33, -59, 72, -27, -57, 59, -10, 1, -48, 70, 26, -27, -50, 59, 73, -31, 47, -10, 50, 21, -25, -31, -15, -57, -31, 57, -29, -71, -66, 18, 39, 19, -34, 61, 66, -28, 60, -56, 44, 13, -21, -30, -1, 21, 39, 70, -5, -70, 47, 71, -25, 41, -40, 68, -57, -29, 49, 37, 57, 11, 8, -4, -66, -45, 64, -17, -65, 0, -33, -14, -54, -28, 8, 11, 50, 51, 51, -62, -3, 54, 49, 44, 66, -60, -18, 36, -37, 11, 21, -55, -23, -67, 2, -26, 61, 50, 70, -49, -42, -36, 70, -3, -39, 17, 17, 41, 61, -38, 33, 26, -26, 5, -32, -54, 15, -15, 67, 55, -30, 47, -45, 4, 19, 40, -52, 55, -44, 5, -60, -67, 19, -34, -69, -17, -71, 21, 29, 43, -17, -49, -18, -19, 22, -62, 51, -50, 2, -59, 32, 10, 72, 44, -31, 38, 51, 54, 74, 32, -26, 31, 59, -39, 17, 8, 23, -2, -42, 66, 73, 31, -55, 30, -2, -32, 28, -24, 17, -34, -34, -47, 26, -12, 49, 13, -30, -31, 57, -18, -7, -45, -38, 30, 46, 37, 50, -15, 3, 49, 3, 16, -47, 59, -28, -31, 3, 39, 45, -72, -19, 30, -50, 37, -71, -41, 65, 22, -46, 2, -72, 58, -18, 44, -56, 74, -39, -33, -56, 50, -40, -63, 39, -48, 60, 1, 69, -62, -31, 8, -41, 74, 42, 55, 51, -62, 25, 2, -32, -43, -17, 11, 33, -38, -56, 48, 40, -3, 37, 26, 65, 21, -37, -34, 11, -21, -37, -12, 13, -19, 45, -34, 17, 6, 69, 69, -6, 8, -13, 42, -18, 24, -25, -39, 72, -56, 44, -34, -44, 15, 65, -51, 61, -21, -59, -33, 57, -38, -9, -37, 42, 25, -49, -60, 50, -29, 62, 43, -11, 7, -12, -51, 69, -44, -12, 48, 50, 49, 11, 41, 27, 29, -51, -12, 62, -9, 4, 44, -46, 2, 0, -70, 31, -71, -2, -48, 70, -58, -31, 61, 61, -15, 64, -44, 33, -74, 17, -60, 18, 43, 33, -71, 44, 56, 11, 23, -16, -57, 5, -68, -50, 63, -73, 1, 1, -43, 18, -32, 39, 43, -21, 1, -3, -41, -1, 44, -25, -30, 24, -20, -65, -11, 41, 4, 24, -30, -44, 55, -28, 64, -68, -9, 12, 64, -65, 6, -30, -1, -1, -41, 7, -48, -30, 71, 47, 21, 45, -33, -71, 55, 47, 45, -6, -67, -57, -69, -69, 30, -28, 67, 0, -36, 2, -2, 12, 47, -72, -3, -41, -29, 1, 35, -9, -33, 41, -45, 9, -45, 45, 60, 14, -34, -6, -51, 71, -64, 16, 62, 72, -73, -27, 63, -53, 31, -3, -3, 4, 37, -37, 9, -7, -46, 13, -55, -71, -71, 66, 22, -9, -44, 49, -49, -48, 34, -18, 47, -11, -24, -15, 52, 23, 66, 47, -1, 69, -55, 63, 2, -64, -3, 27, 72, 20, 45, 25, 61, 69, 72, -30, 48, 42, 45, 11, 12, 69, -11, -38, -11, -11, 48, 16, 6, 45, 63, -28, -62, 8, -67, -8, -15, 64, 23, -71, -31, -53, -55, -6, -53, -48, 67, 28, 33, 52, 16, 39, -69, -27, 41, 31, -62, 1, -52, -67, -47, -70, 47, -17, -13, -1, 69, 43, 31, -45, -45, 17, -7, 7, 7, -38, -56, -5, -6, -5, -38, 9, 25, 63, -67, 68, 66, -26, -35, 28, -54, 18, 2, 56, -36, 42, -45, 19, -41, 32, 58, 39, -21, 45, -54, -18, -22, -74, 21, 14, 45, 72, -4, -58, 7, 23, 17, -14, -67, -34, 3, -26, 22, 55, -34, -20, -38, 66, 71, -48, 38, 24, 35, -15, 26, 20, -71, -56, -40, -59, 71, -68, -5, -31, 43, 8, -43, -37, -31, 29, 4, -25, -31, -5, -23, -48, 36, 13, 64, 30, 39, -34, 43, 43, 66, 58, 15, 68, 15, 45, -53, -42, -24, 38, -6, -3, 62, -8, 72, -42, -57, 12, 23, 47, -61, 39, -27, -4, 65, 43, 47, -24, -55, -34, -69, -12, -70, -17, -29, -41, 57, 44, -69, -58, -60, -15, 41, -20, 17, 20, 17, 39, -27, -48, -36, -69, -57, 12, 57, -73, 68, -28, -70, -37, -4, 28, 36, -14, 43, -23, -29, -35, -62, -33, 45, 40, 68, 24, 36, 27, 10, 59, -14, -35, -31, 24, 1, -35, -10, -37, 15, -56, -58, -16, 58, 17, -22, -40, -72, -50, -11, -17, -59, 26, 60, -20, -10, 26, -21, -30, 29, 2, 57, -20, 67, -26, 68, 64, -68, 58, -50, -63, 46, -38, 52, -12, -46, -39, -40, -19, -40, 59, 29, 65, 4, 50, -13, 30, -4, 18, 43, -41, -9, -14, -39, -68, 3, -53, 41, -56, 39, 8, -67, -26, 38, 64, -70, 23, 11, -22, -68, 73, 72, 51, -14, -40, 64, -2, -25, -22, 17, 41, -34, 1, -69, 56, 39, -35, -8, -58, 69, -60, 21, -38, 2, 36, 34, -38, 18, 35, 62, -57, -18, 52, 1, 21, -65, 6, 47, 8, 53, 20, -3, 47, -9, 20, -53, -14, 18, -42, 1, 17, 56, 2, 37, -61, 31, -26, 35, -31, -72, 4, -6, -32, -70, 4, -14, -34, -1, -58, -14, -27, 16, 66, -42, 43, -13, 72, -4, -64, 66, 65, -5, 12, 24, -65, -50, 46, 7, -9, -62, -48, -59, 69, -4, 5, -61, -17, 48, 56, 7, 23, -27, -50, 66, -56, 13, -9, -18, -31, 27, 38, 16, -34, -45, 24, -74, -38, -54, -20, 59, -60, 14, -26, -28, -57, 18, -58, -57, 19, 2, -14, -20, -54, -70, 46, -30, 33, -21, 46, 22, -17, -1, 31, 61, 71, 18, -56, -46, 59, 40, -64, 16, -16, -26, -25, 62, -57, 36, 11, 3, 62, -3, 37, 1, -68, -52, 44, 25, 40, 51, -53, 28, -38, -22, 63, -25, -23, 50, 65, -60, 20, -43, -43, 58, -70, -55, -46, 72, -35, -58, 50, -67, 40, -45, 61, -64, 67, -56, 11, 57, 8, 19, 13, 61, -25, 12, -26, -42, -44, -35, -65, 65, 62, 13, 67, -16, -10, -23, -1, -66, -63, -7, 1, -23, 7, 39, -8, -18, -68, -18, 21, -44, -70, 51, 43, -22, 21, 21, -9, 41, 59, 64, 61, 54, 70, 42, 35, 44, -64, -22, -58, -43, 39, 40, 53, -33, 70, 41, -38, 63, -13, -57, -15, 53, 28, -65, -55, -38, -61, 14, -35, -33, -49, -3, 59, -32, -72, 27, -32, -46, -21, -26, 63, 19, -3, 64, -63, -59, -21, -48, 4, -10, 34, 70, 27, -62, 20, 70, -68, -32, 31, 63, -69, -28, -42, 17, -7, -2, -57, 62, -5, 43, 15, 32, -61, -3, 29, -15, 33, -23, -3, 28, -9, 26, 47, -72, -6, -60, -60, 67, -43, 36, -19, 34, 35, 51, 12, -65, -16, 44, -71, 51, -58, -65, 43, -22, 30, 12, 19, -37, 56, -19, 43, -43, -51, 39, 70, 0, 18, -4, -5, -23, -52, 2, 10, 6, -3, 19, -15, 17, 50, -29, 27, 38, -57, -31, -37, -16, -40, 28, -39, 41, 48, -47, -68, -20, 60, 67, 13, 43, 42, 27, -20, -43, -27, 59, -19, 68, -31, 9, -59, -54, -63, 69, -34, 23, -35, -25, 26, 37, -34, -10, 3, -70, -19, -42, 23, 18, -23, -58, 53, 57, 71, -42, 36, -25, 31, 62, 37, 8, 16, 10, 4, 34, 12, 66, 60, 14, 50, 74, -1, 9, 53, -17, -37, -34, 68, 22, 10, -10, -42, -51, -14, 57, 48, 30, -3, 64, 0, 57, -37, -17, -20, -69, 17, -39, 39, -9, -6, -72, 69, 5, 58, 5, -10, 66, 51, -61, 26, 63, -19, 71, 40, 8, 49, -74, -68, -34, 39, 46, -47, 41, 55, 32, -51, 31, 33, -7, -39, -72, -51, -42, -48, 29, 29, -37, -4, -24, 33, 27, 65, 68, -61, -36, 1, 73, -7, 61, 17, -30, -54, -24, -71, -5, -1, 19, 74, -17, 59, -66, 57, 70, -46, 4, 20, 53, -28, -72, 69, 12, -70, 16, 47, -24, 2, 71, -71, 20, 61, -43, 64, -60, -37, 54, -67, -5, 71, 20, -4, -50, 41, -28, 60, 39, 31, -21, 60, 8, 29, 53, -32, -48, -41, 61, -52, -26, 3, 74, 11, -28, 46, -4, -30, -57, 66, 30, 35, -3, -67, -56, -33, -10, -68, 46, 9, 26, -5, 21, -52, -6, -25, -71, 11, 31, 52, 28, -40, 56, -55, -18, 14, 27, 17, -27, -46, 73, -55, 6, 65, 13, -47, 68, -70, 46, 40, -49, 59, 47, 11, -28, 26, 5, -5, -34, 16, 18, 48, -52, -10, -65, 42, 63, -34, -5, 55, -15, 45, -16, -53, 34, 58, 31, 61, -17, -49, 7, -20, 72, -53, 39, 71, -63, -34, -16, -59, 68, 0, 6, -27, 4, -63, -16, 29, 32, -13, -19, -72, 65, 29, 32, 13, 48, 10, -3, 26, -47, -38, 25, 53, 65, 15, -50, -29, -39, -4, -49, -17, 18, 53, -41, 2, -18, 12, -73, -22, -29, 49, 36, -18, -22, 38, -38, 34, 60, 1, 0, 17, 13, -57, 34, 64, 41, -71, 2, -37, -69, -44, -68, 16, -37, -16, -5, 37, 22, 51, -69, 4, -30, 49, -16, 27, -16, -24, 67, -61, -20, 34, 3, 72, 68, 34, 34, -28, 28, 35, -51, -69, 20, 48, 22, -14, 2, 62, -65, -35, 24, -17, -49, 52, 19, 26, 66, -15, -22, -32, -9, -68, 45, 48, 71, 12, -59, 50, 44, 45, -71, 2, 5, 28, 17, 24, -51, -74, -39, 38, -66, -38, 67, -32, -31, -50, 7, 28, -45, -68, 58, -34, -46, -18, 50, -52, -17, 36, -63, 11, 72, -62, 32, -26, 37, -9, -52, -8, 30, -1, 20, -17, -4, 18, 8, -29, -51, -32, -52, -22, -13, 11, -67, 17, -10, 61, -29, 55, 22, 36, 13, 1, -53, -54, -32, -15, 18, 24, -33, 67, 42, 25, -29, -34, 73, 4, -54, 45, 35, 60, 49, -45, -47, -39, -4, -22, -57, 65, -60, -60, -48, 23, -22, -58, -12, -29, 53, 17, 48, 66, 45, -35, -37, -46, 70, 14, 13, 47, -62, 44, 49, 7, -20, -1, 4, -69, 24, -45, 36, -10, -63, -26, 52, 11, 38, 13, 43, 19, 38, 48, -4, -16, -33, -53, -1, 19, 2, -66, -3, -72, -20, -30, 71, 65, -21, -72, 6, -61, 57, -45, -11, 41, 24, -46, 23, 21, 44, 27, 54, -19, 24, -57, -51, -62, -15, -36, -62, 2, 37, -51, -51, -16, 10, -59, -9, 45, -19, 20, 28, -23, 20, -13, 69, 40, -2, 15, 33, 73, 47, 17, -69, -24, -66, -8, 25, -45, -30, 17, -43, -68, 51, -70, 14, -67, 60, -20, 61, 24, -64, -71, 26, 71, -10, 59, 38, -52, 51, 2, -14, 69, 14, -12, 37, -4, -27, -38, 50, -56, 5, 26, 13, -72, -63, 53, 59, 44, 57, 47, -43, -17, -58, -5, 62, -16, 63, -4, 64, 20, -54, 9, 36, -42, 29, -32, -31, -31, -68, -42, -33, 17, -55, 23, -64, 33, -46, 64, 31, -61, 41, 59, -50, 3, -10, 0, 48, -51, 58, 0, -64, 39, -58, -18, -54, 70, -55, -15, -20, 11, -14, 55, 2, -25, 23, -68, -34, 9, -54, -25, -37, 9, 72, 8, -38, -64, -52, -42, 54, 14, -26, -53, -67, -6, 29, 21, -55, 71, -8, -51, 61, 25, -13, -51, 38, 56, -69, -54, 71, -70, 21, -62, 57, 69, 19, 72, 40, -24, -17, 58, -28, 26, -21, 58, -64, 57, -65, -64, -68, -46, 35, -69, -46, -70, 32, -56, -65, 58, 3, -70, -52, -33, 10, -34, 23, -9, -37, 34, -15, 43, -41, -52, -63, -26, -10, -30, 37, 2, 34, -37, -18, -58, -21, 66, -30, -29, 6, -74, 39, 16, 73, -18, -16, 59, 12, 55, -1, 69, 72, 55, 48, -8, -24, 59, -26, 70, -49, 49, -16, 51, -34, -65, -40, -3, 29, -23, 50, 26, -4, 42, -8, 53, -62, 56, 26, -56, 43, 12, -18, 63, -21, -29, 31, 7, 31, -5, -38, -35, -51, -59, -48, -59, 27, -58, 60, 12, -72, 48, 27, 30, -68, -56, -16, 53, 27, 13, -63, -19, -37, -28, 52, 12, 45, -46, 71, 30, 61, 14, -25, -2, 8, 35, -66, -49, 72, -17, 32, -52, -32, 7, 54, 72, -34, 65, -46, 14, -51, 0, 4, 11, 70, -33, 11, -38, 17, -67, -38, 60, -31, -62, -31, -53, 41, -10, -61, 63, 60, 34, 46, 16, 61, -47, 8, -28, -9, -1, -59, 0, -47, -20, 50, 13, 12, 64, 23, -60, 55, -10, -70, -48, -27, -22, -19, -12, -48, -39, -63, 6, 70, -47, -61, -34, -28, -60, 60, 72, -18, -56, -53, 7, 74, -37, -46, 27, -19, -62, -64, 7, -11, 72, -27, -68, -66, -8, -6, 14, -72, 44, 49, -4, 71, -48, -37, -2, -62, 68, 24, 40, 46, -16, -51, 61, 49, -44, -51, 17, 51, -18, 54, -33, 20, 1, -48, 27, -56, -12, -56, 45, -29, 11, -67, 49, 24, -31, -1, 31, -37, -74, 52, -43, 45, -23, 24, -21, -1, 60, 31, 14, 39, 27, 47, 59, 7, -3, -48}

#define ADD_CONV2D_KERNEL_0_SHIFT (9)

#define ADD_CONV2D_BIAS_0 {-31, 87, 45, 107, 57, 98, 36, -90, 71, 8, 46, -17, 77, 106, 110, 25}

#define ADD_CONV2D_BIAS_0_SHIFT (8)

#define BATCH_NORMALIZATION_KERNEL_0 {81, 81, 81, 80, 81, 80, 80, 80, 80, 81, 81, 79, 80, 80, 80, 80}

#define BATCH_NORMALIZATION_KERNEL_0_SHIFT (7)

#define BATCH_NORMALIZATION_BIAS_0 {123, 122, 122, 122, 123, 122, 122, 124, 122, 122, 122, 122, 122, 123, 122, 123}

#define BATCH_NORMALIZATION_BIAS_0_SHIFT (7)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define ADD_CONV2D_OUTPUT_SHIFT 0
#define BATCH_NORMALIZATION_OUTPUT_SHIFT 1

/* bias shift and output shift for each layer */
#define ADD_CONV2D_INPUT_LSHIFT (ADD_CONV2D_KERNEL_0_SHIFT-INPUT_1_OUTPUT_SHIFT)
#if ADD_CONV2D_INPUT_LSHIFT > 0
#define ADD_CONV2D_MODE 1
#define ADD_CONV2D_INTER_SHIFT ADD_CONV2D_KERNEL_0_SHIFT
#define ADD_CONV2D_INTER_LSHIFT (ADD_CONV2D_KERNEL_0_SHIFT-INPUT_1_OUTPUT_SHIFT)
#elif ADD_CONV2D_INPUT_LSHIFT < 0
#define ADD_CONV2D_MODE 2
#define ADD_CONV2D_INTER_SHIFT INPUT_1_OUTPUT_SHIFT
#define ADD_CONV2D_INTER_LSHIFT (INPUT_1_OUTPUT_SHIFT-ADD_CONV2D_KERNEL_0_SHIFT)
#else
#define ADD_CONV2D_MODE 0
#define ADD_CONV2D_INTER_SHIFT ADD_CONV2D_KERNEL_0_SHIFT
#define ADD_CONV2D_INTER_LSHIFT 0
#endif
#define ADD_CONV2D_OUTPUT_RSHIFT   (ADD_CONV2D_INTER_SHIFT-ADD_CONV2D_OUTPUT_SHIFT)
#define ADD_CONV2D_BIAS_LSHIFT   (ADD_CONV2D_INTER_SHIFT-ADD_CONV2D_BIAS_0_SHIFT)
#if ADD_CONV2D_OUTPUT_RSHIFT < 0
#error ADD_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if ADD_CONV2D_BIAS_LSHIFT < 0
#error ADD_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif
#define BATCH_NORMALIZATION_OUTPUT_RSHIFT (ADD_CONV2D_OUTPUT_SHIFT+BATCH_NORMALIZATION_KERNEL_0_SHIFT-BATCH_NORMALIZATION_OUTPUT_SHIFT)
#define BATCH_NORMALIZATION_BIAS_LSHIFT   (ADD_CONV2D_OUTPUT_SHIFT+BATCH_NORMALIZATION_KERNEL_0_SHIFT-BATCH_NORMALIZATION_BIAS_0_SHIFT)
#if BATCH_NORMALIZATION_OUTPUT_RSHIFT < 0
#error BATCH_NORMALIZATION_OUTPUT_RSHIFT must be bigger than 0
#endif
#if BATCH_NORMALIZATION_BIAS_LSHIFT < 0
#error BATCH_NORMALIZATION_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t add_conv2d_weights[] = ADD_CONV2D_KERNEL_0;
static const nnom_weight_t add_conv2d_w = { (const void*)add_conv2d_weights, ADD_CONV2D_OUTPUT_RSHIFT};
static const int8_t add_conv2d_bias[] = ADD_CONV2D_BIAS_0;
static const nnom_bias_t add_conv2d_b = { (const void*)add_conv2d_bias, ADD_CONV2D_BIAS_LSHIFT};
static const nnom_addconv_parameter_t add_conv2d_parameter = { (const void*)NULL, ADD_CONV2D_MODE, ADD_CONV2D_INTER_LSHIFT};
static const int8_t batch_normalization_weights[] = BATCH_NORMALIZATION_KERNEL_0;
static const nnom_weight_t batch_normalization_w = { (const void*)batch_normalization_weights, BATCH_NORMALIZATION_OUTPUT_RSHIFT};
static const int8_t batch_normalization_bias[] = BATCH_NORMALIZATION_BIAS_0;
static const nnom_bias_t batch_normalization_b = { (const void*)batch_normalization_bias, BATCH_NORMALIZATION_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(16, 16, 16), nnom_input_data);
	layer[1] = model.hook(AddConv2D(16, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &add_conv2d_w, &add_conv2d_b, &add_conv2d_parameter), layer[0]);
	layer[2] = model.hook(BatchNormalization(16, &batch_normalization_w, &batch_normalization_b), layer[1]);
	layer[3] = model.hook(Output(shape(16, 16, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}