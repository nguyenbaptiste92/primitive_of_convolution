#include "nnom.h"

#define CONV2D_KERNEL_0 {-14, 68, -68, 32, 5, -27, -38, 52, 45, 16, 20, -25, -55, -52, 54, 44, -2, -14, -21, -35, 59, -56, -41, 65, -39, -49, 52, 47, -23, 0, -61, 35, -43, 16, -58, 50, 15, -3, -16, -18, -35, -51, -51, 56, -72, 30, -12, 32, 34, 58, 26, -9, -9, -61, 48, -51, -56, -6, -28, -18, 68, -33, -60, -20, 54, 9, -54, -18, 13, -57, -10, -36, -3, -31, -50, -66, -59, 9, -34, 37, 14, -21, -33, -12, 4, -48, -57, 37, 16, -43, 56, 53, 71, -4, -41, 46, -21, -1, 17, -55, 18, 1, -20, 10, -36, 72, 32, -9, -51, 28, 24, -55, 50, -49, 10, -42, -26, -72, -4, -13, 14, 36, -70, -6, -14, 60, -67, 60, -14, 23, -6, 72, -39, 4, -18, -31, -47, 23, 22, -16, -19, -46, -44, -25, 55, -30, 39, 0, -35, 26, -37, -43, 24, 53, -39, -4, 40, 57, 21, -9, -69, 47, 66, 35, -27, -21, -5, 24, -31, 72, 26, 47, 24, -7, -19, 18, -56, -47, 36, -59, 35, -5, -55, 18, -22, 64, -31, 55, -6, -33, -25, -53, 47, 27, 71, 30, 4, -37, -18, 49, -36, 18, 14, 49, -31, 67, -19, 26, 7, 65, -22, -5, 49, 57, -28, 67, -27, -51, -17, -50, 7, 23, 54, 11, 72, -63, -61, 35, -33, 3, 58, 43, -73, 37, 5, -3, 8, 72, 60, 39, 43, 25, 6, -62, -65, -31, 72, -9, -60, 49, 20, -56, -1, -70, 34, -1, -58, -51, -8, 36, 55, -66, -37, 13, 31, 69, -64, 0, 62, 63, -5, -73, 51, -38, -57, 31, 11, 35, 17, 61, -69, -4, 0, -4, 73, -41, -26, 5, -67, -55, 11, -58, 34, -70, 8, 60, 24, -46, -32, -12, 20, 43, -45, 12, -56, -30, 38, 16, 55, 23, -19, 52, -26, 60, -2, -53, 26, 72, -34, -29, 46, 63, -67, 24, -69, 72, 28, -59, 0, -13, -41, 30, 1, -10, -41, -30, 5, -41, 45, 43, 57, 68, 23, 54, 29, 61, -69, -16, -58, -9, -57, 11, -71, -69, -63, -65, 11, 63, 48, 8, 43, 48, 35, -30, -2, -41, -57, -63, 46, 3, -57, -9, 50, -34, 66, -26, 53, -53, -60, -56, -8, 17, -11, -18, -18, 59, 35, -2, 11, 55, 72, 32, -12, 41, 26, -53, -69, -43, 21, 40, -41, -43, 14, 13, -37, 12, -58, -23, -33, -49, -5, 2, -3, 68, 56, 46, -53, -52, 49, -53, -45, 20, 42, -5, -27, 46, -12, 59, -33, 34, -13, -39, 70, -51, 9, -30, -20, 41, -60, 7, -37, 47, -70, -14, 61, 15, 68, -10, -36, 42, 69, -69, -66, 53, -66, 5, -6, -27, -37, 9, 12, -65, 60, -42, 56, -54, 50, 43, 70, -33, -20, -2, 56, 64, 72, -60, -45, 67, 25, 28, -18, -43, 0, 27, -37, -29, -34, 16, 31, 36, -24, -35, -39, 39, -3, -64, -41, -41, -53, 70, 49, -6, -74, 49, 28, -49, -48, -59, 69, -69, 55, 61, -32, 33, 40, 28, -2, -32, 65, 62, 5, -22, -33, 45, 8, 28, 13, -16, 34, -39, 45, -42, -54, 72, 34, -59, 67, -29, 26, 25, -73, -68, -10, -36, -58, 18, -41, 28, 49, -68, -30, -62, -3, 19, 74, -70, 0, 35, -59, -27, -26, 59, -55, -55, -36, -62, 49, -68, 8, 51, 10, 53, 6, -13, -52, 7, 49, 41, -35, 55, 43, -21, -6, -36, -49, 34, -2, 61, -27, 17, -2, 71, 34, -54, 65, 43, 11, 32, -71, -26, -45, -16, -4, 23, 66, -55, -48, -19, 11, 39, 3, -11, -12, 35, -71, -48, 50, 57, -68, -4, 14, 39, -67, 57, -34, -32, -57, 47, -49, 22, 48, -2, -69, 60, -29, 73, -7, -41, -35, 16, 67, -27, 33, -26, -72, -35, 51, 51, 48, -3, 27, 57, -43, -58, -33, -15, 13, 66, 26, -43, -73, -69, 8, -28, -18, -57, -57, 36, 51, 45, 15, 65, -5, -8, -19, 10, -68, 68, -70, 16, -10, 1, -3, 36, 30, -41, 29, -24, -28, 56, -56, 39, -38, 12, -15, 44, -46, -71, -47, -52, -10, -53, -23, -52, -27, -42, 50, 49, -2, 13, -28, -35, -48, 27, -33, 46, 37, -38, 41, -27, -24, 0, 39, -38, 12, -58, -58, -54, 42, 57, 33, -59, 42, -38, 71, -10, 15, 13, -70, 41, 63, 57, 48, 35, 21, 23, 59, 32, -72, -42, -23, -67, -33, 8, -55, -54, 72, 18, -4, 28, -32, -14, 9, -20, 18, -36, 55, -70, -15, 36, -21, -73, 0, 25, -34, 65, 51, -6, -69, -48, 63, -1, 56, 42, 3, -1, -57, 38, 9, -21, 22, 10, -65, 21, 71, -63, -74, 69, 58, -29, 49, 28, 51, -56, -55, 5, -23, -49, -45, -52, -27, 0, -62, 67, 37, -4, -15, -33, -18, -31, 63, 58, 54, 68, -39, 53, -29, -68, -36, 40, -59, -63, 58, 59, 49, -57, -21, 51, 67, 8, -56, -5, 42, 1, 16, -42, -17, -52, -1, -57, 55, -54, -72, -69, 15, 7, -15, 58, -37, 1, 32, -12, -12, -10, -22, 34, 27, 2, 8, -7, 27, -55, -41, -29, -39, 55, -2, 56, 64, -57, 37, 7, -42, 8, -23, -69, 26, -38, -7, 26, -29, 36, 61, -42, 0, 72, -22, 59, -47, -13, -15, 12, -9, -31, -44, 51, -20, 73, -30, 66, -16, -6, -67, -26, -27, -11, 70, 68, 59, 23, -10, -44, -47, -4, 38, 71, 26, -44, -72, -28, -33, 20, 28, 72, 62, -50, -49, 21, -28, 56, -18, 25, -34, -2, -60, 73, 42, 10, -50, -7, -48, -22, -5, -27, -28, -46, 36, -16, 21, -47, 70, 26, 15, -21, -74, -38, -43, -4, -17, -16, 55, 38, 64, -13, -55, 37, -42, 17, 13, -2, -38, -65, -46, -5, -42, 57, -29, -63, 43, -1, 21, -55, 35, 17, -46, 42, -70, 20, -47, -23, 51, -22, 65, -64, 48, 35, 59, -52, -30, 72, 40, -9, -10, -45, 7, -61, 44, -24, 60, -29, 31, -3, -34, 18, 62, 44, -15, 8, -63, -9, 9, -15, 5, -38, 6, 37, -49, 38, 61, 37, -25, 7, -36, 24, -14, -57, -12, 61, -45, 39, 57, -44, -12, 8, 5, 70, 63, -43, 0, 24, -32, 1, 35, -71, -53, -42, 71, 11, -45, 28, -10, 37, -46, -35, 25, -68, -54, -14, 5, 0, 13, 60, 10, 51, -38, -60, 20, 53, 35, -22, -6, 48, -61, -52, 11, 58, 33, 44, -31, 73, 5, -33, 6, -53, -8, 61, -25, -49, -58, -62, 64, -45, 5, 10, -59, -60, 46, 23, -71, -71, -33, 4, -43, -5, -43, -41, -70, -46, -53, -14, -57, -4, 42, 32, -7, -16, 37, -70, -23, -66, -11, -1, 68, -65, -18, -21, -6, -19, -34, 44, 51, -42, -55, 70, 16, -60, -54, 57, -34, -58, 16, 56, 8, 32, -30, 12, -15, -57, -17, 56, 60, 20, -19, 46, -31, 63, -52, -54, -20, -22, 15, -40, -49, 6, -43, 37, 41, 42, -48, -41, -62, -25, 72, 48, 44, -47, -31, -59, 66, 52, 22, 8, 57, 32, -69, -6, -64, 2, -29, 38, -2, 16, -67, 4, 7, 24, 65, 26, 27, 25, 39, -32, -1, 66, 41, 71, -1, 15, 7, -8, -21, 71, -55, -51, 15, -16, 56, 61, -29, 30, -3, -35, 7, -34, -39, 26, -39, 40, 39, -24, -68, 27, -31, 30, -72, -15, 47, 18, -70, 72, 54, 21, -72, 13, 40, 28, -15, 49, -32, -23, 46, -69, -70, 66, -71, 67, 41, 68, 33, 13, 34, -64, 51, 63, -17, -42, -3, 57, -4, -29, -10, 52, -20, 33, -28, 61, 71, 19, 21, -69, 63, 61, -18, -19, 43, -72, 1, -56, 56, -6, 45, 27, -46, 3, 45, -57, 30, 59, 19, 51, -67, 50, -60, -22, -58, 25, 64, -31, 4, 37, -40, 15, -20, -51, -3, 54, 49, 34, 18, -32, 39, 32, -46, 73, 59, 70, -34, 64, -17, 22, 30, 60, 32, 46, -17, -41, 36, 64, -55, 71, -27, 29, 55, 72, 59, -46, 4, 57, 8, 59, 25, 33, 9, -58, -7, -32, -31, -68, -34, -63, 46, -7, -45, 50, -70, 34, 27, -22, -38, 13, 66, -26, 33, 73, 44, -24, 19, -33, 37, 38, 46, 48, 41, -21, 63, 30, -50, -25, -59, -47, -9, -42, 43, 62, 0, -42, 41, -43, 16, -45, 47, -70, 54, -60, 40, 36, -44, -69, 15, -67, -3, -29, 60, 10, 27, 18, -20, 4, 49, -33, -8, -20, 47, 29, -57, -23, 45, 56, 30, 55, 45, 54, 54, -43, -30, -25, -28, -48, 56, 32, 60, -3, -4, -61, -43, 70, 57, 26, -2, -73, 40, 9, -24, 5, -37, -37, 40, -50, -9, -29, 3, 52, -25, 70, 21, 68, 31, 15, 44, 54, -27, -7, -26, 18, 40, 28, -69, 12, 34, 50, 13, 49, -20, 21, 56, 40, -57, 34, 59, -7, -60, 11, 8, 49, 56, 16, -50, 18, 57, 42, -69, 74, -56, 58, 28, -59, 13, 70, 4, -66, -14, -33, -8, 2, 46, 31, -60, -23, -13, -42, 65, -52, -8, 9, -69, -68, 50, 40, 19, 17, -51, -45, -11, 15, 22, -30, 61, -29, -1, 23, 36, 45, -55, -11, -68, 72, 48, 4, -61, 34, -18, 52, -47, 68, -20, -32, 54, 26, 37, -21, -19, 66, 44, 28, -51, 33, 52, 25, 28, -25, -44, 10, 29, -27, -21, 34, -59, 8, 41, 40, -47, 34, 43, -34, -7, 2, 67, 51, 67, 67, -50, 19, -22, 27, 12, -23, 37, -73, 51, 51, 30, -51, 57, 19, -67, 39, -60, -74, -45, -31, -63, 27, 60, 13, -42, 14, -45, -72, -21, -5, -45, 11, 40, -64, -35, 46, -51, -41, 30, -60, 40, -32, -20, -43, 53, 64, -20, -50, -64, -40, 50, 11, -65, -48, 35, 73, 48, 35, -63, 19, -68, -43, -9, -65, 27, 37, -40, -15, 43, 69, 12, 9, -1, 26, 60, -48, -23, 67, -39, 63, 51, 39, -32, -36, -25, -54, -50, 61, -42, 49, 23, 51, 35, -49, 66, -66, -6, 65, 28, 39, -34, -36, -18, -72, 19, 53, -20, 35, -7, 6, -44, -45, 11, -49, 56, -48, 1, 63, -42, 11, -50, -74, 67, 33, 71, -20, 55, -10, -7, -24, 71, 72, -15, -58, 27, 33, -6, -8, 28, 56, -57, -42, -39, -44, 68, -38, -34, -48, 7, 47, 17, -47, 15, 25, 73, -56, 37, 66, -23, -68, 25, -71, -11, 0, -29, -3, 53, -3, 67, -38, 9, 30, 2, 22, -60, -20, -65, 14, -54, -31, -44, -7, -60, 51, 45, -48, 70, 3, 15, 46, 10, -36, 22, 27, -51, -59, 67, -32, -2, -46, -26, 66, 7, -26, 49, 31, -69, 25, 35, 7, 5, -43, 31, 1, 2, 32, -43, -5, 5, -27, 0, -66, -30, -7, 2, 31, 9, -14, -6, 31, -71, 31, 62, -10, -61, 27, -72, -22, 22, -39, -61, -69, -16, -63, 12, 66, -39, -41, -15, 68, 70, -30, 18, 8, -50, -38, -29, 37, 23, 24, -65, -61, -50, -13, 58, -36, 64, 68, -65, 27, 62, -71, 59, -73, 54, -44, 67, -30, -15, 7, 10, -48, -54, -30, -47, -47, 30, 72, -46, -41, 16, 2, -15, -38, -36, -54, 8, -3, 13, 69, 42, 31, 28, 32, 41, -9, 28, -33, 28, -45, -20, -62, 35, -13, -29, 11, -30, 66, 52, 50, -24, -61, -29, 71, 66, 2, 4, -58, 5, -44, -22, -13, -35, 49, 6, -18, 34, 8, 4, 2, 3, 40, 2, -26, 52, 57, -73, -35, 45, 21, 65, 28, 58, -37, 28, -32, 19, 29, 34, 52, -2, 36, 21, 38, 1, -70, -51, -12, 68, 70, 17, -35, -11, -40, 63, 26, -49, 6, -13, -10, 60, -15, 52, -30, -49, 36, 39, 52, -47, -12, -19, 36, 46, -67, -25, 25, 21, 33, 29, 1, 68, 61, -13, 54, -9, -16, -71, -1, -65, 29, 8, -60, -35, -54, 37, 32, 43, -70, -73, 64, 36, 1, -72, -21, -57, -42, -49, -5, 36, 38, -47, 10, 37, -62, 1, -33, 54, 19, 2, 23, -19, 23, 64, -40, -26, 59, -1, -58, 4, -32, -8, 69, -73, 23, -65, -70, -1, -69, -33, -26, -63, 30, 21, -20, 24, -38, -20, 42, -43, -39, 24, -61, 53, 19, -11, -40, -51, -31, -32, 30, -16, 65, -29, -49, -35, -18, -70, 55, -70, -73, -9, -12, 58, -3, -52, -54, -58, -67, -5, -14, -63, 34, 66, -24, 49, -4, 61, -48, -32, 72, 66, -41, -59, 41, 29, 13, 31, 41, -59, 19, -44, 67, -37, -44, 8, -63, 56, 24, -53, -68, -50, 21, 37, -29, -9, -6, 59, -58, 17, -56, -50, 27, 47, -27, -9, 36, -22, 2, 17, 47, -6, 24, -32, -26, -36, -18, -30, 34, 53, 47, 48, 32, -59, 45, -19, 33, 55, 33, -31, 0, -54, -23, -65, -43, -61, 30, -59, -53, 0, -20, 5, -58, -69, -74, 17, 62, 20, -50, -59, 33, -33, -17, -38, -49, 58, -11, -32, -65, -32, 36, -5, 33, -69, -40, -44, -56, 16, -41, 30, -33, 25, -1, 54, 68, 61, 64, 45, -59, -69, 61, -43, 22, 59, 58, -68, 19, 6, 71, -22, -13, 0, 68, -29, -13, -36, 60, 67, 2, -58, -27, 46, 60, -67, 2, 64, 1, -9, 9, -57, -60, 67, -24, -5, 11, -1, -71, 54, 24, 7, 42, 48, 63, 59, -61, -53, 8, -32, -27, -45, 65, -53, -34, 29, 70, -64, 59, -2, -2, 22, 29, 0, 10, 53, -40, -57, 3, 34, -40, -8, -73, 10, 31, 57, -67}

#define CONV2D_KERNEL_0_SHIFT (9)

#define CONV2D_BIAS_0 {-22, -45, 49, 60, -17, 55, 56, 44, -89, 106, -25, 17, -95, -89, 60, 41}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 5

/* bias shift and output shift for each layer */
#define CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_OUTPUT_SHIFT)
#define CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_BIAS_0_SHIFT)
#if CONV2D_OUTPUT_RSHIFT < 0
#error CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if CONV2D_BIAS_LSHIFT < 0
#error CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t conv2d_weights[] = CONV2D_KERNEL_0;
static const nnom_weight_t conv2d_w = { (const void*)conv2d_weights, CONV2D_OUTPUT_RSHIFT};
static const int8_t conv2d_bias[] = CONV2D_BIAS_0;
static const nnom_bias_t conv2d_b = { (const void*)conv2d_bias, CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[9216];
static int8_t nnom_output_data[9216];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	new_model(&model);

	layer[0] = Input(shape(24, 24, 16), nnom_input_data);
	layer[1] = model.hook(Conv2D(16, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[0]);
	layer[2] = model.hook(Output(shape(24, 24, 16), nnom_output_data), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}