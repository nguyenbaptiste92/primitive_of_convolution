#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {38, -28, 59, -26, -54, 25, 57, -50, -2, -31, 40, -53, -41, 5, 11, -9, 52, -57, -70, 40, -26, 0, -72, -20, 34, -18, -14, 0, -5, 32, 4, 2, -42, 35, 56, 66, 46, -70, -65, -13, 44, 64, -12, 65, -40, 10, -51, -11, -8, 65, 47, 15, -59, 48, 55, -60, 22, 5, -28, 49, 2, -40, 64, 37, 11, 12, 50, 68, -35, -5, -44, 8, 27, -9, -12, -60, -29, 9, 9, 29, 26, 3, 32, 71, 35, 64, -26, -61, 22, 69, -17, 41, -56, -29, -59, -54, 15, -70, -13, 43, -63, -36, 45, -18, 65, -13, 29, 22, 42, -20, -52, -40, -12, 16, -1, -31, -23, -55, 28, -5, 50, -48, 14, 58, -49, 34, -37, 0, 58, -28, 16, -1, -38, -38, 18, -45, 18, 65, 70, -39, -18, 45, 33, -32, -62, 67, 21, -5, 71, -30, 4, 53, 7, 56, -21, 61, -61, 45, 24, -28, -71, -44, -69, 21, -58, 3, 4, 52, -42, -62, -34, -30, 66, 66, 5, -33, 27, 2, -52, 67, 14, 31, -58, 51, 49, 53, 11, -50, -14, 37, -17, -50, 8, 17, -12, 63, -12, 4, 51, 45, -45, -1, 23, 2, 39, -20, -12, 3, -11, -10, 67, 59, -5, 7, 47, -51, -25, -13, -65, 44, 0, -13, -69, 24, 51, 48, -50, 12, 7, 16, -1, 9, 44, -21, -50, 62, -33, -15, 3, -48, -20, 26, 45, -27, 57, 51, 56, 12, -37, 71, -62, -6, -70, -43, 20, 50, -46, 53, 35, -17, -39, -44, -5, -53, -14, 7, 39, 54, -37, 19, -52, 61, 11, 8, -46, 12, -12, -73, -28, -68, -53, 61, 70, -49, -70, 53, 27, -4}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {-58, -26, -64, 48, -39, 35, 23, -40, 0, 61, 45, -78, 75, 36, 4, -68, -12, -45, -28, 3, 12, 24, 23, -7, -58, 48, -57, 11, -55, -71, 63, 38}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-86, -48, 49, -18, -82, 84, -31, 24, 3, -6, -61, -18, -4, -7, -14, -61, -9, 27, -28, 51, 79, 89, -82, -54, 48, -75, 67, 48, -66, 7, -35, -58, 46, -8, -58, -23, -18, 1, -65, 39, 44, -78, -28, -14, 48, -41, 27, 17, -64, 66, -3, 23, 72, 15, 82, 79, 86, 65, -78, 71, 80, -1, -44, -15, 39, 15, 11, -50, 37, 41, -11, 88, -14, 4, -13, 63, -78, -77, 80, -26, -86, 66, -1, -46, -48, 56, 79, 41, -53, -34, 51, 69, -64, 72, 85, -63, 65, 88, -65, -17, -74, 0, -63, 73, -29, -75, -63, -88, -19, -12, -5, -76, -40, -64, -86, -33, -85, -46, 87, -34, -7, 9, 50, 22, -11, 38, 49, 70, 7, -30, -5, -14, -60, 89, -37, -90, -76, -28, 31, -15, -43, 90, 60, 35, 68, -17, 40, 1, -65, -79, 33, -43, -62, -58, 80, 35, -52, 20, 71, 78, 65, 49, -53, -33, 14, 76, 89, 69, 2, 74, -61, 4, 9, 16, 58, -70, -51, 39, -86, -35, 42, 11, -88, -75, -10, 5, -67, -47, -77, 17, 18, 12, 75, 84, 90, 72, 73, -20, -38, 87, 37, -25, 41, -57, -28, 19, -66, 32, 19, 89, 21, 17, -4, 56, 66, -62, 30, -56, 1, -64, -55, 44, 28, 69, 10, -20, -75, 71, 76, 72, 81, -68, -45, -60, 10, -84, 22, 18, -45, -47, -71, -59, -15, -43, -6, 62, -5, -61, -73, -43, -45, 56, 78, 12, -69, 53, 12, 60, 24, -89, -46, 24, -67, 26, 66, -37, -15, -45, 3, 31, 35, -23, -57, 60, -62, -39, -78, 86, -63, 22, -65, 5, 67, -63, -22, 60, -47, 31, 4, 73, -45, 89, 10, 1, 76, -72, 29, -85, -77, -10, 33, 51, -19, 36, 28, 10, -73, 23, 73, -60, 57, -55, 50, -12, 69, -62, 75, 36, -19, -53, -7, -3, -21, 61, 63, -85, 48, 74, 16, -79, 46, -60, 46, 41, 47, -46, 29, -26, -54, 40, 87, -33, -17, 6, -9, -13, 73, -61, 47, 51, 21, 2, 30, -59, -15, 61, -68, -41, 83, 75, 17, -45, 57, 76, 71, -14, -74, -70, -40, 7, 47, -75, -22, -3, -22, -28, 59, 87, -77, -3, 27, 67, 62, 84, 69, -41, 57, -49, 63, 7, -14, -75, -70, -11, 65, -4, -74, -85, 83, -57, 87, 65, 1, 25, -71, 74, -89, 0, 52, -16, -24, -40, -26, 24, 85, 64, 68, -78, 59, 70, -30, 86, 23, -27, 33, 81, -47, -9, 35, 90, 73, 51, -1, -53, 37, -14, 13, 60, 43, -70, -83, -27, -65, 26, 0, -43, 75, -22, -64, -72, 30, -4, 24, -10, 35, -41, 76, -62, 85, 73, 36, 32, -11, -17, -39, -1, -85, 71, 36, -59, -35, 11, -74, 0, 6, -67, -78, -42, 85, -89, 76, -65, -47, -64, -28, -6, -10, -39, 49, -37, -73, 40, -80, 36, -10, 39, 56, -5, -31, 25, -26, 18, -38, 13, -36, 35, -29, -27, 79, 38, -7, 64}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-45, -38, -37, -2, -25, 30, 103, -16, 39, 55, -105, -20, 59, -103, -97, 98}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 7

/* bias shift and output shift for each layer */
#define DEPTHWISE_CONV2D_OUTPUT_RSHIFT (INPUT_1_OUTPUT_SHIFT+DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT-DEPTHWISE_CONV2D_OUTPUT_SHIFT)
#define DEPTHWISE_CONV2D_BIAS_LSHIFT   (INPUT_1_OUTPUT_SHIFT+DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT-DEPTHWISE_CONV2D_BIAS_0_SHIFT)
#if DEPTHWISE_CONV2D_OUTPUT_RSHIFT < 0
#error DEPTHWISE_CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DEPTHWISE_CONV2D_BIAS_LSHIFT < 0
#error DEPTHWISE_CONV2D_BIAS_LSHIFT must be bigger than 0
#endif
#define CONV2D_OUTPUT_RSHIFT (DEPTHWISE_CONV2D_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_OUTPUT_SHIFT)
#define CONV2D_BIAS_LSHIFT   (DEPTHWISE_CONV2D_OUTPUT_SHIFT+CONV2D_KERNEL_0_SHIFT-CONV2D_BIAS_0_SHIFT)
#if CONV2D_OUTPUT_RSHIFT < 0
#error CONV2D_OUTPUT_RSHIFT must be bigger than 0
#endif
#if CONV2D_BIAS_LSHIFT < 0
#error CONV2D_BIAS_LSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t depthwise_conv2d_weights[] = DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0;
static const nnom_weight_t depthwise_conv2d_w = { (const void*)depthwise_conv2d_weights, DEPTHWISE_CONV2D_OUTPUT_RSHIFT};
static const int8_t depthwise_conv2d_bias[] = DEPTHWISE_CONV2D_BIAS_0;
static const nnom_bias_t depthwise_conv2d_b = { (const void*)depthwise_conv2d_bias, DEPTHWISE_CONV2D_BIAS_LSHIFT};
static const int8_t conv2d_weights[] = CONV2D_KERNEL_0;
static const nnom_weight_t conv2d_w = { (const void*)conv2d_weights, CONV2D_OUTPUT_RSHIFT};
static const int8_t conv2d_bias[] = CONV2D_BIAS_0;
static const nnom_bias_t conv2d_b = { (const void*)conv2d_bias, CONV2D_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[32768];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 32), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
