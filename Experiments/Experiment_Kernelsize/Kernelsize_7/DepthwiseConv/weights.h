#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {57, -19, -86, -56, 48, 84, -55, -71, -66, 31, 48, 81, 10, -82, 78, 20, 30, -65, 72, -36, -8, 79, 28, -32, 52, -60, -46, 0, -58, 26, -18, -31, 85, 65, 13, 1, -82, 75, -29, -76, -66, 64, -64, 2, 4, 62, 15, 34, -8, 16, 22, 10, 47, -45, -86, 18, 43, -33, -57, -3, -59, 34, -24, -22, 28, 65, 21, 12, 1, -22, 55, -75, 61, 58, 12, 51, 36, 73, 21, -33, -44, -79, -78, 78, -76, 39, -40, -63, -3, 49, -51, 41, 6, 32, -71, 10, -73, -82, 53, 84, 4, 9, 10, 84, -21, 15, -69, -84, 63, 4, 57, 9, 2, -59, -70, -79, -46, -53, 32, -49, -51, -71, 27, 76, 2, -47, 14, 80, 17, -68, 62, 77, -25, -52, 1, -69, -21, 34, 34, -70, -34, 36, -35, 86, 76, -62, 2, -63, -34, 46, 73, 28, -42, 67, 5, 53, -29, 36, 1, -70, -63, -86, 45, 76, -65, 80, -35, -14, 45, -85, -76, 68, -54, -13, 82, -35, -35, 40, -57, 70, 1, -78, 3, -46, 85, 32, -27, 16, -13, 10, 16, -50, -84, -28, 55, -9, -59, -45, 56, 83, 5, -21, 54, -51, -38, 67, 81, -2, -32, -57, 24, -54, 18, 45, 72, -28, 20, 17, -23, -32, 62, -17, 74, -15, 45, 75, 77, -86, 68, 10, -44, 58, 61, -5, 11, 14, -33, -46, 77, -67, 6, 26, 47, 5, 78, 64, 33, -85, 82, -49, 23, -22, 73, 24, -15, -44, -84, -81, 57, -34, -28, 66, 71, -4, -52, -35, 24, 5, -10, -64, 67, 61, 32, -77, 61, 15, -20, -48, 53, 45, -20, -77, -54, -75, 79, 54, -43, 47, 84, 45, -23, -12, -31, 54, -5, 60, -16, 30, -80, -57, 33, -10, 86, 17, 6, -69, 36, -40, -11, -72, 72, -72, 86, -21, -7, 34, -5, 85, -12, 71, 24, -29, 13, -5, 7, -2, -21, -65, -85, 21, -14, 53, 30, -15, -33, -64, 16, -50, 7, -15, 76, 50, 75, -20, 40, 85, 4, 57, 25, 78, -56, 67, 0, 70, -41, 7, -70, -22, -6, -47, -61, -45, -7, -55, 64, -42, 6, 81, -64, -55, 72, 83, -47, -9, 14, 4, 2, 33, 0, 18, 28, 84, 77, -78, -81, 56, -50, 76, -21, 83, -69, 3, -45, -26, 26, 32, 36, 82, -63, -72, -6, 84, -15, -67, 74, 75, 84, -68, -35, -51, 51, -67, -69, 56, 58, 23, 84, -26, 15, 81, -63, -3, 65, -16, 47, 1, -30, -60, -50, -62, -86, 54, 22, -14, 58, -30, 1, -73, 86, -22, 28, 15, 86, 73, -17, -38, -50, 67, 75, 68, 18, -38, 85, -35, -52, 51, -36, 55, 7, 52, 58, 0, -28, -11, 19, -48, -79, 5, 75, 14, 5, 14, -71, 48, 59, -17, 50, -41, 46, 72, -84, -56, -15, 87, -9, -71, 16, 46, 43, -10, 2, 10, -78, 7, -31, 56, 84, -72, -46, 3, 66, -64, 76, 23, -4, 53, 45, 32, -19, -54, -22, 81, 30, -63, -29, -22, 44, -39, 14, 28, -63, -47, -77, -3, -55, -55, -21, 39, -83, -62, 47, 79, -49, 60, 55, 51, 36, -1, 48, 8, 76, -69, -16, -76, -69, -58, -5, 58, -47, 26, 47, 71, -48, 29, -27, -86, -67, 79, -24, 27, -53, -67, -2, 13, 32, -20, 14, -17, -53, 62, -66, -65, -81, -81, -3, -69, -74, 28, -44, -36, -48, 2, 17, 41, -86, 34, 50, -27, -48, 74, 46, -24, 36, 51, -77, 57, 57, 53, -8, -83, -8, -36, 52, 37, -64, 67, 13, -56, -16, -34, 2, -27, 13, -2, -63, 28, 80, -6, 31, 17, -16, 41, 57, -83, -16, -40, -46, -1, -80, 24, 10, -31, 36, -10, 20, -68, 49, -3, 47, 41, -71, -25, -78, -3, 83, -82, 16, -14, 78, -78, -43, -68, -53, 2, -67, 37, 87, 85, -85, 8, 40, 2, 57, -36, -25, -10, 26, 10, 28, 69, 48, 68, 76, -27, 4, -35, 47, -78, 82, -14, 1, 41, -73, 31, -79, -60, 1, 77, 18, -33, -86, -21, -17, -25, -4, -73, 45, 26, -38, 67, 68, 31, -11, -50, -28, 57, -70, -58, -21, 74, -23, -6, 5, 27, 29, 10, -60, 20, -15, -80, -63, -32, 13, -14, 48, 46, -69, 60, 58, -12, 31, 26, -78, -51, -67, 37, 16, 39, -66, 32, 1, 23, -13, 18, -46, -78, -57, -43, 60, -62, -74, 5, 76, -49, -36, 78, -31, 6, 63, 64, -57, 23, 55, 74, 11, 34, 16, -81, 37, 67, -22, 3, -75, -78, 32, 36, 56, 59, 44, 43, 17, 52, -41, -65}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (10)

#define DEPTHWISE_CONV2D_BIAS_0 {58, 14, 7, -19, -45, -106, -5, 22, 95, -88, -8, 85, 73, -106, 69, -54}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {108, -93, 13, -2, -61, 11, -35, -49, 26, 101, -99, 54, 65, -82, 87, 100, 104, -25, 50, 61, -27, -91, 43, -94, 30, -13, 99, -81, 12, 38, -19, -7, 38, 37, -104, 47, -89, 96, 46, -71, -56, 20, 44, -41, -65, -92, 25, 51, 48, 63, -12, 80, 3, -39, -75, 82, -79, -27, 80, 51, 16, 38, 5, -69, 22, 100, -51, -85, -72, 79, 65, 82, 24, -32, 7, 47, -26, 68, 73, 94, -76, -103, 46, -32, 22, -90, -14, -6, -90, 23, 103, -47, -80, 99, -38, -107, -3, -17, -19, -64, 36, 41, -109, 108, -42, 34, -46, 19, 6, 13, -56, -61, -84, -3, 55, -100, 92, 80, 104, -81, 29, 107, -44, 46, 60, 39, -52, 62, 31, 74, 37, -41, 59, -35, -84, -93, 83, -43, 58, -32, -59, -19, 94, 3, -37, -26, 80, -80, 8, -75, 32, 27, 105, 98, 54, -66, 4, 9, -105, 82, 5, 42, -94, 42, 85, 27, -17, -60, 17, -26, -109, 39, 16, 84, -76, 19, -83, 20, 58, 90, -103, 53, 76, -53, -5, -77, 12, -83, -96, 83, 68, 22, 105, 30, -38, 11, 82, 83, -107, 102, -52, -56, 71, -89, -23, -67, -101, 90, -53, -13, -15, 108, -60, 87, 39, 15, 93, 102, 26, 9, 54, -82, -93, 44, 102, -39, 73, 100, -66, 38, -32, 82, 92, 37, 33, -15, 5, -56, -99, -34, -38, -46, 47, -39, 75, -55, 69, -40, 53, 93, 14, 26, -38, -37, 25, 45}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {24, 68, 72, 66, -8, -102, 25, -49, 84, 1, -14, -19, 47, 14, -1, 94}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 7
#define CONV2D_OUTPUT_SHIFT 6

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
static int8_t nnom_input_data[16384];
static int8_t nnom_output_data[16384];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(7, 7), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
