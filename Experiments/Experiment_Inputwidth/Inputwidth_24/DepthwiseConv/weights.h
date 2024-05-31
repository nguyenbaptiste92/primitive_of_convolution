#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {-76, 81, -74, -57, -18, 17, -5, 44, -31, -3, 29, 93, 59, -55, -12, -19, 9, -64, 23, 6, -20, 24, 64, 69, -89, 9, 81, -5, 50, 51, -17, 11, 59, -34, -89, 1, 26, -41, -54, -72, -31, -92, -68, -27, 85, -65, -57, 97, 52, -97, -75, 60, -17, 96, -9, 10, 41, -3, 93, 25, 64, 101, -2, -20, -82, 24, 16, -66, 40, -38, -80, -81, -46, 15, 2, -81, -48, 90, 38, -91, 50, -60, -28, -81, 56, 10, -43, 35, 35, 72, 32, 31, 14, 4, 33, -34, -62, 36, 52, 87, 72, -32, -32, 67, 52, -66, 71, -29, 19, 35, 89, 9, 63, -31, -96, 90, -39, -21, 58, 57, 54, -100, 56, -70, 32, 5, 100, 48, -10, 70, -6, 79, -89, -47, -19, 96, -91, -30, -47, -93, -39, -32, -41, 67}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {-96, -32, 79, -43, -65, 103, 32, 77, 57, -25, 68, 19, 57, -36, -42, -33}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-31, -9, 14, -74, -56, 80, 100, -12, -94, -71, 17, 74, -73, -51, -109, -95, -49, -26, -109, -33, 64, 59, 105, 21, -91, -86, -105, 79, -6, -67, 111, 91, 108, -107, 39, -80, -58, -40, -63, 44, 80, -65, -105, 77, -19, -2, 92, -87, 63, 49, -77, -18, 77, 38, -5, -50, 42, -36, -33, 41, -82, -107, 98, 4, 69, 35, -25, 19, -29, -102, 82, 68, 80, -75, 19, -11, -31, 20, 71, 109, 57, 37, -95, -54, 0, 15, -84, 41, 6, 50, 107, -83, 39, 26, -48, -30, 69, 19, -86, 61, -43, 90, -37, 80, -63, -67, -106, 103, -70, 39, 103, 95, -13, 56, -49, -104, 1, -15, 41, 35, -40, 11, 83, -35, -57, -1, -70, 33, -3, 46, 75, 36, -93, -69, 75, 56, 1, 102, -109, -22, 34, -79, -104, 28, 63, 71, 1, 82, -102, -94, -43, 25, 108, -74, 7, 26, -27, 25, 70, -98, 11, 12, 8, -31, 98, -9, 10, 106, 26, -101, -104, 3, 7, -89, -42, -32, 32, 37, 7, 82, -99, -53, -74, 72, -14, 27, 81, -27, -103, 64, 49, 88, -53, 51, 61, -54, -76, 28, 108, -41, -9, 12, 4, -12, 52, -98, -7, 30, -66, 93, 72, 7, 78, 35, -9, -42, -54, -99, 103, -46, -71, -88, -61, 43, -22, -95, -14, -24, 21, -107, 6, 52, -32, -69, -64, 31, 50, -7, 98, -71, 48, 81, -64, -80, 95, -108, -70, -22, -15, 43, -33, 66, 103, -97, -4, 16}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-44, 64, 96, 88, -26, 31, 84, -56, -58, 83, -83, -57, 82, 84, 80, -39}

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
static int8_t nnom_input_data[9216];
static int8_t nnom_output_data[9216];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(24, 24, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(24, 24, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
