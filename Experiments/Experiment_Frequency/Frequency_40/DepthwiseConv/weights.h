#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {-58, -61, -16, 73, 5, -1, -59, -21, -55, 5, -84, -4, -2, 91, 88, 80, 9, -55, -63, -90, 71, 13, 69, 38, -14, -83, 82, 24, -96, -76, -20, 26, -9, 28, -33, 69, 86, -67, 87, 45, 93, -23, 101, 46, 77, -44, -5, 97, -9, -87, 76, 93, 47, 5, 10, 91, -85, 24, 63, -55, -68, -29, 1, -29, 85, -70, -29, 73, 53, 61, -86, 49, 32, 3, -14, 29, -58, 79, 39, -43, 50, -96, -93, 51, -20, -41, -4, 86, -10, -6, -98, -15, 98, 39, 70, 79, 15, 40, 21, 55, 47, -48, 65, 67, -96, -28, -21, 52, -43, 91, -61, 6, -25, -16, -47, -33, 22, 22, -72, 70, 96, -61, -41, -10, 14, -47, 86, -27, 54, -17, 66, -15, 25, 39, -86, 57, -30, 43, -43, -50, 86, 7, -33, 19}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {-66, 82, 108, -73, -81, -98, -33, -36, 105, 98, 97, 64, -74, -7, 85, 19}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {105, -18, -99, -73, 6, 106, -86, 66, -57, -1, 92, 28, 82, 103, 5, -102, -105, 10, -46, -39, -83, 94, -52, -60, 40, 2, 41, 40, -96, 13, 68, 23, -53, -31, 77, -50, 53, 86, 19, -38, 86, 109, 97, 98, -27, 20, 70, 107, 56, 87, 9, 92, -58, -68, 37, -91, 79, 40, -49, 69, 91, -29, 45, -48, 45, -40, -28, -13, 38, -15, 61, -86, 60, -110, 38, -20, 68, 52, -72, 55, -9, 103, -100, 96, -107, 11, -108, 26, -6, -77, 90, 51, 8, 47, -63, 59, -19, 2, 69, 5, 30, 111, 22, 85, -61, 35, -20, 60, 77, -41, -71, 70, 107, 30, 15, 12, -22, -73, -88, -82, -108, 25, -106, -90, 107, 10, 93, 93, 98, -38, 101, -52, 97, 98, -81, 34, 19, -8, -48, -102, -61, 18, -31, -65, 30, 86, -17, 89, -16, -72, -69, 32, -58, -61, 25, -82, 56, 24, 33, -76, -106, -93, -50, -111, -102, 73, -29, 64, 77, 29, -66, -76, 92, 70, -25, 29, -30, 109, -103, 15, 62, 15, -82, -97, 68, 3, 87, 53, 60, -42, 25, -61, -37, -66, -56, -11, 27, -103, 14, -103, -34, -79, -8, -10, -9, -11, -88, -1, 48, 41, 74, -63, -2, 39, 89, 64, -101, -65, -53, -11, 21, 40, 82, 82, 36, 13, 64, 99, 63, 56, 73, 43, -7, -29, 98, -6, 89, 92, 66, 14, -31, 25, -68, 103, -24, -62, 84, 109, -82, 64, 99, -64, -100, -109, 17, 88}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-75, -87, -6, -31, -27, 39, 100, -3, 107, 52, -66, -62, -67, -15, 67, 69}

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
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
