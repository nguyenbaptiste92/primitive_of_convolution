#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {79, -31, -88, -83, -100, -45, 48, -69, -58, -64, -66, -43, -28, -67, -17, -78, 50, 20, 78, -21, -63, -5, -7, 20, -97, 85, 49, 40, -75, -75, 13, 36, -50, -88, -18, -91, 98, -29, -85, 9, 58, -74, 59, 95, -22, -2, -74, 20, 58, -18, -86, -17, -25, -73, 79, 92, 73, -53, -65, -36, -82, 39, -10, 2, 101, 0, 27, 26, -70, -89, 61, -76, 51, 28, 25, -33, -81, -50, -16, 79, 62, -55, 96, 81, 57, -92, 50, -87, -45, 53, -74, 85, 16, 26, 82, 26, -37, 90, -98, -64, -76, 86, -82, -93, 98, -92, 37, 40, 20, 78, 53, 72, 84, -25, 85, -70, 85, 27, -49, -49, -30, -52, -64, 80, -94, -39, 93, -4, 73, -81, 87, -78, -71, -34, -96, -47, -21, -98, 78, 68, 29, -28, 57, 4}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {-66, -91, 75, -92, 93, -47, -41, -93, 58, 2, -57, -20, 103, -107, -50, 35}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {68, 15, -23, 4, -1, -56, 29, -107, 20, -49, -19, -19, 24, -106, 60, 20, -51, -74, 59, -14, 20, -76, 58, 49, 51, -21, -82, 16, 40, 70, -86, 102, -64, -12, -72, 43, -95, -101, -2, -30, -10, -54, -54, -47, -8, -57, 45, -90, -10, 28, 93, -13, -16, -83, -64, -109, 79, -29, 42, 16, -50, -7, -98, 108, -77, 52, -105, 100, 11, 71, -43, 47, -86, 11, 83, -102, 10, 74, -101, -58, -37, -21, 70, -91, -45, -43, 76, 13, -13, 60, -74, 10, 28, 80, 66, -69, -29, 50, -95, -96, 21, -67, -49, -22, 108, 25, -46, 24, 84, -48, -72, 110, 70, -7, 88, 38, -3, 38, -101, 89, -34, 72, -101, 49, 110, -33, -9, -101, 84, -76, 38, 72, -6, -45, -75, 6, -97, 107, -80, -48, -92, 71, -82, -34, -81, -86, 69, 95, 78, -110, 72, -85, -85, -19, -83, -96, 77, -8, 45, -8, 64, -87, -15, -38, 78, -54, -100, -106, 13, -90, 67, -56, 59, -62, 34, -66, 49, -43, 48, -3, -10, -43, -73, -19, 22, -40, -108, -79, 79, 9, 80, 22, 57, -73, 77, 22, 97, -10, -103, -72, 31, 89, -55, 8, 52, 60, 62, -61, 74, -81, 79, -16, -41, -85, 67, 103, -41, -41, -10, -1, 71, -81, 100, -33, -36, -50, 106, -40, 37, 80, 26, 90, -45, 38, -44, 13, -93, -72, 92, -67, -58, 43, -4, -59, 109, 52, 4, 1, 77, -7, 72, 50, 29, -48, 39, -18}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {56, -42, -13, -69, -70, -12, 84, -2, 85, -36, 63, 99, 5, 46, 81, 60}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 6
#define CONV2D_OUTPUT_SHIFT 5

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
static int8_t nnom_input_data[4096];
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(16, 16, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(16, 16, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
