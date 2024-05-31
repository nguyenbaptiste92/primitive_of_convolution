#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {5, -29, -41, 11, 65, -75, -55, -5, -63, 63, -75, -90, -83, 25, -67, -51, 73, 50, -87, 3, 46, 24, 77, 39, -83, -98, -28, 88, -32, -71, -65, -62, -66, 7, -68, 45, -1, 31, 49, -64, 93, -84, -64, 58, -99, -7, -18, -67, -75, -55, -10, 86, -45, 31, -87, -29, 88, 25, -23, 52, -75, 7, 93, 99, 87, 75, -7, 80, 79, 43, -77, 99, -35, -96, -81, 9, 12, 4, 88, -51, -17, -13, 6, 3, -77, -39, 12, -81, -31, 39, -82, -75, -94, -26, 35, 12, 86, -40, 73, 11, 72, 67, 100, -90, 99, 15, 20, -64, -91, 74, 0, -99, 87, -42, -73, -89, 1, 15, 35, -47, -27, -36, -80, -39, -9, 73, -92, 92, 66, -93, -52, 0, -4, 53, -40, -78, -18, -57, 28, -7, 51, 3, -55, 48}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {11, -10, 104, 92, -87, 45, 105, -70, -60, 91, -75, -69, -46, 13, 22, -56}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {66, -82, -35, 65, 85, -69, 89, 69, 59, 13, 90, -75, 105, -26, 94, -86, 12, -85, 44, 12, 22, 52, 60, -94, -1, 11, -8, 51, 8, 65, 13, -40, -17, 108, 70, 66, -75, 108, -9, 33, 98, -18, -52, -39, -3, 48, -48, -95, 66, 78, 59, 58, 59, -70, -34, -95, 85, -60, -5, 51, -32, -22, 82, 104, -94, 88, 89, -12, -92, 66, 104, 46, 40, -104, -26, 34, 3, -33, 61, 65, -47, -47, 28, -85, -59, -103, -82, -53, 68, 94, 64, -89, -51, -99, 54, -3, 98, -86, -49, 70, 57, -36, 52, 108, -99, 86, -99, -108, 42, 62, -27, 67, -59, 82, -87, 111, 15, -35, 7, -16, -105, 50, -80, 50, -92, -67, 29, -46, 20, -99, 2, 80, -83, -69, 37, 94, 21, -93, 103, 59, -23, -84, -45, -31, -67, -25, 6, -3, -103, 31, -100, -83, -41, -35, -7, 1, 45, -11, -28, -47, 30, -99, 88, 61, 52, 40, -73, 63, -52, -60, 22, -74, -92, -109, 59, 14, -95, -60, 4, -28, 17, 103, 26, -1, -11, 1, 68, 105, 38, -14, 29, 63, -3, 99, 57, 68, 110, 34, -71, 35, 95, -89, -18, 25, -83, -88, -91, 37, 77, -105, 64, 72, -97, -76, -24, -55, -93, -29, 91, -13, -1, -68, 51, 5, 93, 90, 9, -4, -48, 67, 45, 41, -66, -82, 46, 5, 78, 83, 62, 61, -93, -54, 52, -104, -95, 100, 97, 45, -27, -65, 33, 72, 53, 30, -85, -24}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {2, 78, -63, -88, 83, 9, 35, 65, -15, 94, 3, -20, 109, -63, 7, -100}

#define CONV2D_BIAS_0_SHIFT (8)


/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define DEPTHWISE_CONV2D_OUTPUT_SHIFT 6
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
