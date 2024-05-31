#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {-72, -19, 22, -3, -8, -18, 56, 75, -60, 53, 54, -50, -27, 36, -41, -8}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (7)

#define DEPTHWISE_CONV2D_BIAS_0 {97, -21, 15, 69, 76, 1, 79, 41, -38, -32, 21, -41, 68, 63, -90, -3}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {20, 42, 52, -55, -35, -55, 104, 76, -2, -20, -63, -69, -9, -71, 89, 70, -105, 28, -74, 102, -16, 31, -107, 29, -100, 87, -90, 45, -73, -35, -68, 5, -50, 86, -56, 84, -1, 67, 50, 93, 20, -108, 72, -19, -31, 50, 10, -110, 43, -61, 22, 27, 33, 5, 15, 20, 72, 96, -20, 68, 82, 53, 59, -66, -100, 33, -43, 98, 8, 94, -92, 34, 53, 86, -70, 16, -74, -82, -18, 76, -28, -50, -38, -33, -44, -100, 76, -90, -16, -105, -40, 92, 0, -46, -109, 10, -5, 95, 90, 87, -101, 107, -101, -61, 100, -65, -109, -110, 41, -77, 18, -69, -30, -94, 79, -4, 77, 60, 11, -63, -71, -79, -102, 48, 30, -105, 74, 95, 12, -50, -92, 56, -61, -70, -33, -17, 18, 110, -16, -67, 92, -11, 22, 62, 101, 2, -28, -77, -105, 90, 18, -25, -51, -16, 92, -90, 91, 81, 110, 107, -39, 40, 43, 33, 93, -79, -86, -77, -62, 0, -78, -48, 2, -43, 93, 44, -51, 86, 79, 61, -100, 44, 16, -12, 12, 40, -13, -70, -96, -74, 45, 46, 6, 11, 59, -34, 74, 105, 44, -21, -78, 95, -76, 33, -2, 14, -29, -52, 19, 57, -56, -17, -81, -101, 38, -62, -92, -54, -99, 79, 4, 91, -28, -60, -63, 35, 86, 10, 29, 41, 72, -9, -4, 16, 48, -13, -45, -41, 96, -97, -36, -103, -27, 22, -35, 50, -59, 4, -54, -34, 90, 41, 13, 41, 40, 24}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {-24, -28, -89, -32, 10, -75, 21, -18, 88, -12, -80, -35, -42, 76, 45, -57}

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
	layer[1] = model.hook(DW_Conv2D(1, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(16, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 16), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
