#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {93, -32, -13, 92, -72, -88, 32, 93, -91, -93, 52, -37, -91, 83, 3, 89, -78, -84, -64, 76, 83, 60, 75, 16, 5, 29, 72, 27, 36, 39, 44, -37, 9, 37, 7, 24, 52, -85, 4, 62, 59, 60, 68, -66, -66, -66, 14, 85, 70, 87, 85, 30, -15, 13, -47, -90, -40, -52, 55, 71, 45, 18, 39, -51, -12, -5, 16, -60, 60, -70, -76, 9, -96, -74, -80, 40, 35, -50, 74, 46, 59, 5, -30, 61, 64, 39, 25, 6, -54, -15, -20, -72, -37, 7, 67, -53, 97, 93, 57, 28, -85, 71, 27, -41, 83, -30, -80, -80, 48, -13, -101, 94, 44, -6, 80, -100, -41, 78, -23, -69, 30, -67, -9, -40, -68, -55, -53, 58, -5, -60, -83, -55, 61, 6, -100, -15, -74, 4, -40, 56, 27, 79, -50, 33}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {100, 107, -45, -4, 12, 87, -22, -39, 18, 0, -51, -73, -37, -71, 1, 95}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {-55, 5, 25, 68, -4, 56, 19, -13, 11, 19, 28, 3, 60, -3, 62, -35, -8, 48, -55, -29, -25, 51, -8, 38, -30, -34, -17, 11, 37, 62, 45, 4, -13, 31, 11, 40, -15, 22, 66, 14, -6, -11, 2, 32, 26, 22, 35, -27, 57, 44, -16, -61, -44, 4, -41, -37, 62, -18, -70, -70, -45, 69, -2, -33}

#define CONV2D_KERNEL_0_SHIFT (7)

#define CONV2D_BIAS_0 {-89, -84, 24, -56}

#define CONV2D_BIAS_0_SHIFT (7)


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
static int8_t nnom_output_data[4096];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[4];

	new_model(&model);

	layer[0] = Input(shape(32, 32, 16), nnom_input_data);
	layer[1] = model.hook(DW_Conv2D(1, kernel(3, 3), stride(1, 1), dilation(1, 1), PADDING_SAME, &depthwise_conv2d_w, &depthwise_conv2d_b), layer[0]);
	layer[2] = model.hook(Conv2D(4, kernel(1, 1), stride(1, 1), dilation(1, 1), PADDING_SAME, &conv2d_w, &conv2d_b), layer[1]);
	layer[3] = model.hook(Output(shape(32, 32, 4), nnom_output_data), layer[2]);
	model_compile(&model, layer[0], layer[3]);
	return &model;
}
