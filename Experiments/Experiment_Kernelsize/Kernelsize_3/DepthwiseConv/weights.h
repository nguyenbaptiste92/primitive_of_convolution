#include "nnom.h"

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0 {78, 50, -6, 17, -19, 61, 39, -51, -50, -64, 91, -43, 100, 98, -56, 68, 15, 82, 100, -25, 82, 92, -52, 54, -46, -14, -100, -68, -5, 7, -79, -76, 75, 84, 99, -36, 44, -17, -20, -70, 84, 5, 1, -35, 94, -64, -78, -33, 13, 58, 65, -98, -57, 9, -42, 16, -77, -61, -20, 94, -7, 61, 83, 39, -18, -21, -85, 52, -37, 0, -67, 28, -40, 6, -56, 29, -9, -73, 16, 32, -3, -24, -41, 2, 10, 83, -101, 52, -8, 82, -88, 51, -96, -97, 11, -91, -71, -40, -82, -17, -39, 46, 58, -16, 30, 21, -35, -88, 80, -16, 85, -79, 50, -55, 10, 65, 75, 51, -13, 62, -84, -69, -63, -60, 83, -67, -59, -69, 2, 15, -61, 5, -23, -3, -8, -62, -100, 20, 24, -2, 63, -2, -26, -18}

#define DEPTHWISE_CONV2D_DEPTHWISE_KERNEL_0_SHIFT (9)

#define DEPTHWISE_CONV2D_BIAS_0 {65, -52, 73, 39, 43, -93, -29, 91, -46, -71, 30, 100, -81, -43, 43, -50}

#define DEPTHWISE_CONV2D_BIAS_0_SHIFT (8)

#define CONV2D_KERNEL_0 {38, 32, 69, 105, -35, -38, 48, 89, -3, -56, 85, 61, -70, 61, -67, -23, 69, -111, -37, -4, -74, -108, -10, 8, 80, 25, -30, 20, 107, 79, -3, -70, -20, 34, -12, 32, 42, -110, -65, 16, -48, -19, -6, -39, 42, 25, 96, 76, 30, 89, -22, 20, 67, 48, 12, 15, -64, -13, -78, 2, 40, 3, -48, -70, -86, -95, -56, 68, -35, -3, 109, -49, 60, -37, -71, -9, -41, -43, -91, 66, -101, 51, -97, 59, -96, -39, 48, 43, -50, 90, 21, 8, 11, -75, -64, -37, -36, -79, 89, 8, -87, 89, -49, 36, -64, -63, -57, -73, 69, -79, -13, 60, 72, -99, -88, -80, 15, -64, 6, -38, 84, 75, 6, 106, -53, -59, -59, 24, -51, 51, -38, 24, 98, 64, 41, 76, 103, -16, 50, -44, -30, -70, -5, 46, 70, -24, 48, 105, -37, 106, -8, -34, 18, 17, 88, 30, -16, -75, 43, -24, -106, 46, 99, 48, 1, -103, 92, -59, 16, -78, 102, 101, 56, -55, -55, -82, 97, -95, -62, 104, -93, -9, 77, -84, -14, 87, -11, 109, -87, 33, 85, -80, -77, 110, -12, 43, 109, -36, 18, -11, -35, 17, 78, 20, 55, 27, -4, -14, -109, 50, -88, 79, -14, 81, 24, 71, 18, -54, 64, 81, 67, 80, -16, -40, -13, 46, -93, -13, 19, 82, -82, -21, -17, 33, -107, 71, -95, -31, 21, 65, 48, -28, 1, -25, 72, -71, -109, -34, -44, -92, 2, -51, 55, 6, -54, 77}

#define CONV2D_KERNEL_0_SHIFT (8)

#define CONV2D_BIAS_0 {32, 62, -5, 33, 78, -103, 73, 46, -4, -39, 65, -103, -90, 93, -65, 108}

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
